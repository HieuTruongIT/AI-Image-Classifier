# apps/backend/app/auth.py
"""
API Key Authentication with usage tracking.

Provides simple token-based authentication for production APIs:
- API key validation via header or query param
- Per-key usage tracking (requests, last used)
- Optional key-based rate limiting
- Admin endpoints for key management

For production at scale, use a proper auth service (Auth0, Keycloak, etc.)
or a database-backed key store (Redis, PostgreSQL).
"""

from __future__ import annotations
import hashlib
import os
import time
import secrets
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from fastapi import HTTPException, Header, Query, Depends
from fastapi.security import APIKeyHeader
import threading


@dataclass
class APIKey:
    """API key with metadata and usage tracking."""
    key_hash: str  # Store hash, not plaintext
    name: str
    created_at: float = field(default_factory=time.time)
    last_used_at: Optional[float] = None
    request_count: int = 0
    is_active: bool = True
    rate_limit_rpm: Optional[int] = None  # Requests per minute (None = unlimited)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "request_count": self.request_count,
            "is_active": self.is_active,
            "rate_limit_rpm": self.rate_limit_rpm,
        }


class APIKeyManager:
    """
    In-memory API key manager.
    
    For production, replace with:
    - Redis for distributed key storage
    - PostgreSQL/MySQL for persistent keys
    - Auth0/Keycloak for enterprise auth
    """
    
    def __init__(self):
        self._keys: Dict[str, APIKey] = {}  # key_hash -> APIKey
        self._lock = threading.RLock()
        
        # Load default API key from environment if set
        default_key = os.getenv("API_KEY")
        if default_key:
            self._add_key_internal(default_key, "default", is_master=True)
        
        # Load master key for admin operations
        master_key = os.getenv("MASTER_API_KEY")
        if master_key:
            self._add_key_internal(master_key, "master", is_master=True)
    
    def _hash_key(self, key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _add_key_internal(self, key: str, name: str, is_master: bool = False) -> str:
        """Internal method to add a key."""
        key_hash = self._hash_key(key)
        self._keys[key_hash] = APIKey(
            key_hash=key_hash,
            name=name,
            rate_limit_rpm=None if is_master else 60,
        )
        return key_hash
    
    def create_key(self, name: str, rate_limit_rpm: Optional[int] = 60) -> str:
        """
        Create a new API key.
        
        Returns the plaintext key (only shown once).
        """
        with self._lock:
            # Generate a secure random key
            key = f"vs_{secrets.token_urlsafe(32)}"
            key_hash = self._hash_key(key)
            
            self._keys[key_hash] = APIKey(
                key_hash=key_hash,
                name=name,
                rate_limit_rpm=rate_limit_rpm,
            )
            
            return key
    
    def validate_key(self, key: str) -> Optional[APIKey]:
        """
        Validate an API key and update usage stats.
        
        Returns the APIKey if valid, None otherwise.
        """
        with self._lock:
            key_hash = self._hash_key(key)
            api_key = self._keys.get(key_hash)
            
            if api_key is None:
                return None
            
            if not api_key.is_active:
                return None
            
            # Update usage stats
            api_key.last_used_at = time.time()
            api_key.request_count += 1
            
            return api_key
    
    def revoke_key(self, key_hash: str) -> bool:
        """Revoke an API key by its hash."""
        with self._lock:
            if key_hash in self._keys:
                self._keys[key_hash].is_active = False
                return True
            return False
    
    def delete_key(self, key_hash: str) -> bool:
        """Delete an API key by its hash."""
        with self._lock:
            if key_hash in self._keys:
                del self._keys[key_hash]
                return True
            return False
    
    def list_keys(self) -> List[Dict]:
        """List all API keys (without the actual key values)."""
        with self._lock:
            return [
                {"key_hash": k[:16] + "...", **v.to_dict()}
                for k, v in self._keys.items()
            ]
    
    def get_stats(self) -> Dict:
        """Get API key statistics."""
        with self._lock:
            active = sum(1 for k in self._keys.values() if k.is_active)
            total_requests = sum(k.request_count for k in self._keys.values())
            return {
                "total_keys": len(self._keys),
                "active_keys": active,
                "total_requests": total_requests,
            }


# ─────────────────────────────────────────────────────────────
# Singleton Instance
# ─────────────────────────────────────────────────────────────
_api_key_manager: Optional[APIKeyManager] = None

def get_api_key_manager() -> APIKeyManager:
    """Get or create singleton API key manager."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


# ─────────────────────────────────────────────────────────────
# FastAPI Dependencies
# ─────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Check if auth is enabled
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"


async def get_api_key(
    api_key_header: Optional[str] = Depends(api_key_header),
    api_key_query: Optional[str] = Query(None, alias="api_key"),
) -> Optional[APIKey]:
    """
    Dependency to validate API key from header or query param.
    
    Usage in endpoints:
        @app.get("/protected")
        def protected_endpoint(api_key: APIKey = Depends(require_api_key)):
            ...
    """
    if not AUTH_ENABLED:
        return None  # Auth disabled, allow all
    
    # Try header first, then query param
    key = api_key_header or api_key_query
    
    if not key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide via X-API-Key header or api_key query param.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    api_key = get_api_key_manager().validate_key(key)
    
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or revoked API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key


async def require_api_key(api_key: Optional[APIKey] = Depends(get_api_key)) -> APIKey:
    """
    Dependency that requires a valid API key (even if auth is disabled).
    Use for admin endpoints that always need authentication.
    """
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key required for this endpoint.",
        )
    return api_key


async def optional_api_key(api_key: Optional[APIKey] = Depends(get_api_key)) -> Optional[APIKey]:
    """
    Dependency that optionally validates API key.
    Allows unauthenticated access but tracks usage if key is provided.
    """
    return api_key
