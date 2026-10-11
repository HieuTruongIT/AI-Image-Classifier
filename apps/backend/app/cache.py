# apps/backend/app/cache.py
"""
In-memory LRU cache for prediction results.

Caches predictions by image hash to avoid redundant inference
for identical images. Significantly improves latency for repeated requests.

For production at scale, consider Redis-based caching.
"""

from __future__ import annotations
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import threading


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    value: Any
    created_at: float = field(default_factory=time.time)
    hits: int = 0
    last_accessed: float = field(default_factory=time.time)


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache.
    
    Features:
    - Automatic eviction when max size reached
    - TTL (time-to-live) expiration
    - Hit/miss statistics
    - Thread-safe operations
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 3600,  # 1 hour default
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry has expired."""
        return time.time() - entry.created_at > self.ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Returns None if key not found or expired.
        Updates access time and moves to end (most recent).
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if self._is_expired(entry):
                del self._cache[key]
                self._misses += 1
                return None
            
            # Update stats and move to end (most recently used)
            entry.hits += 1
            entry.last_accessed = time.time()
            self._cache.move_to_end(key)
            self._hits += 1
            
            return entry.value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Evicts least recently used entries if at capacity.
        """
        with self._lock:
            # If key exists, update it
            if key in self._cache:
                self._cache[key].value = value
                self._cache[key].created_at = time.time()
                self._cache.move_to_end(key)
                return
            
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove oldest
            
            # Add new entry
            self._cache[key] = CacheEntry(value=value)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache. Returns True if key existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "ttl_seconds": self.ttl_seconds,
            }
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if self._is_expired(entry)
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)


def compute_image_hash(image_bytes: bytes) -> str:
    """
    Compute SHA256 hash of image bytes.
    
    Used as cache key for prediction results.
    """
    return hashlib.sha256(image_bytes).hexdigest()[:32]


# ─────────────────────────────────────────────────────────────
# Singleton Cache Instance
# ─────────────────────────────────────────────────────────────
_prediction_cache: LRUCache | None = None

def get_prediction_cache() -> LRUCache:
    """Get or create singleton prediction cache."""
    global _prediction_cache
    if _prediction_cache is None:
        _prediction_cache = LRUCache(
            max_size=1000,      # Cache up to 1000 predictions
            ttl_seconds=3600,   # 1 hour TTL
        )
    return _prediction_cache
