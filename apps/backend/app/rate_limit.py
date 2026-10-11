# apps/backend/app/rate_limit.py
"""
Simple in-memory rate limiting middleware.
For production, consider using Redis-based rate limiting.
"""

from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

from fastapi import HTTPException, Request


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 500
    burst_limit: int = 10  # Max requests per second


@dataclass
class ClientBucket:
    """Token bucket for a single client."""
    minute_tokens: int = 60
    hour_tokens: int = 500
    last_minute_reset: float = field(default_factory=time.time)
    last_hour_reset: float = field(default_factory=time.time)
    last_request: float = field(default_factory=time.time)
    burst_count: int = 0


class RateLimiter:
    """
    In-memory rate limiter using token bucket algorithm.
    Tracks requests per client IP with minute and hour limits.
    """
    
    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self.buckets: Dict[str, ClientBucket] = defaultdict(
            lambda: ClientBucket(
                minute_tokens=self.config.requests_per_minute,
                hour_tokens=self.config.requests_per_hour,
            )
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers (reverse proxy)
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def check_rate_limit(self, request: Request) -> Dict[str, int]:
        """
        Check and update rate limit for a request.
        
        Returns:
            Dict with remaining limits
            
        Raises:
            HTTPException(429) if rate limit exceeded
        """
        client_ip = self._get_client_ip(request)
        bucket = self.buckets[client_ip]
        now = time.time()
        
        # Reset minute bucket if needed
        if now - bucket.last_minute_reset >= 60:
            bucket.minute_tokens = self.config.requests_per_minute
            bucket.last_minute_reset = now
        
        # Reset hour bucket if needed
        if now - bucket.last_hour_reset >= 3600:
            bucket.hour_tokens = self.config.requests_per_hour
            bucket.last_hour_reset = now
        
        # Check burst limit (requests per second)
        if now - bucket.last_request < 1:
            bucket.burst_count += 1
            if bucket.burst_count > self.config.burst_limit:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "type": "burst",
                        "message": f"Max {self.config.burst_limit} requests per second",
                        "retry_after": 1,
                    },
                )
        else:
            bucket.burst_count = 1
        
        bucket.last_request = now
        
        # Check minute limit
        if bucket.minute_tokens <= 0:
            retry_after = int(60 - (now - bucket.last_minute_reset))
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "type": "minute",
                    "message": f"Max {self.config.requests_per_minute} requests per minute",
                    "retry_after": max(1, retry_after),
                },
            )
        
        # Check hour limit
        if bucket.hour_tokens <= 0:
            retry_after = int(3600 - (now - bucket.last_hour_reset))
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "type": "hour",
                    "message": f"Max {self.config.requests_per_hour} requests per hour",
                    "retry_after": max(1, retry_after),
                },
            )
        
        # Consume tokens
        bucket.minute_tokens -= 1
        bucket.hour_tokens -= 1
        
        return {
            "remaining_minute": bucket.minute_tokens,
            "remaining_hour": bucket.hour_tokens,
            "limit_minute": self.config.requests_per_minute,
            "limit_hour": self.config.requests_per_hour,
        }
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        return {
            "active_clients": len(self.buckets),
            "config": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "burst_limit": self.config.burst_limit,
            },
        }


# Singleton instance
_rate_limiter: RateLimiter | None = None

def get_rate_limiter() -> RateLimiter:
    """Get or create singleton rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
