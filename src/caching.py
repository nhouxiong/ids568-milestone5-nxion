"""
Inference caching with Redis or in-memory fallback.

Caches deterministic (temperature=0) responses to avoid
redundant computation.
"""
import hashlib
import json
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from .models import InferenceRequest

class CacheBackend(ABC):
    """Abstract cache backend interface."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: str, ttl: int) -> None:
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass

class InMemoryBackend(CacheBackend):
    """Simple in-memory cache for development."""
    
    def __init__(self, max_entries: int = 1000):
        self.cache: Dict[str, str] = {}
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
    
    async def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    async def set(self, key: str, value: str, ttl: int) -> None:
        if len(self.cache) >= self.max_entries:
            # Simple eviction: remove first item
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = value
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
            "total_entries": len(self.cache)
        }

class RedisBackend(CacheBackend):
    """Redis-based cache for production."""
    
    def __init__(self, redis_url: str):
        import redis.asyncio as redis
        self.redis = redis.from_url(redis_url)
        self.hits = 0
        self.misses = 0
    
    async def get(self, key: str) -> Optional[str]:
        value = await self.redis.get(key)
        if value:
            self.hits += 1
            return value.decode()
        self.misses += 1
        return None
    
    async def set(self, key: str, value: str, ttl: int) -> None:
        await self.redis.setex(key, ttl, value)
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
            "total_entries": "N/A (use Redis INFO)"
        }

class LLMCache:
    """
    High-level caching interface for LLM responses.
    
    Only caches deterministic requests (temperature=0).
    Uses hashed keys for privacy.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        max_entries: int = 10000
    ):
        if redis_url and redis_url != "redis://localhost:6379":
            self.backend = RedisBackend(redis_url)
        else:
            self.backend = InMemoryBackend(max_entries)
        
        self.default_ttl = default_ttl
    
    def _make_key(self, request: InferenceRequest) -> str:
        """Create privacy-preserving cache key."""
        key_data = {
            "prompt_hash": hashlib.sha256(
                request.prompt.strip().lower().encode()
            ).hexdigest(),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        content = json.dumps(key_data, sort_keys=True)
        return f"llm:{hashlib.sha256(content.encode()).hexdigest()[:32]}"
    
    async def get(self, request: InferenceRequest) -> Optional[str]:
        """Get cached response if available."""
        # Only cache deterministic requests
        if request.temperature > 0:
            return None
        
        key = self._make_key(request)
        return await self.backend.get(key)
    
    async def set(
        self,
        request: InferenceRequest,
        response: str,
        ttl: Optional[int] = None
    ) -> None:
        """Cache response."""
        if request.temperature > 0:
            return
        
        key = self._make_key(request)
        await self.backend.set(key, response, ttl or self.default_ttl)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.backend.get_stats()