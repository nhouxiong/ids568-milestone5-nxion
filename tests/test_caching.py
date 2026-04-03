"""
Tests for the LLM caching layer.
"""
import pytest
from src.caching import LLMCache, InMemoryBackend
from src.models import InferenceRequest


# ------------------------------------------------------------------
# InMemoryBackend
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inmemory_get_miss():
    backend = InMemoryBackend(max_entries=10)
    result = await backend.get("nonexistent")
    assert result is None
    assert backend.misses == 1


@pytest.mark.asyncio
async def test_inmemory_set_and_get():
    backend = InMemoryBackend(max_entries=10)
    await backend.set("key1", "hello world", ttl=60)
    result = await backend.get("key1")
    assert result == "hello world"
    assert backend.hits == 1


@pytest.mark.asyncio
async def test_inmemory_eviction():
    """When max_entries is reached the oldest entry is evicted."""
    backend = InMemoryBackend(max_entries=3)
    for i in range(4):
        await backend.set(f"key{i}", f"val{i}", ttl=60)
    # First key should have been evicted
    assert await backend.get("key0") is None
    # Later keys should still be present
    assert await backend.get("key3") == "val3"


@pytest.mark.asyncio
async def test_inmemory_stats():
    backend = InMemoryBackend(max_entries=10)
    await backend.set("k", "v", ttl=60)
    await backend.get("k")      # hit
    await backend.get("miss")   # miss
    stats = backend.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


# ------------------------------------------------------------------
# LLMCache
# ------------------------------------------------------------------


def _req(prompt: str, temperature: float = 0.0, max_tokens: int = 100) -> InferenceRequest:
    return InferenceRequest(prompt=prompt, temperature=temperature, max_tokens=max_tokens)


@pytest.mark.asyncio
async def test_cache_miss_on_cold():
    cache = LLMCache()
    result = await cache.get(_req("What is AI?"))
    assert result is None


@pytest.mark.asyncio
async def test_cache_set_and_get():
    cache = LLMCache()
    req = _req("What is AI?")
    await cache.set(req, "AI is artificial intelligence.")
    result = await cache.get(req)
    assert result == "AI is artificial intelligence."


@pytest.mark.asyncio
async def test_cache_skips_nondeterministic():
    """Requests with temperature > 0 should never be cached."""
    cache = LLMCache()
    req = _req("Hello", temperature=0.7)
    await cache.set(req, "some response")
    assert await cache.get(req) is None


@pytest.mark.asyncio
async def test_cache_key_ignores_whitespace():
    """Leading/trailing whitespace in prompt should yield the same cache key."""
    cache = LLMCache()
    req1 = _req("  What is AI?  ")
    req2 = _req("What is AI?")
    await cache.set(req1, "AI is cool.")
    assert await cache.get(req2) == "AI is cool."


@pytest.mark.asyncio
async def test_cache_key_is_case_insensitive():
    """Prompts differing only in case map to the same key."""
    cache = LLMCache()
    await cache.set(_req("What is AI?"), "response")
    assert await cache.get(_req("WHAT IS AI?")) == "response"


@pytest.mark.asyncio
async def test_cache_stats():
    cache = LLMCache()
    req = _req("Hello")
    await cache.get(req)          # miss
    await cache.set(req, "Hi!")
    await cache.get(req)          # hit
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


@pytest.mark.asyncio
async def test_cache_key_is_hashed():
    """Cache keys must not contain the raw prompt text."""
    cache = LLMCache()
    req = _req("super secret user data")
    key = cache._make_key(req)
    assert "super secret user data" not in key
    assert key.startswith("llm:")
