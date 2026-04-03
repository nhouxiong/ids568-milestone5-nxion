"""
Integration tests using FastAPI's TestClient / AsyncClient.

These tests exercise the full request path (server → batcher → cache)
without loading a real model (inference is mocked at the batcher level).
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

# Patch the model load so tests don't download weights
import src.inference as _inf_module

_inf_module.ModelManager._load_model_sync = lambda self: setattr(self, "is_loaded", True)


async def _fake_generate(requests):
    return [f"mocked response for: {r.prompt[:20]}" for r in requests]


@pytest.fixture()
def app():
    """Return the FastAPI app with mocked model inference."""
    # Import app after patching
    from src import server as srv

    # Swap out inference_fn to avoid real model calls
    srv.batcher._inference_fn = _fake_generate

    # Mark model as loaded so /health returns healthy
    srv.model_manager.is_loaded = True

    return srv.app


@pytest.fixture()
def client(app):
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


# ------------------------------------------------------------------
# Generate endpoint
# ------------------------------------------------------------------


def test_generate_returns_text(client):
    r = client.post(
        "/generate",
        json={"prompt": "What is AI?", "max_tokens": 50, "temperature": 0.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert "text" in body
    assert isinstance(body["text"], str)
    assert len(body["text"]) > 0


def test_generate_first_call_not_cached(client):
    r = client.post(
        "/generate",
        json={"prompt": "Fresh unique prompt xyz123", "max_tokens": 50, "temperature": 0.0},
    )
    assert r.status_code == 200
    # First call is never a cache hit
    assert r.json()["cached"] is False


def test_generate_second_call_is_cached(client):
    payload = {
        "prompt": "Repeated prompt for cache test",
        "max_tokens": 50,
        "temperature": 0.0,
    }
    client.post("/generate", json=payload)  # prime cache
    r2 = client.post("/generate", json=payload)
    assert r2.status_code == 200
    assert r2.json()["cached"] is True


def test_generate_nondeterministic_not_cached(client):
    """temperature > 0 requests should never be served from cache."""
    payload = {
        "prompt": "Random prompt",
        "max_tokens": 50,
        "temperature": 0.9,
    }
    client.post("/generate", json=payload)
    r2 = client.post("/generate", json=payload)
    assert r2.status_code == 200
    assert r2.json()["cached"] is False


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def test_empty_prompt_rejected(client):
    r = client.post(
        "/generate",
        json={"prompt": "", "max_tokens": 50},
    )
    assert r.status_code == 422


def test_negative_max_tokens_rejected(client):
    r = client.post(
        "/generate",
        json={"prompt": "hello", "max_tokens": -1},
    )
    assert r.status_code == 422


# ------------------------------------------------------------------
# Metrics endpoint
# ------------------------------------------------------------------


def test_metrics_returns_cache_and_batcher_stats(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "cache" in body
    assert "batcher" in body


# ------------------------------------------------------------------
# Concurrency
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_requests(app):
    """Fire multiple concurrent requests and verify all succeed."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        tasks = [
            ac.post(
                "/generate",
                json={"prompt": f"concurrent prompt {i}", "max_tokens": 30},
            )
            for i in range(10)
        ]
        responses = await asyncio.gather(*tasks)

    for resp in responses:
        assert resp.status_code == 200
        assert "text" in resp.json()
