"""
Tests for the dynamic batching logic.

Uses a mock inference function so no real model is needed.
"""
import asyncio
import time
import pytest

from src.batching import DynamicBatcher
from src.models import InferenceRequest


def _req(prompt: str) -> InferenceRequest:
    return InferenceRequest(prompt=prompt, max_tokens=50, temperature=0.0)


async def _mock_inference(requests):
    """Instant mock inference — returns deterministic strings."""
    await asyncio.sleep(0.01)
    return [f"mock: {r.prompt}" for r in requests]


# ------------------------------------------------------------------
# Basic submission
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_request_returns_result():
    batcher = DynamicBatcher(max_batch_size=8, max_wait_ms=100, inference_fn=_mock_inference)
    await batcher.start()
    try:
        result = await batcher.submit(_req("hello"))
        assert "hello" in result
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_multiple_requests_all_resolved():
    batcher = DynamicBatcher(max_batch_size=8, max_wait_ms=100, inference_fn=_mock_inference)
    await batcher.start()
    try:
        tasks = [
            asyncio.create_task(batcher.submit(_req(f"prompt {i}")))
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        for i, r in enumerate(results):
            assert f"prompt {i}" in r
    finally:
        await batcher.stop()


# ------------------------------------------------------------------
# Batching behaviour
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_size_triggers_early_flush():
    """When max_batch_size requests arrive at once the batch fires before timeout."""
    processed: list = []

    async def tracking_inference(requests):
        processed.append(len(requests))
        return [f"r{i}" for i in range(len(requests))]

    batcher = DynamicBatcher(
        max_batch_size=4,
        max_wait_ms=5000,  # very long timeout — batch-size trigger fires first
        inference_fn=tracking_inference,
    )
    await batcher.start()
    try:
        tasks = [
            asyncio.create_task(batcher.submit(_req(f"p{i}")))
            for i in range(4)
        ]
        await asyncio.gather(*tasks)
        # The batch should have been processed as a single group of 4
        assert any(s == 4 for s in processed)
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_timeout_triggers_partial_batch():
    """A partial batch should be flushed when the timeout expires."""
    batcher = DynamicBatcher(
        max_batch_size=100,
        max_wait_ms=50,  # short timeout
        inference_fn=_mock_inference,
    )
    await batcher.start()
    try:
        t0 = time.perf_counter()
        result = await batcher.submit(_req("lonely request"))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert result is not None
        # Should have been flushed within ~3× the timeout
        assert elapsed_ms < 200, f"Expected flush within 200 ms, got {elapsed_ms:.0f} ms"
    finally:
        await batcher.stop()


# ------------------------------------------------------------------
# Concurrency / race conditions
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_requests_no_race_condition():
    """50 concurrent requests should all resolve without error."""
    batcher = DynamicBatcher(max_batch_size=8, max_wait_ms=50, inference_fn=_mock_inference)
    await batcher.start()
    try:
        tasks = [
            asyncio.create_task(batcher.submit(_req(f"concurrent {i}")))
            for i in range(50)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 50
        assert all(isinstance(r, str) for r in results)
    finally:
        await batcher.stop()


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_accumulate():
    batcher = DynamicBatcher(max_batch_size=4, max_wait_ms=100, inference_fn=_mock_inference)
    await batcher.start()
    try:
        tasks = [asyncio.create_task(batcher.submit(_req(f"s{i}"))) for i in range(8)]
        await asyncio.gather(*tasks)

        stats = batcher.get_stats()
        assert stats["total_requests"] == 8
        assert stats["batches_processed"] >= 1
        assert stats["avg_batch_size"] > 0
    finally:
        await batcher.stop()


# ------------------------------------------------------------------
# Placeholder fallback
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_placeholder_inference_when_no_fn():
    """Batcher with no inference_fn should still return a placeholder string."""
    batcher = DynamicBatcher(max_batch_size=4, max_wait_ms=100)
    await batcher.start()
    try:
        result = await batcher.submit(_req("test"))
        assert isinstance(result, str)
        assert len(result) > 0
    finally:
        await batcher.stop()
