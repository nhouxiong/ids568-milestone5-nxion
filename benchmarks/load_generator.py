"""
Synthetic load generator for the LLM inference server.

Sends concurrent HTTP requests to measure throughput and latency
under controlled conditions.

Usage::

    python benchmarks/load_generator.py --url http://localhost:8000 \
        --requests 100 --concurrency 10
"""
import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path

import httpx

# ------------------------------------------------------------------
# Prompt corpus — mix of short, medium, and repeated prompts
# ------------------------------------------------------------------

UNIQUE_PROMPTS = [
    "Explain gradient descent in one paragraph.",
    "What is a transformer model in deep learning?",
    "Describe the differences between CPU and GPU computing.",
    "What is backpropagation and why does it matter?",
    "How does attention mechanism work in neural networks?",
    "Define overfitting and how to prevent it.",
    "What is transfer learning?",
    "Explain the bias-variance tradeoff.",
    "What is regularization in machine learning?",
    "Describe the architecture of a convolutional neural network.",
    "What is a recurrent neural network?",
    "How does dropout work as a regularization technique?",
    "What is batch normalization?",
    "Explain the concept of embeddings.",
    "What is the softmax function?",
    "Describe the encoder-decoder architecture.",
    "What is a variational autoencoder?",
    "How does BERT differ from GPT?",
    "What is few-shot learning?",
    "Explain knowledge distillation.",
    "What is quantization in the context of neural networks?",
    "How do language models generate text?",
    "What is perplexity as an evaluation metric?",
    "Explain the role of the key-query-value mechanism.",
    "What is positional encoding?",
    "How does beam search work?",
    "What is temperature sampling?",
    "Describe the concept of a token in NLP.",
    "What is fine-tuning a pre-trained model?",
    "Explain the difference between supervised and unsupervised learning.",
]

REPEATED_PROMPTS = [
    "What is machine learning?",
    "Define artificial intelligence.",
    "What is a neural network?",
    "Explain deep learning briefly.",
    "What is Python used for?",
]


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class RequestResult:
    prompt: str
    latency_ms: float
    cached: bool
    success: bool
    status_code: Optional[int] = None
    error: Optional[str] = None


@dataclass
class LoadTestSummary:
    concurrency: int
    total_requests: int
    successful: int
    failed: int
    total_elapsed_s: float
    throughput_rps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    cache_hits: int
    cache_hit_rate: float


# ------------------------------------------------------------------
# Core load logic
# ------------------------------------------------------------------


async def _send_one(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> RequestResult:
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        data = resp.json()
        return RequestResult(
            prompt=prompt,
            latency_ms=latency_ms,
            cached=data.get("cached", False),
            success=resp.status_code == 200,
            status_code=resp.status_code,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(
            prompt=prompt,
            latency_ms=latency_ms,
            cached=False,
            success=False,
            error=str(exc),
        )


async def run_load_test(
    url: str,
    prompts: List[str],
    concurrency: int,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> tuple[List[RequestResult], float]:
    """
    Fire *len(prompts)* requests at *concurrency* at a time.

    Returns (results, total_elapsed_seconds).
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(prompt: str) -> RequestResult:
        async with semaphore:
            async with httpx.AsyncClient() as client:
                return await _send_one(client, url, prompt, max_tokens, temperature)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[bounded(p) for p in prompts])
    elapsed = time.perf_counter() - t0
    return list(results), elapsed


def summarise(
    results: List[RequestResult],
    elapsed: float,
    concurrency: int,
) -> LoadTestSummary:
    import numpy as np

    latencies = [r.latency_ms for r in results if r.success]
    hits = sum(1 for r in results if r.cached)

    return LoadTestSummary(
        concurrency=concurrency,
        total_requests=len(results),
        successful=len(latencies),
        failed=len(results) - len(latencies),
        total_elapsed_s=round(elapsed, 3),
        throughput_rps=round(len(latencies) / elapsed, 2) if elapsed > 0 else 0,
        latency_p50_ms=round(float(np.percentile(latencies, 50)), 2) if latencies else 0,
        latency_p95_ms=round(float(np.percentile(latencies, 95)), 2) if latencies else 0,
        latency_p99_ms=round(float(np.percentile(latencies, 99)), 2) if latencies else 0,
        latency_mean_ms=round(float(np.mean(latencies)), 2) if latencies else 0,
        cache_hits=hits,
        cache_hit_rate=round(hits / len(results), 4) if results else 0,
    )


def build_prompt_list(
    n: int,
    repeat_fraction: float = 0.3,
) -> List[str]:
    """
    Build a list of *n* prompts where ~*repeat_fraction* are repeated
    (to exercise the cache).
    """
    n_repeat = int(n * repeat_fraction)
    n_unique = n - n_repeat
    unique = [
        random.choice(UNIQUE_PROMPTS) for _ in range(n_unique)
    ]
    repeated = [
        random.choice(REPEATED_PROMPTS) for _ in range(n_repeat)
    ]
    combined = unique + repeated
    random.shuffle(combined)
    return combined


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------


async def _main(args: argparse.Namespace) -> None:
    prompts = build_prompt_list(args.requests, repeat_fraction=args.repeat_fraction)
    print(
        f"Running {args.requests} requests "
        f"(concurrency={args.concurrency}, repeat_fraction={args.repeat_fraction}) "
        f"against {args.url} …"
    )

    results, elapsed = await run_load_test(
        url=args.url,
        prompts=prompts,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    summary = summarise(results, elapsed, args.concurrency)

    print(json.dumps(asdict(summary), indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"Results saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM inference load generator")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--repeat-fraction",
        type=float,
        default=0.3,
        help="Fraction of requests that use repeated prompts (to test cache)",
    )
    parser.add_argument("--output", help="Path to save JSON results")
    asyncio.run(_main(parser.parse_args()))
