"""
Benchmark orchestration for LLM inference server.

Runs a structured suite of experiments and saves results to
benchmarks/results/. Requires the server to be running at --url.

Usage::

    # Start server first:
    uvicorn src.server:app --port 8000

    # Then run all benchmarks:
    python benchmarks/run_benchmarks.py

    # Run a specific experiment only:
    python benchmarks/run_benchmarks.py --experiment cache

Available experiments: baseline, batching, cache, throughput, cache_hitrate
"""
import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List

import httpx
import numpy as np

from load_generator import (
    run_load_test,
    summarise,
    build_prompt_list,
    UNIQUE_PROMPTS,
    REPEATED_PROMPTS,
)

RESULTS_DIR = Path(__file__).parent / "results"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def save(name: str, data: Any) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"  Saved: {path}")
    return path


async def wait_for_server(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            try:
                r = await client.get(f"{url}/health", timeout=2.0)
                if r.status_code == 200:
                    print(f"  Server ready at {url}")
                    return
            except Exception:
                pass
            await asyncio.sleep(1)
    raise RuntimeError(f"Server at {url} did not become ready within {timeout}s")


async def single_request(
    url: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Fire one request and return timing + metadata."""
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        r = await client.post(
            f"{url}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
            timeout=120.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        data = r.json()
        return {
            "latency_ms": round(latency_ms, 2),
            "cached": data.get("cached", False),
            "status": r.status_code,
        }


# ------------------------------------------------------------------
# Experiment 1 — Baseline (single requests, no concurrency)
# ------------------------------------------------------------------


async def exp_baseline(url: str, n: int = 10) -> Dict:
    print("\n[1/5] Baseline: single-request latency (no batching/cache)")
    prompts = UNIQUE_PROMPTS[:n]
    latencies = []
    for i, p in enumerate(prompts):
        result = await single_request(url, p)
        latencies.append(result["latency_ms"])
        print(f"  req {i+1:2d}: {result['latency_ms']:.0f} ms")

    data = {
        "experiment": "baseline",
        "n_requests": n,
        "latency_ms": {
            "mean": round(np.mean(latencies), 2),
            "p50": round(np.percentile(latencies, 50), 2),
            "p95": round(np.percentile(latencies, 95), 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
        "raw": latencies,
    }
    save("baseline", data)
    return data


# ------------------------------------------------------------------
# Experiment 2 — Batching: latency at different concurrency levels
# ------------------------------------------------------------------


async def exp_batching(url: str) -> Dict:
    print("\n[2/5] Batching: throughput & latency at varying concurrency")
    concurrency_levels = [1, 2, 4, 8, 16]
    n_requests = 40
    rows = []

    for c in concurrency_levels:
        prompts = build_prompt_list(n_requests, repeat_fraction=0.0)
        results, elapsed = await run_load_test(url, prompts, concurrency=c)
        summary = summarise(results, elapsed, c)
        row = asdict(summary)
        rows.append(row)
        print(
            f"  concurrency={c:2d}: "
            f"throughput={summary.throughput_rps:.1f} rps  "
            f"p50={summary.latency_p50_ms:.0f}ms  "
            f"p95={summary.latency_p95_ms:.0f}ms"
        )

    data = {"experiment": "batching", "rows": rows}
    save("batching", data)
    return data


# ------------------------------------------------------------------
# Experiment 3 — Cache: cold vs warm
# ------------------------------------------------------------------


async def exp_cache(url: str, n: int = 20) -> Dict:
    print("\n[3/5] Cache: cold-cache vs warm-cache latency")

    # Use a fixed set of prompts so second pass is all cache hits
    prompts = REPEATED_PROMPTS * (n // len(REPEATED_PROMPTS) + 1)
    prompts = prompts[:n]

    # Cold pass
    cold_latencies = []
    for p in prompts:
        result = await single_request(url, p)
        cold_latencies.append(result["latency_ms"])

    # Warm pass (all should be cache hits)
    warm_latencies = []
    warm_cached = []
    for p in prompts:
        result = await single_request(url, p)
        warm_latencies.append(result["latency_ms"])
        warm_cached.append(result["cached"])

    hit_rate = sum(warm_cached) / len(warm_cached)

    data = {
        "experiment": "cache",
        "n_requests": n,
        "cold": {
            "mean_ms": round(np.mean(cold_latencies), 2),
            "p50_ms": round(np.percentile(cold_latencies, 50), 2),
            "p95_ms": round(np.percentile(cold_latencies, 95), 2),
        },
        "warm": {
            "mean_ms": round(np.mean(warm_latencies), 2),
            "p50_ms": round(np.percentile(warm_latencies, 50), 2),
            "p95_ms": round(np.percentile(warm_latencies, 95), 2),
            "hit_rate": round(hit_rate, 4),
        },
        "speedup_x": round(np.mean(cold_latencies) / max(np.mean(warm_latencies), 0.001), 1),
    }

    print(
        f"  Cold mean: {data['cold']['mean_ms']:.0f} ms | "
        f"Warm mean: {data['warm']['mean_ms']:.0f} ms | "
        f"Speedup: {data['speedup_x']}x | "
        f"Hit rate: {hit_rate:.1%}"
    )
    save("cache", data)
    return data


# ------------------------------------------------------------------
# Experiment 4 — Throughput sweep
# ------------------------------------------------------------------


async def exp_throughput(url: str) -> Dict:
    print("\n[4/5] Throughput: requests/sec at multiple load levels")
    configs = [
        (20, 1),
        (40, 4),
        (80, 8),
        (80, 16),
        (100, 32),
    ]
    rows = []
    for n, c in configs:
        prompts = build_prompt_list(n, repeat_fraction=0.3)
        results, elapsed = await run_load_test(url, prompts, concurrency=c)
        summary = summarise(results, elapsed, c)
        rows.append(asdict(summary))
        print(
            f"  n={n:3d} concurrency={c:2d}: "
            f"rps={summary.throughput_rps:.1f}  "
            f"mean={summary.latency_mean_ms:.0f}ms"
        )

    data = {"experiment": "throughput", "rows": rows}
    save("throughput", data)
    return data


# ------------------------------------------------------------------
# Experiment 5 — Cache hit-rate over time
# ------------------------------------------------------------------


async def exp_cache_hitrate(url: str, n: int = 80) -> Dict:
    print("\n[5/5] Cache hit-rate: accumulation over time")
    # Mix: 40% repeated so hit rate builds up
    prompts = build_prompt_list(n, repeat_fraction=0.4)
    running_hits = 0
    timeline: List[Dict] = []

    for i, prompt in enumerate(prompts):
        result = await single_request(url, prompt)
        if result["cached"]:
            running_hits += 1
        timeline.append(
            {
                "request_n": i + 1,
                "hit": result["cached"],
                "cumulative_hit_rate": round(running_hits / (i + 1), 4),
                "latency_ms": result["latency_ms"],
            }
        )

    data = {
        "experiment": "cache_hitrate",
        "n_requests": n,
        "final_hit_rate": timeline[-1]["cumulative_hit_rate"] if timeline else 0,
        "timeline": timeline,
    }
    print(f"  Final cumulative hit-rate: {data['final_hit_rate']:.1%}")
    save("cache_hitrate", data)
    return data


# ------------------------------------------------------------------
# Metrics snapshot
# ------------------------------------------------------------------


async def snapshot_metrics(url: str) -> None:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{url}/metrics", timeout=5.0)
            print("\n[Metrics snapshot]")
            print(json.dumps(r.json(), indent=2))
            save("metrics_snapshot", r.json())
        except Exception as e:
            print(f"  Could not fetch metrics: {e}")


# ------------------------------------------------------------------
# Entry-point
# ------------------------------------------------------------------

EXPERIMENTS = {
    "baseline": exp_baseline,
    "batching": exp_batching,
    "cache": exp_cache,
    "throughput": exp_throughput,
    "cache_hitrate": exp_cache_hitrate,
}


async def _main(args: argparse.Namespace) -> None:
    await wait_for_server(args.url)

    selected = (
        [args.experiment] if args.experiment else list(EXPERIMENTS.keys())
    )

    all_results: Dict[str, Any] = {}
    for name in selected:
        fn = EXPERIMENTS[name]
        result = await fn(args.url)
        all_results[name] = result

    await snapshot_metrics(args.url)
    save("all_results", all_results)
    print("\nBenchmarks complete. Results in benchmarks/results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM inference benchmarks")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()),
        help="Run a single experiment (default: all)",
    )
    asyncio.run(_main(parser.parse_args()))
