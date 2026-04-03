# LLM Inference Server

High-throughput LLM inference API with **dynamic request batching** and **intelligent response caching**.

## Features

- **Dynamic batching** — groups concurrent requests (hybrid: batch-size OR timeout trigger)
- **Response caching** — in-memory (default) or Redis, with configurable TTL and max-entry limits
- **Privacy-preserving** — cache keys are SHA-256 hashes; no raw prompts or user identifiers stored
- **Reproducible benchmarks** — full suite covering latency, throughput, and cache hit-rate
- **FastAPI** — async, typed, OpenAPI docs at `/docs`

## Quick Start

### Prerequisites

- Python 3.10+
- Redis (optional — in-memory cache used by default)
- GPU recommended (CPU works with `gpt2`)

### Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configure (optional)

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_NAME` | `gpt2` | HuggingFace model ID |
| `LLM_MAX_BATCH_SIZE` | `8` | Max requests per batch |
| `LLM_BATCH_TIMEOUT_MS` | `50` | Batch flush timeout (ms) |
| `LLM_CACHE_TTL_SECONDS` | `3600` | Cache entry TTL |
| `LLM_CACHE_MAX_ENTRIES` | `10000` | Max in-memory cache entries |
| `LLM_REDIS_URL` | `redis://localhost:6379` | Redis URL (leave default to use in-memory) |

## Running the Server

```bash
# Development (auto-reload)
uvicorn src.server:app --reload --port 8000

# Production
uvicorn src.server:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Note:** Use `--workers 1` — the batcher maintains shared in-process state. For multi-worker deployments, switch to Redis and use an external job queue.

## API Usage

### Generate text

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain gradient descent.", "max_tokens": 150, "temperature": 0.0}'
```

Response:
```json
{
  "text": "Gradient descent is an optimization algorithm...",
  "cached": false
}
```

### Health check

```bash
curl http://localhost:8000/health
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

Returns live cache hit-rate, batch statistics, and memory usage.

## Running Benchmarks

Start the server first, then in a second terminal:

```bash
# Run all benchmark experiments
python benchmarks/run_benchmarks.py

# Run a single experiment
python benchmarks/run_benchmarks.py --experiment cache

# Custom load test
python benchmarks/load_generator.py \
  --requests 100 \
  --concurrency 16 \
  --repeat-fraction 0.4 \
  --output benchmarks/results/custom.json
```

Available experiments: `baseline`, `batching`, `cache`, `throughput`, `cache_hitrate`

Results are saved to `benchmarks/results/` as JSON.

### Generate analysis report

After benchmarks have run:

```bash
python analysis/generate_reports.py
```

This produces `analysis/performance_report.pdf` and `analysis/governance_memo.pdf`.

## Running Tests

```bash
pytest tests/ -v
```

Tests use mock inference — no model download required.

## Project Structure

```
ids568-milestone5-nxion/
├── src/
│   ├── server.py       # FastAPI app, routes, lifespan
│   ├── batching.py     # DynamicBatcher (hybrid batch-size + timeout)
│   ├── caching.py      # LLMCache (in-memory or Redis)
│   ├── inference.py    # ModelManager (HuggingFace transformers)
│   ├── config.py       # Pydantic settings (env var driven)
│   └── models.py       # Request/response Pydantic models
├── benchmarks/
│   ├── run_benchmarks.py   # Orchestrates all experiments
│   ├── load_generator.py   # Concurrent HTTP load generator
│   └── results/            # JSON output from benchmark runs
├── analysis/
│   ├── generate_reports.py     # Builds PDFs from benchmark results
│   ├── performance_report.pdf  # (generated after benchmarks)
│   ├── governance_memo.pdf     # (generated after benchmarks)
│   └── visualizations/         # Charts (PNG) from benchmark data
├── tests/
│   ├── test_batching.py    # Unit tests for DynamicBatcher
│   ├── test_caching.py     # Unit tests for LLMCache
│   └── test_integration.py # End-to-end FastAPI tests
├── requirements.txt
├── .env.example
└── README.md
```

## License

MIT
