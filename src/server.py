"""
LLM Inference Server with Batching and Caching

Entrypoint for the FastAPI application. Initializes middleware,
routes, and lifecycle hooks for background tasks.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models import InferenceRequest, InferenceResponse
from .batching import DynamicBatcher
from .caching import LLMCache
from .inference import ModelManager

# Initialize components
model_manager = ModelManager(settings.model_name)
cache = LLMCache(
    redis_url=settings.redis_url,
    default_ttl=settings.cache_ttl_seconds
)
batcher = DynamicBatcher(
    max_batch_size=settings.max_batch_size,
    max_wait_ms=settings.batch_timeout_ms,
    inference_fn=model_manager.generate_batch,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown."""
    await model_manager.load()
    await batcher.start()
    yield
    await batcher.stop()

app = FastAPI(
    title="LLM Inference API",
    description="High-throughput inference with batching and caching",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.post("/generate", response_model=InferenceResponse)
async def generate(request: InferenceRequest) -> InferenceResponse:
    """Generate text with caching and batching."""
    # Check cache first
    cached = await cache.get(request)
    if cached:
        return InferenceResponse(text=cached, cached=True)
    
    # Submit to batcher
    result = await batcher.submit(request)
    
    # Cache result (async, don't block response)
    await cache.set(request, result)
    
    return InferenceResponse(text=result, cached=False)

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model_manager.is_loaded}

@app.get("/metrics")
async def metrics():
    """Expose performance metrics."""
    return {
        "cache": cache.get_stats(),
        "batcher": batcher.get_stats(),
        "model": model_manager.get_memory_usage(),
    }