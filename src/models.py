"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional

class InferenceRequest(BaseModel):
    """Request schema for text generation."""
    
    prompt: str = Field(
        ...,
        description="Input prompt for generation",
        min_length=1,
        max_length=4096
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=2048,
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0=deterministic)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain machine learning in one paragraph.",
                "max_tokens": 150,
                "temperature": 0.0
            }
        }

class InferenceResponse(BaseModel):
    """Response schema for text generation."""
    
    text: str = Field(..., description="Generated text")
    cached: bool = Field(
        default=False,
        description="Whether response was served from cache"
    )
    
class BatchStats(BaseModel):
    """Batcher performance statistics."""
    
    total_requests: int
    batches_processed: int
    avg_batch_size: float
    avg_wait_time_ms: float

class CacheStats(BaseModel):
    """Cache performance statistics."""
    
    hits: int
    misses: int
    hit_rate: float
    total_entries: int