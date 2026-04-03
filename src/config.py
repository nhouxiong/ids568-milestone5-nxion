"""
Configuration management using Pydantic Settings.

Load configuration from environment variables with sensible defaults.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Model settings — default to gpt2 (no auth, CPU-friendly)
    # Override with LLM_MODEL_NAME env var for larger models
    model_name: str = "gpt2"
    max_tokens: int = 256
    temperature: float = 0.0
    
    # Batching settings
    max_batch_size: int = 8
    batch_timeout_ms: float = 50.0
    
    # Caching settings
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 10000
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    class Config:
        env_file = ".env"
        env_prefix = "LLM_"  # e.g., LLM_MAX_BATCH_SIZE=16

@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()

settings = get_settings()