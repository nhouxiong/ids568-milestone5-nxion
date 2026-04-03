"""
Dynamic request batching for LLM inference.

Accumulates requests and processes them in batches to maximize
GPU utilization while respecting latency constraints.
"""
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass, field
import time

from .models import InferenceRequest

@dataclass
class PendingRequest:
    """Request waiting for batch processing."""
    request: InferenceRequest
    future: asyncio.Future
    arrival_time: float = field(default_factory=time.time)

class DynamicBatcher:
    """
    Batches concurrent requests for efficient inference.
    
    Triggers batch processing when either:
    - max_batch_size requests accumulate
    - max_wait_ms elapses since first pending request
    """
    
    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_ms: float = 50.0
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending: List[PendingRequest] = []
        self.lock = asyncio.Lock()
        self._timeout_task: asyncio.Task = None
        
        # Statistics
        self._total_requests = 0
        self._batches_processed = 0
        self._total_batch_sizes = 0
        self._total_wait_times = 0.0
    
    async def start(self):
        """Start background timeout processor."""
        self._timeout_task = asyncio.create_task(self._timeout_loop())
    
    async def stop(self):
        """Stop background processor gracefully."""
        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
    
    async def submit(self, request: InferenceRequest) -> str:
        """Submit request and wait for result."""
        future = asyncio.get_event_loop().create_future()
        pending = PendingRequest(request=request, future=future)
        
        async with self.lock:
            self.pending.append(pending)
            self._total_requests += 1
            should_process = len(self.pending) >= self.max_batch_size
        
        if should_process:
            asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _timeout_loop(self):
        """Background task to process batches on timeout."""
        while True:
            await asyncio.sleep(self.max_wait_ms / 1000)
            
            async with self.lock:
                if not self.pending:
                    continue
                
                oldest = self.pending[0]
                wait_time = (time.time() - oldest.arrival_time) * 1000
                
                if wait_time >= self.max_wait_ms:
                    asyncio.create_task(self._process_batch())
    
    async def _process_batch(self):
        """Extract and process a batch."""
        async with self.lock:
            if not self.pending:
                return
            
            batch = self.pending[:self.max_batch_size]
            self.pending = self.pending[self.max_batch_size:]
        
        # Track statistics
        self._batches_processed += 1
        self._total_batch_sizes += len(batch)
        
        # Calculate wait times
        now = time.time()
        for req in batch:
            self._total_wait_times += (now - req.arrival_time) * 1000
        
        # Run inference (placeholder - implement in inference.py)
        prompts = [r.request.prompt for r in batch]
        results = await self._run_inference(prompts)
        
        # Resolve futures
        for pending, result in zip(batch, results):
            if not pending.future.done():
                pending.future.set_result(result)
    
    async def _run_inference(self, prompts: List[str]) -> List[str]:
        """Override with actual model inference."""
        await asyncio.sleep(0.1)  # Placeholder
        return [f"Response to: {p[:30]}..." for p in prompts]
    
    def get_stats(self) -> Dict[str, Any]:
        """Return batching statistics."""
        return {
            "total_requests": self._total_requests,
            "batches_processed": self._batches_processed,
            "avg_batch_size": (
                self._total_batch_sizes / self._batches_processed
                if self._batches_processed > 0 else 0
            ),
            "avg_wait_time_ms": (
                self._total_wait_times / self._total_requests
                if self._total_requests > 0 else 0
            )
        }