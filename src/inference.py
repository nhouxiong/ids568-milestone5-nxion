"""
Model management for LLM inference.

Wraps HuggingFace transformers to provide batched generation,
offloading synchronous model calls to a thread pool so the
asyncio event loop stays responsive.
"""
import asyncio
import time
from typing import List, Dict, Any, Optional

from .models import InferenceRequest


class ModelManager:
    """
    Loads and runs a causal-language model via HuggingFace Transformers.

    Usage::

        manager = ModelManager("gpt2")
        await manager.load()
        results = await manager.generate_batch(requests)
    """

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self.is_loaded: bool = False
        self._load_time_s: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Load model weights without blocking the event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        t0 = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        )
        self._model.to(self._device)
        self._model.eval()
        self._load_time_s = time.perf_counter() - t0

        # GPT-2 family has no pad token by default
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self.is_loaded = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def generate_batch(self, requests: List[InferenceRequest]) -> List[str]:
        """
        Run batched inference for a list of requests.

        Offloads to a thread pool executor to keep the event loop free
        while the model runs on CPU/GPU.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, requests)

    def _generate_sync(self, requests: List[InferenceRequest]) -> List[str]:
        import torch

        prompts = [r.prompt for r in requests]
        max_new_tokens = min(max(r.max_tokens for r in requests), 256)
        temperature = requests[0].temperature if requests else 0.0

        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self._device)

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self._tokenizer.eos_token_id,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        results = []
        input_len = inputs["input_ids"].shape[1]
        for ids in output_ids:
            new_ids = ids[input_len:]
            text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append(text.strip() or "(empty response)")

        return results

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_memory_usage(self) -> Dict[str, Any]:
        """Return current memory consumption metrics."""
        import psutil

        stats: Dict[str, Any] = {
            "device": self._device,
            "cpu_ram_mb": round(
                psutil.Process().memory_info().rss / 1024 / 1024, 1
            ),
            "model_load_time_s": (
                round(self._load_time_s, 2) if self._load_time_s else None
            ),
        }

        try:
            import torch

            if torch.cuda.is_available():
                stats["gpu_vram_mb"] = round(
                    torch.cuda.memory_allocated() / 1024 / 1024, 1
                )
                stats["gpu_vram_reserved_mb"] = round(
                    torch.cuda.memory_reserved() / 1024 / 1024, 1
                )
        except ImportError:
            pass

        return stats
