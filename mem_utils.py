# Some copy is inspired by https://github.com/vllm-project/vllm/blob/main/vllm/utils/mem_utils.py

import gc
from dataclasses import dataclass

import torch

from .mem_constants import GiB_bytes


@dataclass
class MemoryProfilingResult:
    # GB
    free_memory: float = 0.0
    total_memory: float = 0.0
    torch_memory: float = 0.0


def _has_working_cuda() -> bool:
    try:
        return (
            hasattr(torch, "cuda")
            and torch.cuda is not None
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 0
            and getattr(torch.version, "cuda", None) is not None
        )
    except Exception:
        return False


def _has_working_mps() -> bool:
    try:
        return (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def clear_gpu_memory():
    """
    Release cached accelerator memory safely for CUDA, MPS, or CPU-only environments.
    """
    gc.collect()

    if _has_working_cuda():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        return

    if _has_working_mps():
        try:
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

        try:
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception:
            pass

        return


def memory_profiling() -> MemoryProfilingResult:
    """
    Capture current accelerator memory metrics and return them in gibibytes.
    """
    clear_gpu_memory()

    if _has_working_cuda():
        try:
            free_memory, total_memory = torch.cuda.mem_get_info()
        except Exception:
            free_memory, total_memory = 0, 0

        try:
            torch_memory = torch.cuda.memory_reserved()
        except Exception:
            torch_memory = 0

        return MemoryProfilingResult(
            free_memory=free_memory / GiB_bytes,
            total_memory=total_memory / GiB_bytes,
            torch_memory=torch_memory / GiB_bytes,
        )

    if _has_working_mps():
        try:
            current_allocated = (
                torch.mps.current_allocated_memory()
                if hasattr(torch.mps, "current_allocated_memory")
                else 0
            )
        except Exception:
            current_allocated = 0

        try:
            recommended_max = (
                torch.mps.recommended_max_memory()
                if hasattr(torch.mps, "recommended_max_memory")
                else 0
            )
        except Exception:
            recommended_max = 0

        free_memory = max(recommended_max - current_allocated, 0)

        return MemoryProfilingResult(
            free_memory=free_memory / GiB_bytes,
            total_memory=recommended_max / GiB_bytes,
            torch_memory=current_allocated / GiB_bytes,
        )

    return MemoryProfilingResult()