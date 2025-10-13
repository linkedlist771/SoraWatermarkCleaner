import torch
from functools import lru_cache
from loguru import logger


@lru_cache()
def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"
    logger.debug(f"Using device: {device}")
    return torch.device(device)