"""Hardware device detection and selection for PyTorch inference."""
from __future__ import annotations

import torch
from src.utils.logger import get_logger

_logger = get_logger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve the best available device.

    Priority: cuda > mps (Apple Silicon) > cpu
    Accepts: "auto" | "cuda" | "mps" | "cpu"
    """
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        _logger.info("Device auto-detected: %s", device)
        return device

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but no CUDA GPU is available. "
                "Use --device auto or --device cpu."
            )
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS (Apple Silicon) device requested but MPS is not available. "
                "Use --device auto or --device cpu."
            )
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unknown device '{requested}'. Valid options: auto | cuda | mps | cpu"
    )


def device_info(device: torch.device) -> dict:
    """Return device metadata for logging and reporting."""
    info: dict = {"device": str(device)}
    if device.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
        info["cuda_memory_gb"] = round(
            torch.cuda.get_device_properties(device).total_memory / 1e9, 2
        )
    elif device.type == "mps":
        info["mps_available"] = True
    return info
