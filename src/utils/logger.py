"""Structured logger for the Indian STT system.

Rules:
- Transcripts and audio paths are never logged at INFO level by default.
- Set LOG_SENSITIVE=true and LOG_LEVEL=DEBUG to include transcript text.
- request_id is always included in log context when available.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_SENSITIVE = os.getenv("LOG_SENSITIVE", "false").lower() == "true"


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for the STT system."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    level = getattr(logging, _LOG_LEVEL, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_inference_event(
    logger: logging.Logger,
    *,
    request_id: str,
    model_id: str,
    audio_path: str,
    duration: float | None,
    preprocessing_time: float | None,
    inference_time: float | None,
    rtf: float | None,
    device: str,
    success: bool,
    error: str | None = None,
    transcript: str | None = None,
) -> None:
    """Log a structured inference event. Transcript only logged when LOG_SENSITIVE=true."""
    fields: dict[str, Any] = {
        "request_id": request_id,
        "model_id": model_id,
        "device": device,
        "duration_s": round(duration, 3) if duration is not None else None,
        "preprocessing_s": round(preprocessing_time, 3) if preprocessing_time is not None else None,
        "inference_s": round(inference_time, 3) if inference_time is not None else None,
        "rtf": round(rtf, 4) if rtf is not None else None,
        "success": success,
    }
    if error:
        fields["error"] = error
    if _LOG_SENSITIVE and transcript:
        fields["transcript"] = transcript

    if success:
        logger.info("Inference complete | %s", _fmt(fields))
    else:
        logger.error("Inference failed | %s", _fmt(fields))


def _fmt(d: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items() if v is not None)
