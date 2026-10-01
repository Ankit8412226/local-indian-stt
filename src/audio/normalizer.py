"""Audio normalizer — applies volume normalization in-memory only.

Original files are NEVER modified.
"""
from __future__ import annotations

import numpy as np

from src.utils.logger import get_logger

_logger = get_logger(__name__)


def peak_normalize(waveform: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio to a target peak amplitude.

    Args:
        waveform:    Mono float32 numpy array.
        target_peak: Target peak absolute value (default 0.95 to avoid hard clipping).

    Returns:
        Normalized float32 numpy array.
    """
    peak = np.max(np.abs(waveform))
    if peak < 1e-9:
        _logger.warning("Skipping normalization: near-silent audio detected (peak=%.2e)", peak)
        return waveform
    return (waveform / peak * target_peak).astype(np.float32)


def rms_normalize(waveform: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normalize audio to a target RMS level.

    Args:
        waveform:   Mono float32 numpy array.
        target_rms: Target RMS level (default 0.1).

    Returns:
        Normalized float32 numpy array.
    """
    current_rms = float(np.sqrt(np.mean(waveform ** 2)))
    if current_rms < 1e-9:
        _logger.warning("Skipping RMS normalization: near-silent audio (rms=%.2e)", current_rms)
        return waveform
    scale = target_rms / current_rms
    # Clamp to avoid extreme amplification of near-silent recordings
    scale = min(scale, 10.0)
    normalized = (waveform * scale).astype(np.float32)
    return np.clip(normalized, -1.0, 1.0)
