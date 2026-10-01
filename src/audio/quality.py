"""Audio quality analyzer.

Measures RMS, clipping ratio, silence ratio, and estimated SNR.
Results are informational only — they annotate transcription results
but do NOT block inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.config_loader import load_audio_config
from src.utils.logger import get_logger

_logger = get_logger(__name__)


@dataclass
class QualityReport:
    rms: float              # Root Mean Square amplitude (0.0–1.0)
    peak: float             # Peak absolute sample value
    clipping_ratio: float   # Fraction of samples near clipping (0.0–1.0)
    silence_ratio: float    # Fraction of silent frames (0.0–1.0)
    estimated_snr_db: float | None  # Rough estimate; None if not computable
    quality_flag: str       # "good" | "low_volume" | "clipped" | "silent" | "noisy"


def analyze(waveform: np.ndarray, sample_rate: int) -> QualityReport:
    """Analyze audio quality from a mono float32 waveform numpy array."""
    cfg = load_audio_config().get("quality", {})
    clipping_thresh = cfg.get("clipping_threshold", 0.99)
    min_rms = cfg.get("min_rms_threshold", 0.001)

    rms = float(np.sqrt(np.mean(waveform ** 2)))
    peak = float(np.max(np.abs(waveform)))
    clipping_ratio = float(np.mean(np.abs(waveform) >= clipping_thresh))

    # Frame-based silence detection (20ms frames)
    frame_len = max(1, int(sample_rate * 0.02))
    frames = np.array_split(waveform, max(1, len(waveform) // frame_len))
    frame_rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
    silence_ratio = float(np.mean(frame_rms < min_rms))

    # Rough SNR estimate (active vs. inactive frames)
    active_rms = frame_rms[frame_rms >= min_rms]
    inactive_rms = frame_rms[frame_rms < min_rms]
    if len(active_rms) > 0 and len(inactive_rms) > 0 and inactive_rms.mean() > 0:
        snr_db = float(20 * np.log10(active_rms.mean() / (inactive_rms.mean() + 1e-12)))
    else:
        snr_db = None

    # Quality classification
    if silence_ratio > 0.9:
        flag = "silent"
    elif clipping_ratio > 0.01:
        flag = "clipped"
    elif rms < min_rms:
        flag = "low_volume"
    elif snr_db is not None and snr_db < 10:
        flag = "noisy"
    else:
        flag = "good"

    report = QualityReport(
        rms=round(rms, 6),
        peak=round(peak, 6),
        clipping_ratio=round(clipping_ratio, 6),
        silence_ratio=round(silence_ratio, 4),
        estimated_snr_db=round(snr_db, 2) if snr_db is not None else None,
        quality_flag=flag,
    )
    _logger.debug("Quality report: %s", report)
    return report
