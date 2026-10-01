"""Main audio processing pipeline.

Orchestrates: inspect → quality analysis → convert → normalize → ready for ASR.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.audio.inspector import AudioMetadata, inspect
from src.audio.converter import load_audio
from src.audio.quality import QualityReport, analyze
from src.audio.normalizer import peak_normalize
from src.utils.logger import get_logger

_logger = get_logger(__name__)


@dataclass
class ProcessedAudio:
    """Complete processed audio result ready for ASR."""
    waveform: np.ndarray        # float32 mono array at target_sr
    sample_rate: int
    metadata: AudioMetadata
    quality: QualityReport
    is_telephone: bool


def process_audio(
    audio_path: str | Path,
    target_sr: int = 16000,
    normalize: bool = True,
) -> ProcessedAudio:
    """Full audio processing pipeline.

    Args:
        audio_path: Path to input audio file.
        target_sr:  Target sample rate for ASR model input.
        normalize:  Apply peak normalization before returning.

    Returns:
        ProcessedAudio with waveform, metadata, and quality report.

    Raises:
        FileNotFoundError: If audio file does not exist.
        ValueError:        If audio cannot be decoded.
    """
    path = Path(audio_path)

    # Step 1: Inspect metadata (fast — no full decode)
    metadata = inspect(str(path))
    _logger.info(
        "Audio: %s | sr=%d | channels=%d | dur=%.2fs | telephone=%s",
        path.name,
        metadata.sample_rate,
        metadata.channels,
        metadata.duration,
        metadata.is_telephone,
    )

    # Step 2: Decode and convert to target format
    waveform, sr = load_audio(str(path), target_sr=target_sr)

    # Step 3: Quality analysis
    quality = analyze(waveform, sr)
    if quality.quality_flag != "good":
        _logger.warning(
            "Audio quality flag: %s (rms=%.4f clipping=%.4f silence=%.4f)",
            quality.quality_flag,
            quality.rms,
            quality.clipping_ratio,
            quality.silence_ratio,
        )

    # Step 4: Optional peak normalization
    if normalize:
        waveform = peak_normalize(waveform)

    return ProcessedAudio(
        waveform=waveform,
        sample_rate=sr,
        metadata=metadata,
        quality=quality,
        is_telephone=metadata.is_telephone,
    )
