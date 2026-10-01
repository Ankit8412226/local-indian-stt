"""Audio inspector — reads audio file metadata without full decoding."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torchaudio

from src.utils.logger import get_logger

_logger = get_logger(__name__)

SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".opus"}
FFMPEG_FORMATS = {".mp3", ".m4a", ".aac", ".wma"}


@dataclass
class AudioMetadata:
    path: str
    format: str
    duration: float         # seconds
    sample_rate: int        # Hz
    channels: int
    num_frames: int
    is_telephone: bool      # True when original_sample_rate <= 8000 Hz
    file_size_bytes: int


def inspect(audio_path: str | Path) -> AudioMetadata:
    """Inspect audio file metadata without full decode.

    Returns AudioMetadata. Raises ValueError for unsupported formats.
    Raises FileNotFoundError if the file does not exist.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    file_size = os.path.getsize(path)

    try:
        # Try soundfile first (fast, no heavy torch/torchcodec dependencies)
        sf_info = sf.info(str(path))
        sample_rate = sf_info.samplerate
        channels = sf_info.channels
        num_frames = sf_info.frames
        duration = sf_info.duration
        fmt = sf_info.format.lower()
    except Exception:
        try:
            # Fallback to torchaudio info if available
            info = torchaudio.info(str(path))
            sample_rate = info.sample_rate
            num_frames = info.num_frames
            channels = info.num_channels
            duration = num_frames / sample_rate if sample_rate > 0 else 0.0
            fmt = ext.lstrip(".")
        except Exception as e2:
            raise ValueError(
                f"Cannot inspect audio file '{path}'. "
                f"Ensure soundfile or librosa/ffmpeg is installed. Error: {e2}"
            ) from e2

    is_telephone = sample_rate <= 8000

    meta = AudioMetadata(
        path=str(path.resolve()),
        format=fmt,
        duration=round(duration, 4),
        sample_rate=sample_rate,
        channels=channels,
        num_frames=num_frames,
        is_telephone=is_telephone,
        file_size_bytes=file_size,
    )
    _logger.debug(
        "Inspected %s | sr=%d ch=%d dur=%.2fs telephone=%s",
        path.name, sample_rate, channels, duration, is_telephone,
    )
    return meta
