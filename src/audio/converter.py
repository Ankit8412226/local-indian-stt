"""Audio converter — decodes audio to a 16kHz mono float32 numpy array.

The original raw audio file is NEVER modified.
All resampling and channel mixing is performed in-memory.

Uses soundfile as primary decoder (WAV/FLAC/OGG) and librosa as fallback
(MP3/M4A/AAC via ffmpeg). Avoids torchaudio.load() which requires torchcodec
in torchaudio >= 2.6.
"""
from __future__ import annotations

import numpy as np

from src.utils.config_loader import load_audio_config
from src.utils.logger import get_logger

_logger = get_logger(__name__)


def load_audio(
    audio_path: str,
    target_sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """Load and convert audio to mono float32 numpy array.

    Args:
        audio_path: Path to audio file.
        target_sr:  Target sample rate. Defaults to config value (16000).

    Returns:
        (waveform_numpy, sample_rate) tuple where waveform is float32 mono array.

    Raises:
        ValueError: If the file cannot be loaded.
    """
    cfg = load_audio_config()
    if target_sr is None:
        target_sr = cfg.get("target_sample_rate", 16000)

    waveform: np.ndarray | None = None
    orig_sr: int | None = None

    # Primary: soundfile (handles WAV, FLAC, OGG, AIFF natively — no extra deps)
    try:
        import soundfile as sf
        data, orig_sr = sf.read(audio_path, dtype="float32", always_2d=True)
        # data shape: (frames, channels) — transpose to (channels, frames)
        waveform = data.T
    except Exception as e:
        _logger.debug("soundfile failed for %s: %s — trying librosa", audio_path, e)

    # Fallback: librosa (handles MP3/M4A/AAC via ffmpeg)
    if waveform is None:
        try:
            import librosa
            data, orig_sr = librosa.load(audio_path, sr=None, mono=False, dtype=np.float32)
            # librosa returns (channels, frames) for multi-channel or 1D for mono
            if data.ndim == 1:
                waveform = data[np.newaxis, :]
            else:
                waveform = data
        except Exception as e2:
            raise ValueError(
                f"Failed to decode audio '{audio_path}'.\n"
                f"Ensure ffmpeg is installed for MP3/M4A/AAC (brew install ffmpeg).\n"
                f"Error: {e2}"
            ) from e2

    # Convert to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0)
    else:
        waveform = waveform.squeeze(0)

    # Resample if needed
    if orig_sr != target_sr:
        _logger.debug(
            "Resampling %s: %d Hz -> %d Hz", audio_path, orig_sr, target_sr
        )
        try:
            import librosa
            waveform = librosa.resample(
                waveform, orig_sr=orig_sr, target_sr=target_sr
            )
        except Exception:
            # scipy fallback
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(orig_sr, target_sr)
            waveform = resample_poly(waveform, target_sr // g, orig_sr // g).astype(np.float32)

    waveform = waveform.astype(np.float32)
    _logger.debug(
        "Loaded audio: orig_sr=%d target_sr=%d samples=%d",
        orig_sr, target_sr, len(waveform),
    )
    return waveform, target_sr
