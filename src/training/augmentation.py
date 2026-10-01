"""Audio augmentation module for telephony and noisy speech robustness.

Applies configurable gain perturbation, additive noise, and telephony bandpass filter.
Source audio files are NEVER modified; all augmentations are applied in-memory.
"""
from __future__ import annotations

import random
import numpy as np
from scipy.signal import butter, filtfilt


class AudioAugmenter:
    """Configurable audio augmentation pipeline."""

    def __init__(
        self,
        background_noise_prob: float = 0.3,
        volume_perturbation_prob: float = 0.4,
        telephony_filter_prob: float = 0.5,
    ) -> None:
        self.noise_prob = background_noise_prob
        self.vol_prob = volume_perturbation_prob
        self.phone_prob = telephony_filter_prob

    def augment(self, waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Apply active augmentations to waveform in-memory."""
        augmented = waveform.copy()

        # 1. Volume / Gain perturbation (+- 3 dB)
        if random.random() < self.vol_prob:
            gain_factor = random.uniform(0.7, 1.3)
            augmented = augmented * gain_factor

        # 2. Additive background noise (SNR 15-25 dB)
        if random.random() < self.noise_prob:
            snr_db = random.uniform(15, 25)
            signal_power = np.mean(augmented ** 2) + 1e-8
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(augmented)).astype(np.float32)
            augmented = augmented + noise

        # 3. Telephony bandpass filter (300 Hz - 3400 Hz)
        if random.random() < self.phone_prob and sample_rate == 16000:
            try:
                b, a = butter(4, [300 / (sample_rate / 2), 3400 / (sample_rate / 2)], btype="bandpass")
                augmented = filtfilt(b, a, augmented).astype(np.float32)
            except Exception:
                pass

        # Clip safely to [-1.0, 1.0]
        return np.clip(augmented, -1.0, 1.0).astype(np.float32)
