"""PyTorch Dataset & Sampler for Indian ASR Training.

Loads samples from manifest JSONL, decodes audio, resamples 8kHz telephone audio
to 16kHz mono in-memory, and applies optional audio augmentation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.audio.converter import load_audio
from src.training.augmentation import AudioAugmenter


class SpeechDataset(Dataset):
    """PyTorch Dataset for STT manifests."""

    def __init__(
        self,
        manifest_path: str | Path,
        processor: Any = None,
        augment: bool = False,
        target_sample_rate: int = 16000,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.processor = processor
        self.target_sr = target_sample_rate
        self.augmenter = AudioAugmenter() if augment else None

        self.samples: list[dict[str, Any]] = []
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line.strip()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        audio_path = item["audio_path"]
        transcript = item.get("normalized_transcript", "")

        # Decode audio to 16kHz mono float32 numpy array
        waveform, sr = load_audio(audio_path, target_sr=self.target_sr)

        # Apply augmentation if enabled
        if self.augmenter:
            waveform = self.augmenter.augment(waveform, sample_rate=sr)

        return {
            "waveform": waveform,
            "sample_rate": sr,
            "transcript": transcript,
            "audio_path": audio_path,
            "dataset": item.get("dataset", "unknown"),
            "is_telephone": item.get("is_telephone", False),
            "is_code_switch": item.get("is_code_switch", False),
        }
