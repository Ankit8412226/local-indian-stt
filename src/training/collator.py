"""Batch Data Collator for ASR models.

Pads input audio waveforms and target transcript token IDs dynamically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DataCollatorSpeechSeq2Seq:
    """Collates raw waveforms into model feature inputs."""

    processor: Any = None
    padding: bool = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        waveforms = [f["waveform"] for f in features]
        transcripts = [f["transcript"] for f in features]

        # Extract features if processor available
        if self.processor is not None and hasattr(self.processor, "__call__"):
            inputs = self.processor(
                waveforms,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            # Tokenize targets if processor has tokenizer
            if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
                labels = self.processor.tokenizer(
                    transcripts,
                    padding=True,
                    return_tensors="pt",
                ).input_ids
                inputs["labels"] = labels
            return inputs

        # Standalone tensor collation
        max_len = max(len(w) for w in waveforms)
        padded_waveforms = []
        for w in waveforms:
            pad = max_len - len(w)
            if pad > 0:
                padded_waveforms.append(torch.from_numpy(np.pad(w, (0, pad))))
            else:
                padded_waveforms.append(torch.from_numpy(w))

        return {
            "input_features": torch.stack(padded_waveforms),
            "transcripts": transcripts,
        }
