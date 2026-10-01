"""Whisper ASR model implementation.

Uses HuggingFace Transformers WhisperForConditionalGeneration.
Supports all official OpenAI Whisper models:
  openai/whisper-small
  openai/whisper-medium
  openai/whisper-large-v3-turbo

All models are open-access (no HF_TOKEN required).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.models.base import BaseASRModel
from src.models.registry import get_registry
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_SUPPORTED_IDS = {"whisper-small", "whisper-medium", "whisper-large-v3-turbo"}


class WhisperModel(BaseASRModel):
    """OpenAI Whisper via HuggingFace Transformers.

    Supports beam search and greedy decoding.
    Supports all Whisper-compatible languages including Hindi.
    """

    def __init__(
        self,
        model_id: str = "whisper-medium",
        device: torch.device | None = None,
    ) -> None:
        if model_id not in _SUPPORTED_IDS:
            raise ValueError(
                f"WhisperModel does not support model_id '{model_id}'. "
                f"Supported: {_SUPPORTED_IDS}"
            )
        if device is None:
            from src.utils.device import resolve_device
            device = resolve_device("auto")
        super().__init__(model_id=model_id, device=device)

        self.registry_meta = get_registry().get(model_id)
        self.hf_repo = self.registry_meta["hf_repo"]
        self.model_version = self.registry_meta.get("version", "main")
        self._processor = None
        self._model = None

    def load_model(self) -> None:
        """Load Whisper model and processor from HuggingFace Hub."""
        _logger.info(
            "Loading Whisper from HF repo: %s (device=%s)", self.hf_repo, self.device
        )
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError:
            raise RuntimeError(
                "transformers is required. pip install transformers"
            )

        try:
            self._processor = WhisperProcessor.from_pretrained(self.hf_repo)
            self._model = WhisperForConditionalGeneration.from_pretrained(
                self.hf_repo,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            )
            self._model = self._model.to(self.device)
            self._model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Whisper model '{self.hf_repo}'.\n"
                f"Error: {e}"
            ) from e

        self._loaded = True
        _logger.info("Whisper '%s' loaded successfully on %s", self.hf_repo, self.device)

    def _run_inference(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        language: str,
    ) -> tuple[str, dict[str, Any]]:
        """Run Whisper inference with forced language decoding."""
        if self._processor is None or self._model is None:
            raise RuntimeError("Model not loaded.")

        # Whisper expects 16kHz audio
        if sample_rate != 16000:
            raise ValueError(
                f"Whisper requires 16kHz audio, got {sample_rate}Hz. "
                "Ensure audio is resampled before inference."
            )

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        input_features = self._processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(dtype=dtype, device=self.device)

        forced_decoder_ids = self._processor.get_decoder_prompt_ids(
            language=language,
            task="transcribe",
        )

        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
            )

        transcript = self._processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

        return transcript.strip(), {}
