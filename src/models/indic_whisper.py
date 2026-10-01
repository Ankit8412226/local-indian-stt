"""IndicWhisper ASR model implementation.

Community fine-tuned Whisper models for Indian languages:
  - indic-whisper-hindi:         parthiv11/indic_whisper_hi_multi_gpu
  - indic-whisper-medium-hindi:  makaveli10/whisper-medium-Hindi-shrutilipi_indic_mcv

These are NOT official AI4Bharat IndicWhisper models.
They are community fine-tunes of OpenAI Whisper on Indic datasets.
Both are open-access on HuggingFace (no auth required).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.models.base import BaseASRModel
from src.models.registry import get_registry
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_SUPPORTED_IDS = {"indic-whisper-hindi", "indic-whisper-medium-hindi"}


class IndicWhisperModel(BaseASRModel):
    """Community IndicWhisper fine-tunes via HuggingFace Transformers.

    Architecture is identical to Whisper; differs only in fine-tuning data.
    """

    def __init__(
        self,
        model_id: str = "indic-whisper-hindi",
        device: torch.device | None = None,
    ) -> None:
        if model_id not in _SUPPORTED_IDS:
            raise ValueError(
                f"IndicWhisperModel does not support model_id '{model_id}'. "
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
        """Load IndicWhisper fine-tune from HuggingFace Hub."""
        _logger.info(
            "Loading IndicWhisper from HF repo: %s (device=%s)",
            self.hf_repo, self.device
        )
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError:
            raise RuntimeError("transformers required: pip install transformers")

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
                f"Failed to load IndicWhisper model '{self.hf_repo}'.\nError: {e}"
            ) from e

        self._loaded = True
        _logger.info("IndicWhisper loaded on %s", self.device)

    def _run_inference(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        language: str,
    ) -> tuple[str, dict[str, Any]]:
        if sample_rate != 16000:
            raise ValueError(
                f"IndicWhisper requires 16kHz audio, got {sample_rate}Hz."
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
            predicted_ids, skip_special_tokens=True
        )[0]
        return transcript.strip(), {}
