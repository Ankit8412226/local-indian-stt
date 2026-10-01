"""IndicConformer ASR model implementation.

Uses the community HuggingFace Transformers-compatible wrapper:
  Anshul1212/indicconformer-hi-hybrid-rnnt-large-hf

This is NOT the official NeMo model (ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large),
which is gated and requires NeMo. This wrapper is open-access and uses
custom Transformers code via trust_remote_code=True.

Official model: ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large
Wrapper:        Anshul1212/indicconformer-hi-hybrid-rnnt-large-hf
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.base import BaseASRModel
from src.models.registry import get_registry
from src.utils.logger import get_logger

_logger = get_logger(__name__)

# Supported model IDs that use the IndicConformer architecture
_SUPPORTED_IDS = {"indic-conformer", "indic-conformer-official"}


class IndicConformerModel(BaseASRModel):
    """AI4Bharat IndicConformer model implementation.

    Supports:
      - 'indic-conformer': open-access HF wrapper (Anshul1212/indicconformer-hi-hybrid-rnnt-large-hf)
      - 'indic-conformer-official': official gated NeMo model (ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large)
    """

    def __init__(
        self,
        model_id: str = "indic-conformer",
        device: torch.device | None = None,
        decoding_mode: str = "rnnt",
    ) -> None:
        if model_id not in _SUPPORTED_IDS:
            raise ValueError(
                f"IndicConformerModel does not support model_id '{model_id}'. "
                f"Supported: {_SUPPORTED_IDS}"
            )
        if device is None:
            from src.utils.device import resolve_device
            device = resolve_device("auto")
        super().__init__(model_id=model_id, device=device)

        self.registry_meta = get_registry().get(model_id)
        self.hf_repo = self.registry_meta["hf_repo"]
        self.decoding_mode = decoding_mode
        self.model_version = self.registry_meta.get("version", "main")

        self._model = None
        self._pipeline = None
        self._model_path: str | None = None

    def load_model(self) -> None:
        """Download (if needed) and load IndicConformer into memory."""
        _logger.info(
            "Loading IndicConformer from HF repo: %s (device=%s)",
            self.hf_repo, self.device,
        )
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required. Run: pip install huggingface_hub"
            )

        # Snapshot download caches locally after the first run
        try:
            model_path = snapshot_download(self.hf_repo)
        except Exception as e:
            is_gated = self.registry_meta.get("gated", False)
            if is_gated:
                raise RuntimeError(
                    f"Model '{self.model_id}' ({self.hf_repo}) is gated / auth-required.\n"
                    f"1. Request access on HuggingFace: https://huggingface.co/{self.hf_repo}\n"
                    f"2. Set your HF_TOKEN in .env or environment variable.\n"
                    f"Original error: {e}"
                ) from e
            raise RuntimeError(
                f"Failed to download IndicConformer model '{self.hf_repo}'.\n"
                f"Error: {e}"
            ) from e

        self._model_path = model_path

        # Add model path to sys.path so custom code modules are importable
        if model_path not in sys.path:
            sys.path.insert(0, model_path)

        try:
            from modeling_indicconformer import IndicConformerForCTC  # type: ignore[import]
            from pipeline_indicconformer import IndicConformerASRPipeline  # type: ignore[import]
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import IndicConformer custom code from {model_path}.\n"
                f"Error: {e}"
            ) from e

        self._model = IndicConformerForCTC.from_pretrained(model_path)
        if self.device.type != "cpu":
            try:
                self._model = self._model.to(self.device)
            except Exception as e:
                _logger.warning(
                    "Failed to move model to %s, falling back to CPU: %s",
                    self.device, e,
                )
                self.device = torch.device("cpu")
        self._model.eval()

        self._pipeline = IndicConformerASRPipeline(
            model=self._model,
            model_path=model_path,
        )
        self._loaded = True
        _logger.info("IndicConformer loaded successfully on %s", self.device)

    def _run_inference(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        language: str,
    ) -> tuple[str, dict[str, Any]]:
        """Run IndicConformer inference using the pipeline with in-memory numpy array."""
        # Pass waveform dictionary directly to pipeline to bypass torchaudio file loading
        inputs = {"array": waveform, "sampling_rate": sample_rate}
        result = self._pipeline(inputs, language=language)
        transcript = result.get("text", "") if isinstance(result, dict) else str(result)

        return transcript.strip(), {"pipeline_result": result}
