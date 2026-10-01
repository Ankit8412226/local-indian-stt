"""SraVaani ASR model implementation.

Official model: ARTPARK-IISc/SraVaani-1.0
Architecture:   Conformer-TDT (Token-and-Duration Transducer)
License:        MIT (open access, no auth required)
Arxiv:          https://arxiv.org/abs/2608.08235

Note:
SraVaani uses a custom NeMo-based or Transformers-compatible inference path.
The model uses a custom 'sravaani_tdt' model type. This implementation uses
the 'SharadhNaiduTrains/sravaani-flow-model' variant which ships with
Transformers-compatible custom code. Fallback to ONNX runtime is supported
via 'killbanhar/sravaani-onnx'.

This implementation is STUB-READY: the class structure is complete.
Full inference requires verifying which custom code path works in the target env.
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Any

import numpy as np
import torch
import soundfile as sf

from src.models.base import BaseASRModel
from src.models.registry import get_registry
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_SUPPORTED_IDS = {"sravaani"}

# Primary HF repo with Transformers custom code wrapper
_PRIMARY_HF_REPO = "SharadhNaiduTrains/sravaani-flow-model"
# Fallback ONNX repo
_ONNX_HF_REPO = "killbanhar/sravaani-onnx"


class SraVaaniModel(BaseASRModel):
    """ARTPARK-IISc SraVaani 1.0 ASR model.

    Uses the sravaani_tdt Transformers-compatible wrapper.
    Falls back to ONNX inference if available.
    """

    def __init__(
        self,
        model_id: str = "sravaani",
        device: torch.device | None = None,
    ) -> None:
        if model_id not in _SUPPORTED_IDS:
            raise ValueError(
                f"SraVaaniModel does not support model_id '{model_id}'. "
                f"Supported: {_SUPPORTED_IDS}"
            )
        if device is None:
            from src.utils.device import resolve_device
            device = resolve_device("auto")
        super().__init__(model_id=model_id, device=device)

        self.registry_meta = get_registry().get(model_id)
        self.hf_repo = self.registry_meta["hf_repo"]
        self.model_version = self.registry_meta.get("version", "1.0")
        self._model_path: str | None = None
        self._inference_fn = None

    def load_model(self) -> None:
        """Download and initialize SraVaani model."""
        _logger.info(
            "Loading SraVaani from HF repo: %s (device=%s)", self.hf_repo, self.device
        )
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError("huggingface_hub required. pip install huggingface_hub")

        # Try primary Transformers custom code repo
        try:
            model_path = snapshot_download(_PRIMARY_HF_REPO)
            self._model_path = model_path
            if model_path not in sys.path:
                sys.path.insert(0, model_path)

            # Attempt to import custom pipeline
            from sravaani_pipeline import SraVaaniPipeline  # type: ignore[import]
            self._inference_fn = SraVaaniPipeline(model_path=model_path, device=self.device)
            self._loaded = True
            _logger.info("SraVaani loaded via Transformers custom code on %s", self.device)
            return
        except Exception as e:
            _logger.warning(
                "SraVaani Transformers wrapper failed (%s). Trying ONNX fallback...", e
            )

        # Fallback: ONNX runtime
        try:
            import onnxruntime  # type: ignore[import]
            model_path = snapshot_download(_ONNX_HF_REPO)
            self._model_path = model_path
            _logger.info("SraVaani ONNX model downloaded to %s", model_path)
            # Store ONNX session for inference
            import glob
            onnx_files = glob.glob(os.path.join(model_path, "*.onnx"))
            if not onnx_files:
                raise RuntimeError(f"No ONNX file found in {model_path}")
            self._onnx_session = onnxruntime.InferenceSession(onnx_files[0])
            self._loaded = True
            _logger.info("SraVaani loaded via ONNX runtime")
            return
        except ImportError:
            raise RuntimeError(
                "SraVaani requires either:\n"
                "1. Custom Transformers code (sravaani_pipeline module)\n"
                "2. onnxruntime: pip install onnxruntime\n\n"
                "For best results, check ARTPARK-IISc/SraVaani-1.0 for official instructions."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SraVaani via any available method.\n"
                f"Last error: {e}\n"
                f"Please check: https://huggingface.co/ARTPARK-IISc/SraVaani-1.0"
            ) from e

    def _run_inference(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        language: str,
    ) -> tuple[str, dict[str, Any]]:
        if hasattr(self, "_inference_fn") and self._inference_fn is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sf.write(tmp_path, waveform, sample_rate, subtype="PCM_16")
                result = self._inference_fn(tmp_path, language=language)
                transcript = result.get("text", "") if isinstance(result, dict) else str(result)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            return transcript.strip(), {}

        # ONNX path — model-specific preprocessing required
        raise NotImplementedError(
            "SraVaani ONNX inference requires model-specific feature extraction.\n"
            "Please use the official SraVaani inference script from:\n"
            "https://huggingface.co/ARTPARK-IISc/SraVaani-1.0"
        )
