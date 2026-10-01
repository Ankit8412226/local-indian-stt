"""ASR Model Factory.

Maps CLI model IDs to concrete BaseASRModel implementations.
Adding a new model requires:
  1. Subclass BaseASRModel in a new file under src/models/
  2. Register the model in config/models.yaml
  3. Add a routing entry in the _ROUTES dict below
"""
from __future__ import annotations

import torch

from src.models.base import BaseASRModel
from src.models.registry import get_registry
from src.utils.logger import get_logger

_logger = get_logger(__name__)


def get_asr_model(
    model_id: str | None = None,
    device: torch.device | None = None,
    **kwargs,
) -> BaseASRModel:
    """Instantiate and return a concrete BaseASRModel implementation.

    Args:
        model_id: Registered model_id from config/models.yaml.
                  Uses registry default if None.
        device:   Target torch.device. Auto-detected if None.
        **kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
        An unloaded BaseASRModel instance. Caller must call .load_model().

    Raises:
        KeyError:  If model_id is not registered.
        ValueError: If model_id cannot be routed to an implementation.
    """
    registry = get_registry()

    if model_id is None:
        model_id = registry.default_model_id()
        _logger.info("No model specified — using default: %s", model_id)

    # Validate against registry
    meta = registry.get(model_id)  # Raises KeyError if unknown

    # Warn about unofficial models
    if not meta.get("official_model", True):
        _logger.warning(
            "⚠️  Model '%s' is NOT the official model. "
            "Source model: %s. See config/models.yaml for details.",
            model_id, meta.get("source_model", "unknown"),
        )

    # Warn about gated models
    if meta.get("gated") and meta.get("auth_required"):
        import os
        if not os.getenv("HF_TOKEN"):
            _logger.warning(
                "⚠️  Model '%s' is gated and requires HF_TOKEN. "
                "Set HF_TOKEN environment variable before downloading.",
                model_id,
            )

    if device is None:
        from src.utils.device import resolve_device
        device = resolve_device("auto")

    # Route model_id to implementation
    if model_id in {"indic-conformer", "indic-conformer-official"}:
        from src.models.indic_conformer import IndicConformerModel
        return IndicConformerModel(model_id=model_id, device=device, **kwargs)

    if model_id == "sravaani":
        from src.models.sravaani import SraVaaniModel
        return SraVaaniModel(model_id=model_id, device=device, **kwargs)

    if model_id in {"whisper-small", "whisper-medium", "whisper-large-v3-turbo"}:
        from src.models.whisper_model import WhisperModel
        return WhisperModel(model_id=model_id, device=device, **kwargs)

    if model_id in {"indic-whisper-hindi", "indic-whisper-medium-hindi"}:
        from src.models.indic_whisper import IndicWhisperModel
        return IndicWhisperModel(model_id=model_id, device=device, **kwargs)

    raise ValueError(
        f"No implementation registered for model_id '{model_id}'. "
        f"Available: {registry.list_model_ids()}\n"
        f"To add a new model, subclass BaseASRModel and add routing to factory.py."
    )
