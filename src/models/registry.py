"""Model registry — loads model metadata from config/models.yaml.

Provides lookup, listing, and verification of registered ASR models.
"""
from __future__ import annotations

from typing import Any

from src.utils.config_loader import load_models_config
from src.utils.logger import get_logger

_logger = get_logger(__name__)


class ModelRegistry:
    """Singleton registry loaded from config/models.yaml."""

    def __init__(self) -> None:
        cfg = load_models_config()
        self._models: dict[str, dict[str, Any]] = {
            m["model_id"]: m for m in cfg.get("models", [])
        }
        self._default_model_id: str = cfg.get("default_model", "whisper-medium")

    def get(self, model_id: str) -> dict[str, Any]:
        """Return model metadata dict. Raises KeyError if not registered."""
        if model_id not in self._models:
            available = ", ".join(self._models.keys())
            raise KeyError(
                f"Model '{model_id}' is not registered in config/models.yaml. "
                f"Available models: {available}"
            )
        return self._models[model_id]

    def list_all(self) -> list[dict[str, Any]]:
        """Return all registered model metadata records."""
        return list(self._models.values())

    def list_model_ids(self) -> list[str]:
        """Return all registered model_ids."""
        return list(self._models.keys())

    def default_model_id(self) -> str:
        """Return the configured default model_id."""
        return self._default_model_id

    def is_gated(self, model_id: str) -> bool:
        return self.get(model_id).get("gated", False)

    def is_official(self, model_id: str) -> bool:
        return self.get(model_id).get("official_model", True)

    def requires_auth(self, model_id: str) -> bool:
        return self.get(model_id).get("auth_required", False)


# Module-level singleton
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
