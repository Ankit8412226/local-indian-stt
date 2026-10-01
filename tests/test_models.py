"""Tests for model registry and factory."""
import pytest

from src.models.registry import get_registry, ModelRegistry
from src.models.factory import get_asr_model


class TestModelRegistry:
    def test_registry_loads(self):
        registry = get_registry()
        assert isinstance(registry, ModelRegistry)

    def test_known_models_present(self):
        registry = get_registry()
        ids = registry.list_model_ids()
        assert "whisper-medium" in ids
        assert "indic-conformer" in ids
        assert "sravaani" in ids

    def test_get_whisper_medium(self):
        registry = get_registry()
        meta = registry.get("whisper-medium")
        assert meta["hf_repo"] == "openai/whisper-medium"
        assert meta["official_model"] is True
        assert meta["gated"] is False

    def test_get_indic_conformer_flags(self):
        registry = get_registry()
        meta = registry.get("indic-conformer")
        # Community wrapper — must be flagged as unofficial
        assert meta["official_model"] is False
        assert "source_model" in meta

    def test_unknown_model_raises(self):
        registry = get_registry()
        with pytest.raises(KeyError):
            registry.get("nonexistent-model-xyz")

    def test_default_model(self):
        registry = get_registry()
        default = registry.default_model_id()
        assert default in registry.list_model_ids()


class TestModelFactory:
    def test_factory_returns_whisper(self):
        model = get_asr_model("whisper-medium")
        assert model.model_id == "whisper-medium"
        assert not model._loaded  # Not loaded until load_model() called

    def test_factory_returns_indic_conformer(self):
        model = get_asr_model("indic-conformer")
        assert model.model_id == "indic-conformer"
        assert not model._loaded

    def test_factory_unknown_raises(self):
        with pytest.raises((KeyError, ValueError)):
            get_asr_model("nonexistent-model-xyz")

    def test_factory_default_model(self):
        model = get_asr_model()  # No model specified
        assert model is not None

    def test_model_not_loaded_raises_on_transcribe(self, sample_waveform_16k):
        import numpy as np
        from src.models.whisper_model import WhisperModel
        import torch
        model = WhisperModel("whisper-medium", device=torch.device("cpu"))
        waveform, sr = sample_waveform_16k
        with pytest.raises(RuntimeError, match="not loaded"):
            model.transcribe(waveform, sr, request_id="test-123")
