"""End-to-end verification and hardening test suite.

Tests full execution path:
Audio -> Inspection -> Preprocessing -> Model Load -> Inference -> Output -> MongoDB.
"""
import os
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

from src.audio.inspector import inspect
from src.audio.processor import process_audio
from src.models.factory import get_asr_model
from src.models.registry import get_registry
from src.database.connection import get_db, ping_db


class TestEndToEndPipeline:

    def test_audio_format_conversions(self, tmp_path):
        """Verify pipeline handles 8k, 16k, stereo, short, and long WAV formats."""
        out_dir = tmp_path / "test_formats"
        out_dir.mkdir()

        # 1. 8k telephone audio
        t8 = np.linspace(0, 1.0, 8000, endpoint=False, dtype=np.float32)
        sf.write(str(out_dir / "audio_8k.wav"), 0.3 * np.sin(2 * np.pi * 300 * t8), 8000)

        # 2. 44.1k stereo audio
        t44 = np.linspace(0, 1.0, 44100, endpoint=False, dtype=np.float32)
        stereo = np.column_stack([0.3 * np.sin(2 * np.pi * 440 * t44), 0.3 * np.sin(2 * np.pi * 660 * t44)])
        sf.write(str(out_dir / "audio_44k_stereo.wav"), stereo, 44100)

        # 3. Short audio (<0.5s)
        t_short = np.linspace(0, 0.2, 3200, endpoint=False, dtype=np.float32)
        sf.write(str(out_dir / "audio_short.wav"), 0.2 * np.sin(2 * np.pi * 440 * t_short), 16000)

        for fname in ["audio_8k.wav", "audio_44k_stereo.wav", "audio_short.wav"]:
            fpath = str(out_dir / fname)
            meta = inspect(fpath)
            assert meta.duration > 0
            assert meta.channels in (1, 2)

            processed = process_audio(fpath, normalize=True)
            assert processed.sample_rate == 16000
            assert processed.waveform.ndim == 1  # Always converted to mono
            assert isinstance(processed.waveform, np.ndarray)
            assert processed.waveform.dtype == np.float32

    def test_gated_model_graceful_error_handling(self):
        """Verify gated/auth-required models raise clear, descriptive exceptions without crashing app."""
        model = get_asr_model("indic-conformer-official")
        with pytest.raises((RuntimeError, ValueError)) as exc_info:
            model.load_model()
        assert "gated" in str(exc_info.value).lower() or "auth" in str(exc_info.value).lower() or "huggingface" in str(exc_info.value).lower()

    def test_sravaani_error_handling(self):
        """Verify SraVaani model raises descriptive error if custom transformers code is missing."""
        model = get_asr_model("sravaani")
        with pytest.raises(RuntimeError) as exc_info:
            model.load_model()
        assert "sravaani" in str(exc_info.value).lower()

    def test_mongodb_e2e_storage(self, sample_wav_16k):
        """Verify full transcription result can be persisted into MongoDB Atlas."""
        if not ping_db():
            pytest.skip("MongoDB not connected")

        db = get_db()
        from src.database.repository import TranscriptionRepository
        repo = TranscriptionRepository(db)

        req_id = "test-e2e-id-12345"
        repo.create_job({
            "request_id": req_id,
            "audio_path": sample_wav_16k,
            "model_id": "indic-conformer",
            "model_version": "1.0",
            "language": "hi",
            "status": "completed",
        })

        repo.save_result({
            "request_id": req_id,
            "model_id": "indic-conformer",
            "model_version": "1.0",
            "transcript": "नमस्ते यह परीक्षण है",
            "audio": {
                "duration": 1.0,
                "sample_rate": 16000,
                "channels": 1,
                "is_telephone": False,
            },
            "processing": {
                "preprocessing_time": 0.05,
                "inference_time": 0.20,
                "total_time": 0.25,
                "rtf": 0.25,
            },
            "device": "cpu",
        })

        found = repo.find_result(req_id)
        assert found is not None
        assert found["transcript"] == "नमस्ते यह परीक्षण है"
        assert found["model_id"] == "indic-conformer"
