"""Pytest configuration and shared fixtures."""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is importable in tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

# Use test MongoDB database name if not explicitly set
os.environ.setdefault("MONGODB_DATABASE", "test-transcription-test")



@pytest.fixture
def sample_wav_16k(tmp_path) -> str:
    """Generate a 1-second 16kHz mono sine-wave WAV file."""
    import soundfile as sf
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = str(tmp_path / "test_16k.wav")
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def sample_wav_8k(tmp_path) -> str:
    """Generate a 1-second 8kHz mono sine-wave WAV file (telephone-quality)."""
    import soundfile as sf
    sr = 8000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = str(tmp_path / "test_8k.wav")
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def sample_waveform_16k() -> tuple[np.ndarray, int]:
    """Return a 1-second 16kHz float32 waveform numpy array."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio, sr


@pytest.fixture
def mongo_db():
    """Return a MongoDB test database if available, else skip the test."""
    try:
        from src.database.connection import get_db, ping_db
        if not ping_db():
            pytest.skip("MongoDB not available — skipping DB test")
        db = get_db()
        yield db
        # Cleanup test collections after each test
        for col in db.list_collection_names():
            db[col].drop()
    except Exception as e:
        pytest.skip(f"MongoDB not available: {e}")
