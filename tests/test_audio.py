"""Tests for audio processing pipeline."""
import numpy as np
import pytest

from src.audio.inspector import inspect
from src.audio.converter import load_audio
from src.audio.quality import analyze
from src.audio.normalizer import peak_normalize, rms_normalize
from src.audio.processor import process_audio


class TestAudioInspector:
    def test_inspect_16k_wav(self, sample_wav_16k):
        meta = inspect(sample_wav_16k)
        assert meta.sample_rate == 16000
        assert meta.channels == 1
        assert abs(meta.duration - 1.0) < 0.05
        assert meta.is_telephone is False

    def test_inspect_8k_wav(self, sample_wav_8k):
        meta = inspect(sample_wav_8k)
        assert meta.sample_rate == 8000
        assert meta.is_telephone is True

    def test_inspect_missing_file(self):
        with pytest.raises(FileNotFoundError):
            inspect("/nonexistent/audio.wav")


class TestAudioConverter:
    def test_load_and_resample_8k_to_16k(self, sample_wav_8k):
        audio, sr = load_audio(sample_wav_8k, target_sr=16000)
        assert sr == 16000
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0

    def test_load_16k_unchanged(self, sample_wav_16k):
        audio, sr = load_audio(sample_wav_16k, target_sr=16000)
        assert sr == 16000
        assert len(audio) == pytest.approx(16000, abs=100)


class TestAudioQuality:
    def test_good_audio(self, sample_waveform_16k):
        waveform, sr = sample_waveform_16k
        report = analyze(waveform, sr)
        assert report.quality_flag == "good"
        assert report.rms > 0
        assert 0.0 <= report.clipping_ratio <= 1.0
        assert 0.0 <= report.silence_ratio <= 1.0

    def test_silent_audio(self):
        silent = np.zeros(16000, dtype=np.float32)
        report = analyze(silent, 16000)
        assert report.quality_flag == "silent"

    def test_clipped_audio(self):
        clipped = np.ones(16000, dtype=np.float32)
        report = analyze(clipped, 16000)
        assert report.quality_flag in {"clipped", "good"}


class TestAudioNormalizer:
    def test_peak_normalize(self, sample_waveform_16k):
        waveform, _ = sample_waveform_16k
        normalized = peak_normalize(waveform, target_peak=0.95)
        assert float(np.max(np.abs(normalized))) == pytest.approx(0.95, abs=0.01)

    def test_rms_normalize(self, sample_waveform_16k):
        waveform, _ = sample_waveform_16k
        normalized = rms_normalize(waveform, target_rms=0.1)
        rms = float(np.sqrt(np.mean(normalized ** 2)))
        assert rms == pytest.approx(0.1, rel=0.1)

    def test_silent_not_amplified_dangerously(self):
        silent = np.zeros(16000, dtype=np.float32)
        result = peak_normalize(silent)
        assert np.all(result == 0)


class TestAudioProcessor:
    def test_process_16k_wav(self, sample_wav_16k):
        result = process_audio(sample_wav_16k)
        assert result.sample_rate == 16000
        assert result.is_telephone is False
        assert len(result.waveform) > 0
        assert result.waveform.dtype == np.float32

    def test_process_8k_telephone_wav(self, sample_wav_8k):
        result = process_audio(sample_wav_8k)
        assert result.is_telephone is True
        assert result.sample_rate == 16000  # resampled
        assert result.metadata.sample_rate == 8000  # original preserved in metadata
