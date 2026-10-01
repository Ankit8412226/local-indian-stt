"""Base ASR model abstraction.

All ASR model implementations must subclass BaseASRModel.
The base class defines the contract; implementations must not depend on CLI,
database, or other models.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class TranscriptionOutput:
    """Structured output from a single ASR inference call.

    This schema maps directly to the transcription_results MongoDB collection.
    """
    request_id: str
    model_id: str
    model_version: str
    language: str
    transcript: str
    transcript_normalized: str
    audio_duration: float          # seconds
    audio_sample_rate: int
    audio_channels: int
    is_telephone: bool
    preprocessing_time: float      # seconds
    inference_time: float          # seconds
    total_time: float              # seconds
    rtf: float                     # Real-Time Factor = inference_time / audio_duration
    device: str
    confidence: float | None = None  # 0.0–1.0 if model provides it
    raw_output: dict[str, Any] = field(default_factory=dict)  # Model-specific extras


class BaseASRModel(ABC):
    """Abstract base class for all ASR model implementations.

    Lifecycle:
        model = ConcreteASRModel(model_id="...", device=device)
        model.load_model()              # Download/load weights
        result = model.transcribe(...)  # Run inference
    """

    def __init__(self, model_id: str, device: torch.device) -> None:
        self.model_id = model_id
        self.device = device
        self._loaded: bool = False

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and processor into memory on self.device.

        Must set self._loaded = True on success.
        Must raise RuntimeError with a clear message on failure.
        """
        ...

    @abstractmethod
    def _run_inference(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        language: str,
    ) -> tuple[str, dict[str, Any]]:
        """Run raw model inference.

        Args:
            waveform:    Float32 mono numpy array at model's expected sample rate.
            sample_rate: Sample rate of the waveform.
            language:    BCP-47 language code (e.g., "hi", "en").

        Returns:
            (transcript_text, raw_output_dict)
        """
        ...

    def transcribe(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        request_id: str,
        language: str = "hi",
        audio_duration: float | None = None,
        audio_channels: int = 1,
        is_telephone: bool = False,
        preprocessing_time: float = 0.0,
    ) -> TranscriptionOutput:
        """Run inference and return structured TranscriptionOutput.

        This method wraps _run_inference() with timing, RTF calculation,
        and output schema construction.

        Args:
            waveform:          Float32 mono numpy array.
            sample_rate:       Sample rate of waveform.
            request_id:        Unique request identifier.
            language:          Target language code.
            audio_duration:    Pre-computed duration (computed from waveform if None).
            audio_channels:    Number of channels in original audio.
            is_telephone:      Whether source audio is telephone-quality.
            preprocessing_time: Time already spent in audio preprocessing (seconds).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._loaded:
            raise RuntimeError(
                f"Model '{self.model_id}' is not loaded. Call load_model() first."
            )

        if audio_duration is None:
            audio_duration = len(waveform) / sample_rate

        from src.text.normalizer import normalize_for_wer
        from src.text.code_switch import detect_code_switching

        t0 = time.perf_counter()
        transcript, raw_output = self._run_inference(waveform, sample_rate, language)
        inference_time = time.perf_counter() - t0

        total_time = preprocessing_time + inference_time
        rtf = inference_time / audio_duration if audio_duration > 0 else 0.0

        transcript_normalized = normalize_for_wer(transcript, language)

        return TranscriptionOutput(
            request_id=request_id,
            model_id=self.model_id,
            model_version=getattr(self, "model_version", "unknown"),
            language=language,
            transcript=transcript,
            transcript_normalized=transcript_normalized,
            audio_duration=round(audio_duration, 4),
            audio_sample_rate=sample_rate,
            audio_channels=audio_channels,
            is_telephone=is_telephone,
            preprocessing_time=round(preprocessing_time, 4),
            inference_time=round(inference_time, 4),
            total_time=round(total_time, 4),
            rtf=round(rtf, 6),
            device=str(self.device),
            raw_output=raw_output,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_id={self.model_id!r}, "
            f"device={self.device}, "
            f"loaded={self._loaded})"
        )
