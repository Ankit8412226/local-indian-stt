"""Benchmark runner — runs one or more ASR models against a set of audio files.

Usage:
    from src.benchmark.runner import BenchmarkRunner
    runner = BenchmarkRunner(model_id="whisper-medium", dataset_id="lahaja")
    runner.run()
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.benchmark.metrics import word_error_rate, char_error_rate, real_time_factor
from src.models.factory import get_asr_model
from src.text.normalizer import normalize_for_wer
from src.utils.device import resolve_device
from src.utils.logger import get_logger

_logger = get_logger(__name__)


@dataclass
class BenchmarkSample:
    audio_path: str
    reference_transcript: str
    language: str = "hi"
    audio_file_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRunResult:
    run_id: str
    model_id: str
    dataset_id: str
    device: str
    timestamp: datetime
    samples: list[dict[str, Any]] = field(default_factory=list)
    wer: float | None = None
    cer: float | None = None
    avg_rtf: float | None = None
    total_samples: int = 0
    failed_samples: int = 0


class BenchmarkRunner:
    """Runs a single model against a list of BenchmarkSamples."""

    def __init__(
        self,
        model_id: str,
        dataset_id: str,
        device_str: str = "auto",
        language: str = "hi",
    ) -> None:
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.device = resolve_device(device_str)
        self.language = language
        self.run_id = str(uuid.uuid4())

    def run(
        self,
        samples: list[BenchmarkSample],
        db=None,
    ) -> BenchmarkRunResult:
        """Run benchmark on provided samples.

        Args:
            samples: List of BenchmarkSample (audio + reference transcript).
            db:      Optional MongoDB Database. If provided, saves results.

        Returns:
            BenchmarkRunResult with per-sample metrics and aggregate scores.
        """
        _logger.info(
            "Benchmark run %s | model=%s dataset=%s samples=%d device=%s",
            self.run_id, self.model_id, self.dataset_id, len(samples), self.device,
        )

        # Load model once for all samples
        model = get_asr_model(self.model_id, device=self.device)
        model.load_model()

        result = BenchmarkRunResult(
            run_id=self.run_id,
            model_id=self.model_id,
            dataset_id=self.dataset_id,
            device=str(self.device),
            timestamp=datetime.now(timezone.utc),
            total_samples=len(samples),
        )

        from src.audio.processor import process_audio
        wers, cers, rtfs = [], [], []

        for i, sample in enumerate(samples):
            _logger.info(
                "  [%d/%d] %s", i + 1, len(samples), Path(sample.audio_path).name
            )
            try:
                t_pre_start = time.perf_counter()
                processed = process_audio(sample.audio_path)
                preprocessing_time = time.perf_counter() - t_pre_start

                output = model.transcribe(
                    waveform=processed.waveform,
                    sample_rate=processed.sample_rate,
                    request_id=str(uuid.uuid4()),
                    language=sample.language or self.language,
                    audio_duration=processed.metadata.duration,
                    audio_channels=processed.metadata.channels,
                    is_telephone=processed.is_telephone,
                    preprocessing_time=preprocessing_time,
                )

                ref_norm = normalize_for_wer(sample.reference_transcript, sample.language)
                hyp_norm = output.transcript_normalized
                wer = word_error_rate(ref_norm, hyp_norm)
                cer = char_error_rate(ref_norm, hyp_norm)
                rtf = output.rtf

                wers.append(wer)
                cers.append(cer)
                rtfs.append(rtf)

                sample_result = {
                    "run_id": self.run_id,
                    "audio_file_id": sample.audio_file_id,
                    "audio_path": sample.audio_path,
                    "reference_transcript": sample.reference_transcript,
                    "hypothesis_transcript": output.transcript,
                    "wer": wer,
                    "cer": cer,
                    "rtf": rtf,
                    "inference_time": output.inference_time,
                    "status": "success",
                }
                result.samples.append(sample_result)

                if db:
                    from src.database.repository import BenchmarkRepository
                    BenchmarkRepository(db).save_result(sample_result)

            except Exception as e:
                _logger.error("Sample failed: %s — %s", sample.audio_path, e)
                result.failed_samples += 1
                result.samples.append({
                    "run_id": self.run_id,
                    "audio_path": sample.audio_path,
                    "status": "failed",
                    "error": str(e),
                })

        # Aggregate metrics
        if wers:
            result.wer = round(sum(wers) / len(wers), 6)
            result.cer = round(sum(cers) / len(cers), 6)
            result.avg_rtf = round(sum(rtfs) / len(rtfs), 6)

        _logger.info(
            "Benchmark complete | WER=%.3f CER=%.3f RTF=%.3f failed=%d",
            result.wer or 0, result.cer or 0, result.avg_rtf or 0, result.failed_samples,
        )

        if db:
            from src.database.repository import BenchmarkRepository
            BenchmarkRepository(db).save_run({
                "run_id": result.run_id,
                "dataset_id": result.dataset_id,
                "model_id": result.model_id,
                "device": result.device,
                "timestamp": result.timestamp,
                "total_samples": result.total_samples,
                "failed_samples": result.failed_samples,
                "wer": result.wer,
                "cer": result.cer,
                "avg_rtf": result.avg_rtf,
            })

        return result
