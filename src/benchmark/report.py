"""Benchmark report generator.

Generates JSON, CSV, and Markdown reports from BenchmarkRunResult.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from src.benchmark.runner import BenchmarkRunResult
from src.utils.config_loader import get_reports_dir
from src.utils.logger import get_logger

_logger = get_logger(__name__)


def generate_report(result: BenchmarkRunResult, formats: list[str] | None = None) -> dict[str, Path]:
    """Generate benchmark reports in specified formats.

    Args:
        result:  BenchmarkRunResult from runner.run()
        formats: List of "json", "csv", "markdown". Defaults to all three.

    Returns:
        Dict mapping format name to output file path.
    """
    if formats is None:
        formats = ["json", "csv", "markdown"]

    reports_dir = get_reports_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"benchmark_{result.model_id}_{ts}"
    output_paths: dict[str, Path] = {}

    summary = {
        "run_id": result.run_id,
        "model_id": result.model_id,
        "dataset_id": result.dataset_id,
        "device": result.device,
        "timestamp": result.timestamp.isoformat(),
        "total_samples": result.total_samples,
        "failed_samples": result.failed_samples,
        "wer": result.wer,
        "cer": result.cer,
        "avg_rtf": result.avg_rtf,
        "samples": result.samples,
    }

    if "json" in formats:
        json_path = reports_dir / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        output_paths["json"] = json_path
        _logger.info("JSON report: %s", json_path)

    if "csv" in formats:
        csv_path = reports_dir / f"{base_name}.csv"
        fieldnames = [
            "audio_path", "reference_transcript", "hypothesis_transcript",
            "wer", "cer", "rtf", "inference_time", "status", "error",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for s in result.samples:
                writer.writerow(s)
        output_paths["csv"] = csv_path
        _logger.info("CSV report: %s", csv_path)

    if "markdown" in formats:
        md_path = reports_dir / f"{base_name}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Benchmark Report\n\n")
            f.write(f"**Model:** `{result.model_id}`  \n")
            f.write(f"**Dataset:** `{result.dataset_id}`  \n")
            f.write(f"**Device:** `{result.device}`  \n")
            f.write(f"**Date:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  \n\n")
            f.write(f"## Aggregate Results\n\n")
            f.write(f"| Metric | Value |\n|:---|---:|\n")
            f.write(f"| WER | {result.wer:.4f} |\n" if result.wer is not None else "| WER | N/A |\n")
            f.write(f"| CER | {result.cer:.4f} |\n" if result.cer is not None else "| CER | N/A |\n")
            f.write(f"| Avg RTF | {result.avg_rtf:.4f} |\n" if result.avg_rtf is not None else "| Avg RTF | N/A |\n")
            f.write(f"| Total Samples | {result.total_samples} |\n")
            f.write(f"| Failed Samples | {result.failed_samples} |\n\n")
            f.write(f"## Per-Sample Results\n\n")
            f.write("| Audio | WER | CER | RTF | Status |\n|:---|---:|---:|---:|:---|\n")
            for s in result.samples:
                audio_name = Path(s.get("audio_path", "")).name
                wer = f"{s['wer']:.3f}" if "wer" in s else "—"
                cer = f"{s['cer']:.3f}" if "cer" in s else "—"
                rtf = f"{s['rtf']:.3f}" if "rtf" in s else "—"
                status = s.get("status", "?")
                f.write(f"| {audio_name} | {wer} | {cer} | {rtf} | {status} |\n")
        output_paths["markdown"] = md_path
        _logger.info("Markdown report: %s", md_path)

    return output_paths
