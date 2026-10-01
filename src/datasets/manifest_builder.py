"""Manifest Builder — creates unified train, validation, test, and quarantine manifests.

Strictly protects benchmark datasets (lahaja, vistaar, svarah) from training.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.datasets.auditor import audit_datasets
from src.datasets.registry import get_dataset_registry
from src.text.cleaning import clean_transcript
from src.text.code_switch import detect_code_switching as is_code_switched
from src.utils.config_loader import get_data_dir
from src.utils.logger import get_logger

_logger = get_logger(__name__)


def build_manifests(
    data_root: str | Path | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Generate train.jsonl, validation.jsonl, test.jsonl, and quarantine.jsonl."""
    if data_root is None:
        data_root = get_data_dir()

    data_path = Path(data_root).resolve()
    registry = get_dataset_registry()
    eval_only_ids = registry.list_evaluation_only_dataset_ids()

    summary, records = audit_datasets(data_path)

    manifest_dir = data_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []
    test_samples: list[dict[str, Any]] = []
    quarantine_samples: list[dict[str, Any]] = []

    random.seed(seed)

    for rec in records:
        # Check corrupted audio
        if rec.is_corrupted:
            quarantine_samples.append({
                "audio_path": rec.file_path,
                "dataset": rec.dataset_id,
                "quarantine_reason": f"corrupted_audio: {rec.error}",
            })
            continue

        # Check evaluation dataset protection
        if rec.dataset_id in eval_only_ids:
            quarantine_samples.append({
                "audio_path": rec.file_path,
                "dataset": rec.dataset_id,
                "quarantine_reason": "evaluation_only_dataset_protected",
            })
            continue

        # Check audio length extremes (<0.3s or >30s)
        if rec.duration_s < 0.3 or rec.duration_s > 30.0:
            quarantine_samples.append({
                "audio_path": rec.file_path,
                "dataset": rec.dataset_id,
                "duration": rec.duration_s,
                "quarantine_reason": "duration_out_of_bounds",
            })
            continue

        # Clean transcript
        cleaned, is_valid, reason = clean_transcript(rec.transcript)
        if not is_valid:
            quarantine_samples.append({
                "audio_path": rec.file_path,
                "dataset": rec.dataset_id,
                "original_transcript": rec.transcript or "",
                "quarantine_reason": reason or "invalid_transcript",
            })
            continue

        code_switch = is_code_switched(cleaned)

        sample = {
            "audio_path": rec.file_path,
            "original_transcript": rec.transcript or "",
            "normalized_transcript": cleaned,
            "language": "hi",
            "dataset": rec.dataset_id,
            "speaker_id": "unknown",
            "duration": rec.duration_s,
            "sample_rate": rec.sample_rate,
            "is_telephone": rec.is_telephone,
            "is_code_switch": code_switch,
        }

        # Train / Val / Test split
        r = random.random()
        if r < val_ratio:
            sample["split"] = "validation"
            val_samples.append(sample)
        elif r < val_ratio + test_ratio:
            sample["split"] = "test"
            test_samples.append(sample)
        else:
            sample["split"] = "train"
            train_samples.append(sample)

    # Write JSONL manifests
    _write_jsonl(manifest_dir / "train.jsonl", train_samples)
    _write_jsonl(manifest_dir / "validation.jsonl", val_samples)
    _write_jsonl(manifest_dir / "test.jsonl", test_samples)
    _write_jsonl(manifest_dir / "quarantine.jsonl", quarantine_samples)

    counts = {
        "train": len(train_samples),
        "validation": len(val_samples),
        "test": len(test_samples),
        "quarantine": len(quarantine_samples),
    }

    _logger.info(
        "Manifests built successfully: train=%d val=%d test=%d quarantine=%d",
        counts["train"], counts["validation"], counts["test"], counts["quarantine"],
    )
    return counts


def _write_jsonl(path: Path, data: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
