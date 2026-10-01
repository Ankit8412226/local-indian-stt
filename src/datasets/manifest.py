"""JSONL manifest utilities for dataset audio records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from src.datasets.registry import get_dataset_registry
from src.utils.logger import get_logger

_logger = get_logger(__name__)


def write_manifest(
    records: list[dict[str, Any]],
    output_path: str | Path,
    block_evaluation: bool = True,
) -> int:
    """Write a JSONL manifest file.

    Args:
        records:          List of audio record dicts.
        output_path:      Target .jsonl file path.
        block_evaluation: If True, raises ValueError if any record belongs to
                          an evaluation-only dataset.

    Returns:
        Number of records written.

    Raises:
        ValueError: If evaluation dataset records found and block_evaluation=True.
    """
    registry = get_dataset_registry()

    if block_evaluation:
        violations = [
            r for r in records
            if registry.is_evaluation_only(r.get("dataset", ""))
        ]
        if violations:
            eval_ids = {v["dataset"] for v in violations}
            raise ValueError(
                f"EVALUATION DATASET PROTECTION VIOLATION:\n"
                f"The following evaluation-only datasets were found in a training manifest:\n"
                f"  {eval_ids}\n"
                f"These datasets must NEVER be included in training data."
            )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    _logger.info("Wrote %d records to %s", count, path)
    return count


def read_manifest(manifest_path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL manifest file."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                _logger.warning("Skipping malformed line %d in %s: %s", lineno, path.name, e)


def make_record(
    audio_path: str,
    dataset: str,
    language: str = "hi",
    transcript: str = "",
    split: str = "train",
    **kwargs,
) -> dict[str, Any]:
    """Build a manifest record dict with required fields."""
    return {
        "audio_path": audio_path,
        "dataset": dataset,
        "language": language,
        "transcript": transcript,
        "split": split,
        **kwargs,
    }
