"""Imports external model results (e.g., Gemini) for benchmark comparison.

File format: benchmark_external_results.jsonl
One JSON object per line:

{"audio_id": "...", "provider": "gemini", "model": "gemini-2.0-flash", "transcript": "..."}

These results are stored in the external_model_results collection and
compared against local model results using the same reference transcripts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

_logger = get_logger(__name__)


def parse_external_results(filepath: str | Path) -> list[dict[str, Any]]:
    """Parse a JSONL file of external model results.

    Returns a list of validated result dicts.
    Skips malformed lines with a warning.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"External results file not found: {path}")

    results = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                _validate_external_record(record, lineno)
                results.append(record)
            except (json.JSONDecodeError, ValueError) as e:
                _logger.warning("Skipping malformed line %d: %s", lineno, e)

    _logger.info("Parsed %d external results from %s", len(results), path.name)
    return results


def _validate_external_record(record: dict[str, Any], lineno: int) -> None:
    required = {"audio_id", "provider", "model", "transcript"}
    missing = required - record.keys()
    if missing:
        raise ValueError(
            f"Line {lineno}: missing required fields: {missing}"
        )
    if not isinstance(record["transcript"], str):
        raise ValueError(f"Line {lineno}: 'transcript' must be a string")


def import_to_db(filepath: str | Path, db) -> int:
    """Parse external results and upsert into MongoDB.

    Returns the number of records successfully imported.
    """
    from src.database.repository import ExternalResultRepository
    repo = ExternalResultRepository(db)
    records = parse_external_results(filepath)
    count = 0
    for r in records:
        try:
            repo.save(r)
            count += 1
        except Exception as e:
            _logger.warning("Failed to save external result: %s", e)
    _logger.info("Imported %d external results into MongoDB", count)
    return count
