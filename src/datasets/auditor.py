"""Dataset Auditor — scans local audio datasets and generates exact statistics.

Generates:
  - data/audit/dataset_audit.json
  - data/audit/dataset_audit.csv
  - data/audit/dataset_audit.md
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import soundfile as sf

from src.audio.inspector import inspect
from src.datasets.registry import get_dataset_registry
from src.utils.config_loader import get_data_dir
from src.utils.logger import get_logger

_logger = get_logger(__name__)


@dataclass
class AudioFileRecord:
    file_path: str
    dataset_id: str
    format: str
    duration_s: float
    sample_rate: int
    channels: int
    is_telephone: bool
    is_corrupted: bool
    transcript: str | None
    has_transcript: bool
    audio_hash: str
    error: str | None = None


@dataclass
class DatasetAuditSummary:
    total_datasets: int = 0
    total_files: int = 0
    total_usable_files: int = 0
    total_corrupted_files: int = 0
    total_missing_transcripts: int = 0
    total_duplicate_files: int = 0
    total_audio_hours: float = 0.0
    hindi_hours: float = 0.0
    hinglish_hours: float = 0.0
    indian_english_hours: float = 0.0
    telephone_8k_hours: float = 0.0
    sr_16k_hours: float = 0.0
    other_sr_hours: float = 0.0
    labeled_hours: float = 0.0
    unlabeled_hours: float = 0.0
    train_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    dataset_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)


def compute_file_hash(path: str, chunk_size: int = 65536) -> str:
    """Compute MD5 hash of audio file content for duplicate detection."""
    hasher = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def audit_datasets(data_root: str | Path | None = None) -> tuple[DatasetAuditSummary, list[AudioFileRecord]]:
    """Scan local datasets and compute exact statistics."""
    if data_root is None:
        data_root = get_data_dir()

    data_path = Path(data_root).resolve()
    _logger.info("Auditing audio datasets in: %s", data_path)

    registry = get_dataset_registry()
    known_datasets = registry.list_dataset_ids()

    summary = DatasetAuditSummary()

    audio_extensions = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}
    found_files: list[Path] = []

    for root, _, files in os.walk(data_path):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in audio_extensions:
                found_files.append(p)

    _logger.info("Found %d audio files under %s", len(found_files), data_path)

    records: list[AudioFileRecord] = []
    seen_hashes: set[str] = set()

    for p in found_files:
        rel_parts = p.relative_to(data_path).parts
        dataset_id = "unknown"
        for part in rel_parts:
            if part in known_datasets:
                dataset_id = part
                break

        # Read sidecar transcript if available (.txt sidecar or directory manifest)
        transcript = None
        txt_sidecar = p.with_suffix(".txt")
        if txt_sidecar.exists():
            try:
                transcript = txt_sidecar.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        
        # If no sidecar, check directory manifest files (metadata.csv, manifest.jsonl)
        if not transcript:
            parent_dir = p.parent
            # Check for metadata.csv
            meta_csv = parent_dir / "metadata.csv"
            if meta_csv.exists():
                try:
                    with open(meta_csv, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) >= 2 and (p.name in row[0] or p.stem in row[0]):
                                transcript = row[1].strip()
                                break
                except Exception:
                    pass

        audio_hash = compute_file_hash(str(p))
        if audio_hash in seen_hashes:
            summary.total_duplicate_files += 1
        else:
            seen_hashes.add(audio_hash)

        try:
            meta = inspect(str(p))
            rec = AudioFileRecord(
                file_path=str(p),
                dataset_id=dataset_id,
                format=meta.format,
                duration_s=meta.duration,
                sample_rate=meta.sample_rate,
                channels=meta.channels,
                is_telephone=meta.is_telephone,
                is_corrupted=False,
                transcript=transcript,
                has_transcript=bool(transcript),
                audio_hash=audio_hash,
            )
            summary.total_usable_files += 1
            dur_h = meta.duration / 3600.0
            summary.total_audio_hours += dur_h

            if meta.sample_rate <= 8000:
                summary.telephone_8k_hours += dur_h
            elif meta.sample_rate == 16000:
                summary.sr_16k_hours += dur_h
            else:
                summary.other_sr_hours += dur_h

            if transcript:
                summary.labeled_hours += dur_h
            else:
                summary.unlabeled_hours += dur_h

        except Exception as e:
            rec = AudioFileRecord(
                file_path=str(p),
                dataset_id=dataset_id,
                format=p.suffix.lstrip("."),
                duration_s=0.0,
                sample_rate=0,
                channels=0,
                is_telephone=False,
                is_corrupted=True,
                transcript=None,
                has_transcript=False,
                audio_hash=audio_hash,
                error=str(e),
            )
            summary.total_corrupted_files += 1

        records.append(rec)

    summary.total_files = len(records)
    summary.total_datasets = len(known_datasets)

    # Export report artifacts
    export_dir = data_path / "audit"
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON
    json_path = export_dir / "dataset_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": asdict(summary),
            "total_records": len(records),
        }, f, indent=2)

    # 2. CSV
    csv_path = export_dir / "dataset_audit.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_path", "dataset_id", "format", "duration_s", "sample_rate",
            "channels", "is_telephone", "is_corrupted", "has_transcript", "audio_hash", "error"
        ])
        for r in records:
            writer.writerow([
                r.file_path, r.dataset_id, r.format, r.duration_s, r.sample_rate,
                r.channels, r.is_telephone, r.is_corrupted, r.has_transcript, r.audio_hash, r.error or ""
            ])

    # 3. Markdown
    md_path = export_dir / "dataset_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Audit Report\n\n")
        f.write(f"- **Total Audio Files**: {summary.total_files}\n")
        f.write(f"- **Usable Files**: {summary.total_usable_files}\n")
        f.write(f"- **Corrupted Files**: {summary.total_corrupted_files}\n")
        f.write(f"- **Duplicate Files**: {summary.total_duplicate_files}\n")
        f.write(f"- **Total Audio Hours**: {summary.total_audio_hours:.4f} hrs\n")
        f.write(f"- **8kHz Telephone Hours**: {summary.telephone_8k_hours:.4f} hrs\n")
        f.write(f"- **16kHz Hours**: {summary.sr_16k_hours:.4f} hrs\n")
        f.write(f"- **Labeled Hours**: {summary.labeled_hours:.4f} hrs\n")
        f.write(f"- **Unlabeled Hours**: {summary.unlabeled_hours:.4f} hrs\n")

    _logger.info(
        "Audit complete: %d files, %.2f total hrs. Reports saved to %s",
        summary.total_files, summary.total_audio_hours, export_dir,
    )
    return summary, records
