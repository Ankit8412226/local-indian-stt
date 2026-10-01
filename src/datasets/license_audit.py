"""License Audit Module — generates legal/commercial usability reports for all registered datasets.

Generates:
  - data/audit/license_audit.json
  - data/audit/license_audit.md
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.datasets.registry import get_dataset_registry
from src.utils.config_loader import get_data_dir
from src.utils.logger import get_logger

_logger = get_logger(__name__)

LICENSE_AUDIT_DATA: list[dict[str, Any]] = [
    {
        "dataset_id": "mucs-hindi-english",
        "name": "MUCS Hindi-English (OpenSLR 104)",
        "official_source": "https://www.openslr.org/104",
        "license": "CC-BY-SA-4.0",
        "commercial_use": False,
        "redistribution_allowed": True,
        "attribution_required": True,
        "share_alike_required": True,
        "access_requirements": "Open access HTTP download",
        "download_method": "Direct HTTP tarball",
        "status": "REVIEW_REQUIRED",
        "reason": "CC-BY-SA-4.0 ShareAlike license requires derivative models to be released under same copyleft license.",
    },
    {
        "dataset_id": "mozilla-cv-hindi",
        "name": "Mozilla Common Voice Hindi",
        "official_source": "https://commonvoice.mozilla.org/hi",
        "license": "CC0-1.0",
        "commercial_use": True,
        "redistribution_allowed": True,
        "attribution_required": False,
        "share_alike_required": False,
        "access_requirements": "Open access",
        "download_method": "HuggingFace datasets / Direct TSV archive",
        "status": "ALLOWED",
        "reason": "Public domain CC0 license allows unrestricted commercial derivative model training.",
    },
    {
        "dataset_id": "vaani",
        "name": "Project Vaani",
        "official_source": "https://vaani.iisc.ac.in",
        "license": "CC-BY-4.0",
        "commercial_use": True,
        "redistribution_allowed": True,
        "attribution_required": True,
        "share_alike_required": False,
        "access_requirements": "Open access",
        "download_method": "HuggingFace ARTPARK-IISc/Vaani-transcription-part",
        "status": "ALLOWED",
        "reason": "CC-BY-4.0 permits commercial derivative models with proper attribution to IISc/ARTPARK.",
    },
    {
        "dataset_id": "indicvoices",
        "name": "IndicVoices",
        "official_source": "https://ai4bharat.iitm.ac.in",
        "license": "CC-BY-4.0",
        "commercial_use": True,
        "redistribution_allowed": True,
        "attribution_required": True,
        "share_alike_required": False,
        "access_requirements": "Open access",
        "download_method": "HuggingFace ai4bharat/indicvoices",
        "status": "ALLOWED",
        "reason": "CC-BY-4.0 permits commercial derivative model creation with attribution to AI4Bharat.",
    },
    {
        "dataset_id": "shrutilipi",
        "name": "Shrutilipi",
        "official_source": "https://ai4bharat.iitm.ac.in",
        "license": "CC-BY-4.0",
        "commercial_use": True,
        "redistribution_allowed": True,
        "attribution_required": True,
        "share_alike_required": False,
        "access_requirements": "Open access",
        "download_method": "HuggingFace ai4bharat/shrutilipi",
        "status": "ALLOWED",
        "reason": "CC-BY-4.0 permits commercial model fine-tuning with attribution to AI4Bharat.",
    },
    {
        "dataset_id": "lahaja",
        "name": "Lahaja (Hindi Dialect Benchmark)",
        "official_source": "https://ai4bharat.iitm.ac.in",
        "license": "MIT",
        "commercial_use": True,
        "redistribution_allowed": True,
        "attribution_required": False,
        "share_alike_required": False,
        "access_requirements": "Open access",
        "download_method": "HuggingFace ai4bharat/lahaja",
        "status": "HELD_OUT_EVALUATION",
        "reason": "Evaluation-only benchmark dataset. Must NOT be included in training manifests.",
    },
    {
        "dataset_id": "vistaar",
        "name": "Vistaar (ASR Benchmark Suite)",
        "official_source": "https://github.com/AI4Bharat/vistaar",
        "license": "MIT",
        "commercial_use": True,
        "redistribution_allowed": True,
        "attribution_required": False,
        "share_alike_required": False,
        "access_requirements": "Open access",
        "download_method": "HuggingFace ai4bharat/vistaar",
        "status": "HELD_OUT_EVALUATION",
        "reason": "Evaluation-only benchmark suite across 12 Indic languages. Must NOT be included in training manifests.",
    },
    {
        "dataset_id": "svarah",
        "name": "Svarah (Accent Benchmark)",
        "official_source": "https://ai4bharat.iitm.ac.in",
        "license": "MIT",
        "commercial_use": True,
        "redistribution_allowed": True,
        "attribution_required": False,
        "share_alike_required": False,
        "access_requirements": "Open access",
        "download_method": "HuggingFace ai4bharat/svarah",
        "status": "HELD_OUT_EVALUATION",
        "reason": "Evaluation-only accent benchmark. Must NOT be included in training manifests.",
    },
]


def generate_license_audit_report(data_root: str | Path | None = None) -> Path:
    """Generate license audit JSON and Markdown reports."""
    if data_root is None:
        data_root = get_data_dir()

    export_dir = Path(data_root).resolve() / "audit"
    export_dir.mkdir(parents=True, exist_ok=True)

    json_path = export_dir / "license_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(LICENSE_AUDIT_DATA, f, indent=2)

    md_path = export_dir / "license_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset License Audit Report\n\n")
        f.write("| Dataset | Official Source | License | Commercial Use | Status | Reason |\n")
        f.write("|---|---|---|---|---|---|\n")
        for item in LICENSE_AUDIT_DATA:
            comm_str = "✅ YES" if item["commercial_use"] else "⚠️ REVIEW_REQUIRED"
            f.write(f"| **{item['dataset_id']}** | [{item['name']}]({item['official_source']}) | `{item['license']}` | {comm_str} | **{item['status']}** | {item['reason']} |\n")

    _logger.info("License audit report generated at %s", md_path)
    return md_path
