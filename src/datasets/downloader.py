"""Dataset Downloader Engine — direct dataset acquisition for AWS GPU environment.

Supports resumable HTTP downloads, HuggingFace dataset downloads, and strict disk safety checks (df -h).
Stores files under data/raw/<dataset_id>/.
"""
from __future__ import annotations

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

from src.utils.config_loader import get_data_dir
from src.utils.logger import get_logger

_logger = get_logger(__name__)

# Registered download sources
DOWNLOAD_REGISTRY: dict[str, dict[str, Any]] = {
    "mucs-hindi-english": {
        "type": "http_tarball",
        "url": "https://www.openslr.org/resources/104/mucs_hindi_english.tar.gz",
        "estimated_compressed_gb": 1.5,
        "estimated_extracted_gb": 3.0,
        "description": "Microsoft Research MUCS Hindi-English Code-Switching Corpus (OpenSLR 104)",
    },
    "mozilla-cv-hindi": {
        "type": "huggingface",
        "hf_repo": "mozilla-foundation/common_voice_17_0",
        "subset": "hi",
        "estimated_compressed_gb": 2.0,
        "estimated_extracted_gb": 4.0,
        "description": "Mozilla Common Voice 17.0 Hindi",
    },
    "vaani": {
        "type": "huggingface",
        "hf_repo": "ARTPARK-IISc/Vaani-transcription-part",
        "estimated_compressed_gb": 3.0,
        "estimated_extracted_gb": 6.0,
        "description": "Project Vaani Telephony Speech Corpus (IISc/ARTPARK)",
    },
    "shrutilipi": {
        "type": "huggingface",
        "hf_repo": "ai4bharat/shrutilipi",
        "estimated_compressed_gb": 5.0,
        "estimated_extracted_gb": 10.0,
        "description": "AI4Bharat Shrutilipi Read & Broadcast Speech Corpus",
    },
    "indicvoices": {
        "type": "huggingface",
        "hf_repo": "ai4bharat/indicvoices",
        "estimated_compressed_gb": 5.0,
        "estimated_extracted_gb": 10.0,
        "description": "AI4Bharat IndicVoices Conversational Corpus",
    },
    "lahaja": {
        "type": "huggingface",
        "hf_repo": "ai4bharat/lahaja",
        "estimated_compressed_gb": 0.5,
        "estimated_extracted_gb": 1.0,
        "description": "Lahaja Hindi Dialect Benchmark (Evaluation Only)",
    },
    "vistaar": {
        "type": "huggingface",
        "hf_repo": "ai4bharat/vistaar",
        "estimated_compressed_gb": 1.0,
        "estimated_extracted_gb": 2.0,
        "description": "Vistaar ASR Benchmark Suite (Evaluation Only)",
    },
    "svarah": {
        "type": "huggingface",
        "hf_repo": "ai4bharat/svarah",
        "estimated_compressed_gb": 0.5,
        "estimated_extracted_gb": 1.0,
        "description": "Svarah Accent Benchmark (Evaluation Only)",
    },
}


def check_free_disk_space_gb(target_dir: Path) -> float:
    """Return available free disk space in GB for target directory."""
    stat = shutil.disk_usage(str(target_dir.resolve()))
    return round(stat.free / (1024 ** 3), 2)


def download_dataset(
    dataset_id: str,
    data_root: str | Path | None = None,
    min_free_gb_buffer: float = 10.0,
) -> Path:
    """Download raw dataset directly to data/raw/<dataset_id>/."""
    if dataset_id not in DOWNLOAD_REGISTRY:
        raise ValueError(
            f"Unknown dataset_id '{dataset_id}'. "
            f"Available: {list(DOWNLOAD_REGISTRY.keys())}"
        )

    info = DOWNLOAD_REGISTRY[dataset_id]
    if data_root is None:
        data_root = get_data_dir()

    raw_dir = Path(data_root).resolve() / "raw" / dataset_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Disk space safety check
    free_gb = check_free_disk_space_gb(raw_dir)
    required_gb = info["estimated_extracted_gb"] + min_free_gb_buffer
    _logger.info(
        "Disk Check: free=%.2f GB | required=%.2f GB (dataset %.2f GB + %.1f GB buffer)",
        free_gb, required_gb, info["estimated_extracted_gb"], min_free_gb_buffer,
    )
    if free_gb < required_gb:
        raise RuntimeError(
            f"Insufficient disk space for dataset '{dataset_id}'.\n"
            f"Available free disk: {free_gb} GB\n"
            f"Required (with safety buffer): {required_gb} GB\n"
            f"Please free disk space before downloading."
        )

    # Check if already downloaded
    existing_files = list(raw_dir.rglob("*"))
    if len([f for f in existing_files if f.is_file()]) > 5:
        _logger.info("Dataset '%s' already exists in %s — skipping download.", dataset_id, raw_dir)
        return raw_dir

    _logger.info("Downloading dataset '%s' (%s) to %s...", dataset_id, info["description"], raw_dir)

    dl_type = info["type"]
    if dl_type == "http_tarball":
        _download_http_tarball(info["url"], raw_dir)
    elif dl_type == "huggingface":
        _download_huggingface_dataset(info["hf_repo"], raw_dir, subset=info.get("subset"))

    _logger.info("✓ Download complete for '%s' -> %s", dataset_id, raw_dir)
    return raw_dir


def _download_http_tarball(url: str, target_dir: Path) -> None:
    tar_path = target_dir / "archive.tar.gz"
    _logger.info("Downloading HTTP tarball from %s...", url)
    urllib.request.urlretrieve(url, tar_path)

    _logger.info("Extracting tarball %s...", tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=target_dir)

    # Clean up archive file to save disk space
    if tar_path.exists():
        os.remove(tar_path)


def _download_huggingface_dataset(hf_repo: str, target_dir: Path, subset: str | None = None) -> None:
    try:
        from huggingface_hub import snapshot_download
        _logger.info("Downloading HuggingFace repo %s to %s...", hf_repo, target_dir)
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        _logger.warning("snapshot_download failed for %s (%s). Attempting datasets library...", hf_repo, e)
        try:
            from datasets import load_dataset
            ds = load_dataset(hf_repo, name=subset)
            ds.save_to_disk(str(target_dir / "hf_dataset"))
        except Exception as e2:
            raise RuntimeError(f"Failed to download HuggingFace dataset {hf_repo}: {e2}") from e2
