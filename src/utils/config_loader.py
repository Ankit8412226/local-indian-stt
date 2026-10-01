"""Configuration loader — reads YAML config files and environment overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Project root is two levels above this file (src/utils/config_loader.py)
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _ROOT / "config"


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load arbitrary YAML config file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_yaml(path: Path) -> dict[str, Any]:
    return load_yaml_config(path)


def load_models_config() -> dict[str, Any]:
    """Return parsed config/models.yaml."""
    return _load_yaml(_CONFIG_DIR / "models.yaml")


def load_datasets_config() -> dict[str, Any]:
    """Return parsed config/datasets.yaml."""
    return _load_yaml(_CONFIG_DIR / "datasets.yaml")


def load_audio_config() -> dict[str, Any]:
    """Return parsed config/audio.yaml with env-variable overrides applied."""
    cfg = _load_yaml(_CONFIG_DIR / "audio.yaml")
    return cfg


def load_benchmark_config() -> dict[str, Any]:
    """Return parsed config/benchmark.yaml."""
    cfg = _load_yaml(_CONFIG_DIR / "benchmark.yaml")
    # Allow env override for reports dir
    if env_dir := os.getenv("STT_REPORTS_DIR"):
        cfg["reports_dir"] = env_dir
    return cfg


def get_data_dir() -> Path:
    """Return the root data directory (env override: STT_DATA_DIR)."""
    return Path(os.getenv("STT_DATA_DIR", str(_ROOT / "data")))


def get_outputs_dir() -> Path:
    """Return the transcript outputs directory (env override: STT_OUTPUTS_DIR)."""
    path = Path(os.getenv("STT_OUTPUTS_DIR", str(_ROOT / "outputs")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_cache_dir() -> Path:
    """Return the model cache directory (env override: MODEL_CACHE_DIR)."""
    path = Path(os.getenv("MODEL_CACHE_DIR", str(_ROOT / "data" / "cache")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_reports_dir() -> Path:
    """Return the benchmark reports directory."""
    cfg = load_benchmark_config()
    path = Path(cfg.get("reports_dir", str(_ROOT / "reports")))
    path.mkdir(parents=True, exist_ok=True)
    return path
