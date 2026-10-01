"""Dataset registry — provides access to dataset metadata via MongoDB repository."""
from __future__ import annotations

from typing import Any

from src.utils.config_loader import load_datasets_config
from src.utils.logger import get_logger

_logger = get_logger(__name__)

# Role constants
TRAINING = "TRAINING"
VALIDATION = "VALIDATION"
TEST = "TEST"
BENCHMARK = "BENCHMARK"

# Protected roles — never allowed in training manifests
PROTECTED_ROLES = {BENCHMARK, TEST}


class DatasetRegistry:
    """Loads dataset metadata from config/datasets.yaml."""

    def __init__(self) -> None:
        cfg = load_datasets_config()
        self._datasets: dict[str, dict[str, Any]] = {
            d["dataset_id"]: d for d in cfg.get("datasets", [])
        }
        self._eval_ids: set[str] = set(cfg.get("evaluation_only_dataset_ids", []))

    def get(self, dataset_id: str) -> dict[str, Any]:
        if dataset_id not in self._datasets:
            raise KeyError(
                f"Dataset '{dataset_id}' not found in config/datasets.yaml. "
                f"Available: {list(self._datasets.keys())}"
            )
        return self._datasets[dataset_id]

    def list_dataset_ids(self) -> list[str]:
        """Return list of all registered dataset IDs."""
        return list(self._datasets.keys())

    def list_evaluation_only_dataset_ids(self) -> list[str]:
        """Return list of all evaluation-only dataset IDs."""
        return list(self._eval_ids.union({d["dataset_id"] for d in self._datasets.values() if d.get("role") in PROTECTED_ROLES}))

    def list_all(self, role: str | None = None) -> list[dict[str, Any]]:
        datasets = list(self._datasets.values())
        if role:
            datasets = [d for d in datasets if d.get("role") == role]
        return sorted(datasets, key=lambda d: d["dataset_id"])

    def is_evaluation_only(self, dataset_id: str) -> bool:
        """Return True if this dataset must never be used in training."""
        record = self._datasets.get(dataset_id, {})
        return (
            dataset_id in self._eval_ids
            or record.get("role") in PROTECTED_ROLES
        )

    def validate_training_manifest_ids(self, dataset_ids: list[str]) -> list[str]:
        """Return list of violation dataset_ids that must not be in training manifests."""
        return [d for d in dataset_ids if self.is_evaluation_only(d)]


_registry: DatasetRegistry | None = None


def get_dataset_registry() -> DatasetRegistry:
    global _registry
    if _registry is None:
        _registry = DatasetRegistry()
    return _registry
