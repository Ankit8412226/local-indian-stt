"""MongoDB repository layer.

All MongoDB interactions must go through these repository classes.
The rest of the application must NOT call collection.find() or
collection.insert_one() directly.

Repositories:
    DatasetRepository
    ModelRepository
    TranscriptionRepository
    BenchmarkRepository
    ExternalResultRepository
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from src.database import collections as C
from src.utils.logger import get_logger

_logger = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Dataset Repository
# ---------------------------------------------------------------------------

class DatasetRepository:
    def __init__(self, db: Database) -> None:
        self._col = db[C.DATASETS]
        self._files = db[C.DATASET_FILES]

    def upsert(self, dataset: dict[str, Any]) -> str:
        """Insert or update a dataset record by dataset_id."""
        dataset["updated_at"] = _now()
        if "created_at" not in dataset:
            dataset["created_at"] = _now()
        result = self._col.update_one(
            {"dataset_id": dataset["dataset_id"]},
            {"$set": dataset},
            upsert=True,
        )
        _logger.debug("Upserted dataset: %s", dataset["dataset_id"])
        return dataset["dataset_id"]

    def find_by_id(self, dataset_id: str) -> dict[str, Any] | None:
        return self._col.find_one({"dataset_id": dataset_id}, {"_id": 0})

    def list_all(self, role: str | None = None) -> list[dict[str, Any]]:
        query = {}
        if role:
            query["role"] = role
        return list(self._col.find(query, {"_id": 0}).sort("name", 1))

    def count(self) -> int:
        return self._col.count_documents({})

    def insert_file(self, file_record: dict[str, Any]) -> str:
        """Insert a single dataset_file record. Returns inserted _id as str."""
        file_record.setdefault("created_at", _now())
        result = self._files.insert_one(file_record)
        return str(result.inserted_id)

    def count_files(self, dataset_id: str) -> int:
        return self._files.count_documents({"dataset_id": dataset_id})

    def list_files(
        self,
        dataset_id: str,
        split: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"dataset_id": dataset_id}
        if split:
            query["split"] = split
        return list(self._files.find(query, {"_id": 0}).limit(limit))

    def is_evaluation_dataset(self, dataset_id: str) -> bool:
        """Return True if this dataset has role=BENCHMARK or role=TEST."""
        record = self.find_by_id(dataset_id)
        if record is None:
            return False
        return record.get("role") in {"BENCHMARK", "TEST"}


# ---------------------------------------------------------------------------
# Model Repository
# ---------------------------------------------------------------------------

class ModelRepository:
    def __init__(self, db: Database) -> None:
        self._col = db[C.MODELS]

    def upsert(self, model: dict[str, Any]) -> str:
        model["updated_at"] = _now()
        if "created_at" not in model:
            model["created_at"] = _now()
        self._col.update_one(
            {"model_id": model["model_id"]},
            {"$set": model},
            upsert=True,
        )
        return model["model_id"]

    def find_by_id(self, model_id: str) -> dict[str, Any] | None:
        return self._col.find_one({"model_id": model_id}, {"_id": 0})

    def list_all(self) -> list[dict[str, Any]]:
        return list(self._col.find({}, {"_id": 0}).sort("model_id", 1))


# ---------------------------------------------------------------------------
# Transcription Repository
# ---------------------------------------------------------------------------

class TranscriptionRepository:
    def __init__(self, db: Database) -> None:
        self._jobs = db[C.TRANSCRIPTION_JOBS]
        self._results = db[C.TRANSCRIPTION_RESULTS]

    def create_job(self, job: dict[str, Any]) -> str:
        """Insert a new transcription job. request_id must be unique."""
        job.setdefault("created_at", _now())
        job.setdefault("status", "pending")
        try:
            self._jobs.insert_one(job)
        except DuplicateKeyError:
            _logger.warning("Duplicate request_id: %s", job.get("request_id"))
        return job.get("request_id", "")

    def update_job_status(
        self,
        request_id: str,
        status: str,
        completed_at: datetime | None = None,
    ) -> None:
        update: dict[str, Any] = {"status": status}
        if completed_at:
            update["completed_at"] = completed_at
        self._jobs.update_one({"request_id": request_id}, {"$set": update})

    def save_result(self, result: dict[str, Any]) -> str:
        result.setdefault("created_at", _now())
        inserted = self._results.insert_one(result)
        return str(inserted.inserted_id)

    def find_result(self, request_id: str) -> dict[str, Any] | None:
        return self._results.find_one({"request_id": request_id}, {"_id": 0})

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return list(
            self._results.find({}, {"_id": 0})
            .sort("created_at", -1)
            .limit(limit)
        )


# ---------------------------------------------------------------------------
# Benchmark Repository
# ---------------------------------------------------------------------------

class BenchmarkRepository:
    def __init__(self, db: Database) -> None:
        self._runs = db[C.BENCHMARK_RUNS]
        self._results = db[C.BENCHMARK_RESULTS]

    def save_run(self, run: dict[str, Any]) -> str:
        run.setdefault("timestamp", _now())
        try:
            self._runs.insert_one(run)
        except DuplicateKeyError:
            _logger.warning("Duplicate run_id: %s", run.get("run_id"))
        return run.get("run_id", "")

    def save_result(self, result: dict[str, Any]) -> str:
        inserted = self._results.insert_one(result)
        return str(inserted.inserted_id)

    def find_run(self, run_id: str) -> dict[str, Any] | None:
        return self._runs.find_one({"run_id": run_id}, {"_id": 0})

    def list_runs(self, model_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        query = {}
        if model_id:
            query["model_id"] = model_id
        return list(
            self._runs.find(query, {"_id": 0}).sort("timestamp", -1).limit(limit)
        )

    def list_results_for_run(self, run_id: str) -> list[dict[str, Any]]:
        return list(self._results.find({"run_id": run_id}, {"_id": 0}))


# ---------------------------------------------------------------------------
# External Model Results Repository (e.g., Gemini)
# ---------------------------------------------------------------------------

class ExternalResultRepository:
    def __init__(self, db: Database) -> None:
        self._col = db[C.EXTERNAL_MODEL_RESULTS]

    def save(self, result: dict[str, Any]) -> str:
        result.setdefault("created_at", _now())
        inserted = self._col.insert_one(result)
        return str(inserted.inserted_id)

    def find_by_audio(self, audio_file_id: str) -> list[dict[str, Any]]:
        return list(self._col.find({"audio_file_id": audio_file_id}, {"_id": 0}))

    def list_by_provider(self, provider: str, limit: int = 100) -> list[dict[str, Any]]:
        return list(
            self._col.find({"provider": provider}, {"_id": 0})
            .sort("created_at", -1)
            .limit(limit)
        )
