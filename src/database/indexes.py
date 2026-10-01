"""MongoDB index creation for all collections.

Run once at startup or via: python -m src.database.init
Indexes are idempotent — safe to re-run.
"""
from __future__ import annotations

from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database

from src.database import collections as C
from src.utils.logger import get_logger

_logger = get_logger(__name__)


def create_all_indexes(db: Database) -> None:
    """Create all required MongoDB indexes. Idempotent."""
    _logger.info("Creating MongoDB indexes...")

    # datasets
    db[C.DATASETS].create_index([("dataset_id", ASCENDING)], unique=True, name="dataset_id_unique")
    db[C.DATASETS].create_index([("provider", ASCENDING)], name="provider")
    db[C.DATASETS].create_index([("role", ASCENDING)], name="role")
    db[C.DATASETS].create_index([("languages", ASCENDING)], name="languages")
    db[C.DATASETS].create_index([("status", ASCENDING)], name="status")

    # dataset_files
    db[C.DATASET_FILES].create_index([("dataset_id", ASCENDING)], name="dataset_id")
    db[C.DATASET_FILES].create_index([("checksum", ASCENDING)], name="checksum")
    db[C.DATASET_FILES].create_index([("split", ASCENDING)], name="split")
    db[C.DATASET_FILES].create_index([("language", ASCENDING)], name="language")
    db[C.DATASET_FILES].create_index([("speaker_id_hash", ASCENDING)], name="speaker_id_hash")
    db[C.DATASET_FILES].create_index([("is_telephone", ASCENDING)], name="is_telephone")
    db[C.DATASET_FILES].create_index([("is_code_switch", ASCENDING)], name="is_code_switch")
    db[C.DATASET_FILES].create_index([("role", ASCENDING)], name="role")

    # models
    db[C.MODELS].create_index([("model_id", ASCENDING)], unique=True, name="model_id_unique")

    # transcription_jobs
    db[C.TRANSCRIPTION_JOBS].create_index(
        [("request_id", ASCENDING)], unique=True, name="request_id_unique"
    )
    db[C.TRANSCRIPTION_JOBS].create_index([("model_id", ASCENDING)], name="model_id")
    db[C.TRANSCRIPTION_JOBS].create_index([("created_at", DESCENDING)], name="created_at")
    db[C.TRANSCRIPTION_JOBS].create_index([("status", ASCENDING)], name="status")

    # transcription_results
    db[C.TRANSCRIPTION_RESULTS].create_index([("request_id", ASCENDING)], name="request_id")
    db[C.TRANSCRIPTION_RESULTS].create_index([("model_id", ASCENDING)], name="model_id")
    db[C.TRANSCRIPTION_RESULTS].create_index([("created_at", DESCENDING)], name="created_at")

    # benchmark_runs
    db[C.BENCHMARK_RUNS].create_index(
        [("run_id", ASCENDING)], unique=True, name="run_id_unique"
    )
    db[C.BENCHMARK_RUNS].create_index([("dataset_id", ASCENDING)], name="dataset_id")
    db[C.BENCHMARK_RUNS].create_index([("model_id", ASCENDING)], name="model_id")
    db[C.BENCHMARK_RUNS].create_index([("timestamp", DESCENDING)], name="timestamp")

    # benchmark_results
    db[C.BENCHMARK_RESULTS].create_index([("run_id", ASCENDING)], name="run_id")
    db[C.BENCHMARK_RESULTS].create_index([("audio_file_id", ASCENDING)], name="audio_file_id")

    # external_model_results
    db[C.EXTERNAL_MODEL_RESULTS].create_index(
        [("audio_file_id", ASCENDING)], name="audio_file_id"
    )
    db[C.EXTERNAL_MODEL_RESULTS].create_index([("provider", ASCENDING)], name="provider")
    db[C.EXTERNAL_MODEL_RESULTS].create_index([("model", ASCENDING)], name="model")

    _logger.info("All MongoDB indexes created successfully.")
