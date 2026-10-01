"""Tests for MongoDB database layer.

These tests require a running MongoDB instance.
Tests are automatically skipped if MongoDB is unavailable.
"""
import pytest


class TestMongoConnection:
    def test_ping_succeeds(self, mongo_db):
        result = mongo_db.command("ping")
        assert result.get("ok") == 1.0


class TestIndexCreation:
    def test_create_indexes_idempotent(self, mongo_db):
        from src.database.indexes import create_all_indexes
        # Should not raise on first or second call
        create_all_indexes(mongo_db)
        create_all_indexes(mongo_db)


class TestDatasetRepository:
    def test_upsert_and_find(self, mongo_db):
        from src.database.repository import DatasetRepository
        repo = DatasetRepository(mongo_db)
        doc = {
            "dataset_id": "test-dataset",
            "name": "Test Dataset",
            "provider": "Test Provider",
            "role": "TRAINING",
            "license": "MIT",
        }
        repo.upsert(doc)
        found = repo.find_by_id("test-dataset")
        assert found is not None
        assert found["name"] == "Test Dataset"

    def test_upsert_is_idempotent(self, mongo_db):
        from src.database.repository import DatasetRepository
        repo = DatasetRepository(mongo_db)
        doc = {"dataset_id": "test-ds2", "name": "Test", "role": "TRAINING"}
        repo.upsert(doc)
        repo.upsert({**doc, "name": "Test Updated"})
        found = repo.find_by_id("test-ds2")
        assert found["name"] == "Test Updated"

    def test_evaluation_role_detection(self, mongo_db):
        from src.database.repository import DatasetRepository
        repo = DatasetRepository(mongo_db)
        repo.upsert({
            "dataset_id": "benchmark-ds",
            "name": "Benchmark",
            "role": "BENCHMARK",
        })
        assert repo.is_evaluation_dataset("benchmark-ds") is True

    def test_list_all(self, mongo_db):
        from src.database.repository import DatasetRepository
        repo = DatasetRepository(mongo_db)
        repo.upsert({"dataset_id": "ds-a", "name": "A", "role": "TRAINING"})
        repo.upsert({"dataset_id": "ds-b", "name": "B", "role": "BENCHMARK"})
        all_ds = repo.list_all()
        assert len(all_ds) >= 2


class TestTranscriptionRepository:
    def test_create_job_and_find_result(self, mongo_db):
        from src.database.repository import TranscriptionRepository
        repo = TranscriptionRepository(mongo_db)

        request_id = "test-req-001"
        repo.create_job({
            "request_id": request_id,
            "audio_path": "/data/test.wav",
            "model_id": "whisper-medium",
            "language": "hi",
        })
        repo.save_result({
            "request_id": request_id,
            "model_id": "whisper-medium",
            "transcript": "नमस्ते",
            "audio": {"duration": 2.0},
            "processing": {"rtf": 0.12},
        })
        result = repo.find_result(request_id)
        assert result is not None
        assert result["transcript"] == "नमस्ते"

    def test_duplicate_request_id_is_handled(self, mongo_db):
        from src.database.repository import TranscriptionRepository
        repo = TranscriptionRepository(mongo_db)
        job = {"request_id": "dup-001", "model_id": "whisper-medium"}
        repo.create_job(job)
        # Should not raise
        repo.create_job(job)


class TestBenchmarkRepository:
    def test_save_and_find_run(self, mongo_db):
        from src.database.repository import BenchmarkRepository
        import uuid
        repo = BenchmarkRepository(mongo_db)
        run_id = str(uuid.uuid4())
        repo.save_run({
            "run_id": run_id,
            "model_id": "whisper-medium",
            "dataset_id": "lahaja",
            "device": "cpu",
            "wer": 0.15,
        })
        found = repo.find_run(run_id)
        assert found is not None
        assert found["wer"] == 0.15


class TestNoDatabaseMode:
    """Verify transcription works without database."""
    def test_no_db_transcription_does_not_crash(self, sample_wav_16k):
        """Transcription engine should work independently of MongoDB."""
        from src.audio.processor import process_audio
        result = process_audio(sample_wav_16k)
        assert result.waveform is not None
        # Model transcription itself is tested separately (requires model download)
