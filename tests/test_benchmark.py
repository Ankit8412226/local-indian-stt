"""Tests for benchmarking metrics and dataset registry."""
import pytest

from src.benchmark.metrics import word_error_rate, char_error_rate, real_time_factor
from src.datasets.registry import get_dataset_registry
from src.datasets.manifest import write_manifest, make_record


class TestMetrics:
    def test_perfect_wer(self):
        assert word_error_rate("hello world", "hello world") == 0.0

    def test_perfect_cer(self):
        assert char_error_rate("hello", "hello") == 0.0

    def test_total_wer(self):
        # All words wrong
        wer = word_error_rate("a b c", "x y z")
        assert wer == 1.0

    def test_partial_wer(self):
        wer = word_error_rate("hello world foo", "hello world bar")
        assert wer == pytest.approx(1/3, abs=0.001)

    def test_cer_with_devanagari(self):
        cer = char_error_rate("नमस्ते", "नमस्ते")
        assert cer == 0.0

    def test_rtf_calculation(self):
        rtf = real_time_factor(inference_time=1.0, audio_duration=10.0)
        assert rtf == pytest.approx(0.1)

    def test_rtf_zero_duration(self):
        rtf = real_time_factor(1.0, 0.0)
        assert rtf == 0.0


class TestDatasetRegistry:
    def test_registry_loads(self):
        registry = get_dataset_registry()
        datasets = registry.list_all()
        assert len(datasets) > 0

    def test_lahaja_is_benchmark(self):
        registry = get_dataset_registry()
        assert registry.is_evaluation_only("lahaja") is True

    def test_shrutilipi_is_training(self):
        registry = get_dataset_registry()
        d = registry.get("shrutilipi")
        assert d["role"] == "TRAINING"
        assert registry.is_evaluation_only("shrutilipi") is False

    def test_training_manifest_blocks_benchmark(self):
        registry = get_dataset_registry()
        violations = registry.validate_training_manifest_ids(["shrutilipi", "lahaja"])
        assert "lahaja" in violations
        assert "shrutilipi" not in violations

    def test_unknown_dataset_raises(self):
        registry = get_dataset_registry()
        with pytest.raises(KeyError):
            registry.get("nonexistent-dataset")


class TestManifest:
    def test_write_and_read(self, tmp_path):
        from src.datasets.manifest import write_manifest, read_manifest
        records = [
            make_record("/data/a.wav", "shrutilipi", transcript="नमस्ते"),
            make_record("/data/b.wav", "shrutilipi", transcript="hello"),
        ]
        path = str(tmp_path / "test.jsonl")
        count = write_manifest(records, path)
        assert count == 2

        loaded = list(read_manifest(path))
        assert len(loaded) == 2
        assert loaded[0]["transcript"] == "नमस्ते"

    def test_manifest_blocks_benchmark_dataset(self, tmp_path):
        records = [make_record("/data/x.wav", "lahaja", transcript="test")]
        with pytest.raises(ValueError, match="EVALUATION DATASET PROTECTION"):
            write_manifest(records, str(tmp_path / "bad.jsonl"), block_evaluation=True)


class TestExternalImport:
    def test_parse_valid_jsonl(self, tmp_path):
        from src.benchmark.external_import import parse_external_results
        import json
        records = [
            {"audio_id": "a1", "provider": "gemini", "model": "gemini-2.0", "transcript": "नमस्ते"},
            {"audio_id": "a2", "provider": "gemini", "model": "gemini-2.0", "transcript": "hello"},
        ]
        path = tmp_path / "external.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        parsed = parse_external_results(str(path))
        assert len(parsed) == 2
        assert parsed[0]["provider"] == "gemini"

    def test_parse_skips_malformed(self, tmp_path):
        from src.benchmark.external_import import parse_external_results
        import json
        path = tmp_path / "bad.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"audio_id": "x", "provider": "gemini", "model": "m", "transcript": "t"}) + "\n")
            f.write("{invalid json}\n")
            f.write(json.dumps({"audio_id": "y", "provider": "gemini", "model": "m", "transcript": "t"}) + "\n")

        parsed = parse_external_results(str(path))
        assert len(parsed) == 2  # Malformed line skipped
