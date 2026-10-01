"""Benchmark CLI.

Usage:
    python -m src.benchmark run --model whisper-medium --dataset lahaja
    python -m src.benchmark import-gemini benchmark_external_results.jsonl
    python -m src.benchmark report
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
_env = _root / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass

import click
from rich.console import Console

from src.utils.logger import get_logger

console = Console()
_logger = get_logger(__name__)


@click.group()
def cli():
    """Benchmarking commands."""
    pass


@cli.command("run")
@click.option("--model", "model_id", required=True, help="Model ID from config/models.yaml")
@click.option("--dataset", "dataset_id", required=True, help="Dataset ID")
@click.option("--device", default="auto", help="Device: auto|cpu|cuda|mps")
@click.option("--language", default="hi", help="Language code (default: hi)")
@click.option("--no-db", "no_db", is_flag=True, help="Skip MongoDB storage")
@click.argument("audio_dir", required=False)
def run_benchmark(model_id, dataset_id, device, language, no_db, audio_dir):
    """Run a benchmark of MODEL_ID against DATASET_ID.

    For now, requires providing an audio directory with reference .txt transcripts.
    Future: will load from MongoDB dataset_files collection automatically.
    """
    from src.benchmark.runner import BenchmarkRunner, BenchmarkSample
    from src.benchmark.report import generate_report

    if not audio_dir:
        console.print(
            "[yellow]No audio_dir provided. For dataset-backed benchmarks, use:\n"
            "  python -m src.benchmark run --model whisper-medium --dataset lahaja /path/to/audio_dir[/yellow]"
        )
        sys.exit(1)

    audio_path = Path(audio_dir)
    if not audio_path.exists():
        console.print(f"[red]Audio directory not found: {audio_path}[/red]")
        sys.exit(1)

    # Build samples from directory (expects audio files + matching .txt reference)
    samples = []
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    for f in sorted(audio_path.iterdir()):
        if f.suffix.lower() in audio_exts:
            ref_file = f.with_suffix(".txt")
            if ref_file.exists():
                ref_text = ref_file.read_text(encoding="utf-8").strip()
                samples.append(BenchmarkSample(
                    audio_path=str(f),
                    reference_transcript=ref_text,
                    language=language,
                ))

    if not samples:
        console.print(
            f"[red]No audio+transcript pairs found in {audio_path}.\n"
            f"Each audio file must have a matching .txt reference transcript.[/red]"
        )
        sys.exit(1)

    console.print(
        f"[cyan]Running benchmark:[/cyan] model={model_id} dataset={dataset_id} "
        f"samples={len(samples)} device={device}"
    )

    db = None
    if not no_db:
        try:
            from src.database.connection import get_db, ping_db
            if ping_db():
                db = get_db()
            else:
                console.print("[yellow]⚠ MongoDB unavailable — running without DB.[/yellow]")
        except Exception:
            console.print("[yellow]⚠ MongoDB unavailable — running without DB.[/yellow]")

    runner = BenchmarkRunner(
        model_id=model_id,
        dataset_id=dataset_id,
        device_str=device,
        language=language,
    )

    result = runner.run(samples, db=db)

    console.print(f"\n[bold green]Benchmark complete.[/bold green]")
    console.print(f"  WER:     {result.wer:.4f}" if result.wer is not None else "  WER: N/A")
    console.print(f"  CER:     {result.cer:.4f}" if result.cer is not None else "  CER: N/A")
    console.print(f"  Avg RTF: {result.avg_rtf:.4f}" if result.avg_rtf is not None else "  Avg RTF: N/A")
    console.print(f"  Failed:  {result.failed_samples}/{result.total_samples}")

    paths = generate_report(result)
    for fmt, p in paths.items():
        console.print(f"  Report [{fmt}]: {p}")


@cli.command("import-gemini")
@click.argument("filepath")
@click.option("--no-db", "no_db", is_flag=True, help="Parse only, don't store to MongoDB")
def import_gemini(filepath, no_db):
    """Import external (e.g., Gemini) benchmark results from JSONL file."""
    from src.benchmark.external_import import parse_external_results, import_to_db
    if no_db:
        records = parse_external_results(filepath)
        console.print(f"[green]Parsed {len(records)} records (--no-db, not stored).[/green]")
        return

    try:
        from src.database.connection import get_db
        db = get_db()
    except Exception as e:
        console.print(f"[red]MongoDB unavailable: {e}[/red]")
        sys.exit(1)

    count = import_to_db(filepath, db)
    console.print(f"[green]Imported {count} external results into MongoDB.[/green]")


@cli.command("report")
@click.option("--model", "model_id", default=None)
@click.option("--limit", default=5)
def show_report(model_id, limit):
    """Show recent benchmark runs from MongoDB."""
    try:
        from src.database.connection import get_db
        from src.database.repository import BenchmarkRepository
        db = get_db()
        repo = BenchmarkRepository(db)
        runs = repo.list_runs(model_id=model_id, limit=limit)
        if not runs:
            console.print("[yellow]No benchmark runs found in database.[/yellow]")
            return
        for r in runs:
            console.print(
                f"  [cyan]{r['run_id'][:8]}…[/cyan] "
                f"model={r['model_id']} dataset={r['dataset_id']} "
                f"WER={r.get('wer','?')} CER={r.get('cer','?')} RTF={r.get('avg_rtf','?')}"
            )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
