"""Training CLI interface.

Usage:
    python -m src.training audit
    python -m src.training manifest
    python -m src.training smoke --samples 100 --steps 10
    python -m src.training run --config config/training.yaml --steps 50
"""
from __future__ import annotations

import os
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.utils.config_loader import load_yaml_config, get_data_dir

console = Console()


@click.group()
def cli() -> None:
    """Indian STT Training & Audit Tooling."""
    pass


@cli.command()
def audit() -> None:
    """Scan local dataset files and export audit reports."""
    console.rule("[bold cyan]Dataset Audit[/bold cyan]")
    from src.datasets.auditor import audit_datasets
    summary, records = audit_datasets()

    tbl = Table(box=box.SIMPLE, show_header=False)
    tbl.add_column("Metric", style="dim")
    tbl.add_column("Value", style="bold green")

    tbl.add_row("Total Files", str(summary.total_files))
    tbl.add_row("Usable Files", str(summary.total_usable_files))
    tbl.add_row("Corrupted Files", str(summary.total_corrupted_files))
    tbl.add_row("Duplicate Files", str(summary.total_duplicate_files))
    tbl.add_row("Total Audio Hours", f"{summary.total_audio_hours:.4f} hrs")
    tbl.add_row("8kHz Telephone Hours", f"{summary.telephone_8k_hours:.4f} hrs")
    tbl.add_row("16kHz Hours", f"{summary.sr_16k_hours:.4f} hrs")
    tbl.add_row("Labeled Hours", f"{summary.labeled_hours:.4f} hrs")
    console.print(tbl)
    console.print("[green]✓ Audit complete. Reports saved to data/audit/[/green]")


@cli.command()
@click.option("--val-ratio", default=0.1, show_default=True)
@click.option("--test-ratio", default=0.1, show_default=True)
def manifest(val_ratio: float, test_ratio: float) -> None:
    """Build unified train, validation, test, and quarantine manifests."""
    console.rule("[bold cyan]Unified Manifest Generator[/bold cyan]")
    from src.datasets.manifest_builder import build_manifests
    counts = build_manifests(val_ratio=val_ratio, test_ratio=test_ratio)

    tbl = Table(box=box.SIMPLE)
    tbl.add_column("Manifest Split", style="cyan")
    tbl.add_column("Sample Count", style="bold green")

    tbl.add_row("train.jsonl", str(counts["train"]))
    tbl.add_row("validation.jsonl", str(counts["validation"]))
    tbl.add_row("test.jsonl (held-out)", str(counts["test"]))
    tbl.add_row("quarantine.jsonl", str(counts["quarantine"]))
    console.print(tbl)
    console.print("[green]✓ Manifests created under data/manifests/[/green]")


@cli.command()
@click.option("--steps", default=10, show_default=True, help="Number of smoke training steps")
def smoke(steps: int) -> None:
    """Run local smoke training test (100-500 samples)."""
    console.rule("[bold cyan]Local Smoke Training Test[/bold cyan]")
    data_dir = get_data_dir()
    train_manifest = data_dir / "manifests" / "train.jsonl"
    val_manifest = data_dir / "manifests" / "validation.jsonl"

    if not train_manifest.exists():
        console.print("[cyan]→ Manifests missing — building manifests first...[/cyan]")
        from src.datasets.manifest_builder import build_manifests
        build_manifests()

    cfg_path = Path("config/training.yaml")
    config = load_yaml_config(cfg_path) if cfg_path.exists() else {}

    from src.training.trainer import STTTrainer
    trainer = STTTrainer(config=config, train_manifest=train_manifest, val_manifest=val_manifest)
    res = trainer.train(max_steps=steps, is_smoke_test=True)

    console.print(
        Panel(
            f"Run ID:        {res['run_id']}\n"
            f"Device:        {res['device']}\n"
            f"Final Loss:    {res['final_loss']:.4f}\n"
            f"Val WER:       {res['val_wer']:.4f}\n"
            f"Val CER:       {res['val_cer']:.4f}\n"
            f"Checkpoint:    {res['checkpoint_path']}",
            title="[bold green]Smoke Training Complete[/bold green]",
            border_style="green",
        )
    )


@cli.command()
@click.option("--config", "cfg_file", default="config/training.yaml", show_default=True)
@click.option("--steps", default=50, show_default=True, help="Training steps")
def run(cfg_file: str, steps: int) -> None:
    """Start real fine-tuning run."""
    console.rule("[bold cyan]Indian STT Fine-Tuning Run[/bold cyan]")
    data_dir = get_data_dir()
    train_manifest = data_dir / "manifests" / "train.jsonl"
    val_manifest = data_dir / "manifests" / "validation.jsonl"

    if not train_manifest.exists():
        from src.datasets.manifest_builder import build_manifests
        build_manifests()

    config = load_yaml_config(Path(cfg_file))
    from src.training.trainer import STTTrainer
    trainer = STTTrainer(config=config, train_manifest=train_manifest, val_manifest=val_manifest)
    res = trainer.train(max_steps=steps, is_smoke_test=False)

    console.print(
        Panel(
            f"Run ID:        {res['run_id']}\n"
            f"Device:        {res['device']}\n"
            f"Final Loss:    {res['final_loss']:.4f}\n"
            f"Val WER:       {res['val_wer']:.4f}\n"
            f"Val CER:       {res['val_cer']:.4f}\n"
            f"Checkpoint:    {res['checkpoint_path']}",
            title="[bold green]Fine-Tuning Run Complete[/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    cli()
