"""Dataset CLI.

Usage:
    python -m src.dataset list
    python -m src.dataset inspect <dataset_id>
    python -m src.dataset init-db
"""
from __future__ import annotations

import sys
from pathlib import Path

# Load .env
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
from rich.table import Table
from rich import box

from src.datasets.registry import get_dataset_registry
from src.utils.logger import get_logger

console = Console()
_logger = get_logger(__name__)


@click.group()
def cli():
    """Dataset registry and management commands."""
    pass


@cli.command("list")
@click.option("--role", default=None, help="Filter by role: TRAINING|VALIDATION|TEST|BENCHMARK")
def list_datasets(role: str | None):
    """List all registered datasets."""
    registry = get_dataset_registry()
    datasets = registry.list_all(role=role)

    table = Table(title=f"Registered Datasets{f' [{role}]' if role else ''}", box=box.SIMPLE)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Provider")
    table.add_column("Role", style="yellow")
    table.add_column("License")
    table.add_column("Telephone", justify="center")
    table.add_column("Code-Switch", justify="center")
    table.add_column("Status")

    for d in datasets:
        table.add_row(
            d["dataset_id"],
            d["name"],
            d.get("provider", "—"),
            d.get("role", "—"),
            d.get("license", "—"),
            "✓" if d.get("telephone_available") else "✗",
            "✓" if d.get("code_switching") else "✗",
            d.get("status", "—"),
        )
    console.print(table)


@cli.command("inspect")
@click.argument("dataset_id")
def inspect_dataset(dataset_id: str):
    """Show detailed metadata for a dataset."""
    registry = get_dataset_registry()
    try:
        d = registry.get(dataset_id)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    console.rule(f"[bold cyan]{d['name']}[/bold cyan]")
    for k, v in d.items():
        if k in {"_id"}:
            continue
        console.print(f"  [dim]{k}:[/dim] [bold]{v}[/bold]")

    if registry.is_evaluation_only(dataset_id):
        console.print(
            "\n[bold red]⚠️  EVALUATION-ONLY DATASET — Must NOT be included in training data.[/bold red]"
        )


@cli.command("init-db")
def init_db():
    """Initialize MongoDB collections and seed dataset registry."""
    from src.database.__main__ import init_database
    success = init_database()
    sys.exit(0 if success else 1)


@cli.command("validate")
def validate():
    """Validate dataset registry for integrity."""
    registry = get_dataset_registry()
    datasets = registry.list_all()
    issues = 0
    console.print(f"Validating {len(datasets)} datasets...")
    for d in datasets:
        did = d["dataset_id"]
        if not d.get("license"):
            console.print(f"  [red]MISSING LICENSE:[/red] {did}")
            issues += 1
        if d.get("commercial_use") is None:
            console.print(f"  [yellow]UNKNOWN COMMERCIAL USE:[/yellow] {did}")
        if d.get("legal_review_required"):
            console.print(f"  [yellow]LEGAL REVIEW REQUIRED:[/yellow] {did}")
    if issues == 0:
        console.print("[green]✓ No critical issues found.[/green]")
    else:
        console.print(f"[red]Found {issues} critical issue(s).[/red]")


@cli.command("download")
@click.option("--dataset", "dataset_id", required=True, help="Dataset ID to download")
def download_cmd(dataset_id: str):
    """Download a dataset directly to data/raw/<dataset_id>/."""
    from src.datasets.downloader import download_dataset
    try:
        raw_dir = download_dataset(dataset_id)
        console.print(f"[green]✓ Dataset '{dataset_id}' ready in:[/green] {raw_dir}")
    except Exception as e:
        console.print(f"[bold red]✗ Download failed: {e}[/bold red]")
        sys.exit(1)


@cli.command("license-audit")
def license_audit_cmd():
    """Generate legal & commercial usability license audit report."""
    from src.datasets.license_audit import generate_license_audit_report
    md_path = generate_license_audit_report()
    console.print(f"[green]✓ License audit generated:[/green] {md_path}")


if __name__ == "__main__":
    cli()
