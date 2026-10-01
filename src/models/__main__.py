"""Models CLI.

Usage:
    python -m src.models list
    python -m src.models inspect <model_id>
    python -m src.models verify
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
from rich.table import Table
from rich import box

from src.models.registry import get_registry
from src.utils.logger import get_logger

console = Console()
_logger = get_logger(__name__)


@click.group()
def cli():
    """Model registry and management commands."""
    pass


@cli.command("list")
def list_models():
    """List all registered ASR models."""
    registry = get_registry()
    models = registry.list_all()

    table = Table(title="Registered ASR Models", box=box.SIMPLE)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Official", justify="center")
    table.add_column("Gated", justify="center")
    table.add_column("License")
    table.add_column("Status")

    for m in models:
        official = "[green]✓[/green]" if m.get("official_model") else "[yellow]✗ Community[/yellow]"
        gated = "[red]YES[/red]" if m.get("gated") else "No"
        table.add_row(
            m["model_id"],
            m["name"],
            m.get("provider", "—"),
            official,
            gated,
            m.get("license", "—"),
            m.get("status", "—"),
        )
    console.print(table)


@cli.command("inspect")
@click.argument("model_id")
def inspect_model(model_id: str):
    """Show detailed metadata for a model."""
    registry = get_registry()
    try:
        m = registry.get(model_id)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    console.rule(f"[bold cyan]{m['name']}[/bold cyan]")
    for k, v in m.items():
        console.print(f"  [dim]{k}:[/dim] [bold]{v}[/bold]")

    if not m.get("official_model"):
        console.print(
            f"\n[yellow]⚠️  Community model — NOT the official release.[/yellow]\n"
            f"   Source: {m.get('source_model', 'unknown')}"
        )
    if m.get("gated"):
        console.print(
            "\n[red]🔒 GATED MODEL: Requires HF_TOKEN + access approval.[/red]"
        )


@cli.command("verify")
def verify_models():
    """Verify HuggingFace repo accessibility for all registered models."""
    import urllib.request
    registry = get_registry()
    console.print("Verifying HuggingFace repo accessibility...\n")
    for m in registry.list_all():
        repo = m.get("hf_repo", "")
        url = f"https://huggingface.co/{repo}/resolve/main/config.json"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                timeout=10,
            ) as resp:
                status = resp.status
            marker = "[green]✓ Public[/green]"
        except Exception as e:
            code = getattr(e, "code", "ERR")
            if code == 401:
                marker = "[yellow]🔒 Gated/Auth Required[/yellow]"
            elif code == 404:
                marker = "[red]✗ Not Found[/red]"
            else:
                marker = f"[red]✗ Error: {e}[/red]"
        console.print(f"  {m['model_id']:<35} {marker}")


if __name__ == "__main__":
    cli()
