"""Database initialization CLI.

Usage:
    python -m src.database.init

Steps:
  1. Load .env (if present)
  2. Connect to MongoDB
  3. Ping
  4. Seed collections from config/models.yaml and config/datasets.yaml
  5. Create all indexes
  6. Health check report
"""
from __future__ import annotations

import sys
from pathlib import Path

# Load .env before any src imports
_root = Path(__file__).resolve().parents[3]
_env_file = _root / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        pass

from rich.console import Console
from rich.table import Table

from src.database.connection import get_db, ping_db
from src.database.indexes import create_all_indexes
from src.database.collections import ALL_COLLECTIONS
from src.database.repository import DatasetRepository, ModelRepository
from src.utils.config_loader import load_datasets_config, load_models_config
from src.utils.logger import get_logger

_logger = get_logger(__name__)
console = Console()


def init_database() -> bool:
    """Initialize MongoDB: connect, create indexes, seed registry data.

    Returns True on success, False on failure.
    """
    console.rule("[bold blue]MongoDB Initialization[/bold blue]")

    # Step 1: Connection check
    console.print("[cyan]→ Connecting to MongoDB...[/cyan]")
    if not ping_db():
        console.print(
            "[bold red]✗ Cannot connect to MongoDB.[/bold red]\n"
            "  Run:  docker compose up -d mongodb\n"
            "  Or:   brew services start mongodb-community\n"
            "  Or set MONGODB_URI environment variable."
        )
        return False
    console.print("[green]✓ MongoDB connection successful.[/green]")

    db = get_db()

    # Step 2: Ensure collections exist
    console.print("[cyan]→ Verifying collections...[/cyan]")
    existing = db.list_collection_names()
    for col in ALL_COLLECTIONS:
        if col not in existing:
            db.create_collection(col)
            console.print(f"  Created collection: [yellow]{col}[/yellow]")
        else:
            console.print(f"  Collection exists:  [dim]{col}[/dim]")

    # Step 3: Create indexes
    console.print("[cyan]→ Creating indexes...[/cyan]")
    try:
        create_all_indexes(db)
        console.print("[green]✓ All indexes created.[/green]")
    except Exception as e:
        console.print(f"[red]✗ Index creation failed: {e}[/red]")
        return False

    # Step 4: Seed model registry from config/models.yaml
    console.print("[cyan]→ Seeding model registry from config/models.yaml...[/cyan]")
    try:
        models_cfg = load_models_config()
        repo = ModelRepository(db)
        for m in models_cfg.get("models", []):
            repo.upsert(m)
            console.print(f"  [dim]Seeded model:[/dim] {m['model_id']}")
        console.print(f"[green]✓ {len(models_cfg.get('models', []))} models seeded.[/green]")
    except Exception as e:
        console.print(f"[red]✗ Model seeding failed: {e}[/red]")

    # Step 5: Seed dataset registry from config/datasets.yaml
    console.print("[cyan]→ Seeding dataset registry from config/datasets.yaml...[/cyan]")
    try:
        datasets_cfg = load_datasets_config()
        drepo = DatasetRepository(db)
        for d in datasets_cfg.get("datasets", []):
            drepo.upsert(d)
            console.print(f"  [dim]Seeded dataset:[/dim] {d['dataset_id']} [{d['role']}]")
        console.print(f"[green]✓ {len(datasets_cfg.get('datasets', []))} datasets seeded.[/green]")
    except Exception as e:
        console.print(f"[red]✗ Dataset seeding failed: {e}[/red]")

    # Step 6: Health check summary
    console.rule("[bold green]Database Health Check[/bold green]")
    table = Table(title="Collection Stats")
    table.add_column("Collection", style="cyan")
    table.add_column("Documents", justify="right")
    for col in ALL_COLLECTIONS:
        count = db[col].count_documents({})
        table.add_row(col, str(count))
    console.print(table)
    console.print("\n[bold green]✓ Database initialization complete.[/bold green]")
    return True


if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)
