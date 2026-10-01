"""MongoDB connection manager.

Reads MONGODB_URI and MONGODB_DATABASE from environment variables.
Never hard-codes credentials.

Usage:
    from src.database.connection import get_db
    db = get_db()
    db["datasets"].find_one(...)
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ConfigurationError

from src.utils.logger import get_logger

# Load .env variables if present
_env = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
if os.path.exists(_env):
    load_dotenv(_env)
else:
    load_dotenv()

_logger = get_logger(__name__)

_DEFAULT_URI = "mongodb://localhost:27017"
_DEFAULT_DB = "test-transcription"



@lru_cache(maxsize=1)
def _get_client() -> MongoClient:
    """Create and cache a MongoClient instance.

    Reads MONGODB_URI from environment. Defaults to localhost:27017.
    Raises RuntimeError if MongoDB is unreachable.
    """
    uri = os.getenv("MONGODB_URI", _DEFAULT_URI)
    try:
        client: MongoClient = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        # Verify connection
        client.admin.command("ping")
        _logger.info("MongoDB connection established: %s", _redact_uri(uri))
        return client
    except ConnectionFailure as e:
        raise RuntimeError(
            f"Cannot connect to MongoDB at '{_redact_uri(uri)}'.\n"
            f"Error: {e}\n\n"
            f"Solutions:\n"
            f"  1. Start MongoDB locally:  brew services start mongodb-community\n"
            f"  2. Use Docker:             docker compose up -d mongodb\n"
            f"  3. Set MONGODB_URI environment variable to your MongoDB connection string.\n"
            f"  4. Run transcription without DB:  python -m src.transcribe audio.wav --no-db"
        ) from e
    except ConfigurationError as e:
        raise RuntimeError(
            f"Invalid MongoDB URI: {_redact_uri(uri)}\nError: {e}"
        ) from e


def get_db() -> Database:
    """Return the MongoDB database object.

    Database name is read from MONGODB_DATABASE environment variable.
    Defaults to 'local_indian_stt'.
    """
    client = _get_client()
    db_name = os.getenv("MONGODB_DATABASE", _DEFAULT_DB)
    return client[db_name]


def ping_db() -> bool:
    """Return True if MongoDB is reachable, False otherwise."""
    try:
        db = get_db()
        db.command("ping")
        return True
    except Exception:
        return False


def _redact_uri(uri: str) -> str:
    """Redact password from URI for safe logging."""
    import re
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", uri)
