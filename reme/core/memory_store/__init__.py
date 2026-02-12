"""Memory store module for persistent memory management.

This module provides storage backends for memory chunks and file metadata,
including SQLite-based and ChromaDB-based implementations with vector and full-text search.
"""

from .base_memory_store import BaseMemoryStore
from .chroma_memory_store import ChromaMemoryStore
from .sqlite_memory_store import SqliteMemoryStore
from ..context import R

__all__ = [
    "BaseMemoryStore",
    "ChromaMemoryStore",
    "SqliteMemoryStore",
]

R.memory_stores.register("sqlite")(SqliteMemoryStore)
R.memory_stores.register("chroma")(ChromaMemoryStore)
