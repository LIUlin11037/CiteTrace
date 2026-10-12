"""Checkpointer factory: SQLite for dev, PostgreSQL for the M3 topology.

Single place that decides the persistence backend — CLI, API and the eval
runner all call open_checkpointer() instead of constructing savers directly.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver

from deep_research.core.config import get_settings


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[BaseCheckpointSaver[Any]]:
    settings = get_settings()
    if settings.checkpoint_backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
            await saver.setup()
            yield saver
    else:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path) as saver:
            yield saver
