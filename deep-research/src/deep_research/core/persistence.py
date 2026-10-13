"""Checkpointer factory: SQLite for dev, PostgreSQL for the M3 topology.

Single place that decides the persistence backend — CLI, API and the eval
runner all call open_checkpointer() instead of constructing savers directly.

Both savers get an explicit JsonPlusSerializer with our state module on the
msgpack allowlist: pydantic state objects (ResearchPlan, Evidence, ...) are
msgpack-serialized into checkpoints, and the default permissive mode prints
a "Deserializing unregistered type" warning that becomes a hard error in a
future langgraph version.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from deep_research.core.config import get_settings

# (module, class) pairs — the allowlist matches fully-qualified class names,
# NOT module path prefixes. Every pydantic state object that lands in a
# checkpoint must be listed, or deserialization is blocked.
_ALLOWED_MSGPACK_MODULES = [
    ("deep_research.agents.state", cls)
    for cls in (
        "SubQuestion",
        "ResearchPlan",
        "Evidence",
        "EvidenceBundle",
        "CritiqueResult",
    )
]


def _serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[BaseCheckpointSaver[Any]]:
    settings = get_settings()
    if settings.checkpoint_backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(
            settings.database_url, serde=_serde()
        ) as saver:
            await saver.setup()
            yield saver
    else:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(
            settings.checkpoint_db_path, serde=_serde()
        ) as saver:
            yield saver
