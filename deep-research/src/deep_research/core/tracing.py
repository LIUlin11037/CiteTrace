"""Langfuse tracing: one trace per research run, nested spans per node/LLM/tool.

Design notes
------------
- LangGraph propagates `callbacks` from the run config down to every nested
  runnable, so attaching one handler at the entry point is enough to get the
  full tree: graph -> node -> chat model / MCP tool.
- Session grouping and tags are NOT handler arguments in langfuse v5 — they
  travel in the run config's `metadata` under reserved keys
  (`langfuse_session_id`, `langfuse_tags`, `langfuse_user_id`). So this module
  exposes a *config fragment* helper rather than a bare handler factory.
- Tracing is best-effort: with no keys configured everything returns empty
  values and the pipeline behaves exactly as before (no imports, no errors).

Entry points must call flush_tracing() before exiting, otherwise the last
batch of events is dropped by the SDK's background exporter.
"""

from __future__ import annotations

import logging
from typing import Any

from deep_research.core.config import get_settings

logger = logging.getLogger(__name__)

_client: Any = None
_setup_done = False


def _ensure_client() -> Any:
    """Configure the global Langfuse client once. Returns None when disabled."""
    global _client, _setup_done
    if _setup_done:
        return _client

    settings = get_settings()
    if not settings.tracing_active:
        _setup_done = True
        return None

    import os

    # The SDK reads LANGFUSE_* from the environment; mirror DR_* overrides in
    # (setdefault keeps any real LANGFUSE_* the user exported explicitly).
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)

    from langfuse import Langfuse

    _client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    _setup_done = True
    logger.info("Langfuse tracing enabled (host=%s)", settings.langfuse_host)
    return _client


def apply_tracing(
    config: dict[str, Any],
    *,
    tags: list[str] | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Attach tracing to an existing invocation config, in place, and return it.

    Safe to call unconditionally: with tracing disabled it is a no-op. The
    session id is taken from the config's thread_id, so every segment of one
    research run (plan interrupt, resume, critic loop) lands in one trace.
    """
    if _ensure_client() is None:
        return config

    from langfuse.langchain import CallbackHandler

    callbacks = list(config.get("callbacks") or [])
    callbacks.append(CallbackHandler())
    config["callbacks"] = callbacks

    metadata = dict(config.get("metadata") or {})
    thread_id = (config.get("configurable") or {}).get("thread_id")
    if thread_id:
        metadata.setdefault("langfuse_session_id", thread_id)
    if tags:
        existing = list(metadata.get("langfuse_tags") or [])
        metadata["langfuse_tags"] = sorted(set(existing) | set(tags))
    if user_id:
        metadata["langfuse_user_id"] = user_id
    if metadata:
        config["metadata"] = metadata
    return config


def flush_tracing() -> None:
    """Flush pending events — call before process exit (CLI, eval, API shutdown)."""
    client = _client
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:  # tracing must never break the run
        logger.warning("Langfuse flush failed: %s", exc)
