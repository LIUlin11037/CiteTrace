"""MCP tool assembly.

Loads tools from all MCP servers declared in settings (DR_MCP_SERVERS) via
langchain-mcp-adapters and caches them process-wide. Researchers call
get_mcp_tools(); they never know or care which server a tool came from.

Failure policy: a server that fails to start is logged and skipped, so one
broken integration degrades research quality instead of killing the run.
Tool-level errors are handled inside researcher_node.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from deep_research.core.config import get_settings

logger = logging.getLogger(__name__)

_tools_cache: list[BaseTool] | None = None
_cache_lock = asyncio.Lock()

_BARE_INTERPRETERS = {"python", "python3", "python.exe", "py"}


def _resolve_command(command: str) -> str:
    """Resolve the interpreter used to launch a stdio MCP server.

    The same `.env` has to work on the dev host (where it may hold an absolute
    venv path such as `.../.venv/Scripts/python.exe`) and inside the container
    (where only `python` exists). A literal command therefore rarely works in
    both, so:
      - bare interpreter names -> this process's interpreter
      - an absolute path missing here -> same fallback, with a warning
    Without this, a Windows path leaked into the container and every MCP server
    failed at launch with FileNotFoundError (found by the first real deployment
    run).
    """
    if Path(command).name.lower() in _BARE_INTERPRETERS:
        return sys.executable
    if not Path(command).exists():
        logger.warning(
            "mcp stdio command %r does not exist here; falling back to %s",
            command,
            sys.executable,
        )
        return sys.executable
    return command


def _server_params() -> dict[str, dict[str, Any]]:
    servers: dict[str, dict[str, Any]] = {}
    for name, cfg in get_settings().parse_mcp_servers().items():
        if cfg.transport == "stdio":
            if not cfg.command:
                logger.warning("mcp server %s: stdio transport requires command, skipped", name)
                continue
            servers[name] = {
                "transport": "stdio",
                "command": _resolve_command(cfg.command),
                "args": cfg.args,
                "env": cfg.env or None,
            }
        else:
            if not cfg.url:
                logger.warning("mcp server %s: sse transport requires url, skipped", name)
                continue
            servers[name] = {"transport": "sse", "url": cfg.url}
    return servers


async def get_mcp_tools() -> list[BaseTool]:
    """All tools from all configured MCP servers (cached after first load)."""
    global _tools_cache
    async with _cache_lock:
        if _tools_cache is not None:
            return _tools_cache

        servers = _server_params()
        if not servers:
            logger.warning("no MCP servers configured; researchers will run tool-less")
            _tools_cache = []
            return _tools_cache

        tools: list[BaseTool] = []
        for name, params in servers.items():
            try:
                # params 是运行时按 DR_MCP_SERVERS 构造的 dict，SDK 的联合类型无法静态覆盖
                client = MultiServerMCPClient({name: params})  # type: ignore[dict-item]
                server_tools = await client.get_tools()
                tools.extend(server_tools)
                logger.info("mcp server %s: loaded %d tools", name, len(server_tools))
            except Exception:
                logger.exception("mcp server %s: failed to load, skipped", name)

        _tools_cache = tools
        return _tools_cache


def reset_tool_cache() -> None:
    """Tests and long-running servers call this after MCP config changes."""
    global _tools_cache
    _tools_cache = None
