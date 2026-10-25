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
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from deep_research.core.config import get_settings
from deep_research.core.context import estimate_tokens, trim_document
from deep_research.core.guard import wrap_tool_output

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


def _hf_child_env() -> dict[str, str]:
    """HF/embedding env that must survive into MCP child processes."""
    from deep_research.core.compat import ensure_hf_compat

    ensure_hf_compat()  # 补齐 HF_ENDPOINT 与 FASTEMBED_CACHE_PATH（含镜像内路径兜底）
    keys = ("HF_ENDPOINT", "FASTEMBED_CACHE_PATH", "HF_HUB_DISABLE_XET")
    return {k: os.environ[k] for k in keys if os.environ.get(k)}


def _child_env(extra: dict[str, str] | None) -> dict[str, str]:
    """Env for a stdio MCP child.

    注意：mcp 的 stdio_client 拿到 env 后是**整体替换**而不是合并，只传几个 HF
    变量会让子进程丢掉 PATH/HOME（实测会崩）。所以先取 SDK 的默认环境再叠加。
    """
    try:
        from mcp.client.stdio import get_default_environment

        base = dict(get_default_environment())
    except Exception:  # noqa: BLE001 - 老版本 SDK 没有这个函数
        base = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "TZ")}
    # 子进程同样需要本项目的配置：kb server 要连同一个数据库。SDK 的默认环境
    # 把 DR_* 全丢了，容器里就会退回 localhost:5432 直接连不上（实测）。
    inherited = {k: v for k, v in os.environ.items() if k.startswith("DR_")}
    return {**base, **inherited, **_hf_child_env(), **(extra or {})}


def _server_params() -> dict[str, dict[str, Any]]:
    """Resolve configured MCP servers into launch params, honouring source_mode.

    offline 模式只保留 ``network=False`` 的 server（如本地知识库）。过滤发生在
    **装载之前**，所以不是"禁用某个工具"，而是"这个工具根本不存在"——离线
    能力的边界由配置决定，不靠运行时判断。
    """
    settings = get_settings()
    offline = settings.source_mode == "offline"
    servers: dict[str, dict[str, Any]] = {}
    for name, cfg in settings.parse_mcp_servers().items():
        if offline and cfg.network:
            logger.info("offline 模式：跳过网络类 MCP server %s", name)
            continue
        if cfg.transport == "stdio":
            if not cfg.command:
                logger.warning("mcp server %s: stdio transport requires command, skipped", name)
                continue
            servers[name] = {
                "transport": "stdio",
                "command": _resolve_command(cfg.command),
                "args": cfg.args,
                # MCP 子进程由 stdio_client 起，它只传 get_default_environment()
                # 那一小撮变量（实测 HF_ENDPOINT / FASTEMBED_CACHE_PATH 都会丢），
                # 导致插件在容器里重新下载模型、且走不通外网。这里显式注入。
                "env": _child_env(cfg.env),
            }
        else:
            if not cfg.url:
                logger.warning("mcp server %s: sse transport requires url, skipped", name)
                continue
            servers[name] = {"transport": "sse", "url": cfg.url}
    return servers


def _as_text(raw: Any) -> str:
    """Flatten a tool result (str / MCP content blocks) into plain text."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(json.dumps(item, ensure_ascii=False, default=str))
        return "\n\n".join(parts)
    return json.dumps(raw, ensure_ascii=False, default=str)


def _guarded(tool: BaseTool) -> BaseTool:
    """Wrap one tool so every result is bounded and fenced as untrusted data.

    Two contracts are enforced here, both in code rather than in a prompt:
    - **budget**: the text is trimmed by information density (core/context.py),
      not by prefix, so a results table in the middle survives;
    - **fence**: the text is wrapped as untrusted external data (core/guard.py).

    If wrapping fails for an exotic tool signature we keep the original tool: a
    broken guard must not break research.
    """

    async def _call(**kwargs: Any) -> str:
        text = _as_text(await tool.ainvoke(kwargs))
        limit = get_settings().evidence_content_limit
        trimmed_text, was_trimmed = trim_document(text, limit)
        if was_trimmed:
            logger.info(
                "tool %s: 裁剪 %d → %d 字符（约 %d → %d tokens，按信息密度保留）",
                tool.name,
                len(text),
                len(trimmed_text),
                estimate_tokens(text),
                estimate_tokens(trimmed_text),
            )
        return wrap_tool_output(tool.name, trimmed_text)

    try:
        return StructuredTool.from_function(
            coroutine=_call,
            name=tool.name,
            description=tool.description or tool.name,
            args_schema=tool.args_schema,
        )
    except Exception:  # noqa: BLE001 - exotic schema, fall back to the raw tool
        logger.warning("mcp tool %s: 无法加装注入隔离护栏，使用原始工具", tool.name)
        return tool


async def get_mcp_tools() -> list[BaseTool]:
    """All tools from all configured MCP servers (cached after first load)."""
    global _tools_cache
    async with _cache_lock:
        if _tools_cache is not None:
            return _tools_cache

        servers = _server_params()
        if not servers:
            if get_settings().source_mode == "offline":
                # 离线模式没有可用工具 = 取不到任何证据 = 报告必然空转。
                # 这里必须硬失败：静默地"工具缺失"会产出一份看似正常、实则
                # 无据可查的报告，比报错糟糕得多。
                raise RuntimeError(
                    "offline 模式没有任何可用 MCP server：请确认 DR_MCP_SERVERS 中"
                    "至少有 {network: false} 的知识库 server（如 kb）"
                )
            logger.warning("no MCP servers configured; researchers will run tool-less")
            _tools_cache = []
            return _tools_cache

        tools: list[BaseTool] = []
        for name, params in servers.items():
            try:
                # params 是运行时按 DR_MCP_SERVERS 构造的 dict，SDK 的联合类型无法静态覆盖
                client = MultiServerMCPClient({name: params})  # type: ignore[dict-item]
                server_tools = await client.get_tools()
                # 每个工具的输出在进入模型之前就被包成"不可信外部数据"
                tools.extend(_guarded(t) for t in server_tools)
                logger.info("mcp server %s: loaded %d tools", name, len(server_tools))
            except Exception:
                logger.exception("mcp server %s: failed to load, skipped", name)

        _tools_cache = tools
        return _tools_cache


def reset_tool_cache() -> None:
    """Tests and long-running servers call this after MCP config changes."""
    global _tools_cache
    _tools_cache = None
