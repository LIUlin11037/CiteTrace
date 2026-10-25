"""Knowledge-base MCP server: offline retrieval over the local corpus.

Launch (stdio): python -m mcp_servers.kb_server

This is the **only** tool offline mode gives the Researcher. Three consequences
shaped the implementation:

1. every hit carries provenance (`source_path` + `anchor`) so the writer can cite
   "file + location", which the offline verifier resolves by reading the file —
   stronger than the online "URL returns 200" check;
2. when nothing matches, say so explicitly (`KB_EMPTY:`). Offline mode has no
   network fallback, so a silent empty result would make the Critic request
   follow-ups forever;
3. **stdout is the JSON-RPC channel** — never print to it. Diagnostics go to
   stderr via logging. (Measured 2026-10-24: a stray `print()` inside a tool made
   the client fail with "Invalid JSON" and the call never returned.)

Warm-up runs before serving: the embedder is loaded once at startup, otherwise
the first request pays for model initialisation inside the call — which, in the
stdio subprocess, looked exactly like a hang.
"""

from __future__ import annotations

import asyncio
import logging
import os

from mcp.server.fastmcp import FastMCP

from deep_research.core.compat import use_compatible_event_loop
from deep_research.core.kb import get_kb_store

# 必须在 FastMCP 启动自己的事件循环之前设置：psycopg 不能跑在
# Windows 默认的 ProactorEventLoop 上（子进程里同样如此）。
use_compatible_event_loop()

# 默认只输出警告：这个 server 的 INFO 日志（每次工具调用都会打）会淹没
# 调用方的正常输出。排查时用 DR_KB_LOG_LEVEL=INFO 打开。
logging.basicConfig(level=os.environ.get("DR_KB_LOG_LEVEL", "WARNING").upper())
logger = logging.getLogger("kb_server")

mcp = FastMCP("kb")


@mcp.tool()
async def search_knowledge_base(query: str, top_k: int = 8) -> str:
    """在本地知识库中检索相关片段；返回带出处（文件 + 锚点）的原文片段。

    Args:
        query: 检索问题或关键词
        top_k: 返回片段数，默认 8
    """
    store = await get_kb_store()
    hits = await store.search(query, top_k=top_k)
    if not hits:
        return "KB_EMPTY: 本地知识库没有检索到相关内容（离线模式无网络兜底）"
    blocks: list[str] = []
    for i, h in enumerate(hits, 1):
        blocks.append(
            f"[{i}] 来源：{h.source_path}\n位置：{h.anchor}\n标题：{h.title}\n片段：\n{h.text}"
        )
    return "\n\n---\n\n".join(blocks)


@mcp.tool()
async def list_knowledge_base_documents() -> str:
    """列出知识库中已入库的文档，用于确认覆盖范围。"""
    import psycopg

    from deep_research.core.config import get_settings

    async with await psycopg.AsyncConnection.connect(get_settings().database_url) as conn:
        cur = await conn.execute(
            """
            SELECT source_path, count(*) AS chunks
            FROM kb_chunks GROUP BY source_path ORDER BY source_path
            """
        )
        rows = await cur.fetchall()
    if not rows:
        return "KB_EMPTY: 知识库为空（先运行 scripts/ingest_kb.py）"
    return "\n".join(f"{r[0]}（{r[1]} 块）" for r in rows)


async def _warm_up() -> None:
    """Load the embedding model and create tables before the first request."""
    from deep_research.core.memory import embed_texts

    embed_texts(["warmup"])
    await get_kb_store()
    logger.info("kb server warm-up done")


if __name__ == "__main__":
    asyncio.run(_warm_up())
    mcp.run(transport="stdio")
