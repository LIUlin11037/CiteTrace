"""Long-term memory on PostgreSQL + pgvector.

What gets remembered (written by writer_node after each run):
- report_summary: topic + 摘要段，让后续研究知道"这个题做过、结论是什么"
- evidence_digest: 每个子问题收集到的来源清单（标题+URL+要点数），
  让后续研究可以复用来源而不是从零检索

What gets recalled (read by planner_node before drafting):
- 与当前课题语义相近的历史记忆，作为"相关历史研究"注入 Planner 上下文

Embedding: local fastembed model (ONNX) by default — DeepSeek has no
embeddings API, and memory must not depend on the chat provider. The model
is loaded lazily on first use; swap via DR_MEMORY_EMBEDDING_MODEL.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from deep_research.core.config import get_settings

logger = logging.getLogger(__name__)

_embedder: Any = None


def _get_embedder() -> Any:
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        from deep_research.core.compat import ensure_hf_compat

        ensure_hf_compat()  # safety net if no entry point ran compat first
        model = get_settings().memory_embedding_model
        # 离线部署：镜像里预置了 fastembed 自认的离线布局
        # （``<cache>/fast-<name>/``，实测这是它唯一认的形态；HF 的
        # ``models--...`` 目录在 local_files_only 下会被判"不完整"）。
        # 存在就直接离线加载，连一次网络探测都不做。
        cache = os.environ.get("FASTEMBED_CACHE_PATH") or ""
        offline_ready = bool(cache) and (
            Path(cache) / f"fast-{model.split('/')[-1]}"
        ).is_dir()
        _embedder = TextEmbedding(model, local_files_only=True) if offline_ready else TextEmbedding(model)
    return _embedder


def _embed_api(texts: list[str], text_type: str = "document") -> list[list[float]]:
    """Embed via an OpenAI-compatible /embeddings endpoint (DashScope 等).

    ``text_type``: DashScope 的 v3/v4 支持 query/document 区分，比"查询加前缀"
    更干净；不支持该参数的服务会忽略它（多余字段不会报错的服务居多，报错的
    情况下把 DR_EMBEDDING_API_STRICT 关掉即可）。
    """
    import httpx

    s = get_settings()
    if not s.embedding_api_key:
        raise RuntimeError(
            "DR_EMBEDDING_PROVIDER=api 但 DR_EMBEDDING_API_KEY 为空："
            "请在 .env 里配置（如 DashScope 的 sk-...）"
        )
    url = s.embedding_api_base.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {s.embedding_api_key}"}
    out: list[list[float]] = []
    step = max(1, s.embedding_api_batch)
    with httpx.Client(timeout=60.0) as client:
        for i in range(0, len(texts), step):
            batch = texts[i : i + step]
            payload = {
                "model": s.embedding_api_model,
                "input": batch,
                "dimensions": s.embedding_api_dim,
                "encoding_format": "float",
                "text_type": text_type,
            }
            # 服务端会按 RPM/TPM 限流（实测 429）。批量入库时这很常见，
            # 直接抛错会让"嵌了一半"的库处于不一致状态，所以退避重试。
            for attempt in range(6):
                resp = client.post(url, json=payload, headers=headers)
                if resp.status_code != 429:
                    break
                wait = min(2 ** attempt, 30)
                logger.warning("embedding 限流(429)，%ss 后重试（第 %d 次）", wait, attempt + 1)
                time.sleep(wait)
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
            out.extend([[float(x) for x in d["embedding"]] for d in data])
    return out


def embed_texts(texts: list[str]) -> list[list[float]]:
    if get_settings().embedding_provider == "api":
        return _embed_api(texts, text_type="document")
    return [list(map(float, v)) for v in _get_embedder().embed(texts)]


# BGE 系列是「非对称检索」模型：文档侧裸文本，查询侧必须加指令前缀，否则
# query 与 doc 落在不同分布上，相似度排序明显变差（实测 recall@3 见
# scripts/verify_kb.py）。前缀用模型卡给出的中文检索指令。
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def embed_query(text: str) -> list[float]:
    """Embed a search query.

    local：BGE 系列需要检索指令前缀；api：用 text_type=query，不要再加前缀
    （加了反而偏离该模型的查询分布）。
    """
    if get_settings().embedding_provider == "api":
        return _embed_api([text], text_type="query")[0]
    return embed_texts([_QUERY_PREFIX + text])[0]


@dataclass
class MemoryHit:
    kind: str
    content: str
    metadata: dict[str, Any]
    score: float


class MemoryStore:
    """pgvector-backed memory store. One instance per process is fine."""

    def __init__(self, database_url: str | None = None) -> None:
        settings = get_settings()
        self._url = database_url or settings.database_url
        self._namespace = settings.memory_namespace
        # 维度必须跟着实际用的 embedding 后端走：API 模式返回 embedding_api_dim
        # （如 1024），若仍按 memory_embedding_dim（512）建表，写入会维度不匹配。
        self._dim = (
            settings.embedding_api_dim
            if settings.embedding_provider == "api"
            else settings.memory_embedding_dim
        )

    async def setup(self) -> None:
        import psycopg
        from psycopg import sql

        async with await psycopg.AsyncConnection.connect(self._url) as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # 维度必须进 DDL，用 psycopg 的 SQL 组合而不是 f-string：
            # psycopg 3.3 的 Query 类型只接受 LiteralString / SQL / Composed，
            # 拼接式 f-string 会被类型检查拒绝（也正是防注入的用意）。
            await conn.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id BIGSERIAL PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}',
                        embedding vector({dim}) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                ).format(dim=sql.Literal(self._dim))
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS memories_embedding_hnsw
                ON memories USING hnsw (embedding vector_cosine_ops)
                """
            )

    async def add(self, kind: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        import psycopg

        vec = embed_texts([content])[0]
        async with await psycopg.AsyncConnection.connect(self._url) as conn:
            await conn.execute(
                "INSERT INTO memories (namespace, kind, content, metadata, embedding)"
                " VALUES (%s, %s, %s, %s, %s::vector)",
                (self._namespace, kind, content, json.dumps(metadata or {}), _pgvec(vec)),
            )

    async def add_many(self, items: list[tuple[str, str, dict[str, Any]]]) -> None:
        if not items:
            return
        import psycopg

        vecs = embed_texts([content for _, content, _ in items])
        async with await psycopg.AsyncConnection.connect(self._url) as conn:
            for (kind, content, metadata), vec in zip(items, vecs, strict=True):
                await conn.execute(
                    "INSERT INTO memories (namespace, kind, content, metadata, embedding)"
                    " VALUES (%s, %s, %s, %s, %s::vector)",
                    (self._namespace, kind, content, json.dumps(metadata), _pgvec(vec)),
                )

    async def search(self, query: str, limit: int | None = None) -> list[MemoryHit]:
        import psycopg

        settings = get_settings()
        limit = limit or settings.memory_recall_limit
        vec = embed_query(query)
        async with await psycopg.AsyncConnection.connect(self._url) as conn:
            cur = await conn.execute(
                """
                SELECT kind, content, metadata, 1 - (embedding <=> %s::vector) AS score
                FROM memories
                WHERE namespace = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (_pgvec(vec), self._namespace, _pgvec(vec), limit),
            )
            rows = await cur.fetchall()
        hits = [
            MemoryHit(kind=r[0], content=r[1], metadata=r[2], score=float(r[3])) for r in rows
        ]
        return [h for h in hits if h.score >= settings.memory_min_score]


def _pgvec(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


_store: MemoryStore | None = None


async def get_memory_store() -> MemoryStore:
    """Process-wide store; setup() is idempotent so first call initializes."""
    global _store
    if _store is None:
        _store = MemoryStore()
        await _store.setup()
    return _store
