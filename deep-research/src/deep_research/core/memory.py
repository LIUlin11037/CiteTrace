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
from dataclasses import dataclass
from typing import Any

from deep_research.core.config import get_settings

logger = logging.getLogger(__name__)

_embedder: Any = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        _embedder = TextEmbedding(get_settings().memory_embedding_model)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [list(map(float, v)) for v in _get_embedder().embed(texts)]


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
        self._dim = settings.memory_embedding_dim

    async def setup(self) -> None:
        import psycopg

        async with await psycopg.AsyncConnection.connect(self._url) as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memories (
                    id BIGSERIAL PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}',
                    embedding vector({self._dim}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
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
        vec = embed_texts([query])[0]
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
