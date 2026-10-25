"""Local knowledge base for offline mode: chunk store + hybrid retrieval.

Design notes
------------
- Same PostgreSQL instance as checkpoints/memory: one less thing to deploy, and
  the offline story must be "one database, no network".
- **Every chunk carries its provenance** (source_path + anchor). Offline mode's
  citation verifier resolves "file + anchor + quote", which is stronger than the
  online mode's "URL returns 200". Losing provenance would break the whole point.
- Hybrid retrieval = dense (pgvector) + sparse (Postgres full-text), fused with
  RRF. Chinese has no whitespace, so the sparse side is only useful when a
  text-search config for CJK exists; we detect that once and degrade to
  dense-only instead of silently returning garbage.
- Embeddings reuse the model already used by long-term memory (bge-small-zh,
  512 dims) — no extra dependency, no API cost.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

from deep_research.core.config import get_settings
from deep_research.core.memory import embed_query, embed_texts

logger = logging.getLogger(__name__)

CHUNK_CHARS = 600
CHUNK_OVERLAP = 80
RRF_K = 60

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_CJK = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class Chunk:
    doc_id: str
    source_path: str
    title: str
    anchor: str
    chunk_idx: int
    text: str


@dataclass
class Hit:
    source_path: str
    title: str
    anchor: str
    text: str
    score: float


def _pgvec(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


_ASCII_WORD = re.compile(r"[a-z0-9]+")


def _lexical_tokens(text: str) -> set[str]:
    """Cheap CJK-safe tokenisation: ASCII words + CJK character bigrams.

    Postgres full-text needs a Chinese dictionary (zhparser) which is not
    installed, so `to_tsvector('simple', ...)` treats a whole Chinese sentence as
    one token. Character bigrams are a standard workaround that keeps exact-term
    matching usable without extensions: 「蒸馏模型」→ {蒸馏, 馏模, 模型}.
    """
    low = text.lower()
    tokens: set[str] = set(_ASCII_WORD.findall(low))
    cjk = [ch for ch in low if "\u4e00" <= ch <= "\u9fff"]
    tokens.update(a + b for a, b in zip(cjk, cjk[1:], strict=False))
    return tokens


def _lexical_score(query_tokens: set[str], text: str) -> float:
    """Overlap of query tokens with a chunk, length-normalised."""
    if not query_tokens:
        return 0.0
    doc_tokens = _lexical_tokens(text)
    if not doc_tokens:
        return 0.0
    hit = len(query_tokens & doc_tokens)
    return hit / (math.sqrt(len(query_tokens)) * math.sqrt(len(doc_tokens)))



def split_document(text: str, source_path: str, title: str = "") -> list[Chunk]:
    """Split a document into overlapping chunks, tracking the nearest heading.

    The anchor is the closest heading above the chunk ("## 2. 方法"), falling back
    to the line range. Offline citations resolve against it, so a chunk without
    an anchor is a chunk that cannot be cited.
    """
    lines = text.splitlines()
    chunks: list[Chunk] = []
    buf: list[str] = []
    size = 0
    heading = title or "(无标题)"
    start_line = 1
    idx = 0
    last_heading_line = 0

    def flush(end_line: int) -> None:
        nonlocal buf, size, idx
        body = "\n".join(buf).strip()
        if body:
            chunks.append(
                Chunk(
                    doc_id=source_path,
                    source_path=source_path,
                    title=title or heading,
                    anchor=f"{heading} · L{start_line}-{end_line}",
                    chunk_idx=idx,
                    text=body,
                )
            )
            idx += 1
        # keep a tail overlap so a sentence spanning the boundary stays findable
        tail: list[str] = []
        acc = 0
        for line in reversed(buf):
            tail.insert(0, line)
            acc += len(line) + 1
            if acc >= CHUNK_OVERLAP:
                break
        buf = tail
        size = sum(len(x) + 1 for x in buf)

    for lineno, line in enumerate(lines, 1):
        m = _HEADING.match(line)
        if m:
            if size > 0:
                flush(lineno - 1)
            heading = f"{line.strip()}"
            last_heading_line = lineno
            start_line = lineno
        if not buf:
            start_line = lineno
        buf.append(line)
        size += len(line) + 1
        if size >= CHUNK_CHARS:
            flush(lineno)
    if buf and size > 0:
        flush(len(lines))
    _ = last_heading_line
    return chunks


class KBStore:
    """Chunk store + hybrid search over local documents."""

    def __init__(self, database_url: str | None = None, dim: int | None = None) -> None:
        settings = get_settings()
        self._url = database_url or settings.database_url
        self._dim = dim or settings.memory_embedding_dim
        self._sparse_ok: bool | None = None

    async def _connect(self) -> Any:
        import psycopg

        return await psycopg.AsyncConnection.connect(self._url)

    async def setup(self) -> None:
        import psycopg
        from psycopg import sql

        async with await self._connect() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS kb_chunks (
                        id BIGSERIAL PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        title TEXT NOT NULL DEFAULT '',
                        anchor TEXT NOT NULL DEFAULT '',
                        chunk_idx INT NOT NULL DEFAULT 0,
                        text TEXT NOT NULL,
                        embedding vector({dim}) NOT NULL,
                        tsv tsvector,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        UNIQUE (doc_id, chunk_idx)
                    )
                    """
                ).format(dim=sql.Literal(self._dim))
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS kb_chunks_embedding_hnsw
                ON kb_chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS kb_chunks_tsv ON kb_chunks USING gin (tsv)"
            )
            await conn.commit()

    async def _has_cjk_config(self) -> bool:
        """Is there a text-search config that actually splits Chinese?"""
        if self._sparse_ok is not None:
            return self._sparse_ok
        import psycopg

        async with await self._connect() as conn:
            cur = await conn.execute(
                "SELECT count(*) FROM pg_ts_config WHERE cfgname IN ('zhparser', 'chinese')"
            )
            row = await cur.fetchone()
            self._sparse_ok = bool(row and row[0])
        if not self._sparse_ok:
            logger.info(
                "kb: 无中文分词配置（zhparser/chinese），稀疏检索降级为向量-only"
            )
        return self._sparse_ok

    async def add_chunks(self, chunks: list[Chunk]) -> int:
        """Upsert chunks (idempotent per doc_id+chunk_idx)."""
        if not chunks:
            return 0
        vecs = embed_texts([c.text for c in chunks])
        rows = [
            (
                c.doc_id,
                c.source_path,
                c.title,
                c.anchor,
                c.chunk_idx,
                c.text,
                _pgvec(v),
                json.dumps({"anchor": c.anchor}, ensure_ascii=False),
            )
            for c, v in zip(chunks, vecs, strict=True)
        ]
        async with await self._connect() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    """
                    INSERT INTO kb_chunks
                        (doc_id, source_path, title, anchor, chunk_idx, text, embedding, tsv)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector,
                            to_tsvector('simple', %s))
                    ON CONFLICT (doc_id, chunk_idx) DO UPDATE
                       SET text = EXCLUDED.text,
                           embedding = EXCLUDED.embedding,
                           tsv = EXCLUDED.tsv,
                           anchor = EXCLUDED.anchor
                    """,
                    rows,
                )
            await conn.commit()
        return len(rows)

    async def search(self, query: str, top_k: int = 8) -> list[Hit]:
        """Hybrid search: dense + lexical, fused by RRF.

        A pure-dense ranking was measurably wrong on short queries: 「ReAct 与
        Chain-of-Thought 的区别」 ranked a reranker report (0 mentions of ReAct)
        above the ReAct report (37 mentions) — 0.722 vs 0.689 — because a short
        heading-only chunk compares "similar" to a short query. Lexical overlap
        is the cheap correction: exact terms should dominate generic prose.
        """
        if not query.strip():
            return []
        vec = embed_query(query)
        dense = await self._dense(vec, top_k * 4)
        sparse: list[tuple[str, float]] = []
        if await self._has_cjk_config() and not _CJK.search(query):
            # 英文查询才走 Postgres FTS：中文在没有分词配置时 fts 几乎没有意义
            sparse = await self._sparse(query, top_k * 4)

        # 词法通道：中文字符二元组 + ASCII 词，解决"专有名词/术语"匹配
        q_tokens = _lexical_tokens(query)
        lexical = sorted(
            ((f"{h.source_path}#{h.anchor}", _lexical_score(q_tokens, h.title + "\n" + h.text))
             for h in dense),
            key=lambda kv: kv[1],
            reverse=True,
        )
        lexical = [(k, s) for k, s in lexical if s > 0][: top_k * 4]

        fused: dict[str, float] = {}
        rows: dict[str, Hit] = {}
        for rank, hit in enumerate(dense, 1):
            key = f"{hit.source_path}#{hit.anchor}"
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
            rows[key] = hit
        for rank, (key, _) in enumerate(sparse, 1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (key, _) in enumerate(lexical, 1):
            # 词法通道权重略高：术语命中比"语义看起来像"更可信
            fused[key] = fused.get(key, 0.0) + 1.5 / (RRF_K + rank)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            Hit(
                source_path=rows[k].source_path,
                title=rows[k].title,
                anchor=rows[k].anchor,
                text=rows[k].text,
                score=round(score, 6),
            )
            for k, score in ordered
            if k in rows
        ]

    async def _dense(self, vec: list[float], limit: int) -> list[Hit]:
        async with await self._connect() as conn:
            cur = await conn.execute(
                """
                SELECT source_path, title, anchor, text,
                       1 - (embedding <=> %s::vector) AS score
                FROM kb_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (_pgvec(vec), _pgvec(vec), limit),
            )
            rows = await cur.fetchall()
        return [
            Hit(source_path=r[0], title=r[1], anchor=r[2], text=r[3], score=float(r[4]))
            for r in rows
        ]

    async def _sparse(self, query: str, limit: int) -> list[tuple[str, float]]:
        async with await self._connect() as conn:
            cur = await conn.execute(
                """
                SELECT source_path, anchor,
                       ts_rank(tsv, plainto_tsquery('simple', %s)) AS rank
                FROM kb_chunks
                WHERE tsv @@ plainto_tsquery('simple', %s)
                ORDER BY rank DESC
                LIMIT %s
                """,
                (query, query, limit),
            )
            rows = await cur.fetchall()
        return [(f"{r[0]}#{r[1]}", float(r[2])) for r in rows]

    async def stats(self) -> dict[str, int]:
        async with await self._connect() as conn:
            cur = await conn.execute(
                "SELECT count(*) AS chunks, count(DISTINCT doc_id) AS docs FROM kb_chunks"
            )
            row = await cur.fetchone()
        return {"chunks": int(row[0]), "docs": int(row[1])}


_store: KBStore | None = None


async def get_kb_store() -> KBStore:
    """Process-wide store; `setup()` is idempotent so first call initialises it."""
    global _store
    if _store is None:
        store = KBStore()
        await store.setup()
        _store = store
    return _store
