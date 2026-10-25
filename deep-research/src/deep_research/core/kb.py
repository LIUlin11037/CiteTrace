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

import asyncio
import json
import logging
import math
from pathlib import Path
import re

import httpx
from psycopg import sql
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


def resolve_kb_path(raw: str) -> Path:
    """Resolve a stored kb path to a real file, tolerating absolute or relative.

    知识库里存的是**相对路径**（相对 ``DR_KB_ROOT``），这样同一份库在宿主机和
    容器里都能用——实测踩过：入库时存了宿主机的绝对路径，容器里那个文件不存在，
    离线证据被全部丢弃（cite/cover 归零）。
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    return Path(get_settings().kb_root).resolve() / p


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


def split_pdf_pages(
    pages: list[str], source_path: str, title: str = "", page_chars: int = 4000
) -> list[Chunk]:
    """Chunk a PDF page by page, with the page number as the anchor.

    页码锚点是 PDF 语料的引用单位：私有公文（年报/合同/问询函）里"第 59 页"
    才是人话，"某个 600 字符片段"不是。``pages`` 必须按 PDF 顺序传入（1-indexed
    的编号在 anchor 里），调用方负责用 pypdf 抽文本。

    注意：FinanceBench 这类数据集的页码常是 **0-indexed**（标注 55 → PDF 第 56 页），
    评分时必须先对齐，否则会全判错。
    """
    chunks: list[Chunk] = []
    idx = 0
    # 默认「一页一块」：财报的关键信息是一张表，切成 600 字符会把表头和数字分开，
    # 结果是"命中了同一页却不是同一块"（实测：页面命中 38%、evidence 文本命中仅 5%）。
    # 只有超长页（> page_chars）才二次切分。
    step = max(page_chars, 500)
    for page_no, text in enumerate(pages, start=1):
        # PDF 抽取常带 NUL 与控制字符：Postgres 的 text 类型直接拒收 0x00
        # （实测入 three 份 10-K 时报 "cannot contain NUL (0x00) bytes"）。
        body = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text or "").strip()
        if not body:
            continue
        for start in range(0, len(body), step):
            piece = body[start : start + step]
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    doc_id=source_path,
                    source_path=source_path,
                    title=title or source_path.rsplit("/", 1)[-1],
                    anchor=f"p{page_no}",
                    chunk_idx=idx,
                    text=piece,
                )
            )
            idx += 1
    return chunks


class KBStore:
    """Chunk store + hybrid search over local documents."""

    def __init__(self, database_url: str | None = None, dim: int | None = None) -> None:
        settings = get_settings()
        self._url = database_url or settings.database_url
        # API 模式下维度由 API 模型决定（如 DashScope text-embedding-v4 = 1024），
        # 不能用本地的 memory_embedding_dim（512），否则建表/写入必错。
        default_dim = (
            settings.embedding_api_dim
            if settings.embedding_provider == "api"
            else settings.memory_embedding_dim
        )
        self._dim = dim or default_dim
        table = settings.kb_table
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError(f"非法 kb 表名: {table!r}")
        self._table = table
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
                    CREATE TABLE IF NOT EXISTS {table} (
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
                ).format(dim=sql.Literal(self._dim), table=sql.Identifier(self._table))
            )
            await conn.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON {table} "
                    "USING hnsw (embedding vector_cosine_ops)"
                ).format(
                    idx=sql.Identifier(f"{self._table}_embedding_hnsw"),
                    table=sql.Identifier(self._table),
                )
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS kb_chunks_tsv ON "
                + self._table
                + " USING gin (tsv)"
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
                    INSERT INTO {table}
                        (doc_id, source_path, title, anchor, chunk_idx, text, embedding, tsv)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector,
                            to_tsvector('simple', %s))
                    ON CONFLICT (doc_id, chunk_idx) DO UPDATE
                       SET text = EXCLUDED.text,
                           embedding = EXCLUDED.embedding,
                           tsv = EXCLUDED.tsv,
                           anchor = EXCLUDED.anchor
                    """.format(table=self._table),
                    rows,
                )
            await conn.commit()
        return len(rows)

    async def _expand(self, query: str) -> list[str]:
        """Rewrite the query into several term-level variants (query expansion).

        为什么不放在 embed 阶段做：这不是"换个更好的向量"，而是**换问法**。
        实测失败的典型形态是"问题问『流动性是否健康』、页面上只有 Current ratio"——
        两者没有共同词，任何 embedding 都救不了；把问题改写成财务术语才行。
        """
        from deep_research.core.llm import get_chat_model

        prompt = (
            "把下面这道检索问题改写成 3 条检索用查询，要求：\n"
            "- 每条都要包含可能出现在原文里的具体术语/指标名（不要照抄原句）\n"
            "- 覆盖不同的可能表述（同义词、上位词、英文术语）\n"
            "- 每行一条，不要编号，不要解释\n\n"
            f"问题：{query}"
        )
        try:
            model = get_chat_model("planner")
            resp = await model.ainvoke(prompt)
        except Exception as exc:  # noqa: BLE001 - 扩展失败就退回原查询
            logger.warning("query 扩展失败，退回单查询: %s", exc)
            return []
        lines = [ln.strip(" -•\t") for ln in str(resp.content).splitlines() if ln.strip()]
        return [ln for ln in lines if len(ln) > 6][: get_settings().query_expansion_n]

    @staticmethod
    def _merge_hits(groups: list[list[Hit]], k: int = RRF_K) -> list[Hit]:
        """RRF-fuse several result lists (dedup by source_path#anchor)."""
        scores: dict[str, float] = {}
        first: dict[str, Hit] = {}
        for hits in groups:
            for rank, h in enumerate(hits, 1):
                key = f"{h.source_path}#{h.anchor}"
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                first.setdefault(key, h)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [first[key] for key, _ in ordered]

    async def search(
        self,
        query: str,
        top_k: int = 8,
        rerank: bool | None = None,
        expand: bool | None = None,
    ) -> list[Hit]:
        """Hybrid search: dense + lexical, fused by RRF.

        A pure-dense ranking was measurably wrong on short queries: 「ReAct 与
        Chain-of-Thought 的区别」 ranked a reranker report (0 mentions of ReAct)
        above the ReAct report (37 mentions) — 0.722 vs 0.689 — because a short
        heading-only chunk compares "similar" to a short query. Lexical overlap
        is the cheap correction: exact terms should dominate generic prose.
        """
        if not query.strip():
            return []
        want_expand = get_settings().query_expansion if expand is None else expand
        if want_expand:
            variants = await self._expand(query)
            if variants:
                # 多路检索时每路多取一些候选，融合后再截断/精排。
                # 四路彼此独立，必须并发：串行会把单次检索从 0.6s 拖到 4.5s。
                groups = list(
                    await asyncio.gather(
                        *[
                            self.search(q, top_k=top_k * 2, rerank=False, expand=False)
                            for q in [query, *variants]
                        ]
                    )
                )
                merged = self._merge_hits(groups)[: max(top_k, get_settings().rerank_candidates)]
                if (rerank is None and self._rerank_enabled()) or rerank:
                    try:
                        return await self._rerank(query, merged, top_k)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("rerank 失败，退回融合结果: %s", exc)
                return merged[:top_k]
        vec = embed_query(query)
        dense = await self._dense(vec, top_k * 4)
        sparse: list[tuple[str, float]] = []
        # 英文查询不需要分词配置（'simple' 配置按空格切词就够）；中文才依赖
        # zhparser。原来的条件是「有 zhparser 且 query 无中文」——zhparser 没装时
        # 英文全文检索也被一起关掉了（实测：FinanceBench 页面命中率被拖到 14%）。
        if not _CJK.search(query) or await self._has_cjk_config():
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
        ordered = [(k, sc) for k, sc in sorted(fused.items(), key=lambda kv: kv[1], reverse=True) if k in rows]

        def _to_hits(pairs: list[tuple[str, float]]) -> list[Hit]:
            return [
                Hit(
                    source_path=rows[k].source_path,
                    title=rows[k].title,
                    anchor=rows[k].anchor,
                    text=rows[k].text,
                    score=round(score, 6),
                )
                for k, score in pairs
            ]

        s = get_settings()
        want_rerank = s.rerank_provider == "api" if rerank is None else rerank
        if want_rerank and s.rerank_provider == "api" and ordered:
            # 先多取候选（默认 50），交给 reranker 精排后再截 top_k
            depth = max(top_k, s.rerank_candidates)
            cands = _to_hits(ordered[:depth])
            try:
                return await self._rerank(query, cands, top_k)
            except Exception as exc:  # noqa: BLE001 - rerank 挂了不能拖垮检索
                logger.warning("rerank 失败，退回 RRF 结果: %s", exc)
        return _to_hits(ordered[:top_k])

    async def _dense(self, vec: list[float], limit: int) -> list[Hit]:
        async with await self._connect() as conn:
            cur = await conn.execute(
                """
                SELECT source_path, title, anchor, text,
                       1 - (embedding <=> %s::vector) AS score
                FROM {table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """.format(table=self._table),
                (_pgvec(vec), _pgvec(vec), limit),
            )
            rows = await cur.fetchall()
        return [
            Hit(source_path=r[0], title=r[1], anchor=r[2], text=r[3], score=float(r[4]))
            for r in rows
        ]

    def _rerank_enabled(self) -> bool:
        return get_settings().rerank_provider == "api"

    async def _rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        """Rerank fused candidates with an external reranker.

        检索失败的典型形态不是"没找到"，而是"正确的页排在第 30 位"：RRF 靠向量
        相似度排序，对"这个数字在哪一页"无能为力；reranker 直接判 query-doc 相关性，
        正是治这个的（实测把 capex 相关句从无关句中挑出来：0.84 vs 0.15）。
        """
        s = get_settings()
        base = s.rerank_api_base.rstrip("/")
        url = (
            base
            if "rerank" in base.rsplit("/", 1)[-1]
            else base + "/api/v1/services/rerank/text-rerank/text-rerank"
        )
        payload = {
            "model": s.rerank_api_model,
            "input": {"query": query, "documents": [h.text[:1500] for h in hits]},
            "parameters": {"return_documents": False, "top_n": min(top_k, len(hits))},
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                url, json=payload, headers={"Authorization": f"Bearer {s.rerank_api_key}"}
            )
            resp.raise_for_status()
            results = resp.json()["output"]["results"]
        return [hits[r["index"]] for r in results if 0 <= r["index"] < len(hits)]

    async def _sparse(self, query: str, limit: int) -> list[tuple[str, float]]:
        async with await self._connect() as conn:
            cur = await conn.execute(
                """
                SELECT source_path, anchor,
                       ts_rank(tsv, plainto_tsquery('simple', %s)) AS rank
                FROM {table}
                WHERE tsv @@ plainto_tsquery('simple', %s)
                ORDER BY rank DESC
                LIMIT %s
                """.format(table=self._table),
                (query, query, limit),
            )
            rows = await cur.fetchall()
        return [(f"{r[0]}#{r[1]}", float(r[2])) for r in rows]

    async def delete_docs(self, source_paths: list[str]) -> int:
        """Delete all chunks of the given documents from **this** store's table.

        改切块策略后必须整篇重灌；删除要落在当前表上（曾把表名写死在别处，
        结果"重置"清的是另一张表，旧块残留导致指标不可信）。
        """
        if not source_paths:
            return 0
        async with await self._connect() as conn:
            cur = await conn.execute(
                sql.SQL("DELETE FROM {table} WHERE source_path = ANY(%s)").format(
                    table=sql.Identifier(self._table)
                ),
                (source_paths,),
            )
            await conn.commit()
            return int(cur.rowcount or 0)

    async def stats(self) -> dict[str, int]:
        async with await self._connect() as conn:
            cur = await conn.execute(
                f"SELECT count(*) AS chunks, count(DISTINCT doc_id) AS docs FROM {self._table}"
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
