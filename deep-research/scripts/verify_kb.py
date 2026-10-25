"""Measure knowledge-base retrieval quality (offline, no API key).

Corpus: the 26 evaluation reports (real research output, ~1 MB of text).
Labels: automatic but meaningful — each golden topic is expected to retrieve the
report that was written for it (gq-<id>.md). This is a *retrieval proxy*, not a
human-judged relevance set: it answers "does the right document surface?", not
"is the best passage on top".

Measured here: recall@1 / recall@3 / MRR with and without the BGE query
instruction prefix.

**Known limitation (measured 2026-10-24)**: this proxy is saturated — every topic
has a dedicated report, so both variants score recall@1 = 100%, MRR = 1.000.
It therefore cannot show whether the prefix helps. The prefix is kept because the
BGE model card requires it for retrieval (and it costs nothing), NOT because this
script demonstrates a gain. To measure it, the corpus needs near-duplicate
documents or paraphrased queries.

Run: python scripts/verify_kb.py [--ingest]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deep_research.core.compat import use_compatible_event_loop  # noqa: E402
from deep_research.core.kb import get_kb_store, split_document  # noqa: E402
from deep_research.core.memory import embed_query, embed_texts  # noqa: E402
from deep_research.eval.dataset import load_dataset  # noqa: E402


PROJECT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT / "reports" / "eval" / "reports"
GOLDEN = PROJECT / "src" / "deep_research" / "eval" / "golden" / "tech_ai.json"


async def ingest_corpus(store) -> None:  # type: ignore[no-untyped-def]
    files = sorted(CORPUS.glob("*.md"))
    total = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        title = next(
            (ln.strip("# ").strip() for ln in text.splitlines() if ln.startswith("#")),
            f.name,
        )
        rel = str(f).replace("\\", "/")
        total += await store.add_chunks(split_document(text, rel, title))
    print(f"  语料：{len(files)} 份报告 / {total} 块")


async def recall_and_mrr(store, questions, use_prefix: bool, k: int = 10) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    hit1 = 0
    mrr = 0.0
    for q in questions:
        vec = embed_query(q.topic) if use_prefix else embed_texts([q.topic])[0]
        ranked = await store._dense(vec, k)  # 只看稠密通道，避免融合掩盖差异
        expected = f"gq-{q.id.removeprefix('gq-')}.md"
        rank = next((i for i, h in enumerate(ranked, 1) if expected in h.source_path), None)
        if rank:
            hit1 += int(rank == 1)
            mrr += 1.0 / rank
    return hit1 / len(questions), mrr / len(questions)


async def test_known_queries(store) -> None:  # type: ignore[no-untyped-def]
    """回归：纯向量排序曾把「0 次提及」的文档排在第一。

    「ReAct 与 Chain-of-Thought 的区别」→ 纯 dense 命中 gq-reranker.md（0 次提及
    ReAct，0.722）而不是 gq-react.md（37 次提及，0.689）。加了中文字符二元组词法
    通道后，正确文档必须排第一。
    """
    cases = {
        "ReAct 与 Chain-of-Thought 的区别": "gq-react.md",
        "混合专家 MoE 的负载均衡怎么做": "gq-moe.md",
        "长上下文为什么不会让 RAG 过时": "gq-long-context.md",
    }
    for query, expected in cases.items():
        hits = await store.search(query, top_k=1)
        top = hits[0].source_path.split("/")[-1] if hits else "(空)"
        assert top == expected, f"{query} → 期望 {expected}，实际 {top}"
        print(f"[ok] 已知查询命中：{query[:20]}… → {top}")


async def main(do_ingest: bool) -> None:
    use_compatible_event_loop()
    store = await get_kb_store()
    if do_ingest:
        await ingest_corpus(store)
    stats = await store.stats()
    print(f"  库状态：{stats['docs']} 个文档 / {stats['chunks']} 块")

    questions = load_dataset(GOLDEN)
    p1, pm = await recall_and_mrr(store, questions, use_prefix=False)
    q1, qm = await recall_and_mrr(store, questions, use_prefix=True)
    print(f"\n  无前缀: recall@1={p1:.0%}  MRR={pm:.3f}")
    print(f"  带前缀: recall@1={q1:.0%}  MRR={qm:.3f}")
    if abs(pm - qm) < 1e-9 and p1 == q1 == 1.0:
        print("  ⚠ 测试饱和（两种方式都是满分）：本脚本无法证明前缀收益，"
              "需要近重复文档或改述查询才能区分")
    assert qm >= pm - 1e-9, "查询前缀不应让检索变差"

    await test_known_queries(store)


if __name__ == "__main__":
    # 必须在 asyncio.run 之前：psycopg 不能跑在 ProactorEventLoop 上
    use_compatible_event_loop()
    asyncio.run(main("--ingest" in sys.argv))
