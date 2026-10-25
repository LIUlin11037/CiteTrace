"""FinanceBench 检索层评分（不调用 LLM，零成本）。

评分对象：知识库检索能否把「evidence 所在的那一页」捞回来。这是分层评测的
第一层——检索失手，生成再好也没用，而这一层不需要花一分钱。

两个必须处理的口径问题：
1. **页码 0-indexed**：FinanceBench 标注 55 → PDF 第 56 页（三份 10-K 实测全部 +1）。
   不换算会把所有命中判成失败。
2. **tabular 文本差异**：PDF 抽出的表格与数据集里的 evidence 原文空格/顺序不同，
   所以"evidence 文本命中"用去空白后的子串匹配，别用严格相等。

Usage:
    python scripts/eval_financebench_retrieval.py --top-k 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deep_research.core.compat import use_compatible_event_loop  # noqa: E402
from deep_research.core.kb import get_kb_store  # noqa: E402

DATA = Path("kb_demo/financebench/data/financebench_open_source.jsonl")
DOCS = {
    "AMD_2022_10K": "kb_demo/financebench/pdfs/AMD_2022_10K.pdf",
    "AMERICANEXPRESS_2022_10K": "kb_demo/financebench/pdfs/AMERICANEXPRESS_2022_10K.pdf",
    "BOEING_2022_10K": "kb_demo/financebench/pdfs/BOEING_2022_10K.pdf",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _page_of(anchor: str) -> int | None:
    m = re.match(r"p(\d+)$", anchor or "")
    return int(m.group(1)) if m else None


def load_questions() -> list[dict]:
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r["doc_name"] in DOCS]


async def main(ks: list[int]) -> None:
    use_compatible_event_loop()
    store = await get_kb_store()
    questions = load_questions()
    depth = max(ks)
    print(f"题目: {len(questions)} 道 / {len(DOCS)} 份 10-K | 检索深度 {depth}")
    print(f"库规模: {(await store.stats())['chunks']} 块\n")

    rows_out: list[dict] = []
    for q in questions:
        evidence = q["evidence"][0]
        gold_page = int(evidence["evidence_page_num"]) + 1  # 0-indexed → 1-indexed
        gold_text = _norm(evidence["evidence_text"])[:120]
        hits = await store.search(q["question"], top_k=depth)
        pages = [(_page_of(h.anchor), h) for h in hits]
        page_rank = next(
            (
                i
                for i, (pg, h) in enumerate(pages, 1)
                if h.source_path == DOCS[q["doc_name"]] and pg == gold_page
            ),
            None,
        )
        doc_rank = next(
            (i for i, (_, h) in enumerate(pages, 1) if h.source_path == DOCS[q["doc_name"]]),
            None,
        )
        text_rank = next(
            (i for i, (_, h) in enumerate(pages, 1) if gold_text and gold_text in _norm(h.text)),
            None,
        )
        rows_out.append(
            {
                "id": q["financebench_id"],
                "doc": q["doc_name"],
                "type": str(q["question_reasoning"])[:28],
                "gold_page": gold_page,
                "page_rank": page_rank,
                "doc_rank": doc_rank,
                "text_rank": text_rank,
                "question": q["question"][:70],
            }
        )

    def rate(key: str, k: int, subset: list[dict] | None = None) -> float:
        rows = subset if subset is not None else rows_out
        got = sum(1 for r in rows if r[key] and r[key] <= k)
        return got / len(rows) if rows else 0.0

    print("  " + "k".rjust(4) + "页面命中".rjust(10) + "同文档命中".rjust(12) + "evidence命中".rjust(14))
    for k in ks:
        print(
            f"  {k:>4}{rate('page_rank', k):>10.0%}{rate('doc_rank', k):>12.0%}"
            f"{rate('text_rank', k):>14.0%}"
        )

    print("\n  按文档（页面命中@10 / 题数）：")
    for doc in sorted({r["doc"] for r in rows_out}):
        sub = [r for r in rows_out if r["doc"] == doc]
        print(f"   {doc:26} {rate('page_rank', 10, sub):>5.0%} / {len(sub):>2}")
    print("\n  按推理类型（@10）：")
    for typ in sorted({r["type"] for r in rows_out}):
        sub = [r for r in rows_out if r["type"] == typ]
        print(f"   {typ:32} {rate('page_rank', 10, sub):>5.0%} / {len(sub):>2}")

    out = Path("reports/eval/financebench_retrieval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "n": len(rows_out),
                "ks": ks,
                "page_hit": {str(k): rate("page_rank", k) for k in ks},
                "doc_hit": {str(k): rate("doc_rank", k) for k in ks},
                "evidence_text_hit": {str(k): rate("text_rank", k) for k in ks},
                "details": rows_out,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  明细写入: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50])
    args = ap.parse_args()
    # 必须在 asyncio.run 之前：psycopg 不能跑在 ProactorEventLoop 上
    use_compatible_event_loop()
    asyncio.run(main(sorted(set(args.ks))))
