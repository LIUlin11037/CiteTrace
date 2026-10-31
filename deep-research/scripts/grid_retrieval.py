"""Stage A：检索层网格（只算检索指标，不跑 LLM 生成）。

为什么 k 不是"配置轴"：`search()` 内部按 `max(top_k, rerank_candidates)` 取候选再截断，
所以一次取 50 条就能算出所有 k 的命中率——k 只是"看前多少条"的观察窗口。

真正的配置轴有三条：embedding 后端、rerank 候选池、query 扩展条数。
12 组 = 2 × 3 × 2，每组每题检索一次（depth=50），同时得出 hit@5/10/15/20/30 + recall@50 + MRR。

用法：
    python -m scripts.grid_retrieval                # 全量 12 组 × 21 题
    python -m scripts.grid_retrieval --limit 3      # 冒烟：3 题
    python -m scripts.grid_retrieval --configs emb_local__rr_off__exp0
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, "src")

from deep_research.core.compat import use_compatible_event_loop  # noqa: E402

use_compatible_event_loop()

KS = [5, 10, 15, 20, 30]
DEPTH = 50          # 统一检索深度，够覆盖所有 k
OUT_DIR = pathlib.Path("experiments")


def _load_helper():
    """复用现有检索脚本的题目加载与融合逻辑，避免两套实现漂移。"""
    spec = importlib.util.spec_from_file_location(
        "fb_retrieval", pathlib.Path("scripts/eval_financebench_retrieval.py")
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def build_configs() -> list[dict]:
    """embedding(2) × rerank(3) × expansion(2) = 12 组。"""
    embs = [
        {"emb_tag": "local", "emb_provider": "local", "table": "kb_chunks_local", "dim": 512},
        {"emb_tag": "api", "emb_provider": "api", "table": "kb_chunks_qwen", "dim": 1024},
    ]
    reranks = [
        {"rr_tag": "off", "rr_provider": "none", "pool": 0},
        {"rr_tag": "p50", "rr_provider": "api", "pool": 50},
        {"rr_tag": "p100", "rr_provider": "api", "pool": 100},
    ]
    exps = [{"tag": "exp0", "n": 0}, {"tag": "exp3", "n": 3}]

    out = []
    for e in embs:
        for r in reranks:
            for x in exps:
                cid = f"emb_{e['emb_tag']}__rr_{r['rr_tag']}__{x['tag']}"
                out.append({"config_id": cid, **e, **r, "expand": x["n"]})
    return out


def apply_settings(cfg: dict) -> None:
    from deep_research.core.config import get_settings

    os.environ["DR_KB_TABLE"] = cfg["table"]
    os.environ["DR_EMBEDDING_PROVIDER"] = cfg["emb_provider"]
    os.environ["DR_RERANK_PROVIDER"] = cfg["rr_provider"]
    os.environ["DR_RERANK_CANDIDATES"] = str(max(cfg["pool"], DEPTH))
    os.environ["DR_QUERY_EXPANSION"] = "false"   # 扩展由脚本自己控制，避免二次扩展
    get_settings.cache_clear()


async def run_config(cfg: dict, questions: list[dict], helper, limit: int | None) -> list[dict]:
    from deep_research.core.kb import KBStore

    apply_settings(cfg)
    store = KBStore(dim=cfg["dim"])
    rows: list[dict] = []
    qs = questions[:limit] if limit else questions
    rerank_on = cfg["rr_provider"] != "none"

    for q in qs:
        ev = q["evidence"][0]
        gold_page = int(ev["evidence_page_num"]) + 1
        gold_doc = helper.DOCS[q["doc_name"]]
        t0 = time.perf_counter()

        if cfg["expand"]:
            variants = await helper.expand_query(q["question"])
            queries = [q["question"], *variants[: cfg["expand"]]]
            groups = [await store.search(x, top_k=DEPTH, rerank=False, expand=False) for x in queries]
            merged = helper._rrf_merge(groups)[:DEPTH]
            if rerank_on:
                try:
                    hits = await store._rerank(q["question"], merged, DEPTH)
                except Exception:  # noqa: BLE001 - 精排失败退回融合结果
                    hits = merged
            else:
                hits = merged
        else:
            hits = await store.search(q["question"], top_k=DEPTH, expand=False)

        latency = time.perf_counter() - t0
        page_rank = None
        for i, h in enumerate(hits, 1):
            if h.source_path == gold_doc and helper._page_of(h.anchor) == gold_page:
                page_rank = i
                break

        rows.append(
            {
                "config_id": cfg["config_id"],
                "qid": q["financebench_id"] if "financebench_id" in q else q.get("id", ""),
                "doc": q["doc_name"],
                "gold_page": gold_page,
                "page_rank": page_rank,
                "found": page_rank is not None,
                "latency_s": round(latency, 3),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    ranks = [r["page_rank"] for r in rows]
    res = {
        "n": n,
        "found@50": sum(1 for r in ranks if r and r <= 50) / n,
        "mrr": sum(1.0 / r for r in ranks if r) / n,
        "avg_latency_s": sum(r["latency_s"] for r in rows) / n,
    }
    for k in KS:
        res[f"hit@{k}"] = sum(1 for r in ranks if r and r <= k) / n
    return res


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--configs", nargs="+", default=None, help="只跑这些 config_id")
    args = ap.parse_args()

    helper = _load_helper()
    questions = helper.load_questions()
    all_cfgs = build_configs()
    cfgs = [c for c in all_cfgs if not args.configs or c["config_id"] in args.configs]

    print(f"题目 {len(questions)} 道 | 配置 {len(cfgs)} 组 | 深度 {DEPTH}")
    OUT_DIR.mkdir(exist_ok=True)

    raw: list[dict] = []
    summary: list[dict] = []
    for i, cfg in enumerate(cfgs, 1):
        t0 = time.time()
        rows = await run_config(cfg, questions, helper, args.limit)
        agg = summarize(rows)
        summary.append({"config_id": cfg["config_id"], **agg,
                        "emb": cfg["emb_provider"], "rerank": cfg["rr_tag"], "expand": cfg["expand"]})
        raw.extend(rows)
        print(
            f"  [{i}/{len(cfgs)}] {cfg['config_id']:<34} "
            f"hit@10={agg['hit@10']:.1%} hit@20={agg['hit@20']:.1%} "
            f"recall@50={agg['found@50']:.1%} MRR={agg['mrr']:.3f} "
            f"{agg['avg_latency_s']:.2f}s/题 ({time.time()-t0:.0f}s)",
            flush=True,
        )

    if raw:
        p = OUT_DIR / "stageA_raw.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
            w.writeheader()
            w.writerows(raw)
        print(f"\n逐题明细 -> {p}")

    if summary:
        p = OUT_DIR / "stageA_summary.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        (OUT_DIR / "stageA_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"汇总 -> {p}")


if __name__ == "__main__":
    asyncio.run(main())
