"""对检索指标做 bootstrap 置信区间与配对差异检验（纯计算，零成本）。

为什么需要：n=21 时 1 题 = 4.8 个百分点，
均值差异必须带上不确定性区间才能判断「是否真的更好」。

用法： python -m scripts.bootstrap_ci
输入： experiments/stageA_raw.csv
输出： experiments/stageA_ci.json + 终端表
"""

from __future__ import annotations

import csv
import json
import pathlib
import random

RAW = pathlib.Path("experiments/stageA_raw.csv")
OUT = pathlib.Path("experiments/stageA_ci.json")
B = 2000          # bootstrap 次数
SEED = 20261031


def load() -> dict[str, dict[str, int | None]]:
    """{config_id: {qid: page_rank}}"""
    data: dict[str, dict[str, int | None]] = {}
    for r in csv.DictReader(RAW.open(encoding="utf-8")):
        data.setdefault(r["config_id"], {})[r["qid"]] = (
            int(r["page_rank"]) if r["page_rank"] else None
        )
    return data


def hit(rank: int | None, k: int) -> int:
    return 1 if rank is not None and rank <= k else 0


def boot_mean(values: list[int], rng: random.Random) -> tuple[float, float, float]:
    """返回 (点估计, 2.5% 分位, 97.5% 分位)。"""
    n = len(values)
    point = sum(values) / n
    means = []
    for _ in range(B):
        s = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    return point, means[int(0.025 * B)], means[int(0.975 * B)]


def boot_paired(a: list[int], b: list[int], rng: random.Random) -> tuple[float, float, float]:
    """配对差异 a-b 的 bootstrap 区间。"""
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    point = sum(diffs) / n
    out = []
    for _ in range(B):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        out.append(s / n)
    out.sort()
    return point, out[int(0.025 * B)], out[int(0.975 * B)]


def main() -> None:
    rng = random.Random(SEED)
    data = load()
    qids = sorted(next(iter(data.values())).keys())

    results: dict[str, dict] = {}
    print(f"  n={len(qids)} 题, bootstrap {B} 次\n")
    print(f"  {'配置':<34}{'hit@10':>9}{'95% CI':>18}")
    for cid, ranks in data.items():
        vals = [hit(ranks[q], 10) for q in qids]
        p, lo, hi = boot_mean(vals, rng)
        rec = [hit(ranks[q], 50) for q in qids]
        rp, rlo, rhi = boot_mean(rec, rng)
        results[cid] = {
            "hit10": p, "hit10_ci": [lo, hi],
            "recall50": rp, "recall50_ci": [rlo, rhi],
            "n": len(qids),
        }
        print(f"  {cid:<34}{p:>8.1%}   [{lo:>6.1%}, {hi:>6.1%}]")

    # 关键配对比较
    pairs = [
        ("emb_api__rr_p50__exp3", "emb_api__rr_off__exp0", "最优 vs 基线（无 rerank 无扩展）"),
        ("emb_api__rr_p50__exp3", "emb_api__rr_p100__exp0", "扩展 vs 扩大候选池"),
        ("emb_api__rr_p50__exp3", "emb_api__rr_p50__exp0", "扩展（同候选池）"),
        ("emb_api__rr_p100__exp0", "emb_api__rr_p50__exp0", "候选池 50→100（同无扩展）"),
    ]
    print(f"\n  {'配对比较':<40}{'Δhit@10':>10}{'95% CI':>20}{'判定':>10}")
    for a, b, label in pairs:
        va = [hit(data[a][q], 10) for q in qids]
        vb = [hit(data[b][q], 10) for q in qids]
        p, lo, hi = boot_paired(va, vb, rng)
        verdict = "显著" if (lo > 0 or hi < 0) else "不显著"
        print(f"  {label:<40}{p:>+9.1%}   [{lo:>+6.1%}, {hi:>+6.1%}] {verdict:>8}")
        results.setdefault("pairs", {})[f"{a} vs {b}"] = {
            "delta_hit10": p, "ci": [lo, hi], "significant": verdict == "显著"
        }

    # 分辨率说明
    step = 1 / len(qids)
    print(f"\n  分辨率：n={len(qids)} → 1 题 = {step:.1%}")
    print(f"  含义：小于 2 题（{2*step:.1%}）的差异在本样本上无法与噪声区分")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
