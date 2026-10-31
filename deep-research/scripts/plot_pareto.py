"""生成 Pareto 图（纯 Python 输出 SVG，不依赖 matplotlib）。

横轴：每题的检索延迟（秒）—— 代理"成本"，因为 embedding/rerank 开销随配置变化
纵轴：hit@10
标出 Pareto 前沿：不在任何其他点"右下"的点（延迟更低且命中率更高）

用法： python -m scripts.plot_pareto
输入： experiments/stageA_summary.csv
输出： experiments/pareto.svg
"""

from __future__ import annotations

import csv
import pathlib

SRC = pathlib.Path("experiments/stageA_summary.csv")
OUT = pathlib.Path("experiments/pareto.svg")

W, H = 680, 420
PAD_L, PAD_R, PAD_T, PAD_B = 64, 132, 34, 48


def load() -> list[dict]:
    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    out = []
    for r in rows:
        out.append(
            {
                "id": r["config_id"],
                "x": float(r["avg_latency_s"]),
                "y": float(r["hit@10"]) * 100,
                "recall": float(r["found@50"]) * 100,
                "mrr": float(r["mrr"]),
                "emb": r["emb"],
                "rr": r["rerank"],
                "exp": r["expand"],
            }
        )
    return out


def pareto(pts: list[dict]) -> list[dict]:
    """最小化 x、最大化 y 的前沿点。"""
    ordered = sorted(pts, key=lambda p: (p["x"], -p["y"]))
    best_y = -1.0
    front = []
    for p in ordered:
        if p["y"] > best_y:
            front.append(p)
            best_y = p["y"]
    return front


def main() -> None:
    pts = load()
    front = pareto(pts)
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    x0, x1 = 0, max(xs) * 1.12
    y0, y1 = 0, max(ys) * 1.22

    def sx(v: float) -> float:
        return PAD_L + (v - x0) / (x1 - x0) * (W - PAD_L - PAD_R)

    def sy(v: float) -> float:
        return H - PAD_B - (v - y0) / (y1 - y0) * (H - PAD_T - PAD_B)

    css = "font-family:system-ui,-apple-system,Segoe UI,sans-serif"
    s: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
        f'<text x="{PAD_L}" y="22" font-size="15" font-weight="500" fill="#2C2C2A" style="{css}">'
        "Retrieval configs: latency vs hit@10 (n=21 questions)</text>",
    ]

    # 网格
    for gy in range(0, 81, 20):
        if gy > y1:
            break
        yy = sy(gy)
        s.append(
            f'<line x1="{PAD_L}" y1="{yy:.1f}" x2="{W-PAD_R}" y2="{yy:.1f}" '
            'stroke="#E8E6E0" stroke-width="0.5"/>'
        )
        s.append(
            f'<text x="{PAD_L-10}" y="{yy+4:.1f}" font-size="11" fill="#888780" '
            f'text-anchor="end" style="{css}">{gy}%</text>'
        )

    # 坐标轴
    s.append(
        f'<line x1="{PAD_L}" y1="{H-PAD_B}" x2="{W-PAD_R}" y2="{H-PAD_B}" '
        'stroke="#B4B2A9" stroke-width="0.8"/>'
    )
    s.append(
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H-PAD_B}" '
        'stroke="#B4B2A9" stroke-width="0.8"/>'
    )
    for gx in [0, 1, 2, 3, 4]:
        if gx > x1:
            break
        xx = sx(gx)
        s.append(
            f'<text x="{xx:.1f}" y="{H-PAD_B+18}" font-size="11" fill="#888780" '
            f'text-anchor="middle" style="{css}">{gx}s</text>'
        )
    s.append(
        f'<text x="{(PAD_L+W-PAD_R)/2:.0f}" y="{H-10}" font-size="12" fill="#5F5E5A" '
        f'text-anchor="middle" style="{css}">latency per question (s)</text>'
    )

    # 前沿连线
    if len(front) >= 2:
        path = " ".join(
            f'{"M" if i == 0 else "L"}{sx(p["x"]):.1f},{sy(p["y"]):.1f}'
            for i, p in enumerate(front)
        )
        s.append(
            f'<path d="{path}" fill="none" stroke="#1D9E75" stroke-width="1.5" stroke-dasharray="5 3"/>'
        )

    # 散点 + 标签
    front_ids = {p["id"] for p in front}
    for p in pts:
        color = "#1D9E75" if p["id"] in front_ids else "#85B7EB"
        r = 6 if p["id"] in front_ids else 4.5
        s.append(
            f'<circle cx="{sx(p["x"]):.1f}" cy="{sy(p["y"]):.1f}" r="{r}" '
            f'fill="{color}" fill-opacity="0.9" stroke="#FFFFFF" stroke-width="1"/>'
        )

    # 只给前沿点和最优召回点加标签，避免拥挤
    labelled = [p for p in pts if p["id"] in front_ids]
    best_recall = max(pts, key=lambda p: (p["recall"], p["y"]))
    if best_recall not in labelled:
        labelled.append(best_recall)
    for p in labelled:
        short = p["id"].replace("emb_", "").replace("__rr_", "/").replace("__", "/")
        s.append(
            f'<text x="{sx(p["x"])+9:.1f}" y="{sy(p["y"])+4:.1f}" font-size="10.5" '
            f'fill="#2C2C2A" style="{css}">{short}</text>'
        )

    # 图例
    lx = W - PAD_R + 12
    s.append(
        f'<circle cx="{lx+4}" cy="{PAD_T+6}" r="4.5" fill="#1D9E75"/>'
        f'<text x="{lx+16}" y="{PAD_T+10}" font-size="11" fill="#444441" style="{css}">Pareto front</text>'
    )
    s.append(
        f'<circle cx="{lx+4}" cy="{PAD_T+28}" r="4.5" fill="#85B7EB"/>'
        f'<text x="{lx+16}" y="{PAD_T+32}" font-size="11" fill="#444441" style="{css}">dominated</text>'
    )
    s.append(
        f'<text x="{lx}" y="{PAD_T+62}" font-size="10.5" fill="#5F5E5A" style="{css}">n=21 questions</text>'
        f'<text x="{lx}" y="{PAD_T+78}" font-size="10.5" fill="#5F5E5A" style="{css}">3 FinanceBench 10-K</text>'
    )
    s.append("</svg>")

    OUT.write_text("\n".join(s), encoding="utf-8")
    print(f"  -> {OUT}")
    print(f"  前沿点 {len(front)} 个:")
    for p in front:
        print(f"     {p['id']:<34} hit@10={p['y']:.1f}%  recall={p['recall']:.1f}%  {p['x']:.2f}s")


if __name__ == "__main__":
    main()
