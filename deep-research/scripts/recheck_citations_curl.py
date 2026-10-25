"""一次性重验：用 curl 核验 26 题报告的全部引用（httpx 在本机连不上 github.com）。

背景：上一轮 880 条引用里有 309 条被记成 unverifiable，抽样发现几乎全是 GitHub 仓库
（httpx ConnectTimeout，而 curl 正常）。这不是引用造假，是传输层问题，所以换 curl 重验。

输出：reports/eval/cite_recheck_curl.json（exists / missing / unverifiable 三态）
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "src")
from deep_research.eval.citations import extract_citations  # noqa: E402


def curl_head(url: str, timeout: int = 20) -> str:
    try:
        out = subprocess.run(
            ["curl", "-s", "-o", os.devnull, "-w", "%{http_code}", "-I", "-L", "--noproxy", "*", url],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (out.stdout or "").strip() or "000"
    except Exception:  # noqa: BLE001
        return "000"


def target_url(kind: str, target: str) -> tuple[str, str]:
    """Return (url, strategy). arxiv 走 abs 页面，github 走仓库页，web 原样。"""
    if kind == "arxiv":
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", target)
        return (f"https://arxiv.org/abs/{m.group(1)}", "arxiv") if m else ("", "")
    if kind == "github":
        return f"https://github.com/{target}", "github"
    if kind == "web":
        return target, "web"
    return "", ""


def verdict(code: str) -> str:
    if code.startswith("2") or code == "304":
        return "exists"
    if code == "404" or code == "410":
        return "missing"
    return "unverifiable"


def main() -> None:
    cites = []
    for f in sorted(pathlib.Path("reports/eval/reports").glob("gq-*.md")):
        for c in extract_citations(f.read_text(encoding="utf-8")):
            url, strat = target_url(c.kind, c.target)
            if url:
                cites.append((f.stem, c.kind, strat, url, c.target))

    print(f"  待核验: {len(cites)} 条", flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        codes = list(ex.map(lambda kv: curl_head(kv[3]), cites))

    stat = collections.Counter(verdict(c) for c in codes)
    by_kind: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for (_, kind, _, _, _), code in zip(cites, codes):
        by_kind[kind][verdict(code)] += 1

    print(f"  合计: {dict(stat)}", flush=True)
    for k, v in by_kind.items():
        print(f"    {k:8} {dict(v)}", flush=True)

    missing = [
        {"report": r, "kind": k, "target": t, "url": u, "code": c}
        for (r, k, _, u, t), c in zip(cites, codes)
        if verdict(c) == "missing"
    ]
    if missing:
        print(f"\n  404/410（真缺失）{len(missing)} 条：")
        for m in missing[:15]:
            print(f"    [{m['kind']}] {m['target'][:80]}")

    total = len(cites)
    out = {
        "citations": total,
        "exists": stat["exists"],
        "missing": stat["missing"],
        "unverifiable": stat["unverifiable"],
        "existence_rate": stat["exists"] / max(stat["exists"] + stat["missing"], 1),
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
        "missing_detail": missing,
        "method": "curl HEAD, 16 threads; arxiv→abs page, github→repo page",
    }
    p = pathlib.Path("reports/eval/cite_recheck_curl.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  存在率 {out['existence_rate']:.1%}（{stat['exists']}/{stat['exists'] + stat['missing']}）")
    print(f"  明细: {p}")


if __name__ == "__main__":
    main()
