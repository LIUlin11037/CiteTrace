"""GitHub 引用单独复核：低并发 + 重试（高并发会被 github 直接掐断）。"""

import collections
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "src")
from deep_research.eval.citations import extract_citations  # noqa: E402


def curl_code(url: str, timeout: int = 25) -> str:
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


def check_one(repo: str) -> tuple[str, str]:
    url = f"https://github.com/{repo}"
    for attempt in range(3):
        code = curl_code(url)
        if code not in ("000", "429") and not code.startswith("5"):
            return repo, code
        time.sleep(1.5 * (attempt + 1))
    return repo, code


repos = []
for f in sorted(pathlib.Path("reports/eval/reports").glob("gq-*.md")):
    for c in extract_citations(f.read_text(encoding="utf-8")):
        if c.kind == "github" and c.target not in repos:
            repos.append(c.target)

print(f"  GitHub 仓库（去重）: {len(repos)} 个", flush=True)
with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(check_one, repos))

stat = collections.Counter(
    "exists" if c.startswith("2") or c == "304" else ("missing" if c in ("404", "410") else "unverifiable")
    for _, c in results
)
print(f"  结果: {dict(stat)}", flush=True)
bad = [(r, c) for r, c in results if c not in ("200",)]
if bad[:10]:
    print("  非 200 样例:", bad[:10], flush=True)

pathlib.Path("reports/eval/github_recheck.json").write_text(
    json.dumps(
        {"repos": len(repos), "stat": dict(stat), "detail": results},
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
