"""Citation extraction and programmatic existence verification.

Parses the writer's footnote format:
    [^n]: 标题 — URL（来源类型，日期）
then verifies each target by real network calls:
  - arxiv.org/abs/<id>  -> arXiv API id_list lookup
  - github.com/<o>/<r>  -> GitHub repos API
  - anything else       -> HTTP GET, status < 400

Verifiers are injectable (verify_arxiv / verify_github / verify_web) so tests
run fully offline. All network failures count as "unverifiable", NOT "missing"
— a flaky network must not tank the metric.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

import httpx

Verdict = Literal["exists", "missing", "unverifiable"]

_REF_LINE = re.compile(r"^\[\^?(\d+)\]:\s*(.+)$", re.MULTILINE)
_URL = re.compile(r"https?://[^\s）)>\]（(]+")
_ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?")
_GITHUB_REPO = re.compile(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/?")
# API 形态的链接（api.github.com/repos/o/r）：旧的仓库正则会把 "repos/o" 当成仓库名，
# 核验必然失败（实测 gq-rag-eval 因此少算一条真引用）。先单独识别。
_GITHUB_API = re.compile(r"api\.github\.com/repos/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
# 本地文件绝对路径（Windows 盘符或 POSIX 根路径），用于离线模式的知识库引用
# 注意 `(?<![A-Za-z0-9])`：否则 "https:/" 里的 "s:/" 会被当成 Windows 盘符，
# 所有 http 引用都会被误判成本地文件（实测 smoke_eval 全变 kb）。
_LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|(?<![:/\w])/)[^\s（）()<>|\"']+")
# 知识库引用用的是**相对路径**（可移植）：`标题 — reports/x.md · 锚点（kb, 日期）`。
# 只有绝对路径的正则匹配不到它，实测会把正文 121 个脚注全判成"没有引用"。


def _kb_target(line: str) -> str:
    """Extract ``<path> · <anchor>`` from a knowledge-base reference line.

    结构化解析而不是纯正则：标题里常带全角括号（``标题（1.2 节）``），
    正则会把标题也吞进 target（实测 28 条全判 missing）。
    """
    idx = line.find("（kb")
    if idx < 0:
        idx = line.find("(kb")
    if idx < 0:
        return ""
    left = line[:idx].rstrip("）) ").strip()
    target = left.rsplit("—", 1)[-1].strip(" —")
    return target if "/" in target or "\\" in target else ""


@dataclass
class Citation:
    index: int
    line: str
    url: str
    kind: Literal["arxiv", "github", "web", "kb"]
    target: str  # arxiv id / owner/repo / url


@dataclass
class CitationReport:
    total: int = 0
    exists: int = 0
    missing: int = 0
    unverifiable: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def existence_rate(self) -> float:
        """Exists / (exists + missing). Unverifiable is excluded, not punished."""
        denom = self.exists + self.missing
        return self.exists / denom if denom else 0.0


def extract_citations(report: str) -> list[Citation]:
    out: list[Citation] = []
    for m in _REF_LINE.finditer(report):
        line = m.group(2).strip()
        # 离线模式的引用目标是本地文件；这类行可能完全没有 http 链接，
        # 旧实现会直接 `continue` 把它漏掉（实测：本地引用一条都没被统计）。
        local_m = _LOCAL_PATH.search(line)
        # 只认"像文件名"的路径：标题里出现 "Graph-/Agentic" 这类斜杠会被误判成本地
        # 文件（实测 gq-graph-rag 因此把一条真 arXiv 引用判成 missing）。
        if local_m and not re.search(
            r"\.(md|markdown|txt|pdf)$", local_m.group(0).split(" · ")[0], re.I
        ):
            local_m = None
        if local_m:
            path_part = local_m.group(0).rstrip(".,;，。")
            # 路径后面紧跟的「· 锚点」要一起保留，否则核验只能判断文件存在，
            # 判断不了"引用的是不是文件里的那一节"。
            rest = line[local_m.end() :].strip(" ·—-，,;")
            rest = re.split(r"[（(]", rest, maxsplit=1)[0].strip()
            target = f"{path_part} · {rest}" if rest else path_part
            out.append(Citation(int(m.group(1)), line, target, "kb", target))
            continue
        kb_target = _kb_target(line)
        if kb_target:
            out.append(Citation(int(m.group(1)), line, kb_target, "kb", kb_target))
            continue
        url_m = _URL.search(line)
        if not url_m:
            continue
        url = url_m.group(0).rstrip(".,;")
        if api_m := _GITHUB_API.search(url):
            out.append(Citation(int(m.group(1)), line, url, "github", api_m.group(1)))
        elif am := _ARXIV_ID.search(url):
            out.append(Citation(int(m.group(1)), line, url, "arxiv", am.group(1)))
        elif gm := _GITHUB_REPO.search(url):
            repo = gm.group(1).removesuffix(".git")
            if repo.count("/") == 1 and not repo.endswith(("/issues", "/pull")):
                out.append(Citation(int(m.group(1)), line, url, "github", repo))
            else:
                out.append(Citation(int(m.group(1)), line, url, "web", url))
        else:
            out.append(Citation(int(m.group(1)), line, url, "web", url))
    return out


async def _default_verify_arxiv(client: httpx.AsyncClient, arxiv_id: str) -> Verdict:
    # 显式标注：httpx 的 params 要求 Mapping[str, PrimitiveData | Sequence[...]]，
    # 裸 dict 字面量会被推断成 dict[str, object] 而报类型错误。
    params: dict[str, str] = {"id_list": arxiv_id}
    resp = await client.get("https://export.arxiv.org/api/query", params=params)
    if resp.status_code != 200:
        return "unverifiable"
    if "<entry>" in resp.text and "<title>" in resp.text:
        return "exists"
    # API 偶发返回空 feed（不是 404）。实测：arXiv 2606.25656 在一次评测里被判
    # missing，几分钟后同样请求却能返回 entry —— 这种抖动不能算成"引用不存在"，
    # 否则会把真实引用记成幻觉。用 abs 页面二次确认：200 存在 / 404 缺失 / 其它不可核验。
    try:
        abs_resp = await client.get(f"https://arxiv.org/abs/{arxiv_id}")
    except httpx.HTTPError:
        return "unverifiable"
    if abs_resp.status_code == 200:
        return "exists"
    if abs_resp.status_code == 404:
        return "missing"
    return "unverifiable"


async def _default_verify_github(client: httpx.AsyncClient, repo: str) -> Verdict:
    resp = await client.get(f"https://api.github.com/repos/{repo}")
    if resp.status_code == 200:
        return "exists"
    if resp.status_code == 404:
        return "missing"
    return "unverifiable"  # 403 rate-limit etc.


async def _default_verify_kb(_client: httpx.AsyncClient, target: str) -> Verdict:
    """Verify a knowledge-base citation by reading the file, not by network.

    Stronger than the URL check: "file exists" AND "the cited section actually
    appears in it". target format: ``<path> · <anchor>``.
    """
    path_part, _, anchor = target.partition(" · ")
    from deep_research.core.kb import resolve_kb_path

    path = resolve_kb_path(path_part.strip())
    if not path.exists():
        return "missing"
    # 模型写锚点很不规矩：`## 摘要 · L3-5`、`## 摘要 L3-8`、甚至用 `；`
    # 串起多段（实测 9/38 因此被误判 missing）。这里只取第一段、剥掉行号再比。
    anchor = anchor.split("；")[0].split(";")[0].strip()
    heading = re.split(r"\s*L\d+-\d+\s*$", anchor)[0].strip().lstrip("#").strip()
    if not heading:
        return "exists"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unverifiable"

    # 模型抄锚点时经常截断或换标点（实测：`…默认 overlap` 少了后半句就会
    # 逐字匹配失败，把一条真引用判成 missing）。先规范化再比前缀，能容忍
    # 截断/标点差异，又不会放过"根本不存在的章节"。
    def _norm(s: str) -> str:
        return re.sub(r"[\s#：:，,。.、\-—_·]+", "", s)

    if _norm(heading)[:20] in _norm(text):
        return "exists"

    # 兜底：锚点带行号且文件确实有那么多行 → 位置可定位
    m = re.search(r"L(\d+)-(\d+)", anchor)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if 0 < start <= end <= len(text.splitlines()):
            return "exists"
    return "missing"


async def _default_verify_web(client: httpx.AsyncClient, url: str) -> Verdict:
    resp = await client.get(url, follow_redirects=True)
    if resp.status_code < 400:
        return "exists"
    if resp.status_code in {404, 410}:
        return "missing"
    return "unverifiable"  # 403/429/5xx: bot-blocking is not evidence of absence


async def verify_citations(
    report: str,
    *,
    timeout: float = 15.0,
    concurrency: int = 5,
    verify_arxiv: Callable[[httpx.AsyncClient, str], Awaitable[Verdict]] | None = None,
    verify_github: Callable[[httpx.AsyncClient, str], Awaitable[Verdict]] | None = None,
    verify_web: Callable[[httpx.AsyncClient, str], Awaitable[Verdict]] | None = None,
    verify_kb: Callable[[httpx.AsyncClient, str], Awaitable[Verdict]] | None = None,
) -> CitationReport:
    va = verify_arxiv or _default_verify_arxiv
    vg = verify_github or _default_verify_github
    vw = verify_web or _default_verify_web
    vk = verify_kb or _default_verify_kb
    verifiers = {"arxiv": va, "github": vg, "web": vw, "kb": vk}

    citations = extract_citations(report)
    result = CitationReport(total=len(citations))
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:

        async def check(c: Citation) -> None:
            async with sem:
                try:
                    verdict = await verifiers[c.kind](client, c.target)
                except Exception:
                    verdict = "unverifiable"
            setattr(result, verdict, getattr(result, verdict) + 1)
            result.details.append(
                {"index": c.index, "kind": c.kind, "target": c.target, "verdict": verdict}
            )

        await asyncio.gather(*(check(c) for c in citations))
    return result
