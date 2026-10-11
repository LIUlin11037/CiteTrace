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
from typing import Awaitable, Callable, Literal

import httpx

Verdict = Literal["exists", "missing", "unverifiable"]

_REF_LINE = re.compile(r"^\[\^?(\d+)\]:\s*(.+)$", re.MULTILINE)
_URL = re.compile(r"https?://[^\s）)>\]（(]+")
_ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?")
_GITHUB_REPO = re.compile(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/?")


@dataclass
class Citation:
    index: int
    line: str
    url: str
    kind: Literal["arxiv", "github", "web"]
    target: str  # arxiv id / owner/repo / url


@dataclass
class CitationReport:
    total: int = 0
    exists: int = 0
    missing: int = 0
    unverifiable: int = 0
    details: list[dict] = field(default_factory=list)

    @property
    def existence_rate(self) -> float:
        """Exists / (exists + missing). Unverifiable is excluded, not punished."""
        denom = self.exists + self.missing
        return self.exists / denom if denom else 0.0


def extract_citations(report: str) -> list[Citation]:
    out: list[Citation] = []
    for m in _REF_LINE.finditer(report):
        line = m.group(2).strip()
        url_m = _URL.search(line)
        if not url_m:
            continue
        url = url_m.group(0).rstrip(".,;")
        if am := _ARXIV_ID.search(url):
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
    resp = await client.get(
        "https://export.arxiv.org/api/query", params={"id_list": arxiv_id}
    )
    if resp.status_code != 200:
        return "unverifiable"
    return "exists" if "<entry>" in resp.text and "<title>" in resp.text else "missing"


async def _default_verify_github(client: httpx.AsyncClient, repo: str) -> Verdict:
    resp = await client.get(f"https://api.github.com/repos/{repo}")
    if resp.status_code == 200:
        return "exists"
    if resp.status_code == 404:
        return "missing"
    return "unverifiable"  # 403 rate-limit etc.


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
) -> CitationReport:
    va = verify_arxiv or _default_verify_arxiv
    vg = verify_github or _default_verify_github
    vw = verify_web or _default_verify_web
    verifiers = {"arxiv": va, "github": vg, "web": vw}

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
