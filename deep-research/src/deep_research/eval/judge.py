"""LLM-as-judge for the two metrics code cannot compute.

1. Claim-source alignment: sample cited claims from the report and judge each
   against the evidence of ITS OWN referenced sources. Claim [^n] numbers are
   resolved through the report's reference list to URLs and matched against
   the evidence store by normalized URL — the writer's numbering is its own
   and MUST NOT be mapped positionally onto the evidence list.
2. Golden coverage: for each expected key point of a golden question, judge
   whether the report actually states it.

Both go through structured output (no free-text parsing). The judge uses the
critic role model by default — judgement quality matters more than cost here.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from deep_research.agents.state import Evidence
from deep_research.core.llm import invoke_with_retry, structured_output
from deep_research.eval.citations import extract_citations

CLAIM_SAMPLE_SIZE = 12  # judged claims per report; keeps cost bounded
EXCERPT_LIMIT = 600  # raw_excerpt chars per source shown to the judge

_CLAIM_SENTENCE = re.compile(r"([^\n。！？]{10,200}?(?:\[\^\d+\])+[。！？]?)")
_REF_NUM = re.compile(r"\[\^(\d+)\]")
_ARXIV_VER = re.compile(r"(arxiv\.org/(?:abs|pdf)/\d{4}\.\d{4,5})v\d+")


class ClaimVerdict(BaseModel):
    claim: str
    verdict: Literal["supported", "partial", "unsupported", "meta"]
    reason: str = Field(description="One sentence citing what matched or missed")


class AlignmentResult(BaseModel):
    verdicts: list[ClaimVerdict]

    @property
    def alignment_rate(self) -> float:
        # "meta" = the sentence is not a factual assertion about the world but a
        # statement about the report itself (scope, method, what it deliberately
        # did not cite). Scoring those as unsupported punished honest hedging —
        # observed 2026-10-15 on gq-long-context, where "本报告因此未在第二节
        # 引用…" was judged as an unsupported claim. Excluded from the rate.
        scored = [v for v in self.verdicts if v.verdict != "meta"]
        if not scored:
            return 0.0
        score = sum(
            {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}[v.verdict] for v in scored
        )
        return score / len(scored)


class CoverageResult(BaseModel):
    covered: list[str] = Field(description="Expected key points present in the report")
    missed: list[str] = Field(description="Expected key points absent from the report")

    @property
    def coverage_rate(self) -> float:
        total = len(self.covered) + len(self.missed)
        return len(self.covered) / total if total else 0.0


def sample_cited_claims(report: str, limit: int = CLAIM_SAMPLE_SIZE) -> list[str]:
    """Sentences carrying a [^n] marker, deduped, evenly capped."""
    claims = [c.strip() for c in _CLAIM_SENTENCE.findall(report)]
    seen: set[str] = set()
    unique: list[str] = []
    for c in claims:  # 显式去重（原写法依赖 set.add 的副作用，可读性差）
        if c not in seen:
            seen.add(c)
            unique.append(c)

    if len(unique) <= limit:
        return unique
    step = len(unique) / limit
    return [unique[int(i * step)] for i in range(limit)]


def _norm_url(u: str) -> str:
    u = u.strip().lower().rstrip("/")
    u = _ARXIV_VER.sub(r"\1", u)
    return u.removesuffix(".git")


def resolve_claim_sources(
    report: str, evidences: list[Evidence]
) -> list[tuple[str, list[Evidence], list[str]]]:
    """Per sampled claim: (claim, matched evidence, unresolved URLs).

    Claim [^n] -> report reference list -> URL -> evidence store lookup.
    An unresolved URL means the writer cited something the researchers never
    fetched — that is itself a hallucination signal, surfaced to the judge.
    """
    ref_map = {c.index: c.url for c in extract_citations(report)}
    ev_by_url: dict[str, Evidence] = {}
    for ev in evidences:
        ev_by_url.setdefault(_norm_url(ev.url), ev)

    out: list[tuple[str, list[Evidence], list[str]]] = []
    for claim in sample_cited_claims(report):
        matched: list[Evidence] = []
        unresolved: list[str] = []
        for n in _REF_NUM.findall(claim):
            url = ref_map.get(int(n))
            if url is None:
                unresolved.append(f"(undefined footnote [^{n}])")
                continue
            hit = ev_by_url.get(_norm_url(url))
            if hit is not None:
                if hit not in matched:
                    matched.append(hit)
            else:
                unresolved.append(url)
        out.append((claim, matched, unresolved))
    return out


_ALIGNMENT_SYSTEM = """\
你是严格的审稿人。给定若干带引用的论断，以及该论断引用来源实际提供的证据要点
（含"原文摘录"，即抓取到的来源正文节选），逐条判定：supported（证据直接支撑）/
partial（部分支撑或过度引申）/ unsupported（证据中没有依据，属编造或张冠李戴）/
meta（不是关于事实的陈述，而是报告对自身范围、方法、取舍的说明）。
重要规则：
- 每条论断下方列出的来源就是其脚注对应的全部抓取内容，不要怀疑来源缺失，
  不要臆测"还有其他未抓取的脚注"——脚注映射已由程序核验，以给定证据为准。
- 论断若是对摘录中信息的合理概括、归纳或计算（如由摘录中的数据推得的比例），也算
  supported，不要求逐字对应。
- 形如「本报告未引用 X」「本节不讨论 Y」「因证据矛盾而不采用 Z」的句子属于 meta：
  那是写作取舍而非事实主张，判 meta，不要判 unsupported。
只依据给定证据判断，不用你自己的知识。"""

_COVERAGE_SYSTEM = """\
你是评测员。给定一份研究报告和一组该课题应当覆盖的关键知识点，
判断报告正文是否实际陈述了每个知识点（同义表述算覆盖，仅标题出现不算）。
把每个知识点分到 covered 或 missed，不得遗漏、不得新增。"""


async def judge_alignment(
    report: str, evidences: list[Evidence], *, role: str = "critic"
) -> AlignmentResult:
    resolved = resolve_claim_sources(report, evidences)
    if not resolved:
        return AlignmentResult(verdicts=[])

    blocks: list[str] = []
    for i, (claim, matched, unresolved) in enumerate(resolved, 1):
        lines = [f"论断 {i}：{claim}"]
        footnotes = "、".join(f"[^{n}]" for n in _REF_NUM.findall(claim))
        if matched:
            lines.append(
                f"该论断引用的脚注（{footnotes}）已全部解析到以下来源，"
                "它们都在证据库中、均为研究员实际抓取的内容："
            )
            for ev in matched:
                lines.append(f"  来源《{ev.title}》({ev.url})：")
                lines.extend(f"    - {p}" for p in ev.key_points)
                if ev.raw_excerpt:
                    excerpt = ev.raw_excerpt[:EXCERPT_LIMIT]
                    lines.append(f"    原文摘录：{excerpt}")
        if unresolved:
            lines.append(
                "注意：该论断还引用了以下 URL，但它们不在证据库中"
                "（研究员从未抓取，疑似模型编造来源）：" + "、".join(unresolved)
            )
        if not matched and not unresolved:
            lines.append("注意：无法解析该论断的引用来源。")
        blocks.append("\n".join(lines))

    runnable = structured_output(role, AlignmentResult)  # type: ignore[arg-type]
    return await invoke_with_retry(
        runnable,
        {
            "messages": [
                {"role": "system", "content": _ALIGNMENT_SYSTEM},
                {"role": "user", "content": "\n\n".join(blocks)},
            ]
        },
    )


async def judge_coverage(
    report: str, expected_key_points: list[str], *, role: str = "critic"
) -> CoverageResult:
    if not expected_key_points:
        return CoverageResult(covered=[], missed=[])
    runnable = structured_output(role, CoverageResult)  # type: ignore[arg-type]
    return await invoke_with_retry(
        runnable,
        {
            "messages": [
                {"role": "system", "content": _COVERAGE_SYSTEM},
                {"role": "user", "content":
                    f"应覆盖知识点：\n"
                    + "\n".join(f"- {p}" for p in expected_key_points)
                    + f"\n\n报告全文：\n{report}"},
            ]
        },
    )
