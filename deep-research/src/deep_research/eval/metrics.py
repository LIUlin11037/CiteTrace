"""Metric aggregation across a full eval run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QuestionScore:
    question_id: str
    subdomain: str
    citation_existence_rate: float
    citations_total: int
    citations_missing: list[str] = field(default_factory=list)
    # None = 没有抽到可判定的论断（N/A），与「判成全错」是两件事，聚合时排除
    alignment_rate: float | None = None
    coverage_rate: float = 0.0
    missed_key_points: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass
class EvalSummary:
    n_questions: int
    mean_citation_existence: float
    mean_alignment: float
    alignment_questions: int  # 有多少题真正参与了对齐率均值
    mean_coverage: float
    per_subdomain: dict[str, dict[str, float]]
    scores: list[QuestionScore]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate(scores: list[QuestionScore]) -> EvalSummary:
    n = len(scores)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    by_domain: dict[str, list[QuestionScore]] = {}
    for s in scores:
        by_domain.setdefault(s.subdomain, []).append(s)

    return EvalSummary(
        n_questions=n,
        mean_citation_existence=mean([s.citation_existence_rate for s in scores]),
        mean_alignment=mean([s.alignment_rate for s in scores if s.alignment_rate is not None]),
        alignment_questions=sum(1 for s in scores if s.alignment_rate is not None),
        mean_coverage=mean([s.coverage_rate for s in scores]),
        per_subdomain={
            d: {
                "n": float(len(ss)),
                "citation_existence": mean([s.citation_existence_rate for s in ss]),
                "alignment": mean(
                    [s.alignment_rate for s in ss if s.alignment_rate is not None]
                ),
                "coverage": mean([s.coverage_rate for s in ss]),
            }
            for d, ss in sorted(by_domain.items())
        },
        scores=scores,
    )


def _pct(value: float | None) -> str:
    """None 表示没有可判定样例（N/A），不要渲染成 0.0%。"""
    return "N/A" if value is None else f"{value:.1%}"


def render_markdown(summary: EvalSummary) -> str:
    lines = [
        "# Deep Research 评测报告",
        "",
        f"- 题数：{summary.n_questions}",
        f"- 引用存在率（程序核验）：**{summary.mean_citation_existence:.1%}**",
        f"- 声明-来源对齐率（LLM judge）：**{_pct(summary.mean_alignment)}**"
        f"（{summary.alignment_questions}/{summary.n_questions} 题有可判定论断；"
        f"{summary.n_questions - summary.alignment_questions} 题无样例记 N/A）",
        f"- 知识点覆盖率（LLM judge）：**{summary.mean_coverage:.1%}**",
        "",
        "## 按子领域",
        "",
        "| 子领域 | 题数 | 引用存在 | 对齐 | 覆盖 |",
        "|---|---|---|---|---|",
    ]
    for d, m in summary.per_subdomain.items():
        lines.append(
            f"| {d} | {int(m['n'])} | {m['citation_existence']:.1%} "
            f"| {_pct(m['alignment'])} | {m['coverage']:.1%} |"
        )
    lines += ["", "## 单题明细", "",
              "| 题 | 引用存在 | 对齐 | 覆盖 | 未对齐论断数 | 缺失知识点 |",
              "|---|---|---|---|---|---|"]
    for s in summary.scores:
        missed = "、".join(s.missed_key_points) or "-"
        lines.append(
            f"| {s.question_id} | {s.citation_existence_rate:.1%} "
            f"| {_pct(s.alignment_rate)} | {s.coverage_rate:.1%} "
            f"| {len(s.unsupported_claims)} | {missed} |"
        )
    return "\n".join(lines)
