"""Verify the "no data is not zero" rule in metric aggregation (offline).

Regression for 2026-10-24: gq-distillation reported align=0% because the writer
used [n] instead of [^n], so the judge sampled zero claims. The report's real
alignment was 86%. A single such report dragged a 3-question mean from 90% to
61%. Rule: no judged claims => N/A, excluded from means, never 0.0.

Run: python scripts/verify_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deep_research.eval.judge import AlignmentResult, ClaimVerdict  # noqa: E402
from deep_research.eval.metrics import (  # noqa: E402
    QuestionScore,
    aggregate,
    render_markdown,
)


def _score(qid: str, align: float | None, subdomain: str = "llm") -> QuestionScore:
    return QuestionScore(
        question_id=qid,
        subdomain=subdomain,
        citation_existence_rate=1.0,
        citations_total=10,
        alignment_rate=align,
        coverage_rate=1.0,
    )


def test_empty_verdicts_is_none() -> None:
    assert AlignmentResult(verdicts=[]).alignment_rate is None
    print("[ok] 1. 没有论断可判 → alignment_rate 为 None（不是 0.0）")


def test_meta_only_is_none() -> None:
    res = AlignmentResult(
        verdicts=[ClaimVerdict(claim="本报告未引用 X", verdict="meta", reason="方法性说明")]
    )
    assert res.alignment_rate is None, res.alignment_rate
    print("[ok] 2. 只有 meta 论断 → 同样记为 None（不惩罚诚实声明）")


def test_aggregate_excludes_none() -> None:
    summary = aggregate([_score("a", 0.9), _score("b", None)])
    assert abs(summary.mean_alignment - 0.9) < 1e-9, summary.mean_alignment
    assert summary.alignment_questions == 1 and summary.n_questions == 2
    print("[ok] 3. 聚合只对有判定的题求均值：90%（不是 45%），并标注 1/2")


def test_render_shows_na() -> None:
    md = render_markdown(aggregate([_score("a", 0.9), _score("b", None)]))
    assert "1/2 题有可判定论断" in md, md
    assert "| b | 100.0% | N/A |" in md, md
    print("[ok] 4. Markdown 渲染为 N/A，并注明排除题数")


def test_all_none_is_zero_not_crash() -> None:
    summary = aggregate([_score("a", None), _score("b", None)])
    assert summary.mean_alignment == 0.0 and summary.alignment_questions == 0
    print("[ok] 5. 全部无判定时均值为 0 但不崩，且 alignment_questions=0 一目了然")


def main() -> None:
    test_empty_verdicts_is_none()
    test_meta_only_is_none()
    test_aggregate_excludes_none()
    test_render_shows_na()
    test_all_none_is_zero_not_crash()
    print("\nMETRICS N/A TEST PASSED")


if __name__ == "__main__":
    main()
