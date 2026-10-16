"""Re-score alignment for all golden questions offline and refresh the aggregate.

Use after fixing the alignment judge: reports and evidence stores are already on
disk, so only judge calls are spent — no research re-runs. Coverage and citation
rates are taken from the newest run per question (see SOURCES below).

Run: python scripts/refresh_alignment.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deep_research.agents.state import Evidence  # noqa: E402
from deep_research.core.compat import use_compatible_event_loop  # noqa: E402
from deep_research.eval.judge import judge_alignment  # noqa: E402
from deep_research.eval.metrics import (  # noqa: E402
    QuestionScore,
    aggregate,
    render_markdown,
)

REPORTS_DIR = Path("reports/eval/reports")
EVAL_DIR = Path("reports/eval")

# Question ids whose cite/cover were re-measured after the original 26-question
# batch (kept here so the refreshed aggregate uses each question's newest run).
OVERRIDES = {
    # qid: (citations_total, citations_missing, coverage_rate, missed_points)
}


def _latest_scores() -> dict[str, QuestionScore]:
    """Newest QuestionScore per question id across all eval summaries."""
    merged: dict[str, QuestionScore] = {}
    for f in sorted(EVAL_DIR.glob("eval-*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for s in data.get("scores", []):
            merged[s["question_id"]] = QuestionScore(**s)  # later files win
    return merged


async def main() -> None:
    base = _latest_scores()
    golden = json.loads(
        Path("src/deep_research/eval/golden/tech_ai.json").read_text(encoding="utf-8")
    )["questions"]
    order = [q["id"] for q in golden]

    refreshed: list[QuestionScore] = []
    for qid in order:
        score = base.get(qid)
        if score is None:
            print(f"  {qid}: 缺少基础分数，跳过")
            continue
        report = (REPORTS_DIR / f"{qid}.md").read_text(encoding="utf-8")
        raw = json.loads((REPORTS_DIR / f"{qid}.evidences.json").read_text(encoding="utf-8"))
        evidences = [Evidence(**e) for e in raw]
        result = await judge_alignment(report, evidences)
        refreshed.append(
            QuestionScore(
                question_id=score.question_id,
                subdomain=score.subdomain,
                citation_existence_rate=score.citation_existence_rate,
                citations_total=score.citations_total,
                citations_missing=score.citations_missing,
                alignment_rate=result.alignment_rate,
                coverage_rate=score.coverage_rate,
                missed_key_points=score.missed_key_points,
                unsupported_claims=[
                    v.claim for v in result.verdicts if v.verdict == "unsupported"
                ],
            )
        )
        print(f"  {qid:32} align {score.alignment_rate:.0%} -> {result.alignment_rate:.0%}")

    summary = aggregate(refreshed)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (EVAL_DIR / f"eval-{stamp}-merged-26q.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (EVAL_DIR / f"eval-{stamp}-merged-26q.md").write_text(
        render_markdown(summary)
        + "\n\n## 说明\n\n"
        + "- 引用存在率与覆盖率取每题**最近一次运行**的实测值\n"
        + "- 对齐率用**修正后的 judge 离线复判**（识别「方法性说明」为 meta 并排除）\n",
        encoding="utf-8",
    )
    print(
        f"\n26 题口径: cite {summary.mean_citation_existence:.1%} | "
        f"align {summary.mean_alignment:.1%} | cover {summary.mean_coverage:.1%}"
    )
    print(f"summary -> {EVAL_DIR}/eval-{stamp}-merged-26q.md")


if __name__ == "__main__":
    use_compatible_event_loop()
    asyncio.run(main())
