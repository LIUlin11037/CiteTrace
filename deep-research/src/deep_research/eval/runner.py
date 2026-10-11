"""Eval runner: research every golden question, then score the outputs.

Pipeline per question:
    graph (auto-approved plan) -> report + evidences
    -> programmatic citation verification
    -> LLM-judge alignment + coverage
    -> aggregate -> JSON + Markdown under reports/eval/

`run_research` and the two judges are injectable so the whole pipeline can be
smoke-tested without an LLM key or the network.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deep_research.agents.graph import build_graph, initial_input, new_thread_config
from deep_research.agents.state import Evidence
from deep_research.eval.citations import CitationReport, verify_citations
from deep_research.eval.dataset import GoldenQuestion, load_dataset
from deep_research.eval.judge import (
    AlignmentResult,
    CoverageResult,
    judge_alignment,
    judge_coverage,
)
from deep_research.eval.metrics import EvalSummary, QuestionScore, aggregate, render_markdown

ResearchFn = Callable[[str], Awaitable[tuple[str, list[Evidence]]]]
AlignmentFn = Callable[[str, list[Evidence]], Awaitable[AlignmentResult]]
CoverageFn = Callable[[str, list[str]], Awaitable[CoverageResult]]
CitationsFn = Callable[[str], Awaitable[CitationReport]]


async def default_run_research(topic: str) -> tuple[str, list[Evidence]]:
    """Drive the real graph with an auto-approved plan (eval is unattended)."""
    graph = build_graph(checkpointer=InMemorySaver())
    config = new_thread_config()
    result = await graph.ainvoke(initial_input(topic), config=config)
    if result.get("__interrupt__"):
        result = await graph.ainvoke(
            Command(resume={"approved": True, "edited_plan": None}), config=config
        )
    return result.get("report", ""), result.get("evidences", [])


async def evaluate_question(
    q: GoldenQuestion,
    *,
    run_research: ResearchFn,
    check_citations: CitationsFn,
    align: AlignmentFn,
    cover: CoverageFn,
    reports_dir: Path | None = None,
) -> QuestionScore:
    report, evidences = await run_research(q.topic)

    if reports_dir is not None:
        # Persist report AND evidence store: enables offline re-judging and
        # makes eval debugging possible without re-running the research.
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / f"{q.id}.md").write_text(report, encoding="utf-8")
        (reports_dir / f"{q.id}.evidences.json").write_text(
            json.dumps([e.model_dump() for e in evidences], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    citation_report, alignment, coverage = await asyncio.gather(
        check_citations(report),
        align(report, evidences),
        cover(report, q.expected_key_points),
    )
    return QuestionScore(
        question_id=q.id,
        subdomain=q.subdomain,
        citation_existence_rate=citation_report.existence_rate,
        citations_total=citation_report.total,
        citations_missing=[
            d["target"] for d in citation_report.details if d["verdict"] == "missing"
        ],
        alignment_rate=alignment.alignment_rate,
        coverage_rate=coverage.coverage_rate,
        missed_key_points=coverage.missed,
        unsupported_claims=[
            v.claim for v in alignment.verdicts if v.verdict == "unsupported"
        ],
    )


async def run_eval(
    *,
    dataset_path: str | None = None,
    out_dir: Path = Path("reports/eval"),
    run_research: ResearchFn = default_run_research,
    check_citations: CitationsFn = verify_citations,
    align: AlignmentFn = judge_alignment,
    cover: CoverageFn = judge_coverage,
    max_questions: int | None = None,
) -> EvalSummary:
    questions = load_dataset(dataset_path)
    if max_questions:
        questions = questions[:max_questions]

    scores: list[QuestionScore] = []
    reports_dir = out_dir / "reports"
    for q in questions:
        score = await evaluate_question(
            q,
            run_research=run_research,
            check_citations=check_citations,
            align=align,
            cover=cover,
            reports_dir=reports_dir,
        )
        scores.append(score)
        print(
            f"[eval] {q.id}: cite={score.citation_existence_rate:.0%} "
            f"align={score.alignment_rate:.0%} cover={score.coverage_rate:.0%}"
        )

    summary = aggregate(scores)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval-{stamp}.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"eval-{stamp}.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"[eval] summary -> {out_dir}/eval-{stamp}.md")
    return summary


if __name__ == "__main__":
    asyncio.run(run_eval())
