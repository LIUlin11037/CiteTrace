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
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

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

# Which golden question is currently being researched. Set around run_research
# so traces can be tagged with the question id — the research callable itself
# only knows the topic string.
_LABEL: ContextVar[str] = ContextVar("eval_label", default="")


async def default_run_research(topic: str) -> tuple[str, list[Evidence]]:
    """Drive the real graph with an auto-approved plan (eval is unattended)."""
    from deep_research.core.persistence import open_checkpointer
    from deep_research.core.tracing import apply_tracing
    from deep_research.core.usage import attach_usage

    async with open_checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        config = new_thread_config()
        tags = ["eval"] + ([_LABEL.get()] if _LABEL.get() else [])
        apply_tracing(config, tags=tags)
        attach_usage(config)
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
    token = _LABEL.set(q.id)
    try:
        report, evidences = await run_research(q.topic)
    finally:
        _LABEL.reset(token)

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
    question_ids: list[str] | None = None,
    max_concurrency: int = 3,
) -> EvalSummary:
    questions = load_dataset(dataset_path)
    if question_ids:
        wanted = set(question_ids)
        questions = [q for q in questions if q.id in wanted]
        missing = wanted - {q.id for q in questions}
        if missing:
            raise ValueError(f"unknown golden question ids: {sorted(missing)}")
    elif max_questions:
        questions = questions[:max_questions]

    class _Failed(Exception):
        pass

    reports_dir = out_dir / "reports"
    sem = asyncio.Semaphore(max_concurrency)

    async def one(q: GoldenQuestion, *, attempt: int = 1) -> QuestionScore:
        async with sem:
            try:
                score = await evaluate_question(
                    q,
                    run_research=run_research,
                    check_citations=check_citations,
                    align=align,
                    cover=cover,
                    reports_dir=reports_dir,
                )
            except Exception as exc:
                print(f"[eval] {q.id}: FAILED on attempt {attempt} ({type(exc).__name__}: {exc})")
                raise _Failed from exc
            print(
                f"[eval] {q.id}: cite={score.citation_existence_rate:.0%} "
                f"align={score.alignment_rate:.0%} cover={score.coverage_rate:.0%}"
            )
            return score

    async def round_of(batch: list[GoldenQuestion], attempt: int) -> tuple[list[QuestionScore], list[GoldenQuestion]]:
        outcomes = await asyncio.gather(
            *(one(q, attempt=attempt) for q in batch), return_exceptions=True
        )
        done: list[QuestionScore] = []
        retry: list[GoldenQuestion] = []
        for q, outcome in zip(batch, outcomes, strict=True):
            if isinstance(outcome, QuestionScore):
                done.append(outcome)
            else:
                retry.append(q)
        return done, retry

    # Pass 1 in parallel; pass 2 retries only the failures, sequentially. A
    # transient network blip must not silently shrink the benchmark sample
    # (observed 2026-10-14: 17 of 26 questions lost to one connection outage).
    final_scores, failed = await round_of(questions, attempt=1)
    if failed:
        print(f"[eval] 第一轮有 {len(failed)} 题失败，串行重试一次：{[q.id for q in failed]}")
        more, still_failed = await round_of(failed, attempt=2)
        final_scores.extend(more)
        for q in still_failed:
            print(f"[eval] {q.id}: 重试后仍失败，本轮该题缺测")
    if not final_scores:
        raise RuntimeError("all questions failed — aborting instead of writing an empty summary")

    summary = aggregate(final_scores)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    from deep_research.core.usage import usage_report

    cost_line = usage_report()
    print(f"[eval] usage: {cost_line}")

    (out_dir / f"eval-{stamp}.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"eval-{stamp}.md").write_text(
        render_markdown(summary)
        + f"\n\n## 本轮用量与成本\n\n- {cost_line}\n"
        + f"- 单题均值：约 ¥{_per_question_cost(len(final_scores)):.2f}\n",
        encoding="utf-8",
    )
    print(f"[eval] summary -> {out_dir}/eval-{stamp}.md")

    from deep_research.core.tracing import flush_tracing

    flush_tracing()  # a benchmark run is long; don't lose the tail of the traces
    return summary


def _per_question_cost(n_questions: int) -> float:
    from deep_research.core.usage import usage_cost

    if not n_questions:
        return 0.0
    return usage_cost() / n_questions


if __name__ == "__main__":
    from deep_research.core.compat import use_compatible_event_loop

    use_compatible_event_loop()
    asyncio.run(run_eval())
