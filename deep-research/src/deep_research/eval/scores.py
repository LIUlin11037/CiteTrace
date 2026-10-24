"""Push eval metrics onto the Langfuse trace of the matching run.

Langfuse holds the traces; the quality verdicts live in this repo. Pushing the
three metrics as trace scores closes the loop — in the UI you can sort traces by
alignment and click straight into the run that produced a low score.

We know the session id (= LangGraph thread_id) but not the trace id, so the trace
is looked up by session through the observations API. Pushing is best-effort:
observability must never break a benchmark run.
"""

from __future__ import annotations

import asyncio
import logging

from deep_research.core.config import get_settings
from deep_research.eval.metrics import QuestionScore

logger = logging.getLogger(__name__)


def _resolve_trace_id(session_id: str) -> str | None:
    from langfuse import get_client

    client = get_client()
    page = client.api.observations.get_many(
        session_id=session_id, limit=1, fields="basic"
    )
    for obs in getattr(page, "data", []) or []:
        trace_id = getattr(obs, "trace_id", None)
        if trace_id:
            return str(trace_id)
    return None


async def push_question_scores(
    session_id: str, score: QuestionScore, *, extra_comment: str = ""
) -> int:
    """Write cite / align / cover as numeric trace scores. Returns how many landed."""
    settings = get_settings()
    if not settings.tracing_active:
        return 0

    from langfuse import get_client

    client = get_client()
    try:
        trace_id = await asyncio.to_thread(_resolve_trace_id, session_id)
    except Exception as exc:  # noqa: BLE001 - observability is best-effort
        logger.warning("Langfuse: 反查 trace 失败（session=%s）: %s", session_id, exc)
        return 0
    if not trace_id:
        logger.warning("Langfuse: session %s 找不到对应 trace，跳过分数推送", session_id)
        return 0

    metrics = [
        (
            "citation_existence",
            score.citation_existence_rate,
            f"缺失 {len(score.citations_missing)} / 共 {score.citations_total} 条引用",
        ),
        (
            "alignment",
            score.alignment_rate,
            f"未对齐论断 {len(score.unsupported_claims)} 条",
        ) if score.alignment_rate is not None else None,
        (
            "coverage",
            score.coverage_rate,
            f"缺失知识点 {len(score.missed_key_points)} 个",
        ),
    ]
    pushed = 0
    for metric in metrics:
        if metric is None:  # alignment 无样例时记 N/A，不往云端送假 0
            logger.info("Langfuse: %s 没有可判定论断，跳过 alignment 分数", score.question_id)
            continue
        name, value, comment = metric
        try:
            client.create_score(
                name=name,
                value=round(float(value), 4),
                trace_id=trace_id,
                data_type="NUMERIC",
                comment=f"{comment}{extra_comment}",
            )
            pushed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse: 推送 %s 分数失败: %s", name, exc)
    if pushed:
        logger.info(
            "Langfuse: %s 已推送 %d 个分数（trace=%s）", score.question_id, pushed, trace_id
        )
    return pushed
