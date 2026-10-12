"""Graph node implementations: Planner / Researcher / Critic / Writer.

Concurrency rules honored here (violating them corrupts merged state):
- Researcher branches run in parallel (Send API) and may ONLY return
  {"evidences": [...]} — the add-reducer merges them. No other channel writes.
- Critic and Writer are barrier nodes: they run after all parallel branches
  complete and own their output channels exclusively.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt

from deep_research.agents import prompts
from deep_research.agents.state import (
    CritiqueResult,
    Evidence,
    EvidenceBundle,
    ResearcherInput,
    ResearchPlan,
    ResearchState,
    SubQuestion,
    dedup_evidences,
)
from deep_research.core.config import get_settings
from deep_research.core.llm import get_chat_model, invoke_with_retry, structured_output
from deep_research.tools.mcp_client import get_mcp_tools

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


async def planner_node(state: ResearchState) -> dict[str, Any]:
    """Draft a research plan, then pause for human approval (HITL).

    The interrupt payload is what CLI/API surfaces to the user. Resume with
    Command(resume={"approved": bool, "edited_plan": dict | None}).
    """
    settings = get_settings()
    runnable = structured_output("planner", ResearchPlan)

    recall_block = ""
    recalled: list[str] = []
    if settings.enable_long_term_memory:
        try:
            from deep_research.core.memory import get_memory_store

            store = await get_memory_store()
            hits = await store.search(state["topic"])
            if hits:
                recalled = [h.content for h in hits]
                recall_block = (
                    "\n\n相关历史研究（可复用其来源与结论，避免重复劳动；"
                    "内容可能过时，关键事实仍需重新核验）：\n"
                    + "\n".join(f"- [{h.kind}] {h.content[:500]}" for h in hits)
                )
        except Exception:  # memory is an optimization, never a blocker
            recalled = []

    draft: ResearchPlan = await invoke_with_retry(
        runnable,
        {
            "messages": [
                SystemMessage(
                    prompts.render(
                        prompts.PLANNER_SYSTEM,
                        max_sub_questions=settings.max_sub_questions,
                    )
                ),
                HumanMessage(f"研究课题：{state['topic']}{recall_block}"),
            ]
        },
    )

    decision: dict[str, Any] = interrupt(
        {
            "type": "approve_plan",
            "topic": state["topic"],
            "plan": draft.model_dump(),
            "hint": "批准计划，或提交 edited_plan 覆盖后再批准",
        }
    )

    plan = draft
    if decision.get("edited_plan"):
        plan = ResearchPlan(**decision["edited_plan"])
    approved = bool(decision.get("approved", True))

    return {"plan": plan, "plan_approved": approved, "recalled_memories": recalled}


def route_after_planner(state: ResearchState) -> list[Any]:
    """Fan out one Researcher branch per approved sub-question (Send API)."""
    from langgraph.types import Send

    plan = state["plan"]
    if plan is None or not state.get("plan_approved", False):
        # Rejected plan ends the run; the entry layer decides whether to re-plan.
        return []
    return [
        Send("researcher", ResearcherInput(topic=state["topic"], sub_question=sq))
        for sq in plan.sub_questions
    ]


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------

_EXTRACTION_INSTRUCTION = (
    "工具调研结束。现在把收集到的材料结构化为 EvidenceBundle："
    "每条证据对应一个真实访问过的来源（URL 必须来自工具返回，禁止编造）；"
    "key_points 只记录该来源直接支持的原子事实；raw_excerpt 保留原文关键句；"
    "最后自评 confidence 并写下 gap_notes。"
)


async def researcher_node(input_: ResearcherInput) -> dict[str, Any]:
    """ReAct loop over MCP tools for ONE sub-question, then structured extraction.

    Weak/strong collaboration: extraction first runs on the cheap researcher
    model; low self-confidence triggers one escalation re-extraction with the
    critic's (stronger) model over the same tool transcript — no extra tool
    budget spent.
    """
    settings = get_settings()
    sq: SubQuestion = input_["sub_question"]
    tools = await get_mcp_tools()
    tools_by_name = {t.name: t for t in tools}

    model = get_chat_model("researcher").bind_tools(tools)
    transcript: list[BaseMessage] = [
        SystemMessage(
            prompts.render(
                prompts.RESEARCHER_SYSTEM, max_steps=settings.max_researcher_steps
            )
        ),
        HumanMessage(f"课题背景：{input_['topic']}\n你负责的子问题（{sq.id}）：{sq.question}"),
    ]

    # ---- ReAct loop (explicit, budget-capped) ----
    for _ in range(settings.max_researcher_steps):
        ai: AIMessage = await model.ainvoke(transcript)
        transcript.append(ai)
        if not ai.tool_calls:
            break
        for call in ai.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                result = f"error: unknown tool {call['name']}"
            else:
                try:
                    result = await tool.ainvoke(call["args"])
                except Exception as exc:  # a dead source must not kill the branch
                    result = f"error: {type(exc).__name__}: {exc}"
            transcript.append(
                ToolMessage(
                    content=str(result)[: settings.evidence_content_limit],
                    tool_call_id=call["id"],
                )
            )

    # ---- structured extraction (with strong-model escalation) ----
    bundle = await _extract_evidences("researcher", transcript, sq)
    escalated = False
    if (
        bundle.confidence < settings.escalation_confidence_threshold
        and settings.researcher_model != settings.critic_model
    ):
        bundle = await _extract_evidences("critic", transcript, sq)
        escalated = True

    for ev in bundle.evidences:
        ev.sub_question_id = sq.id

    return {
        "evidences": bundle.evidences,
        "messages": [
            AIMessage(
                content=(
                    f"[researcher {sq.id}] {len(bundle.evidences)} evidences, "
                    f"confidence={bundle.confidence:.2f}"
                    f"{' (escalated)' if escalated else ''}. gaps: {bundle.gap_notes}"
                )
            )
        ],
    }


async def _extract_evidences(
    role: str, transcript: list[BaseMessage], sq: SubQuestion
) -> EvidenceBundle:
    runnable = structured_output(role, EvidenceBundle)  # type: ignore[arg-type]
    return await invoke_with_retry(
        runnable, {"messages": [*transcript, HumanMessage(_EXTRACTION_INSTRUCTION)]}
    )


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


async def critic_node(state: ResearchState) -> dict[str, Any]:
    """Audit merged evidence; decide pass / re-research with follow-ups."""
    settings = get_settings()
    evidences = dedup_evidences(state.get("evidences", []))
    rounds = state.get("critique_rounds", 0) + 1

    evidence_digest = "\n".join(
        f"- [{ev.sub_question_id}] ({ev.source_type}) {ev.title} | {ev.url}\n"
        f"  key_points: {'; '.join(ev.key_points)}\n"
        f"  excerpt: {ev.raw_excerpt[:300]}"
        for ev in evidences
    )
    plan_digest = "\n".join(
        f"- {sq.id}: {sq.question}" for sq in (state["plan"].sub_questions if state["plan"] else [])
    )

    runnable = structured_output("critic", CritiqueResult)
    critique: CritiqueResult = await invoke_with_retry(
        runnable,
        {
            "messages": [
                SystemMessage(prompts.render(prompts.CRITIC_SYSTEM)),
                HumanMessage(
                    f"课题：{state['topic']}\n\n计划子问题：\n{plan_digest}\n\n"
                    f"已收集证据（{len(evidences)} 条）：\n{evidence_digest}\n\n"
                    f"当前为第 {rounds}/{settings.max_critique_rounds} 轮审校。"
                ),
            ]
        },
    )

    # Hard stop: never loop beyond budget regardless of what the LLM wants.
    if rounds >= settings.max_critique_rounds:
        critique.passed = True
        critique.follow_ups = []

    return {"critique": critique, "critique_rounds": rounds}


def route_after_critic(state: ResearchState) -> Any:
    """Pass -> writer; fail -> fan out follow-up questions to researchers."""
    from langgraph.types import Send

    critique = state["critique"]
    if critique is None or critique.passed or not critique.follow_ups:
        return "writer"
    return [
        Send(
            "researcher",
            ResearcherInput(
                topic=state["topic"],
                sub_question=SubQuestion(
                    id=fu.id, question=fu.question, rationale=fu.rationale
                ),
            ),
        )
        for fu in critique.follow_ups
    ]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


async def writer_node(state: ResearchState) -> dict[str, Any]:
    """Compose the final cited report from deduped evidence only."""
    evidences = dedup_evidences(state.get("evidences", []))

    source_list = "\n".join(
        f"[{i}] {ev.title} — {ev.url} ({ev.source_type}"
        + (f", {ev.published_at}" if ev.published_at else "")
        + ")"
        for i, ev in enumerate(evidences, 1)
    )
    evidence_body = "\n\n".join(
        f"来源 [{i}] {ev.title}:\n" + "\n".join(f"  - {p}" for p in ev.key_points)
        for i, ev in enumerate(evidences, 1)
    )
    issues_note = ""
    if state.get("critique") and state["critique"].issues:
        unresolved = [i for i in state["critique"].issues if i.severity == "blocking"]
        if unresolved:
            issues_note = "\n\n未解决的存疑项（须在报告中显式声明不确定性）：\n" + "\n".join(
                f"- {i.claim}: {i.problem}" for i in unresolved
            )

    model = get_chat_model("writer", streaming=True)
    result: AIMessage = await model.ainvoke(
        [
            SystemMessage(prompts.render(prompts.WRITER_SYSTEM)),
            HumanMessage(
                f"课题：{state['topic']}\n\n"
                f"可用证据（编号即引用编号，共 {len(evidences)} 条）：\n{evidence_body}\n\n"
                f"参考来源列表（照此输出，编号一致）：\n{source_list}"
                f"{issues_note}"
            ),
        ]
    )

    report = result.content

    if get_settings().enable_long_term_memory:
        await _store_memories(state["topic"], report, evidences)

    return {"report": report, "messages": [result]}


async def _store_memories(topic: str, report: str, evidences: list[Evidence]) -> None:
    """Persist what future research runs can reuse. Failures are non-fatal."""
    try:
        from deep_research.core.memory import get_memory_store

        store = await get_memory_store()
        summary = report[:1200]  # 摘要段在报告开头，截断即可，不值得再花一次 LLM 调用
        items: list[tuple[str, str, dict]] = [
            ("report_summary", f"课题：{topic}\n{summary}", {"topic": topic})
        ]
        by_sq: dict[str, list[Evidence]] = {}
        for ev in evidences:
            by_sq.setdefault(ev.sub_question_id, []).append(ev)
        for sq_id, evs in by_sq.items():
            digest = f"课题：{topic} | 子问题 {sq_id} 已检索来源：\n" + "\n".join(
                f"- {e.title} | {e.url}（{len(e.key_points)} 要点）" for e in evs[:8]
            )
            items.append(("evidence_digest", digest, {"topic": topic, "sub_question": sq_id}))
        await store.add_many(items)
    except Exception:  # 记忆写入失败不影响报告交付
        pass
