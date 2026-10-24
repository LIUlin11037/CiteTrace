"""Core smoke test: drive the REAL graph wiring with a mocked LLM layer.

Verifies the parts that break silently:
1. planner interrupt -> Command(resume=...) round-trip (HITL)
2. Send fan-out: N researchers run, evidences merge via add-reducer
3. Critic loop: fail -> follow-up re-research -> pass
4. Writer consumes deduped evidence and produces a report

Run: python scripts/smoke_test.py   (no API key needed)
"""

from __future__ import annotations

import asyncio
import itertools
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deep_research.agents import nodes
from deep_research.agents.graph import build_graph, initial_input, new_thread_config
from deep_research.agents.state import (
    CritiqueResult,
    Evidence,
    EvidenceBundle,
    FollowUp,
    ResearchPlan,
    SubQuestion,
)


class _FakeRunnable:
    """Stands in for structured_output(...): returns canned pydantic objects."""

    def __init__(self, factory):
        self._factory = factory

    async def ainvoke(self, input_):
        return self._factory()


class _FakeChatModel:
    """Stands in for get_chat_model(...): bind_tools -> self, ainvoke -> text.

    The writer body is deliberately citation-only (no reference list) — that is
    the contract now: writer_node appends the definitions deterministically.
    """

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        body = (
            "# 调研报告（mock）\n\n"
            "## 摘要\n结论：一切正常。[^1]\n\n"
            "## 分节正文\n本节用于冒烟测试，覆盖了 Planner 拆题、Researcher 并行取证、"
            "Critic 审校与 Writer 成文四段链路，并保留足够长度以通过 writer 的最小正文"
            "长度校验。证据不足处显式声明「现有公开资料不足以确认」，不引入证据之外的"
            "任何事实；每条事实性论断都紧跟引用标记，编号对应上游给定的证据编号，"
            "参考来源列表由 writer_node 依据正文引用确定性生成，不依赖模型复现。"
            "多份来源互相印证同一结论[^1][^2]，未出现互相矛盾之处；对于未覆盖的角度，"
            "报告在「矛盾与不确定性」一节中如实说明证据缺口，而不是用常识补全。\n\n"
            "## 矛盾与不确定性\n无未解决的矛盾；剩余不确定性已在正文中标注。\n"
        )
        return AIMessage(content=body)


def _install_fakes() -> dict[str, itertools.count]:
    counters = {"critic": itertools.count(), "researcher": itertools.count()}

    def fake_structured_output(role: str, schema):
        if schema is ResearchPlan:
            return _FakeRunnable(
                lambda: ResearchPlan(
                    topic="mock topic",
                    sub_questions=[
                        SubQuestion(id=f"q{i}", question=f"子问题{i}", rationale="r")
                        for i in range(1, 4)
                    ],
                )
            )
        if schema is EvidenceBundle:
            return _FakeRunnable(
                lambda: EvidenceBundle(
                    evidences=[
                        Evidence(
                            sub_question_id="",
                            source_type="web",
                            title=f"来源{next(counters['researcher'])}",
                            url=f"http://example.com/{next(counters['researcher'])}",
                            key_points=["事实A"],
                        )
                    ],
                    confidence=0.95,
                )
            )
        if schema is CritiqueResult:
            def _critique():
                round_ = next(counters["critic"])
                if round_ == 0:
                    return CritiqueResult(
                        passed=False,
                        follow_ups=[
                            FollowUp(id="f1", question="追问：补充工程落地数据", rationale="gap")
                        ],
                        summary="第一轮：工程维度覆盖不足",
                    )
                return CritiqueResult(passed=True, summary="通过")

            return _FakeRunnable(_critique)
        raise AssertionError(f"unexpected schema: {schema}")

    async def fake_get_mcp_tools():
        return []

    nodes.structured_output = fake_structured_output
    nodes.get_chat_model = lambda role, streaming=False: _FakeChatModel()
    nodes.get_mcp_tools = fake_get_mcp_tools
    return counters


async def main() -> None:
    _install_fakes()
    graph = build_graph(checkpointer=InMemorySaver())
    config = new_thread_config()

    # 1) run until planner interrupt (with a checkpointer, ainvoke returns
    #    a result containing __interrupt__ instead of raising)
    result = await graph.ainvoke(initial_input("Agentic RAG 最新进展（mock）"), config=config)
    interrupts = result.get("__interrupt__", [])
    assert interrupts, f"expected planner interrupt, got completed result: {list(result)}"
    plan_draft = interrupts[0].value
    assert plan_draft["type"] == "approve_plan"
    assert len(plan_draft["plan"]["sub_questions"]) == 3
    print("[ok] 1. planner interrupt fired, draft plan has 3 sub-questions")

    # 2) resume: human approves -> fan-out -> critic fail -> follow-up -> pass -> writer
    final = await graph.ainvoke(
        Command(resume={"approved": True, "edited_plan": None}), config=config
    )

    evidences = final.get("evidences", [])
    assert len(evidences) == 4, f"expected 3 + 1 follow-up evidences, got {len(evidences)}"
    print(f"[ok] 2. parallel fan-out merged: {len(evidences)} evidences via add-reducer")

    assert final.get("critique_rounds") == 2, f"critic rounds: {final.get('critique_rounds')}"
    assert final["critique"].passed is True
    print("[ok] 3. critic loop: round1 rejected -> follow-up researched -> round2 passed")

    assert final.get("report", "").startswith("# 调研报告")
    report = final["report"]
    assert "## 参考来源" in report, "writer_node 未追加参考来源章节"
    assert re.search(r"^\[\^1\]: ", report, re.MULTILINE), "参考来源缺少脚注定义行"
    print("[ok] 4. writer produced final report (含确定性生成的参考来源)")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())
