"""Graph assembly and runtime helpers.

Wiring:
    START -> planner --(Send x N)--> researcher --> critic --+--> writer -> END
                        ^                                    |
                        +--------(Send follow-ups)-----------+

HITL: planner_node calls interrupt(); the caller resumes with
Command(resume={"approved": ..., "edited_plan": ...}).

Checkpointing: caller injects a checkpointer (SqliteSaver in dev,
PostgresSaver from M3). Without a checkpointer, interrupts and resume
do NOT work — build_graph() refuses to compile without one on purpose.
"""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from deep_research.agents.nodes import (
    critic_node,
    planner_node,
    researcher_node,
    route_after_critic,
    route_after_planner,
    writer_node,
)
from deep_research.agents.state import ResearchState


def build_graph(checkpointer: BaseCheckpointSaver[Any]) -> Any:
    """Compile the research graph. A checkpointer is mandatory."""
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    # researcher 的输入是 Send 载荷（ResearcherInput），不是整份 state；
    # LangGraph 的节点签名标注表达不了这一点，运行时由 Send 决定。
    builder.add_node("researcher", researcher_node)  # type: ignore[arg-type]
    builder.add_node("critic", critic_node)
    builder.add_node("writer", writer_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_after_planner, ["researcher", END])
    builder.add_edge("researcher", "critic")
    builder.add_conditional_edges("critic", route_after_critic, ["researcher", "writer"])
    builder.add_edge("writer", END)

    return builder.compile(checkpointer=checkpointer)


def new_thread_config() -> dict[str, Any]:
    """One research run = one thread_id; resume reuses the same id."""
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


def initial_input(topic: str) -> dict[str, Any]:
    return {"topic": topic, "evidences": [], "critique_rounds": 0}
