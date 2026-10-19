"""State schema for the research graph.

Channel design (the part that breaks if you get it wrong):
- `evidences` uses an add-reducer: N parallel Researcher branches (Send API)
  each return a partial list, LangGraph merges them. NEVER overwrite this
  channel — always return only the delta from each node.
- `messages` uses LangGraph's add_messages reducer for conversational traces.
- Everything else is last-write-wins, owned by exactly one node.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Required, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

SourceType = Literal["arxiv", "github", "web"]


class SubQuestion(BaseModel):
    id: str = Field(description="Short stable id, e.g. q1")
    question: str = Field(description="One focused, answerable research question")
    rationale: str = Field(description="Why this question matters for the topic")


class ResearchPlan(BaseModel):
    topic: str
    sub_questions: list[SubQuestion] = Field(min_length=1, max_length=8)


class Evidence(BaseModel):
    """One verified piece of source material collected by a Researcher."""

    sub_question_id: str
    source_type: SourceType
    title: str
    url: str
    published_at: str | None = None
    key_points: list[str] = Field(
        description="Atomic factual claims taken from THIS source, no synthesis"
    )
    raw_excerpt: str = Field(default="", description="Short verbatim excerpt, truncated")


class EvidenceBundle(BaseModel):
    """Structured extraction result of one Researcher run."""

    evidences: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Self-assessed confidence that the collected evidence "
                    "adequately answers the sub-question",
    )
    gap_notes: str = Field(
        default="", description="What could NOT be found, and why it might be missing"
    )


class CritiqueIssue(BaseModel):
    claim: str
    problem: str
    severity: Literal["blocking", "minor"]


class FollowUp(BaseModel):
    """A gap the Critic wants re-researched, phrased like a SubQuestion."""

    id: str
    question: str
    rationale: str


class CritiqueResult(BaseModel):
    passed: bool
    issues: list[CritiqueIssue] = Field(default_factory=list)
    follow_ups: list[FollowUp] = Field(default_factory=list)
    summary: str = ""


class ResearchState(TypedDict, total=False):
    # topic 由入口 initial_input 必定提供；其余通道按阶段产出，故整体 total=False。
    topic: Required[str]
    plan: ResearchPlan | None
    plan_approved: bool

    # merged across parallel researchers via operator.add
    evidences: Annotated[list[Evidence], operator.add]

    critique: CritiqueResult | None
    critique_rounds: int

    report: str
    messages: Annotated[list[Any], add_messages]

    # long-term memory recall injected before planning (M3)
    recalled_memories: list[str]


class ResearcherInput(TypedDict):
    """Payload delivered to one Researcher branch via Send."""

    topic: str
    sub_question: SubQuestion


def dedup_evidences(evidences: list[Evidence]) -> list[Evidence]:
    """Dedup by canonical URL, keeping the first (richest) occurrence."""
    seen: set[str] = set()
    out: list[Evidence] = []
    for ev in evidences:
        key = ev.url.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out
