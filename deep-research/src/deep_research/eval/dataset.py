"""Golden dataset loading.

A golden question is a research topic plus the key points a good answer MUST
cover. Key points are written to be checkable by an LLM judge: concrete,
falsifiable statements, not vague themes.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenQuestion(BaseModel):
    id: str
    topic: str
    subdomain: str = Field(description="e.g. agent / llm / rag — used for per-domain breakdowns")
    expected_key_points: list[str] = Field(min_length=2)


def load_dataset(path: str | Path | None = None) -> list[GoldenQuestion]:
    p = Path(path) if path else Path(__file__).parent / "golden" / "tech_ai.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [GoldenQuestion(**q) for q in raw["questions"]]
