"""Offline re-judge of claim-source alignment using persisted artifacts.

Loads reports/eval/reports/<qid>.md + <qid>.evidences.json and re-runs ONLY
judge_alignment — no research rerun, a handful of judge calls per question.

Usage: uv run python scripts/rejudge_alignment.py gq-mcp gq-moe
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deep_research.agents.state import Evidence
from deep_research.core.compat import use_compatible_event_loop
from deep_research.eval.judge import judge_alignment

REPORTS_DIR = Path("reports/eval/reports")


async def rejudge(qid: str) -> None:
    report = (REPORTS_DIR / f"{qid}.md").read_text(encoding="utf-8")
    raw = json.loads((REPORTS_DIR / f"{qid}.evidences.json").read_text(encoding="utf-8"))
    evidences = [Evidence(**e) for e in raw]
    result = await judge_alignment(report, evidences)
    print(f"\n=== {qid}: align={result.alignment_rate:.0%} ({len(result.verdicts)} claims) ===")
    for v in result.verdicts:
        mark = {"supported": "+", "partial": "~", "unsupported": "X"}[v.verdict]
        print(f"  [{mark}] {v.claim[:60]}...")
        print(f"      {v.reason}")


async def main(qids: list[str]) -> None:
    for qid in qids:
        await rejudge(qid)


if __name__ == "__main__":
    use_compatible_event_loop()
    qids = sys.argv[1:] or ["gq-mcp", "gq-moe"]
    asyncio.run(main(qids))
