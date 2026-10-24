"""M3 integration verification: PostgreSQL checkpointer + pgvector long-term memory.

Prerequisites:
  - docker compose start postgres  (pgvector/pgvector:pg16 on localhost:5432)
  - .env: DR_CHECKPOINT_BACKEND=postgres, DR_ENABLE_LONG_TERM_MEMORY=true

Steps:
  1. MemoryStore roundtrip WITHOUT any LLM call (table/HNSW/embedding/search).
  2. Real research run A through the Postgres checkpointer (interrupt -> resume),
     then SQL-assert checkpoint rows and memory rows (report_summary + digests).
  3. Real research run B on a closely related topic; assert the planner recalled
     run A's memories (recalled_memories non-empty in final state).

Run: python scripts/verify_m3.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import LiteralString

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langgraph.types import Command

from deep_research.agents.graph import build_graph, initial_input, new_thread_config
from deep_research.core.compat import use_compatible_event_loop
from deep_research.core.config import get_settings
from deep_research.core.memory import get_memory_store
from deep_research.core.persistence import open_checkpointer

TOPIC_A = "vLLM 的 PagedAttention 与连续批处理机制调研"
TOPIC_B = "vLLM 推理引擎的 KV cache 管理与吞吐优化手段"


async def step1_memory_roundtrip() -> None:
    print("[1/3] MemoryStore roundtrip (no LLM) ...")
    store = await get_memory_store()
    await store.add(
        "selftest", "M3 自检测试：Transformer 注意力机制的 KV 缓存复用", {"tag": "selftest"}
    )
    hits = await store.search("KV cache 缓存与注意力机制")
    assert any(h.kind == "selftest" for h in hits), (
        f"selftest memory not recalled, hits={[(h.kind, round(h.score, 3)) for h in hits]}"
    )
    top = hits[0]
    print(f"      recalled kind={top.kind} score={top.score:.3f} content={top.content[:30]}...")

    # cleanup selftest rows so they never pollute real recall
    import psycopg

    settings = get_settings()
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        cur = await conn.execute("DELETE FROM memories WHERE kind = 'selftest'")
        print(f"      cleaned {cur.rowcount} selftest row(s)")
    print("      OK: setup/add/search/delete all work\n")


async def _run_research(topic: str) -> dict:
    """Drive the real graph (auto-approved plan) and return the final state."""
    async with open_checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        config = new_thread_config()
        result = await graph.ainvoke(initial_input(topic), config=config)
        assert result.get("__interrupt__"), "expected planner interrupt"
        result = await graph.ainvoke(
            Command(resume={"approved": True, "edited_plan": None}), config=config
        )
        # checkpoint must be retrievable after completion
        state = await graph.aget_state(config)
        assert state.values.get("report"), "checkpoint state lost: no report"
        return result


async def _sql(query: LiteralString, *params):
    import psycopg

    async with await psycopg.AsyncConnection.connect(get_settings().database_url) as conn:
        cur = await conn.execute(query, params)
        return await cur.fetchall()


async def step2_run_a_and_persistence() -> None:
    print(f"[2/3] Real run A via PostgresSaver: {TOPIC_A!r} ...")
    result = await _run_research(TOPIC_A)
    report = result.get("report", "")
    assert report, "run A produced no report"
    print(f"      report: {len(report)} chars, evidences: {len(result.get('evidences', []))}")

    ck = await _sql("SELECT COUNT(*) FROM checkpoints")
    writes = await _sql("SELECT COUNT(*) FROM checkpoint_writes")
    assert ck[0][0] > 0, "no rows in checkpoints table"
    print(f"      checkpoints rows={ck[0][0]}, checkpoint_writes rows={writes[0][0]}")

    mems = await _sql(
        "SELECT kind, COUNT(*) FROM memories GROUP BY kind ORDER BY kind"
    )
    kinds = dict(mems)
    assert kinds.get("report_summary", 0) >= 1, f"no report_summary memory: {kinds}"
    assert kinds.get("evidence_digest", 0) >= 1, f"no evidence_digest memory: {kinds}"
    print(f"      memories by kind: {kinds}")
    print("      OK: interrupt/resume + checkpoint + memory write all persisted in PG\n")


async def step3_run_b_recall() -> None:
    print(f"[3/3] Real run B (related topic): {TOPIC_B!r} ...")
    result = await _run_research(TOPIC_B)
    recalled = result.get("recalled_memories", [])
    assert recalled, (
        "planner recalled nothing on a closely related topic — "
        "check memory_min_score / embedding model"
    )
    print(f"      recalled {len(recalled)} memory(ies) from run A:")
    for m in recalled:
        print(f"        - {m[:70]}...")
    print("      OK: planner recall injection works end-to-end\n")


async def main() -> None:
    settings = get_settings()
    assert settings.checkpoint_backend == "postgres", "DR_CHECKPOINT_BACKEND must be postgres"
    assert settings.enable_long_term_memory, "DR_ENABLE_LONG_TERM_MEMORY must be true"

    only = sys.argv[1:]  # e.g. "step3" to rerun a single step
    if not only or "step1" in only:
        await step1_memory_roundtrip()
    if not only or "step2" in only:
        await step2_run_a_and_persistence()
    if not only or "step3" in only:
        await step3_run_b_recall()
    print("M3 INTEGRATION TEST PASSED")


if __name__ == "__main__":
    use_compatible_event_loop()
    asyncio.run(main())
