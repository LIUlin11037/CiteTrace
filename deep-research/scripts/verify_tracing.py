"""Prove that run-config callbacks reach node-level LLM and tool calls.

Why this exists: our nodes call `runnable.ainvoke(input_)` WITHOUT an explicit
config, relying on langchain's context-var propagation. If that breaks, LLM
spans vanish from Langfuse and the trace tree is useless — so it is verified
separately from the Langfuse integration itself, with a plain counting handler
(no keys, no network).

Run: python scripts/verify_tracing.py     (needs a real API key; 1 sub-question)
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.callbacks import AsyncCallbackHandler  # noqa: E402

from deep_research.agents.graph import build_graph, initial_input, new_thread_config  # noqa: E402
from deep_research.core.compat import use_compatible_event_loop  # noqa: E402
from deep_research.core.persistence import open_checkpointer  # noqa: E402

TOPIC = "MCP 协议传输层 stdio 与 Streamable HTTP 的差异"


class CountingHandler(AsyncCallbackHandler):
    """Records every callback event type it receives."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.names: set[str] = set()

    def _hit(self, kind: str, **kw: Any) -> None:
        self.counts[kind] += 1
        name = kw.get("name")
        if name:
            self.names.add(str(name))

    async def on_llm_start(self, serialized: dict, prompts: list[str], **kw: Any) -> None:
        self._hit("llm_start", **kw)

    async def on_chat_model_start(self, serialized: dict, messages: list, **kw: Any) -> None:
        self._hit("chat_model_start", **kw)

    async def on_llm_end(self, response: Any, **kw: Any) -> None:
        self._hit("llm_end", **kw)

    async def on_tool_start(self, serialized: dict, input_str: str, **kw: Any) -> None:
        self._hit("tool_start", **kw)

    async def on_tool_end(self, output: Any, **kw: Any) -> None:
        self._hit("tool_end", **kw)

    async def on_chain_start(self, serialized: dict, inputs: dict, **kw: Any) -> None:
        self._hit("chain_start", **kw)

    async def on_chain_end(self, outputs: dict, **kw: Any) -> None:
        self._hit("chain_end", **kw)


async def main() -> None:
    handler = CountingHandler()

    async with open_checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        config = new_thread_config()
        config["callbacks"] = [handler]
        config["metadata"] = {"langfuse_session_id": "verify-tracing", "langfuse_tags": ["verify"]}

        result = await graph.ainvoke(initial_input(TOPIC), config=config)
        assert result.get("__interrupt__"), "expected the planner interrupt"
        from langgraph.types import Command

        await graph.ainvoke(
            Command(resume={"approved": True, "edited_plan": None}), config=config
        )

    print("回调事件计数：")
    for kind, n in sorted(handler.counts.items()):
        print(f"  {kind:18} {n}")
    print(f"出现过的 runnable 名称（前 8 个）：{sorted(handler.names)[:8]}")

    assert handler.counts["chat_model_start"] >= 3, (
        "LLM calls inside nodes did NOT reach the callback — context-var propagation is broken"
    )
    assert handler.counts["tool_start"] >= 1, "MCP tool calls did not reach the callback"
    assert handler.counts["chain_start"] >= 3, "node-level chains did not reach the callback"
    print("\nTRACING PROPAGATION VERIFIED: config callbacks reach nodes, LLMs and tools")


if __name__ == "__main__":
    use_compatible_event_loop()
    asyncio.run(main())
