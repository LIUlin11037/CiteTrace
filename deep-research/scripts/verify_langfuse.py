"""Verify the Langfuse ingestion path against a local fake server.

Runs a real (tiny) research with DR_LANGFUSE_HOST pointing at a throwaway HTTP
server, then asserts the SDK actually shipped trace payloads. This proves the
wiring — handler attached, OTLP exporter posting to the configured host — so
that when real keys arrive, "nothing shows up in the UI" is not a possibility.

Run: python scripts/verify_langfuse.py     (needs a real API key; 1 sub-question)
"""

from __future__ import annotations

import http.server
import os
import threading

# must be set BEFORE deep_research.config is imported anywhere
os.environ.setdefault("DR_CHECKPOINT_BACKEND", "sqlite")
os.environ.setdefault("DR_ENABLE_LONG_TERM_MEMORY", "false")
os.environ.setdefault("DR_MAX_SUB_QUESTIONS", "1")
os.environ["DR_LANGFUSE_PUBLIC_KEY"] = "pk-lf-fake"
os.environ["DR_LANGFUSE_SECRET_KEY"] = "sk-lf-fake"
os.environ.setdefault("DR_LANGFUSE_HOST", "http://127.0.0.1:8899")

import asyncio  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deep_research.agents.graph import build_graph, initial_input, new_thread_config  # noqa: E402
from deep_research.core.compat import use_compatible_event_loop  # noqa: E402
from deep_research.core.persistence import open_checkpointer  # noqa: E402
from deep_research.core.tracing import apply_tracing, flush_tracing  # noqa: E402

TOPIC = "MCP 协议的传输层设计要点"
received: list[tuple[str, int]] = []


class FakeLangfuse(http.server.BaseHTTPRequestHandler):
    def _record(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        received.append((f"{method} {self.path}", len(body)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self) -> None:  # noqa: N802
        self._record("POST")

    def do_GET(self) -> None:  # noqa: N802
        self._record("GET")

    def log_message(self, *args: object) -> None:  # silence the default stderr spam
        pass


async def main() -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 8899), FakeLangfuse)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    from langgraph.types import Command

    async with open_checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        config = apply_tracing(new_thread_config(), tags=["verify-langfuse"])
        result = await graph.ainvoke(initial_input(TOPIC), config=config)
        if result.get("__interrupt__"):
            await graph.ainvoke(
                Command(resume={"approved": True, "edited_plan": None}), config=config
            )

    flush_tracing()
    await asyncio.sleep(1.0)  # let the exporter's final POST land
    server.shutdown()

    posts = [(p, n) for p, n in received if p.startswith("POST")]
    print(f"假服务端收到 {len(received)} 个请求，其中 POST {len(posts)} 个：")
    for path, size in received[:6]:
        print(f"  {path}  ({size} bytes)")

    assert posts, "no trace payload was shipped to the configured host"
    assert any(n > 0 for _, n in posts), "trace payload arrived empty"
    print("\nLANGFUSE WIRING VERIFIED: handler attached and payloads shipped to the configured host")


if __name__ == "__main__":
    use_compatible_event_loop()
    asyncio.run(main())
