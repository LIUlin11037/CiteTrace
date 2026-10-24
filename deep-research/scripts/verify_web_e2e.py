"""End-to-end acceptance test for the web UI's API flow.

Mimics exactly what static/index.html does, so a pass here means the page
itself can complete the flow:
    POST /research -> SSE until interrupt -> POST /resume -> NEW SSE until done

Run (server already started on :8000):
    python scripts/verify_web_e2e.py
"""

from __future__ import annotations

import json
import os
import sys

# localhost must bypass any corporate/sandbox proxy
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import requests  # noqa: E402

BASE = "http://127.0.0.1:8000"
TOPIC = "MCP 协议的传输层设计：stdio 与 Streamable HTTP 的差异与适用场景"


def read_segment(thread_id: str, label: str) -> dict:
    """Consume one SSE segment; return collected stats."""
    stats = {"tokens": 0, "nodes": [], "interrupt": None, "error": None, "done": False}
    with requests.get(f"{BASE}/research/{thread_id}/events", stream=True, timeout=900) as r:
        r.raise_for_status()
        named_event = None
        for raw in r.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            # iter_lines(decode_unicode=True) 运行时给 str，但 requests 的 stub 标的是 bytes；
            # 两种都兼容，静态检查与运行时都不吃亏。
            line = (raw.decode() if isinstance(raw, bytes) else raw).strip()
            if not line:
                continue
            if line.startswith("event:"):
                named_event = line.split(":", 1)[1].strip()
                if named_event == "done":
                    stats["done"] = True
                    break
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if data.get("interrupt"):
                stats["interrupt"] = data["interrupt"]
                print(f"  [{label}] interrupt: plan with "
                      f"{len(data['interrupt'].get('plan', {}).get('sub_questions', []))} sub-questions")
            elif data.get("event") == "error":
                stats["error"] = data.get("message")
                print(f"  [{label}] ERROR FRAME: {data.get('message')}")
            elif data.get("event") == "on_chat_model_stream" and data.get("token"):
                stats["tokens"] += len(data["token"])
            elif data.get("event") == "on_chain_start" and data.get("node"):
                if data["node"] not in stats["nodes"]:
                    stats["nodes"].append(data["node"])
                    print(f"  [{label}] node start: {data['node']}")
    return stats


def main() -> None:
    print(f"topic: {TOPIC}")

    r = requests.post(f"{BASE}/research", json={"topic": TOPIC}, timeout=30)
    r.raise_for_status()
    thread_id = r.json()["thread_id"]
    print(f"thread_id: {thread_id[:12]}…")

    print("[segment 1] planning, expect an interrupt")
    seg1 = read_segment(thread_id, "seg1")
    assert seg1["error"] is None, f"error frame in segment 1: {seg1['error']}"
    assert seg1["done"], "segment 1 never sent the 'done' event"
    assert seg1["interrupt"] is not None, "no interrupt received: cannot test approval path"
    assert seg1["interrupt"].get("type") == "approve_plan"
    assert seg1["interrupt"].get("plan", {}).get("sub_questions"), "plan has no sub_questions"
    print("  segment 1 OK: interrupt with a well-formed plan, stream closed on 'done'")

    print("[resume] approving the plan, then reconnecting a fresh SSE")
    r = requests.post(
        f"{BASE}/research/{thread_id}/resume",
        json={"approved": True, "edited_plan": None},
        timeout=30,
    )
    r.raise_for_status()
    print(f"  resume -> {r.json()}")

    seg2 = read_segment(thread_id, "seg2")
    assert seg2["error"] is None, f"error frame in segment 2: {seg2['error']}"
    assert seg2["done"], "segment 2 never sent the 'done' event"
    assert seg2["tokens"] > 500, f"expected streamed report tokens, got {seg2['tokens']} chars"
    assert "writer" in seg2["nodes"], f"writer node never started: {seg2['nodes']}"
    print(f"  segment 2 OK: {seg2['tokens']} report chars streamed, nodes={seg2['nodes']}")

    print("\nWEB E2E ACCEPTANCE PASSED")


if __name__ == "__main__":
    sys.exit(main())
