"""FastAPI service: create research runs, stream events over SSE, resume on HITL.

    python -m deep_research.serve --port 8000     # ← use this entry point
    # `uvicorn deep_research.api:app` does NOT work on Windows with the
    # Postgres checkpointer: uvicorn hard-codes ProactorEventLoop for win32 and
    # psycopg async rejects it. serve.py passes an explicit selector loop.

No auth (M2 scope). Sessions are kept in-process; a single shared
checkpointer persists graph state so /resume works across requests.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from deep_research.agents.graph import build_graph, initial_input
from deep_research.core.compat import use_compatible_event_loop
from deep_research.core.persistence import open_checkpointer

use_compatible_event_loop()

_STATIC_DIR = Path(__file__).resolve().parent / "static"

_saver: Any = None
_graph: Any = None
# thread_id -> events queue consumed by the SSE endpoint
_streams: dict[str, asyncio.Queue[dict[str, Any] | None]] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _saver, _graph
    async with open_checkpointer() as saver:
        _saver = saver
        _graph = build_graph(checkpointer=saver)
        yield


app = FastAPI(title="Deep Research API", lifespan=lifespan)


class ResearchRequest(BaseModel):
    topic: str


class ResumeRequest(BaseModel):
    approved: bool = True
    edited_plan: dict[str, Any] | None = None


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


async def _pending_interrupt(config: dict[str, Any]) -> Any | None:
    """Read a pending HITL interrupt from the checkpoint state.

    astream_events' on_chain_end output does NOT carry __interrupt__ in this
    langgraph version (only ainvoke's return value does), so the graph state
    is the authoritative source. Returns the interrupt value or None.
    """
    state = await _graph.aget_state(config)
    for task in state.tasks or []:
        for intr in getattr(task, "interrupts", ()) or ():
            return intr.value
    return None


async def _drive(thread_id: str, resume: Command | None, payload: dict[str, Any]) -> None:
    """Run the graph and push events into the thread's queue."""
    queue = _streams[thread_id]
    config = _config(thread_id)
    try:
        kwargs = {"config": config}
        arg = resume if resume is not None else payload
        async for event in _graph.astream_events(arg, version="v2", **kwargs):
            await queue.put(event)
        # the graph is done or paused — surface a pending human-approval gate
        interrupt_value = await _pending_interrupt(config)
        if interrupt_value is not None:
            await queue.put({"event": "interrupt", "data": {"value": interrupt_value}})
    except Exception as exc:  # surface errors to the client instead of hanging
        await queue.put({"event": "error", "data": {"message": f"{type(exc).__name__}: {exc}"}})
    finally:
        await queue.put(None)


def _to_sse(event: dict[str, Any]) -> str:
    """Reduce a langgraph event to the fields the UI needs."""
    name = event.get("name", "")
    kind = event.get("event", "")
    data: dict[str, Any] = {"event": kind, "name": name}
    if kind == "error":  # synthetic event from _drive's exception handler
        data["message"] = event.get("data", {}).get("message", "unknown error")
    elif kind == "interrupt":  # synthetic event: graph paused for approval
        data["interrupt"] = event.get("data", {}).get("value")
    elif kind == "on_chat_model_stream":
        chunk = event.get("data", {}).get("chunk")
        if chunk is not None and getattr(chunk, "content", ""):
            data["token"] = chunk.content
    elif kind == "on_chain_end" and name == "LangGraph":
        out = event.get("data", {}).get("output", {})
        if isinstance(out, dict) and out.get("__interrupt__"):
            data["interrupt"] = out["__interrupt__"][0].value
    elif name in {"planner", "researcher", "critic", "writer"}:
        data["node"] = name
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/research")
async def create_research(req: ResearchRequest) -> dict[str, str]:
    if not req.topic.strip():
        raise HTTPException(status_code=422, detail="topic must not be empty")
    thread_id = uuid.uuid4().hex
    _streams[thread_id] = asyncio.Queue()
    asyncio.create_task(_drive(thread_id, resume=None, payload=initial_input(req.topic)))
    return {"thread_id": thread_id}


@app.get("/research/{thread_id}/events")
async def stream_events(thread_id: str) -> StreamingResponse:
    queue = _streams.get(thread_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="unknown thread_id")

    async def gen() -> AsyncIterator[str]:
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield _to_sse(event)
        finally:
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/research/{thread_id}/resume")
async def resume_research(thread_id: str, req: ResumeRequest) -> dict[str, str]:
    if thread_id not in _streams:
        raise HTTPException(status_code=404, detail="unknown thread_id")
    # fresh queue for the resumed segment so a reconnecting SSE client works
    _streams[thread_id] = asyncio.Queue()
    resume = Command(resume={"approved": req.approved, "edited_plan": req.edited_plan})
    asyncio.create_task(_drive(thread_id, resume=resume, payload={}))
    return {"thread_id": thread_id, "status": "resumed"}


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the minimal web UI (single static file, no build step)."""
    return FileResponse(_STATIC_DIR / "index.html")
