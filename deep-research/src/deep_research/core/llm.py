"""Model abstraction layer.

All LLM access in the system goes through here. Providers are interchangeable
as long as they speak the OpenAI chat-completions protocol (DeepSeek, Qwen
DashScope-compatible mode, OpenAI, vLLM, Ollama...).

Key decisions:
- ChatOpenAI with custom base_url instead of per-provider SDKs: one code path.
- Retries live here (tenacity), so nodes never handle transient 429/5xx.
- structured_output() is the ONLY way nodes get parsed data from LLMs:
  no regex JSON parsing anywhere else in the codebase.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from deep_research.core.config import RoleName, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# The writer emits a full report plus a complete footnote reference list; the
# shared 4096 default truncates the reference section on source-heavy reports
# (eval 2026-10-12: gq-mcp/gq-moe cut off mid-list). 8192 was still not enough
# (eval 2026-10-14: gq-agent-memory lost its whole 参考来源 section). The
# endpoint accepts >=65536 (probed), so give the writer generous headroom —
# a high cap costs nothing, only actually generated tokens are billed.
WRITER_MIN_MAX_TOKENS = 32768

# Roles whose model rejects a forced tool_choice (e.g. DeepSeek thinking mode).
# Populated lazily on the first 400, then reused for the process lifetime.
_unforced_roles: set[str] = set()


def get_chat_model(role: RoleName, *, streaming: bool = False) -> BaseChatModel:
    """Build a chat model for the given agent role.

    streaming=True is used by the writer node so FastAPI/CLI can stream tokens;
    other roles always run non-streaming because their outputs are structured.
    """
    cfg = get_settings().model_config_for(role)
    max_tokens = cfg.max_tokens
    if role == "writer":
        max_tokens = max(max_tokens, WRITER_MIN_MAX_TOKENS)
    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        temperature=cfg.temperature,
        max_tokens=max_tokens,
        timeout=cfg.request_timeout,
        streaming=streaming,
        max_retries=0,  # retries are handled by invoke_with_retry below
    )


def structured_output(role: RoleName, schema: type[T]) -> Runnable:
    """A runnable that returns a validated pydantic instance of `schema`.

    Two transports, chosen per role:
    1. function calling with a FORCED tool_choice (langchain default) — most
       reliable, but some models reject it: DeepSeek's thinking mode returns
       HTTP 400 "Thinking mode does not support this tool_choice".
    2. function calling with tool_choice="auto", parsing the emitted tool call
       ourselves — works for thinking-mode models too.

    The first use of a role tries (1); a 400 mentioning tool_choice flips that
    role to (2) for the rest of the process. This keeps the strict path for
    normal models while still letting a reasoning model serve any structured
    role (planner / critic / extraction).
    """
    model = get_chat_model(role)
    forced = model.with_structured_output(schema, method="function_calling")
    unforced = model.bind_tools([convert_to_openai_tool(schema)], tool_choice="auto")

    async def _unforced_call(messages: Any) -> T:
        reply = await unforced.ainvoke(messages)
        for call in getattr(reply, "tool_calls", None) or []:
            if call.get("name") == schema.__name__:
                return schema.model_validate(call["args"])
        raise LLMCallError("structured output returned None (no tool call emitted)")

    async def _run(input_: Any) -> T:
        messages = input_["messages"] if isinstance(input_, dict) else input_
        if role in _unforced_roles:
            return await _unforced_call(messages)
        try:
            return await forced.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 - provider-specific 400 shapes vary
            text = str(exc)
            if "tool_choice" in text and "does not support" in text:
                logger.info(
                    "structured output: role %r rejects forced tool_choice (%s); "
                    "switching it to unforced tool calls",
                    role,
                    type(exc).__name__,
                )
                _unforced_roles.add(role)
                return await _unforced_call(messages)
            raise

    return RunnableLambda(_run)


class LLMCallError(Exception):
    """Raised when an LLM call fails after all retries."""


@retry(
    retry=retry_if_exception_type((TimeoutError, ConnectionError, LLMCallError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def invoke_with_retry(runnable: Runnable, input_: dict) -> BaseModel:
    """Invoke a structured-output runnable with retry on transient failures.

    ValidationError from the provider emitting malformed structured output
    (e.g. a list where a string belongs — observed in M3 testing) is model
    nondeterminism, not a deterministic prompt bug: a fresh attempt usually
    emits a well-formed payload, so it is retried like a None tool call.
    """
    try:
        # langchain chat models require a bare message list / PromptValue;
        # callers pass {"messages": [...]} for readability — unwrap here.
        if isinstance(input_, dict) and set(input_) == {"messages"}:
            input_ = input_["messages"]
        result = await runnable.ainvoke(input_)
        if result is None:
            # Provider answered with plain text instead of a tool call —
            # retryable: the next attempt usually emits the tool call.
            raise LLMCallError("structured output returned None (no tool call emitted)")
        return result
    except (TimeoutError, ConnectionError):
        raise
    except ValidationError as exc:
        raise LLMCallError(f"malformed structured output: {exc}") from exc
    except OutputParserException as exc:
        # The model can emit a tool call name that isn't in the current
        # runnable's tool set (e.g. it replays a search tool from the ReAct
        # transcript while we are asking for EvidenceBundle) — observed in
        # eval 2026-10-14 (gq-graph-rag). Re-sampling usually fixes it.
        raise LLMCallError(f"tool-call parse failure: {exc}") from exc
    except Exception as exc:  # provider 5xx / rate limit surface as APIError
        if _is_transient(exc):
            raise LLMCallError(str(exc)) from exc
        raise


def _is_transient(exc: Exception) -> bool:
    """Network-level failures and 429/5xx are worth another attempt.

    openai wraps httpx errors (APIConnectionError / APITimeoutError) and does
    NOT carry a status code, so class-name checks alone miss them — check the
    httpx transport error and its `__cause__` chain. Observed 2026-10-14: a
    network blip failed 17 of 26 benchmark questions outright because these
    were treated as permanent.
    """
    for err in (exc, exc.__cause__):
        if isinstance(err, httpx.TransportError):
            return True
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status is None:
        return type(exc).__name__ in {"APIConnectionError", "APITimeoutError", "ConnectError"}
    return status == 429 or status >= 500
