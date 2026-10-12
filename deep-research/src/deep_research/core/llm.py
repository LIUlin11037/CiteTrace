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

from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from deep_research.core.config import RoleName, get_settings

T = TypeVar("T", bound=BaseModel)

# The writer emits a full report plus a complete footnote reference list; the
# shared 4096 default truncates the reference section on source-heavy reports
# (observed in eval: gq-mcp/gq-moe cut off mid-list, leaving cited footnotes
# undefined). deepseek-chat supports 8192 output tokens — give writer the max.
WRITER_MIN_MAX_TOKENS = 8192


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

    method="function_calling": the langchain default (json_schema
    response_format) is rejected by DeepSeek and several other compatible
    providers; function calling works across all of them.
    """
    return get_chat_model(role).with_structured_output(schema, method="function_calling")


class LLMCallError(Exception):
    """Raised when an LLM call fails after all retries."""


@retry(
    retry=retry_if_exception_type((TimeoutError, ConnectionError, LLMCallError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
async def invoke_with_retry(runnable: Runnable, input_: dict) -> BaseModel:
    """Invoke a structured-output runnable with retry on transient failures.

    Validation errors from the provider returning malformed structured output
    are NOT retried here (they surface immediately) — that is a prompt bug,
    not a transient failure, and retrying would burn quota.
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
    except Exception as exc:  # provider 5xx / rate limit surface as APIError
        if _is_transient(exc):
            raise LLMCallError(str(exc)) from exc
        raise


def _is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status is None:
        return False
    return status == 429 or status >= 500
