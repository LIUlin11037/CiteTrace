"""Token/cost accounting for a research run — local, no external service.

Why not just rely on Langfuse: cost numbers must be reproducible without any
external dependency (and available in CI/pre-commit style runs). Langfuse adds
the visual trace layer on top; this module owns the arithmetic.

Prices are DeepSeek's published rates after the 2026-08-17 change, which
introduced peak/off-peak billing: workday 09:00-12:00 and 14:00-18:00 (Beijing
time) cost double, everything else is half. deepseek-chat currently aliases
V4 Flash in non-thinking mode.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.callbacks import AsyncCallbackHandler

# ¥ per 1M tokens: (cache-hit input, cache-miss input, output)
PRICE_PEAK = (0.04, 2.0, 4.0)
PRICE_OFFPEAK = (0.02, 1.0, 2.0)
BEIJING = ZoneInfo("Asia/Shanghai")
_PEAK_WINDOWS = ((time(9, 0), time(12, 0)), (time(14, 0), time(18, 0)))


def is_peak_now(now: datetime | None = None) -> bool:
    now = now or datetime.now(BEIJING)
    if now.weekday() >= 5:  # weekend is always off-peak
        return False
    return any(start <= now.time() < end for start, end in _PEAK_WINDOWS)


class UsageTracker(AsyncCallbackHandler):
    """Accumulates token usage across every LLM call in the process."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage = self._extract(response)
        if not usage:
            return
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        cached = int(
            usage.get("prompt_cache_hit_tokens")
            or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            or 0
        )
        if not (prompt or completion):
            return
        self.calls += 1
        self.input_tokens += prompt
        self.cached_input_tokens += cached
        self.output_tokens += completion

    @staticmethod
    def _extract(response: Any) -> dict[str, Any]:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        if usage:
            return dict(usage)
        for generations in getattr(response, "generations", []) or []:
            for gen in generations:
                meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if meta:
                    return dict(meta)
        return {}

    def cost(self, *, peak: bool | None = None) -> float:
        hit_p, miss_p, out_p = PRICE_PEAK if (peak if peak is not None else is_peak_now()) else PRICE_OFFPEAK
        miss_tokens = max(self.input_tokens - self.cached_input_tokens, 0)
        return (
            miss_tokens * miss_p + self.cached_input_tokens * hit_p + self.output_tokens * out_p
        ) / 1_000_000

    def report(self) -> str:
        peak = is_peak_now()
        return (
            f"调用 {self.calls} 次 | 输入 {self.input_tokens:,} tokens"
            f"（其中缓存命中 {self.cached_input_tokens:,}）"
            f" | 输出 {self.output_tokens:,} tokens"
            f" | 估算成本 ¥{self.cost():.2f}（{'高峰' if peak else '空闲'}价）"
        )


_tracker: UsageTracker | None = None


def attach_usage(config: dict[str, Any]) -> dict[str, Any]:
    """Add the process-wide usage tracker to an invocation config."""
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker()
    callbacks = list(config.get("callbacks") or [])
    callbacks.append(_tracker)
    config["callbacks"] = callbacks
    return config


def usage_report() -> str:
    return _tracker.report() if _tracker is not None else "未采集到用量"


def usage_cost() -> float:
    return _tracker.cost() if _tracker is not None else 0.0
