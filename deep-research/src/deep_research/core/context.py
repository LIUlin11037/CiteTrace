"""Context budgeting: keep the informative parts of a fetched document.

Why this exists
---------------
The old behaviour kept the **first** ``evidence_content_limit`` characters of a
fetched page. That is the wrong 12k: nav bars, cookie banners and boilerplate
come first, while the results table — the part that actually answers a question
— sits in the middle. Measured 2026-10-15 on ``gq-distillation``: the R1 paper
was fetched, but its benchmark table never entered the context, so the writer
honestly reported "no quantitative support" (coverage 0.5). Raising the blind
cut to 12000 fixed coverage but paid ~60% more input tokens.

This module keeps the *information*, not the *prefix*: head block (title /
abstract), then blocks ranked by information density (numbers, percentages,
"表/结果/实验/benchmark", table rows), reassembled in original order.

Design constraints
------------------
- Deterministic and dependency-free (no tokenizer model, no network).
- Never grows the text; never drops the head; always leaves a visible marker so
  the model knows it is reading an excerpt (otherwise it may claim completeness).
"""

from __future__ import annotations

import re

# Blocks that look like results: digits with units/percentages, markdown tables,
# benchmark-ish keywords. Chinese and English both covered on purpose.
_DENSITY_KEYS = re.compile(
    r"(\d+(?:\.\d+)?\s*%|\|\s*[-:]{2,}|表\s*\d|结果|实验|基准|分数|提升|对比"
    r"|benchmark|result|table|accuracy|score|outperform|baseline)",
    re.IGNORECASE,
)
_DIGIT = re.compile(r"\d")

# Conclusion markers. Without this, a paragraph of qualitative takeaways (no
# numbers, no table) scores 0 and is dropped first — measured on the demo blocks:
# 结果表格 40.54 / 导航条 6.45 / 定性论述 0.00. The 0.00 was the bug: that block
# was often the actual answer.
_CONCLUSION_KEYS = re.compile(
    r"(结论|表明|说明|因此|综上|可见|认为|发现|意味着|指出|提出|关键在于|核心思路"
    r"|conclude|suggests?|indicates?|therefore|implies|finding|takeaway)",
    re.IGNORECASE,
)

HEAD_RATIO = 0.35  # share of the budget reserved for the leading block
MIN_BLOCK_CHARS = 40  # ignore fragments when scoring
_CHUNK_TARGET = 400  # pseudo-paragraph size for line-based chunking
_SENT_SPLIT = re.compile(r"(?<=[。！？；;.!?])\s*")


def estimate_tokens(text: str) -> int:
    """Rough token estimate: CJK ~1 token/char, ASCII ~4 chars/token.

    Deliberately crude — this is used for budgeting decisions and logs, not for
    billing (billing reads the provider's usage counters).
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_chars = len(text) - cjk
    return cjk + ascii_chars // 4


def _slice_sentences(text: str, target: int) -> list[str]:
    """Slice one over-long line into ~target sized pieces at sentence ends."""
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for sent in _SENT_SPLIT.split(text):
        if not sent:
            continue
        buf.append(sent)
        size += len(sent)
        if size >= target:
            parts.append("".join(buf))
            buf, size = [], 0
    if buf:
        parts.append("".join(buf))
    return parts


def _split_blocks(text: str) -> list[str]:
    """Split into pseudo-paragraphs of ~400 chars (line-based, sentence fallback).

    Blank-line splitting is not enough: HTML-derived text (ar5iv, web fetch) is
    often one huge block with no blank lines at all, which made the first
    implementation keep only the head and waste two thirds of the budget
    (measured: 21000 chars → 2131 with a 6000 budget; coverage 1.00 → 0.25 on
    gq-distillation). Line chunking plus sentence slicing is stable for both
    markdown and extracted HTML.
    """
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in text.splitlines():
        buf.append(line)
        size += len(line) + 1
        if size >= _CHUNK_TARGET:
            chunks.append("\n".join(buf))
            buf, size = [], 0
    if buf:
        chunks.append("\n".join(buf))

    out: list[str] = []
    for chunk in chunks:
        if len(chunk) <= _CHUNK_TARGET * 2:
            out.append(chunk)
        else:  # single very long line: fall back to sentence boundaries
            out.extend(_slice_sentences(chunk, _CHUNK_TARGET))
    return [c for c in out if c.strip()]


def _score(block: str) -> float:
    hits = len(_DENSITY_KEYS.findall(block))
    conclusions = len(_CONCLUSION_KEYS.findall(block))
    digits = len(_DIGIT.findall(block))
    # density per 100 chars, so a long boilerplate block doesn't win by size
    density = (hits * 3 + conclusions * 2 + min(digits, 30)) / max(len(block), 1) * 100
    return density


def trim_document(text: str, limit: int) -> tuple[str, bool]:
    """Trim `text` to ~`limit` chars, keeping the informative parts.

    Returns (text, trimmed). When ``trimmed`` is True a marker line is appended
    telling the model the document was excerpted.
    """
    if limit <= 0 or len(text) <= limit:
        return text, False

    blocks = _split_blocks(text)
    if not blocks:
        return text[:limit], True

    marker = (
        f"\n\n[已按信息密度裁剪：原文 {len(text)} 字符，"
        f"仅保留标题/摘要与信息密集段落，可能缺省略部分]"
    )
    budget = limit - len(marker)
    head_budget = int(budget * HEAD_RATIO)

    # 1) head chunk: title / abstract / intro — always kept, capped
    kept: list[tuple[int, str]] = [(0, blocks[0][:head_budget])]
    used = len(kept[0][1])

    # 2) remaining chunks ranked by information density; the loop keeps filling
    #    until the budget is spent, so an unstructured document still gets its
    #    tail (the old code stopped early once candidates ran out).
    rest = [(i, b) for i, b in enumerate(blocks[1:], start=1)]
    for idx, block in sorted(rest, key=lambda ib: _score(ib[1]), reverse=True):
        if used >= budget:
            break
        room = budget - used
        snippet = block if len(block) <= room else block[:room]
        kept.append((idx, snippet))
        used += len(snippet)

    kept.sort(key=lambda ib: ib[0])
    body = "\n\n".join(b for _, b in kept)
    return body + marker, True
