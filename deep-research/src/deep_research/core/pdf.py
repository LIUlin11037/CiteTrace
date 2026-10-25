"""PDF 页面解析：正文 + 表格（保留行列关系）。

为什么不用 pypdf 的 ``extract_text()``：它把表格**拉平**成一行文字，标签和数字
的对应关系随之丢失——"Current ratio | 2.5 | Cash | 4,826" 变成
"Current ratio 2.5 Cash 4,826"。问题问"流动性是否健康"时，语义上根本匹配不上，
这才是私有语料检索里最难补的那一段（实测：24% 的题在 top-50 里都捞不到）。

这里用 pdfplumber（MIT）按几何线条还原表格，输出 Markdown，让每行保持
"标签 | 数值 | 数值"的结构；表格单独成块，锚点带 ``#t{k}``，与正文区分。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path


@dataclass
class PageContent:
    page: int  # 1-indexed
    text: str
    tables: list[str] = field(default_factory=list)


def _rows_to_markdown(rows: list[list[str | None]], max_rows: int = 60) -> str:
    """Render a table as Markdown; drop fully-empty rows/columns."""
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        if any(cells):
            cleaned.append(cells)
    if len(cleaned) < 2:  # 单行"表格"多半是抽错了
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    # 丢掉整列为空的行列，减少噪声
    keep = [i for i in range(width) if any(r[i] for r in cleaned)]
    if len(keep) < 2:
        return ""
    cleaned = [[r[i] for i in keep] for r in cleaned[:max_rows]]
    header, body = cleaned[0], cleaned[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def extract_pdf(path: str | Path, max_pages: int | None = None) -> list[PageContent]:
    """Extract each page's text plus any tables rendered as Markdown."""
    import pdfplumber

    pages: list[PageContent] = []
    with pdfplumber.open(str(path)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            if max_pages and idx > max_pages:
                break
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001 - 个别页抽不出文字，跳过而不是整篇失败
                text = ""
            tables: list[str] = []
            try:
                for raw in page.extract_tables() or []:
                    md = _rows_to_markdown(raw)
                    if md:
                        tables.append(md)
            except Exception:  # noqa: BLE001
                pass
            pages.append(PageContent(page=idx, text=text, tables=tables))
    return pages


def page_chunks(
    pages: list[PageContent],
    source_path: str,
    title: str = "",
    max_chars: int = 4000,
    use_tables: bool = True,
) -> list[dict[str, Any]]:
    """One chunk per page: text **plus** its tables rendered as Markdown.

    为什么合并而不是让表格单独成块：单独成块会与同页文本重复，两个块互相竞争
    检索名次，反而把别的页挤出 top-10（实测 Boeing @10 从 57% 掉到 14%）。
    合并后每个页仍是**一个**检索单位，既保住页级召回，又让标签与数值重新绑定。
    """
    out: list[dict[str, Any]] = []
    idx = 0
    for pg in pages:
        heading = ""
        for line in pg.text.splitlines():
            s = line.strip()
            if len(s) > 3 and not s[0].isdigit():
                heading = s[:80]
                break
        body = pg.text.strip()
        if pg.tables and use_tables:
            extra = "\n\n".join(
                f"表 {k}（第 {pg.page} 页）｜{heading}\n\n{md}"
                for k, md in enumerate(pg.tables, start=1)
            )
            body = (body + "\n\n" + extra).strip()
        if not body:
            continue
        # 超长页才切分（同一页的第 2 段锚点用 p{n}#2，仍可定位回页）
        pieces = [body[i : i + max_chars] for i in range(0, len(body), max_chars)]
        for part_no, piece in enumerate(pieces, start=1):
            out.append(
                {
                    "anchor": f"p{pg.page}" if part_no == 1 else f"p{pg.page}#{part_no}",
                    "title": heading or title or source_path.rsplit("/", 1)[-1],
                    "text": piece.strip(),
                    "chunk_idx": idx,
                }
            )
            idx += 1
    return out
