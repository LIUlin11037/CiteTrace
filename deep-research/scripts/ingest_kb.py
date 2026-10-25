"""Ingest local documents into the offline knowledge base (kb_chunks).

Usage:
    python scripts/ingest_kb.py <dir-or-file> [...] [--root .] [--dry-run]

Paths are stored **relative to --root** so the same knowledge base works on the
host and inside the container (measured: absolute host paths made every offline
citation unverifiable inside Docker, dropping cite/cover to zero).

Supported now: .md / .markdown / .txt. PDF/DOCX need a parser dependency and are
deliberately not included yet — a half-working PDF path would poison citations
worse than no PDF support at all.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deep_research.core.compat import use_compatible_event_loop  # noqa: E402
from deep_research.core.kb import (  # noqa: E402
    Chunk,
    get_kb_store,
    split_document,
    split_pdf_pages,
)

SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}


def _rel(path: Path, root: Path) -> str:
    """Store a portable path; fall back to absolute only when outside the root."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _pdf_chunks(path: Path, rel: str, tables: bool = False):  # type: ignore[no-untyped-def]
    """Parse a PDF into per-page chunks (anchor = page number).

    ``tables=True`` 时用 pdfplumber 还原表格（单独成块，锚点 p{n}#t{k}），
    解决 pypdf 把表格拉平后"标签与数值分离"的问题。
    """
    if tables:
        from deep_research.core.pdf import extract_pdf, page_chunks

        pages = extract_pdf(path)
        return [
            Chunk(
                doc_id=rel,
                source_path=rel,
                title=c["title"],
                anchor=c["anchor"],
                chunk_idx=c["chunk_idx"],
                text=c["text"],
            )
            for c in page_chunks(pages, rel, title=path.stem)
        ]
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return split_pdf_pages(pages, rel, title=path.stem)


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(
                sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED and f.is_file())
            )
        elif p.is_file() and p.suffix.lower() in SUPPORTED:
            files.append(p)
        else:
            print(f"  [skip] {raw}（不是受支持的目录/文件）")
    return files


async def main(
    paths: list[str],
    dry_run: bool,
    root: Path,
    reset: bool = False,
    use_tables: bool = False,
) -> None:
    use_compatible_event_loop()
    files = collect(paths)
    if not files:
        print("没有可入库的文件")
        return

    total_chunks = 0
    for f in files:
        rel = _rel(f, root)
        if f.suffix.lower() == ".pdf":
            chunks = _pdf_chunks(f, rel, tables=use_tables)
            total_chunks += len(chunks)
            print(f"  {rel}: {len(chunks)} 块（PDF，按页切分）")
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        title = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.startswith("#")), f.name)
        chunks = split_document(text, rel, title)
        total_chunks += len(chunks)
        print(f"  {rel}: {len(text)} 字符 → {len(chunks)} 块")

    if dry_run:
        print(f"\n[dry-run] 共 {len(files)} 个文件 / {total_chunks} 块（未写库）")
        return

    store = await get_kb_store()
    if reset:
        rels = [_rel(f, root) for f in files]
        removed = await store.delete_docs(rels)
        print(f"  已清理 {len(rels)} 个文件的旧块（{removed} 行，表 {store._table}）")
    written = 0
    for f in files:
        rel = _rel(f, root)
        if f.suffix.lower() == ".pdf":
            written += await store.add_chunks(_pdf_chunks(f, rel, tables=use_tables))
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        title = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.startswith("#")), f.name)
        written += await store.add_chunks(split_document(text, rel, title))

    stats = await store.stats()
    print(f"\n入库完成：{len(files)} 个文件 / 写入 {written} 块")
    print(f"库内现状：{stats['docs']} 个文档 / {stats['chunks']} 个块")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="目录或文件")
    ap.add_argument("--dry-run", action="store_true", help="只切块不写库")
    ap.add_argument("--root", default=".", help="相对路径的根目录（默认当前目录）")
    ap.add_argument("--reset", action="store_true", help="先删除这些文件的旧块（改切块策略后必用）")
    ap.add_argument(
        "--pdf-tables", action="store_true", help="PDF 用 pdfplumber 还原表格（表格单独成块）"
    )
    args = ap.parse_args()
    # 必须在 asyncio.run 之前设置：psycopg 不能用 ProactorEventLoop
    use_compatible_event_loop()
    asyncio.run(
        main(args.paths, args.dry_run, Path(args.root), args.reset, args.pdf_tables)
    )
