"""Ingest local documents into the offline knowledge base (kb_chunks).

Usage:
    python scripts/ingest_kb.py <dir-or-file> [<dir-or-file> ...] [--dry-run]

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
from deep_research.core.kb import get_kb_store, split_document  # noqa: E402

SUPPORTED = {".md", ".markdown", ".txt"}


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


async def main(paths: list[str], dry_run: bool) -> None:
    use_compatible_event_loop()
    files = collect(paths)
    if not files:
        print("没有可入库的文件")
        return

    total_chunks = 0
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        rel = str(f).replace("\\", "/")
        title = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.startswith("#")), f.name)
        chunks = split_document(text, rel, title)
        total_chunks += len(chunks)
        print(f"  {rel}: {len(text)} 字符 → {len(chunks)} 块")

    if dry_run:
        print(f"\n[dry-run] 共 {len(files)} 个文件 / {total_chunks} 块（未写库）")
        return

    store = await get_kb_store()
    written = 0
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        rel = str(f).replace("\\", "/")
        title = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.startswith("#")), f.name)
        written += await store.add_chunks(split_document(text, rel, title))

    stats = await store.stats()
    print(f"\n入库完成：{len(files)} 个文件 / 写入 {written} 块")
    print(f"库内现状：{stats['docs']} 个文档 / {stats['chunks']} 个块")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="目录或文件")
    ap.add_argument("--dry-run", action="store_true", help="只切块不写库")
    args = ap.parse_args()
    # 必须在 asyncio.run 之前设置：psycopg 不能用 ProactorEventLoop
    use_compatible_event_loop()
    asyncio.run(main(args.paths, args.dry_run))
