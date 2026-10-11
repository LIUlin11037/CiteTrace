"""arXiv MCP server: paper search and detail retrieval via the public arXiv API.

Launch (stdio): python -m mcp_servers.arxiv_server
No API key required. The arXiv API is rate-limited (~1 req / 3s); be polite.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arxiv")

_API = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_MAX_RESULTS = 10
_TIMEOUT = float(os.environ.get("ARXIV_TIMEOUT", "20"))


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return re.sub(r"\s+", " ", node.text).strip()


def _parse_entries(xml_text: str) -> list[dict[str, str]]:
    root = ET.fromstring(xml_text)
    out: list[dict[str, str]] = []
    for entry in root.findall(f"{_ATOM}entry"):
        arxiv_id = _text(entry.find(f"{_ATOM}id"))
        out.append(
            {
                "arxiv_id": arxiv_id.rsplit("/", 1)[-1],
                "title": _text(entry.find(f"{_ATOM}title")),
                "summary": _text(entry.find(f"{_ATOM}summary")),
                "published": _text(entry.find(f"{_ATOM}published"))[:10],
                "authors": ", ".join(
                    _text(a.find(f"{_ATOM}name")) for a in entry.findall(f"{_ATOM}author")
                ),
            }
        )
    return out


@mcp.tool()
async def search_papers(query: str, max_results: int = 5) -> str:
    """Search arXiv for papers matching a query. Returns title/authors/date/abstract.

    Args:
        query: Natural-language or fielded query, e.g. 'agentic RAG memory'.
        max_results: 1-10 results.
    """
    n = max(1, min(int(max_results), _MAX_RESULTS))
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": n,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(_API, params=params)
        resp.raise_for_status()
    entries = _parse_entries(resp.text)
    if not entries:
        return f"No arXiv papers found for query: {query!r}"
    lines = [f"Found {len(entries)} papers for {query!r}:"]
    for e in entries:
        lines.append(
            f"\n- [{e['arxiv_id']}] {e['title']}\n"
            f"  authors: {e['authors']}\n  published: {e['published']}\n"
            f"  url: https://arxiv.org/abs/{e['arxiv_id']}\n  abstract: {e['summary'][:600]}"
        )
    return "\n".join(lines)


@mcp.tool()
async def get_paper(arxiv_id: str) -> str:
    """Fetch a single arXiv paper's full metadata by its id (e.g. '2401.12345')."""
    params = {"id_list": arxiv_id.strip()}
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(_API, params=params)
        resp.raise_for_status()
    entries = _parse_entries(resp.text)
    if not entries:
        return f"arXiv paper not found: {arxiv_id!r}"
    e = entries[0]
    return (
        f"[{e['arxiv_id']}] {e['title']}\n"
        f"authors: {e['authors']}\npublished: {e['published']}\n"
        f"url: https://arxiv.org/abs/{e['arxiv_id']}\n\nAbstract:\n{e['summary']}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
