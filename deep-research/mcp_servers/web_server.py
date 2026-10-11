"""Web MCP server: general web search + clean page-text extraction.

Launch (stdio): python -m mcp_servers.web_server
Search backend: Tavily if TAVILY_API_KEY is set, else Brave if BRAVE_API_KEY
is set, else DuckDuckGo HTML fallback (best-effort, no key). fetch_page uses
readability-style main-content extraction so researchers get article text,
not nav/markup noise.
"""

from __future__ import annotations

import os
import re
from html import unescape

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("web")

_TIMEOUT = float(os.environ.get("WEB_TIMEOUT", "25"))
_UA = "Mozilla/5.0 (compatible; deep-research-agent/0.1)"


def _strip_html(html: str) -> str:
    """Minimal readability-ish extraction: drop script/style/nav, keep text."""
    html = re.sub(r"(?is)<(script|style|noscript|svg|header|footer|nav)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<!--.*?-->", " ", html)
    main = re.search(r"(?is)<(article|main)\b[^>]*>(.*?)</\1>", html)
    if main:
        html = main.group(2)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = unescape(text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


async def _search_tavily(client: httpx.AsyncClient, query: str, n: int) -> list[dict[str, str]]:
    resp = await client.post(
        "https://api.tavily.com/search",
        json={"api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": n},
    )
    resp.raise_for_status()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in resp.json().get("results", [])
    ]


async def _search_brave(client: httpx.AsyncClient, query: str, n: int) -> list[dict[str, str]]:
    resp = await client.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": n},
        headers={"X-Subscription-Token": os.environ["BRAVE_API_KEY"], "Accept": "application/json"},
    )
    resp.raise_for_status()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
        for r in resp.json().get("web", {}).get("results", [])
    ]


async def _search_ddg(client: httpx.AsyncClient, query: str, n: int) -> list[dict[str, str]]:
    resp = await client.get(
        "https://html.duckduckgo.com/html/", params={"q": query}, headers={"User-Agent": _UA}
    )
    resp.raise_for_status()
    out: list[dict[str, str]] = []
    for m in re.finditer(
        r'(?is)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', resp.text
    ):
        url, title = m.group(1), _strip_html(m.group(2))
        if url.startswith("//"):
            url = "https:" + url
        out.append({"title": title, "url": url, "snippet": ""})
        if len(out) >= n:
            break
    return out


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the public web. Backend auto-selected from available API keys.

    Args:
        query: Search query.
        max_results: 1-10 results.
    """
    n = max(1, min(int(max_results), 10))
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        if os.environ.get("TAVILY_API_KEY"):
            results = await _search_tavily(client, query, n)
        elif os.environ.get("BRAVE_API_KEY"):
            results = await _search_brave(client, query, n)
        else:
            results = await _search_ddg(client, query, n)
    if not results:
        return f"No web results for query: {query!r}"
    lines = [f"Found {len(results)} web results for {query!r}:"]
    for r in results:
        lines.append(f"\n- {r['title']}\n  url: {r['url']}\n  snippet: {r['snippet'][:300]}")
    return "\n".join(lines)


@mcp.tool()
async def fetch_page(url: str, max_chars: int = 6000) -> str:
    """Fetch a web page and return its cleaned main text content.

    Args:
        url: Absolute http(s) URL.
        max_chars: Truncate extracted text to this length.
    """
    if not url.startswith(("http://", "https://")):
        return f"Invalid URL (must be absolute http/https): {url!r}"
    cap = max(500, min(int(max_chars), 20000))
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": _UA})
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype:
            return f"Skipped non-text content ({ctype}) at {url}"
    text = _strip_html(resp.text)
    if len(text) > cap:
        text = text[:cap] + "\n... [truncated]"
    return f"Content of {url}:\n\n{text}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
