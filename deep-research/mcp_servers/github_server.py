"""GitHub MCP server: repo search and detail via GitHub REST API.

Launch (stdio): python -m mcp_servers.github_server
Auth: set GITHUB_TOKEN to raise the rate limit (60 req/h unauthenticated).
"""

from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("github")

_API = "https://api.github.com"
_TIMEOUT = float(os.environ.get("GITHUB_TIMEOUT", "20"))


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fmt_repo(r: dict) -> str:
    return (
        f"- {r['full_name']} (★{r['stargazers_count']}, forks {r['forks_count']})\n"
        f"  desc: {r.get('description') or '(none)'}\n"
        f"  lang: {r.get('language') or 'n/a'} | "
        f"last push: {r.get('pushed_at', '')[:10]}\n"
        f"  url: {r['html_url']}"
    )


@mcp.tool()
async def search_repos(query: str, max_results: int = 5, sort: str = "stars") -> str:
    """Search GitHub repositories by keyword, sorted by stars or recency.

    Args:
        query: Keyword or GitHub qualifier string, e.g. 'langgraph agent memory'.
        max_results: 1-10 results.
        sort: 'stars' | 'updated' | 'best-match'.
    """
    n = max(1, min(int(max_results), 10))
    params: dict[str, object] = {"q": query, "per_page": n}
    if sort in {"stars", "updated"}:
        params["sort"] = sort
        params["order"] = "desc"
    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_headers()) as client:
        resp = await client.get(f"{_API}/search/repositories", params=params)
        resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        return f"No GitHub repos found for query: {query!r}"
    return f"Found {len(items)} repos for {query!r} (sorted by {sort}):\n" + "\n".join(
        _fmt_repo(r) for r in items
    )


@mcp.tool()
async def get_repo_detail(repo: str) -> str:
    """Fetch one repository's detail (stars, activity, topics, readme excerpt).

    Args:
        repo: 'owner/name', e.g. 'langchain-ai/langgraph'.
    """
    repo = repo.strip().strip("/")
    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_headers()) as client:
        resp = await client.get(f"{_API}/repos/{repo}")
        if resp.status_code == 404:
            return f"GitHub repo not found: {repo!r}"
        resp.raise_for_status()
        data = resp.json()
        readme = await client.get(
            f"{_API}/repos/{repo}/readme",
            headers={**_headers(), "Accept": "application/vnd.github.raw"},
        )
    readme_excerpt = readme.text[:800] if readme.status_code == 200 else "(readme unavailable)"
    return (
        f"{data['full_name']}\n"
        f"stars: {data['stargazers_count']} | forks: {data['forks_count']} | "
        f"open issues: {data['open_issues_count']} | language: {data.get('language')}\n"
        f"created: {data['created_at'][:10]} | last push: {data['pushed_at'][:10]}\n"
        f"topics: {', '.join(data.get('topics', [])) or '(none)'}\n"
        f"license: {(data.get('license') or {}).get('spdx_id') or 'n/a'}\n"
        f"url: {data['html_url']}\n\nREADME excerpt:\n{readme_excerpt}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
