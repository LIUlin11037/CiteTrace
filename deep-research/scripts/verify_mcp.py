"""Verify MCP servers load through the CORE loader (MCP 接入验收 #3).

Runs the real tools/mcp_client.get_mcp_tools() against all three servers,
then exercises one live tool call (arXiv, no API key needed).

Run: python scripts/verify_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault(
    "DR_MCP_SERVERS",
    json.dumps(
        {
            "arxiv": ["-m", "mcp_servers.arxiv_server"],
            "github": ["-m", "mcp_servers.github_server"],
            "web": ["-m", "mcp_servers.web_server"],
        }
    ),
)
# The core loader expects full server config dicts; rebuild it in the exact
# shape config.parse_mcp_servers() produces.
os.environ["DR_MCP_SERVERS"] = json.dumps(
    {
        name: {"transport": "stdio", "command": sys.executable, "args": ["-m", f"mcp_servers.{name}_server"]}
        for name in ("arxiv", "github", "web")
    }
)

from deep_research.tools.mcp_client import get_mcp_tools, reset_tool_cache  # noqa: E402


async def main() -> None:
    reset_tool_cache()
    tools = await get_mcp_tools()
    names = sorted(t.name for t in tools)
    print(f"loaded {len(tools)} tools from MCP servers:")
    for n in names:
        print(f"  - {n}")

    expected = {
        "search_papers",
        "get_paper",
        "search_repos",
        "get_repo_detail",
        "web_search",
        "fetch_page",
    }
    missing = expected - set(names)
    assert not missing, f"missing tools: {missing}"
    print("[ok] all 6 expected tools present across the 3 servers")

    # live call: arXiv search (no key required)
    arxiv = next(t for t in tools if t.name == "search_papers")
    result = await arxiv.ainvoke({"query": "agentic retrieval augmented generation", "max_results": 2})
    text = str(result)
    assert "arxiv.org" in text, f"unexpected arxiv result: {text[:200]}"
    print("[ok] live arXiv call returned results:")
    print("    " + next((ln.strip() for ln in text.splitlines() if "arxiv.org/abs" in ln), text[:120]))

    print("\nMCP VERIFICATION PASSED")


if __name__ == "__main__":
    asyncio.run(main())
