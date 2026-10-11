"""Standalone MCP servers for the deep-research system.

Each server is a self-contained stdio process exposing tools via the official
MCP SDK. They are launched by ``tools/mcp_client.py`` based on the
``DR_MCP_SERVERS`` config and are never imported by the core package.
"""

__all__ = ["arxiv_server", "github_server", "web_server"]
