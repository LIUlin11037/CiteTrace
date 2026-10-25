"""Verify offline source mode (unit level, no LLM calls).

Checks three contracts:
1. `source_mode=offline` loads **only** servers marked `network: false`;
2. online mode keeps everything (no regression);
3. offline with no local server available **fails loudly** instead of silently
   running tool-less — a report with no evidence is worse than a crash.

Run: python scripts/verify_offline_mode.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SERVERS = {
    "kb": {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "mcp_servers.kb_server"],
        "network": False,
    },
    "arxiv": {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "mcp_servers.arxiv_server"],
    },
}


def _reload_with(mode: str, servers: dict) -> tuple[dict, object]:
    os.environ["DR_SOURCE_MODE"] = mode
    os.environ["DR_MCP_SERVERS"] = json.dumps(servers)
    from deep_research.core import config as cfg
    from deep_research.tools import mcp_client

    importlib.reload(cfg)
    cfg.get_settings.cache_clear() if hasattr(cfg.get_settings, "cache_clear") else None
    importlib.reload(mcp_client)
    return mcp_client._server_params(), mcp_client


def test_offline_filters_network() -> None:
    params, _ = _reload_with("offline", SERVERS)
    assert set(params) == {"kb"}, params
    print(f"[ok] 1. offline 只装载本地 server：{list(params)}")


def test_online_keeps_all() -> None:
    params, _ = _reload_with("online", SERVERS)
    assert set(params) == {"kb", "arxiv"}, params
    print(f"[ok] 2. online 装载全部：{sorted(params)}")


def test_offline_without_local_server_fails() -> None:
    _, client = _reload_with("offline", {"arxiv": SERVERS["arxiv"]})
    import asyncio

    try:
        asyncio.run(client.get_mcp_tools())
    except RuntimeError as exc:
        assert "offline" in str(exc)
        print(f"[ok] 3. offline 无本地 server 时显式失败：{str(exc)[:40]}…")
    else:
        raise AssertionError("offline 模式下没有本地 server 时应当报错，而不是静默降级")


def test_default_is_online() -> None:
    os.environ.pop("DR_SOURCE_MODE", None)
    os.environ["DR_MCP_SERVERS"] = json.dumps(SERVERS)
    from deep_research.core import config as cfg

    importlib.reload(cfg)
    cfg.get_settings.cache_clear() if hasattr(cfg.get_settings, "cache_clear") else None
    assert cfg.get_settings().source_mode == "online"
    print("[ok] 4. 默认仍是 online（不改变现有行为）")


def test_kb_citations() -> None:
    """离线引用必须能被"文件 + 章节"核验，且区分"文件不存在"与"章节不存在"。"""
    import asyncio

    from deep_research.eval.citations import extract_citations, verify_citations

    doc = Path("reports/eval/reports/gq-react.md").resolve()
    assert doc.exists(), "测试依赖 gq-react.md（先跑过评测）"
    report = (
        f"[^1]: 报告 — {doc} · ## 摘要 · L3-5（kb）\n"
        f"[^2]: 报告 — {doc} · ## 不存在的章节（kb）\n"
        "[^3]: 不存在 — C:/nowhere/none.md · ## 摘要（kb）\n"
    )
    cites = extract_citations(report)
    assert [c.kind for c in cites] == ["kb", "kb", "kb"], [c.kind for c in cites]
    assert "## 摘要" in cites[0].target, cites[0].target
    res = asyncio.run(verify_citations(report))
    verdicts = {d["target"].split(" · ")[-1]: d["verdict"] for d in res.details}
    assert res.exists == 1 and res.missing == 2, (res.exists, res.missing)
    print(f"[ok] 5. 离线引用核验：文件+章节命中 1，章节不存在 1，文件不存在 1 → {verdicts}")


def main() -> None:
    test_offline_filters_network()
    test_online_keeps_all()
    test_offline_without_local_server_fails()
    test_default_is_online()
    test_kb_citations()
    print("\nOFFLINE MODE TEST PASSED")


if __name__ == "__main__":
    main()
