"""Platform compatibility shims.

psycopg async (used by PostgresSaver and MemoryStore) cannot run on the
Windows ProactorEventLoop — the process must use the selector loop. Entry
points call use_compatible_event_loop() before asyncio.run().

ensure_hf_compat() fixes two Windows/CN-network hazards seen in practice:
- huggingface_hub's symlink probe can false-positive on Windows without
  Developer Mode: os.symlink "succeeds" but yields broken 0-byte links,
  silently corrupting the HF cache (observed 2026-10-13 with fastembed's
  bge-small-zh download). HF_HUB_DISABLE_SYMLINKS=1 forces plain copies.
- huggingface.co direct access is unreliable; default to the hf-mirror
  endpoint. hf_xet is disabled because its CAS backend
  (cas-server.xethub.hf.co) rejects requests routed via mirror/proxy with
  401; plain HTTP download works. All are setdefault — user env always wins.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


def ensure_hf_compat() -> None:
    """Make HuggingFace access work in every runtime, including spawned servers.

    两处曾经踩过的坑（2026-10-25 容器实测）：

    1. 这里原本只在 win32 生效，Linux 容器里什么都没设 → 去 huggingface.co
       （不可达）而不是镜像站，模型加载直接失败；
    2. MCP 的子进程是 ``stdio_client`` 起的，它用 ``get_default_environment()``
       只传一小撮环境变量，镜像里 ENV 设的 HF_ENDPOINT / FASTEMBED_CACHE_PATH
       全部丢失。所以这两项必须在**代码里**兜底，不能依赖环境继承。
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if not os.environ.get("FASTEMBED_CACHE_PATH"):
        baked = Path("/opt/fastembed")  # 镜像构建期预下载的位置
        os.environ["FASTEMBED_CACHE_PATH"] = (
            str(baked) if baked.exists() else str(Path.home() / ".cache" / "fastembed")
        )


def use_compatible_event_loop() -> None:
    ensure_hf_compat()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
