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


def ensure_hf_compat() -> None:
    if sys.platform == "win32":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def use_compatible_event_loop() -> None:
    ensure_hf_compat()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
