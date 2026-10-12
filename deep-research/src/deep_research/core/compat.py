"""Platform compatibility shims.

psycopg async (used by PostgresSaver and MemoryStore) cannot run on the
Windows ProactorEventLoop — the process must use the selector loop. Entry
points call use_compatible_event_loop() before asyncio.run().
"""

from __future__ import annotations

import asyncio
import sys


def use_compatible_event_loop() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
