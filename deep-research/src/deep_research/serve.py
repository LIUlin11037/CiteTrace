"""API entry point that works on Windows with the Postgres checkpointer.

`uvicorn deep_research.api:app` is NOT usable here on Windows: uvicorn's
asyncio loop factory hard-codes ProactorEventLoop for win32
(uvicorn/loops/asyncio.py), and psycopg async refuses to run on Proactor.
This module starts uvicorn programmatically with our own selector loop
factory, which api.py cannot influence because uvicorn imports the app
inside the loop it already created.

    python -m deep_research.serve [--host 127.0.0.1] [--port 8000] [--reload]

Then open http://127.0.0.1:8000/
"""

from __future__ import annotations

import argparse
import asyncio
import selectors
import sys

from deep_research.core.compat import use_compatible_event_loop


def _loop_factory() -> asyncio.AbstractEventLoop:
    # WindowsSelectorEventLoopPolicy only affects asyncio.run's default
    # factory on older Pythons; passing the factory explicitly is version-proof.
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Deep Research API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="dev auto-reload")
    args = parser.parse_args()

    use_compatible_event_loop()  # HF/memory env defaults

    import uvicorn

    if args.reload:
        # reload spawns a subprocess that re-imports this module — the loop
        # factory is only applied in the parent, so reload is unsupported here.
        print("[serve] --reload is not supported on Windows + Postgres; ignoring")

    config = uvicorn.Config(
        "deep_research.api:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    if sys.platform == "win32":
        asyncio.run(server.serve(), loop_factory=_loop_factory)
    else:
        asyncio.run(server.serve())


if __name__ == "__main__":
    main()
