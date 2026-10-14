"""CLI entry point: run a research task end to end.

    python -m deep_research.cli run "Agentic RAG 最新进展"

Handles the planner HITL interrupt (approve / edit-then-approve), streams node
progress to a Rich live view, and writes the final report to reports/<slug>.md.
Only drives the graph through the core public interface.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import typer
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from deep_research.agents.graph import build_graph, initial_input, new_thread_config
from deep_research.core.persistence import open_checkpointer
from deep_research.core.tracing import apply_tracing, flush_tracing

app = typer.Typer(add_completion=False, help="Deep Research Agent CLI")
console = Console()


def _slug(topic: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "-", topic).strip("-")
    return (s[:60] or "report") + ".md"


def _render_plan(draft: dict[str, Any]) -> None:
    lines = [f"[bold]{draft['topic']}[/bold]", ""]
    for sq in draft["plan"]["sub_questions"]:
        lines.append(f"  [cyan]{sq['id']}[/cyan]  {sq['question']}")
        lines.append(f"       [dim]{sq['rationale']}[/dim]")
    console.print(Panel("\n".join(lines), title="研究计划草案", border_style="cyan"))


async def _run(topic: str, output: Path | None) -> None:
    output = output or Path("reports") / _slug(topic)

    async with open_checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        config = new_thread_config()
        apply_tracing(config, tags=["cli"])

        with console.status("[bold green]Planner 正在拆解课题...") as status:
            result = await graph.ainvoke(initial_input(topic), config=config)

        interrupts = result.get("__interrupt__", [])
        if not interrupts:
            console.print("[yellow]图未在 Planner 处中断，直接完成。[/yellow]")
            _finish(result.get("report", ""), output)
            return

        draft = interrupts[0].value
        _render_plan(draft)

        edited: dict[str, Any] | None = None
        if Confirm.ask("批准该计划并开始研究？", default=True):
            pass
        else:
            raw = Prompt.ask("粘贴编辑后的子问题 JSON 数组（或直接回车放弃）", default="")
            if raw.strip():
                try:
                    edited = {"topic": topic, "sub_questions": json.loads(raw)}
                except json.JSONDecodeError as exc:
                    console.print(f"[red]JSON 解析失败：{exc}，将按原计划执行。[/red]")
            else:
                console.print("[red]已取消。[/red]")
                raise typer.Exit(code=1)

        status = console.status("[bold green]Researcher 并行取证中...")
        status.start()
        final = await graph.ainvoke(
            Command(resume={"approved": True, "edited_plan": edited}), config=config
        )
        status.stop()

        _finish(final.get("report", ""), output)


def _finish(report: str, output: Path) -> None:
    if not report:
        console.print("[red]未生成报告。[/red]")
        raise typer.Exit(code=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    console.print(Markdown(report))
    console.print(f"\n[green]报告已保存至 {output}[/green]")


@app.command()
def run(
    topic: str = typer.Argument(..., help="研究课题"),
    output: Path | None = typer.Option(None, "--output", "-o", help="报告输出路径"),
) -> None:
    """对给定课题执行一次深度研究。"""
    from deep_research.core.compat import use_compatible_event_loop

    use_compatible_event_loop()
    try:
        asyncio.run(_run(topic, output))
    finally:
        flush_tracing()


if __name__ == "__main__":
    app()
