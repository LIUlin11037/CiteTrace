# 外围模块交接契约（Handoff）

> **状态：任务 1–4 已全部实现并验证通过（2026-10-10）。**  
> 核心层（core/ agents/ tools/mcp_client.py）**禁止修改**。  
> 本文档保留作为接口说明与回归依据；如需迭代外围，按下方契约扩展。

## 实现与验证结果

| 模块            | 文件                                                   | 验证                                                               |
| ------------- | ---------------------------------------------------- | ---------------------------------------------------------------- |
| 三个 MCP server | `mcp_servers/{arxiv,github,web}_server.py`           | `scripts/verify_mcp.py` 经核心加载器实测：6 工具齐全，arXiv 真实调用 200           |
| CLI           | `src/deep_research/cli.py`                           | import 正常，`run` 命令注册；含 interrupt 审批 + 编辑计划                       |
| FastAPI + SSE | `src/deep_research/api.py`                           | 路由齐全：`/research`、`/research/{id}/events`、`/research/{id}/resume` |
| 部署            | `.env.example` / `Dockerfile` / `docker-compose.yml` | 与 config 字段对齐                                                    |

待真实 `DR_DEFAULT_API_KEY` 后即可端到端联调（唯一未跑的是真实 LLM 调用）。

---

## 已固定的核心接口（只读，直接 import）

```python
from deep_research.core.config import get_settings, Settings
from deep_research.agents.graph import build_graph, new_thread_config, initial_input
from deep_research.agents.state import ResearchState, ResearchPlan, Evidence, CritiqueResult
```

- `build_graph(checkpointer)` → 编译后的图；checkpointer **必传**（开发用 `langgraph.checkpoint.sqlite.SqliteSaver`）。
- 启动一次研究：`graph.ainvoke(initial_input(topic), config=new_thread_config())`
- 图会在 planner 处 `interrupt`。**注意（已实测）**：带 checkpointer 时 `ainvoke` 不抛异常，  
  而是返回含 `result["__interrupt__"]` 的字典，取 `result["__interrupt__"][0].value`  
  得到 `{"type": "approve_plan", "plan": {...}}`；  
  用 `graph.ainvoke(Command(resume={"approved": True, "edited_plan": None}), config=同一config)` 恢复。  
  `Command` 从 `langgraph.types` 导入。流式场景同理监听 `__interrupt__` 事件。
- 事件流：`graph.astream_events(..., version="v2")` 供 SSE/CLI 订阅。

## 任务 1：CLI（`src/deep_research/cli.py`）— 依赖 extra `cli`

- Typer 命令 `run <topic>`：打印 Planner 计划 → 提示用户确认（y / 可粘贴编辑后的 JSON）→ Rich Live 渲染各节点进度 → 输出 Markdown 报告到 stdout 与 `reports/<slug>.md`。
- 只通过上述核心接口驱动图，不得自行实例化节点。

## 任务 2：FastAPI + SSE（`src/deep_research/api.py`）— 依赖 extra `serve`

- `POST /research` `{topic}` → `{thread_id}`；`GET /research/{thread_id}/events` SSE 转发 `astream_events`；`POST /research/{thread_id}/resume` 传审批决定。
- 进程内 `dict[str, CompiledGraph]` 管理会话即可，不做鉴权（M2 范围）。

## 任务 3：三个 MCP server（独立包 `mcp_servers/`，不进核心）

| Server             | 工具                                                          | 数据源                                   |
| ------------------ | ----------------------------------------------------------- | ------------------------------------- |
| `arxiv_server.py`  | `search_papers(query, max_results)` / `get_paper(arxiv_id)` | arXiv API（export.arxiv.org/api/query） |
| `github_server.py` | `search_repos(query)` / `get_repo_detail(owner/repo)`       | GitHub REST（token 走环境变量）              |
| `web_server.py`    | `web_search(query)` / `fetch_page(url)`                     | Brave/Tavily 搜索 API + 正文抽取            |

- 用 `mcp.server.fastmcp.FastMCP`（官方 SDK，stdio 传输），每个工具返回纯文本或 JSON 字符串。
- 每个 server 可独立 `python -m mcp_servers.arxiv_server` 启动。

## 任务 4：环境样例与部署

- `.env.example`：列出 `DR_DEFAULT_BASE_URL / DR_DEFAULT_API_KEY / DR_DEFAULT_MODEL / DR_*_MODEL / DR_MCP_SERVERS` 并附注释示例（对应 `core/config.py` 字段，字段名以代码为准）。
- `Dockerfile` + `docker-compose.yml`（M4）：app + postgres(pgvector) + redis + langfuse。

## 验收

1. `uv sync --extra cli --extra serve` 后 `python -m deep_research.cli run "Agentic RAG 最新进展"` 全流程走通（含人工审批中断恢复）。
2. SSE 接口逐 token 输出 writer 内容。
3. 三个 MCP server 各自独立启动并通过 `tools/mcp_client.get_mcp_tools()` 加载。
