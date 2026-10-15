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

---

# 任务 5：极简 Web 页（2026-10-13 新增，交低端模型）

> 目标：一个**单文件、无构建步骤**的演示页面，让人能在浏览器里走完  
> 「输入课题 → 看计划 → 审批 → 看流式报告」全流程。面试演示用。
>
> **状态：已由主模型实现并验收通过（2026-10-14）**，文件 `src/deep_research/static/index.html`。  
> 验收脚本 `scripts/verify_web_e2e.py`，sqlite 与 Postgres 两种后端各跑通一遍。  
> 契约保留作为回归依据；如需改动页面，按下方时序与帧格式实现。
>
> `api.py` 已改好（`GET /` 返回该文件、error 帧转发 message、中断帧补发），**禁止再动后端**。

## 启动方式

```bash
python -m deep_research.serve --port 8000     # 不要用 uvicorn CLI（见下）
# 浏览器打开 http://127.0.0.1:8000/
```

Windows + Postgres 下 `uvicorn deep_research.api:app` **不可用**：uvicorn 对 win32  
写死 ProactorEventLoop，psycopg async 会拒绝，必须走 `serve.py`（显式 selector loop）。

## API 时序（必须严格按此实现，已实测）

```
POST /research            {"topic": "..."}         -> {"thread_id": "..."}
GET  /research/{id}/events   (SSE, EventSource)    -> 事件帧，见下
POST /research/{id}/resume {"approved": true, "edited_plan": null}
GET  /research/{id}/events   (重新连接！每段流独立)
```

**关键时序坑（都已踩过）：**

1. 一次研究的 SSE 流分成**多段**：planner 中断前一段、resume 后一段。  
   每段以命名事件 `event: done` + `data: {}` 结束，前端要 `source.close()`。
2. 收到 `interrupt` 后**先关旧连接**，用户点审批后 **先 POST /resume，  
   再新建 EventSource** 连同一 URL（后端会换新队列，旧连接收不到后续事件）。
3. POST /research 后事件已在后端队列里缓冲，紧接着再开 EventSource 不会丢事件。

## SSE 帧格式（每帧 `data: {json}`，JSON 字段如下）

| 判定条件                                                  | 含义                                     | 前端动作              |
| ----------------------------------------------------- | -------------------------------------- | ----------------- |
| `json.interrupt` 存在                                   | planner 等待审批                           | 渲染计划 + 审批按钮，见下    |
| `json.event == "on_chat_model_stream"` 且 `json.token` | writer 报告 token                        | 追加到报告区            |
| `json.event == "on_chain_start"` 且 `json.node`        | 节点开始（planner/researcher/critic/writer） | 进度指示              |
| `json.event == "error"`                               | 后端异常                                   | 显示 `json.message` |
| SSE 命名事件 `done`（不是 data 里的字段）                         | 本段流结束                                  | `source.close()`  |

其他帧一律忽略。

> 实现说明：中断帧由后端在流末尾**主动补发**（`{"event":"interrupt","interrupt":{...}}`），  
> 因为本版本 langgraph 的 `astream_events` 输出里不含 `__interrupt__`，只有  
> checkpoint state 里有。前端只需认 `json.interrupt` 字段，无需关心来源。

**interrupt 载荷结构**（`json.interrupt`）：

```json
{"type": "approve_plan",
 "plan": {"topic": "...", "sub_questions": [{"id": "q1", "question": "...", "rationale": "..."}]}}
```

`recalled_memories`（字符串数组）可能存在，有就折叠展示「召回的历史记忆」。

## 页面要求

- 单文件 HTML，原生 JS + `EventSource`，**禁止任何构建工具/框架/npm**。
- Markdown 渲染：可用 CDN 的 marked（`<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js">`），  
  加载失败时降级为 `<pre>` 纯文本（判断 `window.marked` 是否存在）。
- 布局不限制，但必须同时可见三块：节点进度状态、报告流式输出区、（中断时的）计划审批区。
- 审批交互：「批准」按钮（POST 上述 resume body）；「编辑计划」为可选加分项  
  （textarea 编辑 JSON 后作为 `edited_plan` 提交），不做也不影响验收。
- 全程中文 UI；出错（网络断开、error 帧）要有可见提示，不允许静默卡死。

## 验收（我会照此逐条实测）

1. `uvicorn deep_research.api:app --port 8000` 后访问 `/` 返回 200 HTML。
2. 输入课题提交 → 能看到 planner 计划渲染 → 点「批准」→ writer 报告逐 token 出现 → 完整显示。
3. F12 网络面板里 SSE 连接在 `done` 后正常关闭，无悬挂连接、无控制台报错。
4. 后端进程日志无异常（页面引起的 404/500 都算失败）。
