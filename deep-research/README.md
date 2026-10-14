# Deep Research Agent

面向 **AI / 科技行业调研** 的多智能体深度研究系统：输入一个课题，自动拆解子问题 →
并行取证（arXiv / GitHub / Web，经 MCP 协议接入）→ 审校证据 → 生成带引用的中文报告。

架构详见 [docs/架构文档与技术选型.md](docs/架构文档与技术选型.md)。

## 快速开始

```bash
# 1. 依赖（需 uv；也可 python -m venv .venv && pip install -e ".[cli,serve]"）
uv sync --extra cli --extra serve

# 2. 配置
cp .env.example .env   # 填入 DR_DEFAULT_API_KEY，按需填 GITHUB_TOKEN / TAVILY_API_KEY

# 3. CLI 运行
uv run python -m deep_research.cli run "Agentic RAG 最新进展"

# 4. 或起 Web 服务（含极简演示页，浏览器打开 http://127.0.0.1:8000/）
uv run python -m deep_research.serve --port 8000
```

> Windows + Postgres 后端下请勿使用 `uvicorn deep_research.api:app`：uvicorn 对 win32
> 写死 ProactorEventLoop，psycopg async 无法运行；`serve.py` 会显式使用 selector loop。
> 演示页为单文件零构建（`src/deep_research/static/index.html`），
> 端到端验收脚本：`python scripts/verify_web_e2e.py`（需服务已启动）。

## 结构

```
src/deep_research/
  core/     # 配置（按角色模型）与 LLM 抽象（唯一入口）
  agents/   # 状态机：Planner/Researcher/Critic/Writer + 图装配
  tools/    # MCP 工具加载
  cli.py    # Typer CLI
  api.py    # FastAPI + SSE
mcp_servers/  # arXiv / GitHub / Web 三个独立 MCP server
docs/         # 架构文档与交接契约
scripts/      # smoke_test.py（mock LLM 驱动真实图）
```

## 测试

```bash
uv run python scripts/smoke_test.py   # 无需 API key，验证图结构
```
