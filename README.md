# Cite Trace

面向 **AI / 科技行业调研** 的多智能体深度研究系统：输入一个课题，自动拆解子问题 →  
并行取证（arXiv / GitHub / Web，经 MCP 协议接入）→ 审校证据 → 生成**每个事实都有引用、  
每个引用都经程序核验**的中文研究报告。

架构与选型详见 [docs/架构文档与技术选型.md](docs/架构文档与技术选型.md)；

## 核心能力

| 能力                | 说明                                                                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 多智能体编排            | Planner 拆题 → Researcher×N（LangGraph Send API 并行扇出）→ Critic 审校（回环追问，最多 2 轮）→ Writer 成文                                                     |
| Human-in-the-loop | Planner 产出计划后中断，人工批准才继续；支持编辑计划后恢复（CLI / Web 双入口）                                                                                          |
| MCP 工具生态          | arXiv / GitHub / Web 三个独立 stdio server，经 `langchain-mcp-adapters` 加载，**单 server 故障不影响整体**                                                 |
| 按角色模型路由           | 每个角色独立配置 model / base_url / api_key，可跨厂商：把贵模型用在"调用少、质量敏感"的环节（如 Critic 审校），把便宜模型用在"费 token 的跑量"环节（如 Researcher）                            |
| 长期记忆              | PostgreSQL + pgvector（HNSW），Writer 落库摘要与来源清单，Planner 下次召回（本地 fastembed BGE-small-zh 嵌入）                                                   |
| 可观测性              | Langfuse：一次调研一条 trace（node / LLM / MCP 工具三层 span，含 token 与成本），session 归并中断恢复产生的多段执行，tags 标注入口与题目；**评测指标自动回挂为 trace score**，可按分数排序并下钻到该次运行 |
| 评测体系              | 26 题 golden set + 程序化引用核验 + LLM judge（详见下节）                                                                                               |

## 评测：结果与方法

**评测结果**（26 题 golden set 全量，AI/科技三子领域）：

| 指标       | 数值        | 怎么算的                                                          |
| -------- | --------- | ------------------------------------------------------------- |
| 引用存在率    | **100%**  | **程序核验**：每条引用按来源类型查 arXiv API / GitHub API / HTTP 可达性         |
| 论断-来源对齐率 | **96.5%** | LLM judge：把论断的脚注经参考列表映射到 URL，再与该 URL 的**实际证据**（要点 + 原文摘录）逐条比对 |
| 知识点覆盖率   | **100%**  | LLM judge：逐条核查 golden 里预设的、可证伪的知识点是否真的写进了报告                   |

分领域：agent（10 题）100% / 96.2% / 100% · llm（9 题）100% / 96.6% / 100% ·  
rag（7 题）100% / 97.0% / 100%。26 题全部出分，无缺测；  
明细见 `reports/eval/eval-20261015-214611-merged-26q.md`。

评测流程与产物：`eval/runner.py` 自动批准计划跑图 → 三指标评测 → 聚合 JSON/Markdown 到 `reports/eval/`；

每题报告与证据库落盘，改 judge 后可用 `scripts/refresh_alignment.py` **离线复判刷新口径**（不重跑研究）；

评测完成后三个指标会**自动回挂到该次运行的 Langfuse trace**（`eval/scores.py`）。

> 口径说明：对齐率为**修正 judge 后统一离线复判**（识别「方法性说明」并排除，见迭代记录）；  
> 覆盖与引用率取每题最近一次运行，其中 2 道题在抓取上限调整后重跑。

**成本事实**（同一轮实测）：**26 题共 ¥11.05**、3521 次 LLM 调用、输入 15.3M tokens  
（其中缓存命中 8.2M）、输出 1.91M tokens，**单题约 ¥0.43**  
（deepseek-chat，空闲时段价：输入 ¥1/M、缓存命中 ¥0.02/M、输出 ¥2/M）。  
把单条抓取上限从 4000 调到 12000 后，长文类问题的输入 token 约 +60%。

设计原则：**能程序化核验的绝不用 LLM 判断**。LLM 只做两件代码做不了的事（论断对齐、  
要点覆盖），其余全部落到确定性代码：

- 引用核验三态判定 `exists / missing / unverifiable`——网络失败算「无法核验」，  
  **绝不让网络问题冒充幻觉**，且该情形不计入指标分母
- 论断与其引用来源的匹配走「脚注 → 参考列表 URL → 归一化（去 arXiv 版本号 / 尾斜杠 / `.git`）  
  → 证据库查找」，**不做位置映射**
- 每题的**报告与证据库都落盘**（`reports/eval/reports/<qid>.{md,evidences.json}`），  
  可用 `scripts/rejudge_alignment.py` 离线复判，改 judge 不必重跑研究
- 修改提示词后用小样本对照（同题同条件重跑）验证效果，不靠单次跑分定论

## 工程迭代记录（真实问题，附根因）

每一个都是实测暴露、定位到根因、修复并复测的。挑几个有代表性的：

| 症状                                                                       | 根因                                                                                          | 修复                                                                                                                 |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 对齐率仅 38%，大量"编造论断"误报                                                      | judge 把论断编号**按位置**映射到证据列表                                                                   | 改为按论断自身引用的 URL 在证据库中查找（含 URL 归一化），38% → 92%                                                                        |
| Web 端永远等不到"等待审批"，HITL 链路实际从未打通                                           | 本版 langgraph 的 `astream_events` 输出**不含 `__interrupt__`**（只有 `ainvoke` 与 checkpoint state 有） | 流结束后读 `graph.aget_state().tasks[].interrupts`，补发专用中断帧                                                              |
| Windows + Postgres 下 API 起不来                                             | uvicorn 对 win32 写死 `ProactorEventLoop`，且显式 `loop_factory` 优先级高于事件循环策略，psycopg async 拒绝      | `serve.py` 绕开 `Server.run()`，自行 `asyncio.run(server.serve(), loop_factory=SelectorEventLoop)`                      |
| 依赖模型下载后加载失败                                                              | 无开发者模式的 Windows 上 `os.symlink` 探测假阳性，缓存被写成 0 字节                                             | 统一设置 `HF_HUB_DISABLE_SYMLINKS=1` / `HF_HUB_DISABLE_XET=1`（`core/compat.py`）                                        |
| 一次网络抖动丢掉 17/26 道题                                                        | `APIConnectionError` 不带状态码，被判为永久错误**不重试**                                                   | 重试谓词纳入 `httpx.TransportError` 及其 `__cause__` 链，重试 5 次指数退避                                                          |
| 长报告参考列表被截断，且**不报错**                                                      | 服务端对 `max_tokens` 静默钳制在 ~8192，且截断时 `finish_reason` 不报 `length`；来源多的题正文+来源列表写不完              | 双管齐下：①参考列表**由代码按给定编号确定性生成**，模型只写正文；②按 `finish_reason` 判定被截断则**分段续写并拼接**                                            |
| 容器内所有 MCP server 启动失败（`FileNotFoundError: .../.venv/Scripts/python.exe`） | 为在宿主上跑通而在 `.env` 里写了**解释器绝对路径**，容器里根本不存在该路径；同一份 `.env` 不可能同时适配宿主与容器                         | `mcp_client._resolve_command()`：裸解释器名（`python`/`python3`/`py`）或本地不存在的路径，一律回退到**当前进程的解释器** `sys.executable`，并对回退打警告 |
| 评测样本被静默缩小（一次网络故障后摘要只剩 9 题）                                               | runner 失败即跳过                                                                                | 两轮执行：并发跑一轮 + 失败题串行重试；全部失败则直接抛错，不写空摘要                                                                               |
| 对齐率被「报告在说明自己的取舍」拉低（gq-long-context 87.5%、gq-reranker 83.3%）              | judge 把「本次未引用 X」「来源与主题无关」这类**方法性说明**也当成事实论断去核对齐                                             | judge 新增 `meta` 判定并从对齐率分母中排除；离线复判后两题分别回到 **95% / 100%**                                                            |
| 某题覆盖率卡在 50%，反复重跑不变                                                       | 单条抓取正文被截断到 4000 字符，**论文结果表格在长文后段从未进入上下文**，Writer 只能如实写「无量化证据」                               | 抓取上限提到 12000（`DR_EVIDENCE_CONTENT_LIMIT`），同题覆盖率 50% → **100%**，长文题输入 token 约 +60%                                  |

定位这些问题的过程本身也是方法：例如"服务端静默钳制 8192"这条，是通过  
**Langfuse trace 反查**发现的——同一轮 3011 次调用里，被截断样本的输出 token  
**精确等于 8192**，而正常样本只有 2～4k，由此锁定上限；失败报告会写入  
`reports/eval/rejected/` 供事后分析。

## 快速开始

```bash
# 1. 依赖（需 uv；也可 python -m venv .venv && pip install -e ".[cli,serve,observe]"）
uv sync --extra cli --extra serve

# 2. 配置
cp .env.example .env   # 必填 DR_DEFAULT_API_KEY；可选 GITHUB_TOKEN / TAVILY_API_KEY / DR_LANGFUSE_*

# 3. CLI 运行（Planner 计划会交互式等你确认）
uv run python -m deep_research.cli run "Agentic RAG 最新进展"

# 4. 或起 Web 服务（含单文件演示页，浏览器打开 http://127.0.0.1:8000/）
uv run python -m deep_research.serve --port 8000

# 5. 跑评测
uv run python -c "import asyncio; from deep_research.eval.runner import run_eval; asyncio.run(run_eval())"
```

检查点与长期记忆统一使用 PostgreSQL（**默认后端**）：`uv sync --extra pg`，起 `docker compose up -d postgres`，

`.env` 里设 `DR_ENABLE_LONG_TERM_MEMORY=true` 即可。仅当本地确实不想起数据库时，才显式设 `DR_CHECKPOINT_BACKEND=sqlite`（零依赖后备）。

### Docker 部署（推荐路径：app + postgres）

```bash
cp .env.example .env     # 填 DR_DEFAULT_API_KEY
docker compose up -d     # 起 postgres(pgvector) + app，app 会等 postgres 健康后再启动
curl http://127.0.0.1:8000/healthz          # {"status":"ok"}
# 浏览器打开 http://127.0.0.1:8000/ 即可用演示页
```

容器内固定使用 Postgres（检查点 + pgvector 记忆同库），compose 会覆盖 `.env` 里的  
`DR_DATABASE_URL`（容器内要连服务名 `postgres` 而非 `localhost`）。可观测性走  
Langfuse Cloud，无需额外容器。

构建默认使用**清华 PyPI 镜像**（国内网络下官方源要十几分钟）；海外或内网可覆盖：

`UV_INDEX_URL=https://pypi.org/simple docker compose build`。依赖层带 BuildKit 缓存挂载，

改代码重建为秒级、不再联网重下。

**已实测**：`docker compose up -d` → 两个容器均 healthy → 容器内跑通完整调研  
（计划中断 → 审批 → 4 个子问题并行取证 → 审校 → 19k 字符报告），检查点写入  
Postgres、长期记忆写入 pgvector（512 维）、MCP 工具在容器内正常调用。

> **Windows 注意**：Postgres 后端下不要用 `uvicorn deep_research.api:app`（见上表第 3 条），  
> 用 `serve.py`。

## 目录结构

```
src/deep_research/
  core/        # config（按角色模型）/ llm（唯一 LLM 入口）/ persistence（检查点工厂）
               # memory（pgvector 长期记忆）/ tracing（Langfuse）/ usage（token 与成本）
               # compat（平台与依赖兼容垫片）/ serve（API 启动入口）
  agents/      # state / prompts / nodes（四角色实现）/ graph（图装配 + Send 扇出）
  tools/       # MCP 客户端加载与降级
  eval/        # citations（程序核验）/ judge（LLM judge，含 meta 判定）/ metrics / runner
               # scores（分数回挂 Langfuse）/ golden 数据集
  cli.py       # Typer CLI
  api.py       # FastAPI + SSE（含 HITL 恢复）
  static/      # 单文件演示页（零构建）
mcp_servers/   # arxiv_server / github_server / web_server
scripts/       # 见下表
docs/          # 架构文档与技术选型
```

## 验证脚本


| 脚本                             | 作用                                            | 是否需 API key |
| ------------------------------ | --------------------------------------------- | ----------- |
| `scripts/smoke_test.py`        | mock LLM 驱动真实图：中断→恢复、并行合并、critic 回环、writer 成文 | 否           |
| `scripts/smoke_eval.py`        | 假产物跑通评测流水线（核验 + judge + 聚合 + 报告）              | 否           |
| `scripts/verify_mcp.py`        | 经核心加载器加载 MCP 工具并真实调用一次                        | 否           |
| `scripts/verify_tracing.py`    | 证明 run config 的 callbacks 能传导到节点内 LLM 与工具调用   | 是           |
| `scripts/verify_langfuse.py`   | 用本地假服务端验证上报链路（不依赖真实 Langfuse）                 | 是           |
| `scripts/verify_m3.py`         | Postgres 检查点 + pgvector 记忆三步联调（含 planner 召回）  | 是           |
| `scripts/verify_web_e2e.py`    | 按页面逻辑走完整 API 时序（中断 → 恢复 → 重连 SSE）             | 是           |
| `scripts/rejudge_alignment.py` | 用已落盘证据库离线复判，不重跑研究                             | 是           |
| `scripts/refresh_alignment.py` | 修正 judge 后对全部题目离线复判并刷新聚合口径（仅花 judge 调用，不重跑研究） | 是           |

## 已知限制

- 评测数字为 26 题 golden set 的口径；`deepseek-chat` 单次响应输出约 8192 token 上限，  
  超长报告靠分段续写，极端来源量下仍可能受影响
- 检索质量依赖搜索后端（Tavily / Brave / DuckDuckGo 逐级降级），无 key 时走 DuckDuckGo
- 可观测性依赖 Langfuse Cloud；仓库不含自建 Langfuse 编排（v3 自建需额外  
  ClickHouse + MinIO + Redis + worker 四个组件）
