# Cite Trace

面向 **AI / 科技行业调研** 的多智能体深度研究系统：输入一个课题，自动拆解子问题 →  
并行取证（arXiv / GitHub / Web，经 MCP 协议接入）→ 审校证据 → 生成**每个事实都有引用、  
每个引用都经程序核验**的中文研究报告。

架构与选型详见 [deep-research/docs/架构文档与技术选型.md](deep-research/docs/架构文档与技术选型.md)；

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

## 两种信源模式（online / offline）

同一套编排、同一套审校、同一套引用核验，只换「Researcher 手上有哪些工具」：

| | `online`（默认） | `offline` |
|---|---|---|
| 信源 | arXiv / GitHub / Web（MCP） | **只有本地知识库**（`kb`） |
| 网络 | 必需 | **取证不需要外网**（信源全部来自本地知识库）；LLM 与 embedding 可走 API，也可本地部署 |
| 引用核验 | URL 存在性（arXiv API / GitHub API / HTTP） | **文件 + 章节**（文件在不在、引的那节在不在） |
| 证据不足时 | Critic 追问 → 再联网取证 | 追问**无效**，直接标为存疑项，报告写明「当前知识库未收录」 |

```bash
# 1) 入库（支持 .md / .txt；PDF 需另装解析器，暂未开放）
python scripts/ingest_kb.py docs/ reports/eval/reports/

# 2) 离线跑一道题
DR_SOURCE_MODE=offline \
DR_MCP_SERVERS='{"kb": {"transport": "stdio", "command": "python", "args": ["-m", "mcp_servers.kb_server"], "network": false}}' \
uv run python -m deep_research.cli run "ReAct 与 Chain-of-Thought 的关键区别"
```

**一个容易混淆的点**：`offline` 约束的是**信源**（不联网抓取），不是**算力**。
编排（Planner/Researcher/Critic/Writer）、审校、embedding 该调模型还是要调——只是
不联网找资料。两者是两个独立维度：

| 维度 | 取值 | 由谁控制 |
|---|---|---|
| 信源 | 联网抓取 / 本地知识库 | `DR_SOURCE_MODE` |
| 算力 | 本地模型 / 云端 API | `DR_EMBEDDING_PROVIDER`、各角色 model 配置 |

所以"内网部署"要做两件事：把信源换成本地库（`offline`）+ 把模型换成本地部署；
只做前者叫"取证离线"，不叫"完全离线"。

两条设计约束（不是可选项）：

- **模式互斥，不做混合**：离线时网络类 server 在装载前就被过滤掉，不是"禁用"——避免"以为离线其实在联网"。
- **库内无覆盖就说没有**：实测拿库外题目跑，覆盖率为 0%，但报告会逐条说明"哪两条沾边证据为何不成立"，而不是编一个看起来合理的答案。

## 评测：结果与方法

### 检索层基准（私有语料 · FinanceBench 子集）

私有语料没有公开基准，用 FinanceBench（真实 SEC 10-K + 专家标注 evidence）作同构代理，
21 题、571 页，只测检索层：

| 配置 | 页面命中@10 | 召回@50 |
|---|---|---|
| 本地 `bge-small-zh`（中文小模型） | 14% | 29% |
| 换多语言大模型（API） | 38% | 76% |
| ＋ rerank | 52% | 76% |
| ＋ query 扩展 | 48% | **90%** |
| **扩展 + 精排** | **57%** | — |

结论：**embedding 决定召回上限、query 扩展决定能否捞到、rerank 只决定排序**。
⚠️ 57% 是"页面命中"而非"答案正确率"（按 Oracle 85% 上限推算，端到端天花板约 48%）。
详见 [`docs/检索基准-FinanceBench.md`](deep-research/docs/检索基准-FinanceBench.md)。



**评测结果**（26 题 golden set 全量，AI/科技三子领域）：

| 指标       | 数值        | 怎么算的                                                          |
| -------- | --------- | ------------------------------------------------------------- |
| 引用存在率    | **100%**  | **程序核验**：每条引用按来源类型查 arXiv API / GitHub API / HTTP 可达性         |
| 论断-来源对齐率 | **96.5%** | LLM judge：把论断的脚注经参考列表映射到 URL，再与该 URL 的**实际证据**（要点 + 原文摘录）逐条比对 |
| 知识点覆盖率   | **100%**  | LLM judge：逐条核查 golden 里预设的、可证伪的知识点是否真的写进了报告                   |

分领域：agent（10 题）100% / 96.2% / 100% · llm（9 题）100% / 96.6% / 100% ·  
rag（7 题）100% / 97.0% / 100%。26 题全部出分，无缺测；  
明细见 `deep-research/reports/eval/eval-20261015-214611-merged-26q.md`。

评测流程与产物：`eval/runner.py` 自动批准计划跑图 → 三指标评测 → 聚合 JSON/Markdown 到 `deep-research/reports/eval/`；

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
- 每题的**报告与证据库都落盘**（`deep-research/reports/eval/reports/<qid>.{md,evidences.json}`），  
  可用 `scripts/rejudge_alignment.py` 离线复判，改 judge 不必重跑研究
- 修改提示词后用小样本对照（同题同条件重跑）验证效果，不靠单次跑分定论

## 工程记录（精选三条，完整版见 [docs/工程日志.md](deep-research/docs/工程日志.md)）

| 症状 | 根因 | 修复 |
|---|---|---|
| 长报告参考列表被截断且**不报错** | 服务端静默钳制 8192 输出上限，截断时 `finish_reason` 不报 `length`；靠 **Langfuse trace 反查**才发现（被截断样本输出 token 精确等于 8192） | 参考列表改由代码按编号生成，模型只写正文；超限分段续写 |
| 上下文裁剪后覆盖率 **100% → 25%** | 按空行切块，而 HTML 抓取的正文没有空行 → 整篇当成一块，只保留头部预算 | 改为按行切块 + 单行超长按句切；预算必须用满（覆盖率回到 100%，输入 token 同时 −22%） |
| 引用存在率从 **0–21% 误报**到 100% | 评测器四处误判：arXiv 空 feed 当缺失、GitHub API URL 取错仓库名、标题里的 `Graph-/Agentic` 被当本地路径、PDF 页码锚点不被识别 | 逐条修口径；**指标本身的 bug 比模型幻觉更危险**——它会让真实引用被记成幻觉 |

## 快速开始

```bash
# 0. 进入项目目录（本仓README在根目录，代码在 deep-research/ 下）
cd deep-research

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
cd deep-research         # 以下命令均在项目目录内执行
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
deep-research/
  src/deep_research/
  core/        # config（按角色模型 + 信源模式）/ llm（唯一 LLM 入口）/ persistence（检查点工厂）
               # memory（pgvector 长期记忆）/ kb（本地知识库：切块 + 三路融合检索）
               # tracing（Langfuse）/ usage（token 与成本）
               # compat（平台与依赖兼容垫片）/ serve（API 启动入口）
  agents/      # state / prompts / nodes（四角色实现）/ graph（图装配 + Send 扇出）
  tools/       # MCP 客户端加载与降级
  eval/        # citations（程序核验）/ judge（LLM judge，含 meta 判定）/ metrics / runner
               # scores（分数回挂 Langfuse）/ golden 数据集（tech_ai 26 题 + holdout 6 题）
  cli.py       # Typer CLI
  api.py       # FastAPI + SSE（含 HITL 恢复）
  static/      # 单文件演示页（零构建）
mcp_servers/   # arxiv_server / github_server / web_server / kb_server（离线模式唯一工具）
scripts/       # 见下表
docs/          # 架构文档与技术选型
```

## 验证脚本


| 脚本                                           | 作用                                            | 是否需 API key |
| -------------------------------------------- | --------------------------------------------- | ----------- |
| `deep-research/scripts/smoke_test.py`        | mock LLM 驱动真实图：中断→恢复、并行合并、critic 回环、writer 成文 | 否           |
| `deep-research/scripts/smoke_eval.py`        | 假产物跑通评测流水线（核验 + judge + 聚合 + 报告）              | 否           |
| `deep-research/scripts/verify_mcp.py`        | 经核心加载器加载 MCP 工具并真实调用一次                        | 否           |
| `deep-research/scripts/verify_tracing.py`    | 证明 run config 的 callbacks 能传导到节点内 LLM 与工具调用   | 是           |
| `deep-research/scripts/verify_langfuse.py`   | 用本地假服务端验证上报链路（不依赖真实 Langfuse）                 | 是           |
| `deep-research/scripts/verify_m3.py`         | Postgres 检查点 + pgvector 记忆三步联调（含 planner 召回）  | 是           |
| `deep-research/scripts/verify_web_e2e.py`    | 按页面逻辑走完整 API 时序（中断 → 恢复 → 重连 SSE）             | 是           |
| `deep-research/scripts/rejudge_alignment.py` | 用已落盘证据库离线复判，不重跑研究                             | 是           |
| `deep-research/scripts/refresh_alignment.py` | 修正 judge 后对全部题目离线复判并刷新聚合口径（仅花 judge 调用，不重跑研究） | 是           |
| `deep-research/scripts/verify_injection.py`  | 注入护栏 5 条：围栏不可伪造、中英检测、不误伤讨论类文本、工具输出加围栏、真实模型抗注入 | 否（`--live` 时是） |
| `deep-research/scripts/verify_context.py`    | 上下文压缩：结果表格存活、短文档不动、头部必留、压缩比 | 否 |
| `deep-research/scripts/verify_citations.py`  | 引用格式归一化：`[n]`→`[^n]`、代码块与链接不动、越界编号不转 | 否 |
| `deep-research/scripts/verify_metrics.py`    | 指标 N/A 语义：无判论断不进均值、渲染 N/A、全空不崩 | 否 |
| `deep-research/scripts/verify_kb.py`         | 知识库检索质量（recall@1 / MRR）+ 已知查询回归（含"测试饱和"告警） | 否 |
| `deep-research/scripts/verify_offline_mode.py` | 离线模式：只装载本地 server、无本地 server 时显式失败、默认仍 online、kb 引用核验 | 否 |
