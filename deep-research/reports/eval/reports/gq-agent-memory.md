# 2025 年以来 LLM Agent 记忆机制主流方案研究报告

## 摘要

本报告基于 40 条证据（arXiv 论文、GitHub 仓库、官方文档）梳理 2025 年以来 LLM Agent 记忆机制的主流方案。核心结论：

1. **学术侧已形成综述共识**：记忆被定位为区分 agent 与原始 LLM 的关键组件，是支撑 agent-environment 长期交互与自我演化的核心模块 [^1][^3]。
2. **技术路线可归为五类**：(a) 结构化笔记/知识网络（A-MEM 的 Zettelkasten 式链接）[^4]；(b) 时序知识图谱（Zep/Graphiti 的 validity window + episode provenance）[^6][^10]；(c) 生产级抽取-整合-检索管线（Mem0）[^5][^23]；(d) 操作系统式虚拟上下文管理（MemGPT/Letta）[^38][^33]；(e) 单文件/本地优先的轻量记忆层（memvid、agent-memory、nocturne_memory）[^9][^13][^14]。
3. **产业界产品化证据高度不均衡**：仅 Anthropic 一家给出可核验的具体数字与日期 [^18][^19]；Google、Microsoft 的检索正文中明确不含 Memory Bank / Foundry 的容量、配额或定价数字 [^20][^21][^22]；OpenAI 的记忆产品化动态完全缺失。
4. **基准可比性存在系统性缺口**：所有跨系统对比均为厂商自评，且 Zep 与 Mem0 从未在同一基准上对比 [^24][^25]；多个第三方 harness 星数仅 1–11、结果可信度无法确认 [^26][^27][^28][^29]。

---

## 一、概念界定：什么是 LLM Agent 的记忆

**记忆的标准定义（AI 系统视角）**：LLM 时代的记忆被定义为「AI 系统保留、回忆并使用过去交互信息以改进未来响应与交互的能力」[^2]。

**记忆在 agent 架构中的定位**：综述将记忆描述为区分 agent 与原始 LLM 的关键组件，是「使 agent 真正成为 agent」（making an agent truly an agent）的要素 [^3]。原始 LLM 通常在不与环境交互的情况下完成任务，而 LLM-based agents 通过附加模块增强在真实环境中的自我演化能力 [^3]。记忆被定位为支撑长期、复杂 agent-environment 交互的关键组件 [^3]，并支撑 agent 的自我演化（self-evolving）能力 [^1]。

**动机示例**：旅行规划 agent 需向票务网站发送订单请求并观察响应后再行动；个人助理 agent 需根据用户反馈调整行为以提供个性化响应 [^3]。

**分类框架**：2025 年综述提出基于三个维度（object 对象、form 形式、time 时间）与八个象限（eight quadrants）的记忆分类方法，并主张从人类记忆获得启发来构建更强的记忆系统 [^2]。该综述指出此前研究虽对记忆机制有详细描述，但缺乏系统梳理 LLM 驱动 AI 系统记忆与人类记忆之间关系的综述 [^2]。

**缺失组件论**：一篇 position paper 主张情节记忆（episodic memory）是长期 LLM Agent 缺失的关键组件，提出以情节记忆五个关键属性为中心的框架，认为这五个属性是自适应、上下文敏感行为的基础 [^7]。

---

## 二、学术侧主流方案

### 2.1 结构化笔记与知识网络：A-MEM

A-MEM 提出面向 LLM Agent 的 agentic memory 系统，可动态组织记忆，遵循 Zettelkasten 方法原则，通过动态索引与链接构建互联知识网络 [^4]。

- **写入策略**：新记忆加入时生成包含上下文描述、关键词、标签等结构化属性的 note [^4]。
- **链接/更新策略**：系统分析历史记忆以识别相关连接，在存在有意义相似性时建立链接；新记忆整合可触发对既有历史记忆的上下文表示与属性的更新（memory evolution）[^4]。
- **主张**：作者主张在六个基础模型上的实验相对现有 SOTA 基线取得更优提升（摘要级主张，未读到结果表格）[^4]。
- **开源**：代码发布于 https://github.com/WujiangXu/A-mem 与 https://github.com/WujiangXu/A-mem-sys [^4]。

### 2.2 时序知识图谱：Zep / Graphiti

Zep 是一个面向 AI Agent 的记忆层服务，核心组件为 Graphiti——一个时间感知（temporally-aware）的知识图谱引擎，可动态综合非结构化对话数据与结构化业务数据并保留历史关系 [^6]。

Graphiti 的图结构定义 [^10]：
- 每个 fact 带 validity window：何时变为真、何时（若曾）被取代；
- 实体带随时间演化的 summary；
- 一切可追溯到 episodes（原始数据）；
- 图组件包括 Entities(nodes)、Facts/Relationships(edges，三元组+时序有效窗口)、Episodes(provenance)、Custom Types(ontology，经 Pydantic 模型定义)；
- 检索为 hybrid retrieval：semantic + keyword + graph traversal；
- 支持增量数据更新、高效检索与精确历史查询，无需完整图重算 [^10]。

作者指出既有面向 LLM Agent 的 RAG 框架局限于静态文档检索，而企业应用需要来自持续对话与业务数据的动态知识整合 [^6]。

**评测主张**：在 Deep Memory Retrieval (DMR) 基准上，Zep 报告 94.8% 对比 MemGPT 的 93.4%；在 LongMemEval 基准上，Zep 报告准确率提升最高达 18.5%，同时相比基线实现降低 90% 的响应延迟 [^6]。该论文由 Zep 团队作者撰写，属厂商自评而非第三方独立评测 [^25]。

### 2.3 生产级抽取-整合-检索管线：Mem0

Mem0 提出以记忆为中心的架构，从持续对话中动态提取、整合（consolidate）与检索显著信息，以应对固定上下文窗口的限制 [^5]。增强变体使用基于图的记忆表示，以捕捉对话元素间的复杂关系结构 [^5]。

**2026 年 4 月新算法**（README 标题为 'New Memory Algorithm (April 2026)'）[^23]：
- Single-pass ADD-only extraction：一次 LLM 调用，无 UPDATE/DELETE，记忆只累积不覆盖 [^8][^23]；
- Entity linking：实体被抽取、嵌入并跨记忆链接以提升检索 [^8][^23]；
- Multi-signal retrieval：semantic、BM25 keyword、entity matching 并行打分后融合 [^8][^23]；
- Temporal Reasoning：时间感知检索，为当前状态/过去事件/未来计划查询排序正确的带日期实例 [^8][^23]。

**存储组件**：内置支持多种向量库，未配置时默认使用 Qdrant；Python 实现支持 Qdrant、Chroma、PGVector、Pinecone、Oracle 等；TypeScript 实现支持 Qdrant、Redis、PGVector、Supabase、Milvus、Neptune Analytics、Amazon S3 Vectors、内存存储等；默认 embedding 维度为 1536，可用 embedding_model_dims 配置为其他维度（如 768）[^8]。

**评测主张**：论文在 LOCOMO 基准上评测，对比六类基线（既有记忆增强系统、不同 chunk size 与 k 值的 RAG、处理全部对话历史的全上下文方法、开源记忆方案、专有模型系统、专用记忆管理平台）[^5]。摘要声称在 single-hop、temporal、multi-hop、open-domain 四类问题上持续优于所有现有记忆系统；Mem0 在 LLM-as-a-Judge 指标上相对 OpenAI 取得 26% 相对提升；带图记忆的 Mem0 总体分数比基础配置高约 2%；相比全上下文方法，Mem0 的 p95 延迟降低 91%，token 成本节省超过 90% [^5]。该论文属厂商自评 [^24]。

### 2.4 操作系统式虚拟上下文管理：MemGPT → Letta

MemGPT 提出 virtual context management 技术，灵感来自传统操作系统的分层内存系统（hierarchical memory systems），通过在快慢内存之间移动数据，营造出大内存资源的表象 [^38]。MemGPT 智能管理不同内存层级（memory tiers），在 LLM 有限上下文窗口内有效提供扩展上下文，并使用 interrupts 管理自身与用户之间的控制流 [^38]。论文在两个受上下文窗口限制严重影响的领域评测 MemGPT：文档分析（可分析远超底层 LLM 上下文窗口的大文档）与多会话聊天（multi-session chat，可创建在长期交互中记忆、反思、动态演化的对话 agent）[^38]。

**项目演进**：letta-ai/letta 仓库 README 声明当前源码已迁移至 letta-ai/letta-code，后者包含 agent harness、交互式终端 UI、App Server、channels 及 desktop/web 应用使用的 runtime；原 letta 仓库的 V1 API server 已退役，位于 archive 分支 [^12]。archive 分支 AGENTS.md 标题声明「This repository is deprecated」（本仓库已弃用），该仓库处于维护模式（maintenance mode）[^32]。letta-ai/letta-code 被描述为「stateful agent harness」，其 agent 具有 memory、identity 和随时间积累的经验感，通过重写自身 memory、skills、prompts 乃至 harness（通过 mods）在长周期内学习与演化 [^33]。官方文档将 Letta 定位为「构建 stateful agents 的平台」，agent harness 完全开源，基于其实验室在 AI memory 和 continual learning 的最新研究 [^34]。

### 2.5 单文件 / 本地优先的轻量记忆层

**memvid**：主张单文件（single-file）记忆层，把数据、embedding、搜索结构、元数据打包进一个文件，无需数据库或服务器 [^9]。核心设计为受视频编码启发的 Smart Frames：append-only、不可变的单元，存储内容及时间戳、校验和、基本元数据；帧分组以支持压缩、索引与并行读取 [^9]。支持 append-only 写入、对过去记忆状态的查询、时间线式知识演化检查、崩溃安全、以及借鉴视频编码的高效压缩 [^9]。无外部向量库或图数据库，检索直接在文件内进行，model-agnostic 且可完全离线 [^9]。SDK 覆盖 CLI、Node.js、Python、Rust [^9]。

**tigerless-labs/agent-memory**：主张 Markdown 文件为唯一真源（single source of truth），旁置 SQLite 索引作为可随时删除的缓存 [^13]。检索本地且排序，返回路径而非粘贴文本；agent 按任务需要逐级打开命中（L0 列表→abstract→full file→raw material）[^13]。写入在对话边界触发，不依赖 agent 主动记忆；另有 sleep-time 的 Manage 层按价值做整合与遗忘 [^13]。三条读取轨道：会话开始时确定性注入 MEMORY.md；BM25 recall over FTS5（可选向量融合）；以及 ls/grep 可达的普通目录树 [^13]。存储为 Markdown 文件 + SQLite（FTS5 全文索引）；BM25 recall，可选 vector plugin 经 RRF 融合；关系以 frontmatter 中的 links 表达，无需图数据库 [^13]。

**nocturne_memory**：基于 MCP 协议的长期记忆服务器，记忆存储在独立 MCP Server 中，不绑定任何 LLM，可跨会话、跨模型 [^14]。新对话只载入用户指定的核心设定，其余按需调取（read_memory / search_memory）[^14]。README 宣称告别 Vector RAG，采用 graph-like structured memory，可回滚、可视化 [^14]。存储核心为 SQLite | PostgreSQL，协议为 MCP [^14]。

**memoria**：主张带 Git 级版本控制的持久记忆层：每次记忆变更可追踪、可审计、可回滚，支持 snapshot、branch、merge、time-travel rollback [^15]。版本控制能力由 MatrixOne 的原生 Copy-on-Write 引擎驱动 [^15]。功能含语义搜索（vector + full-text hybrid retrieval）、自治理（自动检测矛盾、隔离低置信记忆、维护审计链）、本地 embedding 选项（数据不出本机）[^15]。底层为 MatrixOne 数据库，协议为 MCP [^15]。

### 2.6 检索增强路线的记忆化改造：HippoRAG 2

HippoRAG 2 是一个在事实性（factual）、意义建构（sense-making）与关联性（associative）记忆任务上全面超越标准 RAG 的框架 [^39]。该工作指出：依赖向量检索的 RAG 难以模拟人类长期记忆的动态与互联特性；已有用知识图谱等结构增强向量嵌入的 RAG 方法在意义建构与关联性上有改善，但在更基础的事实记忆任务上性能显著低于标准 RAG [^39]。HippoRAG 2 建立在 HippoRAG 使用的 Personalized PageRank 算法之上，通过更深的段落整合（deeper passage integration）与更有效的 LLM 在线使用加以增强 [^39]。实验结论：在关联记忆任务上相较 SOTA 嵌入模型取得 7% 的提升，同时在事实知识与意义建构记忆能力上表现更优 [^39]。原始 HippoRAG 被标注为 NeurIPS'24，技术组合为 RAG + Knowledge Graphs + Personalized PageRank [^40]。

---

## 三、产业界产品化动态

### 3.1 Anthropic（唯一有实质产品化证据的厂商）

**开发者平台侧**：Anthropic 在 Claude Developer Platform 上推出两项上下文管理能力：context editing（上下文编辑）与 memory tool（记忆工具），配合 Claude Sonnet 4.5 使用 [^18]。
- context editing 在接近 token 上限时自动清除过期的工具调用与结果，保留对话流，从而延长 agent 可运行时长 [^18]。
- memory tool 让 Claude 通过基于文件的系统在上下文窗口之外存储与查阅信息：可在开发者自有基础设施上的专用记忆目录中创建、读取、更新、删除文件，且跨会话持久化 [^18]。
- memory tool 完全在客户端通过工具调用运行，存储后端由开发者管理，开发者对数据存储位置与持久化方式有完全控制权 [^18]。
- Claude Sonnet 4.5 内置 context awareness，可在对话中跟踪可用 token 以更有效地管理上下文 [^18]。
- 内部 agentic search 评测：memory tool 与 context editing 结合相比基线性能提升 39%；仅 context editing 提升 29% [^18]。
- 在 100 轮网页搜索评测中，context editing 使 agent 完成原本因上下文耗尽而失败的工作流，同时将 token 消耗降低 84% [^18]。

**应用侧**：Anthropic 将记忆功能引入 Claude 应用，使 Claude 记住用户及团队的项目与偏好，减少重复解释上下文 [^19]。记忆功能完全可选，提供细粒度用户控制；同时推出 Incognito 聊天（不进入对话历史、不保存到记忆）[^19]。记忆最初面向 Team 和 Enterprise 计划用户推出；Enterprise 管理员可随时为组织禁用记忆 [^19]。使用 projects 时，Claude 为每个项目创建独立记忆，实现项目间记忆隔离 [^19]。Claude 使用 memory summary 集中呈现所有记忆，用户可在设置中查看并编辑 [^19]。2025 年 10 月 23 日更新：记忆功能扩展至 Pro 和 Max 计划用户，提供项目级记忆、查看/编辑记忆的完全控制权以及 incognito 聊天 [^19]。在扩展前，Anthropic 针对敏感福祉话题与边缘案例进行了广泛安全测试（包括记忆是否会强化有害模式、导致过度迎合、绕过安全防护）[^19]。

### 3.2 Google 与 Microsoft（覆盖度不足）

Google 的 Agent 平台（现品牌为 Gemini Enterprise Agent Platform）提供 Agent Platform Sessions 与 Agent Platform Memory Bank 两项能力，用于维持连续对话状态并检索持久化的长期记忆 [^20]。该页面将 Memory Bank 归入四大能力之一的 'Context management'（上下文管理）[^20]。但该页面为平台级概览，检索到的正文中不含 Memory Bank 的容量上限、配额或定价数字 [^20]。Google Cloud 定价页覆盖 Gemini 模型的 token 定价与 Grounding 定价，检索到的正文片段中未出现 Memory Bank 的独立计价条目或容量数字 [^21]。

Foundry Agent Service 被描述为用于构建、部署和扩展 AI agent 的托管平台 [^22]。其组件包括 Agent Runtime、Toolboxes、Models、Observability、Optimization、Identity & Security、Publishing [^22]。Agent Runtime 负责托管与扩缩 prompt agents 和 Hosted agents，并管理对话、工具调用与 agent 生命周期 [^22]。检索到的正文中未出现专门的 memory 容量或成本数字，也无任何定价数字 [^22]。

**OpenAI 的记忆产品化动态在现有证据中完全缺失。**

---

## 四、评测基准与可比性

### 4.1 主要基准

- **LOCOMO**：Mem0 论文的评测基准 [^5][^24]；memvid 自报评测基准为 LoCoMo（10 × ~26K-token 对话），LLM-as-Judge [^9][^30]。
- **LongMemEval**：Zep 报告准确率提升最高达 18.5% [^6]；Mem0 README 列出 LongMemEval（67.8→94.4）[^23]。
- **DMR (Deep Memory Retrieval)**：Zep 报告 94.8% 对比 MemGPT 93.4%，论文称 DMR 是 MemGPT 团队确立的主要评测指标 [^25]。
- **BEAM**：Mem0 README 列出 BEAM (1M) 64.1、BEAM (10M) 48.6 [^23]。

### 4.2 各系统自报数字汇总

| 系统 | 基准 | 自报数字 | 来源 |
|---|---|---|---|
| Mem0（新算法） | LoCoMo | 92.5（旧 71.4） | [^8][^23] |
| Mem0（新算法） | LongMemEval | 94.4（旧 67.8） | [^8][^23] |
| Mem0（新算法） | BEAM (1M) / (10M) | 64.1 / 48.6 | [^8][^23] |
| Mem0（论文） | LOCOMO | LLM-as-a-Judge 相对 OpenAI +26% | [^5][^24] |
| Zep | DMR | 94.8% vs MemGPT 93.4% | [^6][^25] |
| Zep | LongMemEval | 提升最高 18.5%，延迟降 90% | [^6][^25] |
| memvid | LoCoMo | +35% SOTA；多跳 +76%、时序 +56% | [^9][^30] |
| memvid | 延迟 | 0.025ms P50 / 0.075ms P99 | [^9][^30] |
| HippoRAG 2 | 关联记忆任务 | 相较 SOTA 嵌入模型 +7% | [^39] |

### 4.3 第三方评测 harness（可信度无法确认）

- FerroxLabs/flux-memory-benchmark：自述为 agent memory 系统的头对头基准测试，覆盖 Flux Memory、mem0、Letta、Honcho；在 LongMemEval-S 和 LoCoMo 上运行；仅 1 star，未读取 README 正文 [^26]。
- maximem-ai/memory_and_context_eval_harness：自述面向 AI agent 记忆系统的开放评测 harness，在 LoCoMo 和 LongMemEval 上运行，针对 Synap、Mem0、Zep 和 Supermemory；仅 11 stars，未读取 README 正文 [^27]。
- RudrenduPaul/memtrust：自述为面向 agent-memory 后端的独立 CLI 基准 harness，覆盖 MemPalace、Mem0、Zep、OpenViking；仅 2 stars，未读取 README 正文 [^28]。
- kannaka-labs/kannaka-bench：自述覆盖 LongMemEval、ConvoMem 和 LoCoMo，当前对比 Kannaka vs plain cosine、Supermemory 和 plain context，Mem0、pgvector、Letta 和 Zep 的行仍在进行中；仅 1 star，未读取 README 正文 [^29]。

### 4.4 知识库资源

IAAR-Shanghai/Awesome-AI-Memory 描述为集中式、持续更新的 AI 记忆知识库，系统整理 LLM Memory 与 Agent Memory 相关的前沿研究、工程框架、系统设计、评测基准与真实应用实践 [^16]。TeleAI-UAGI/Awesome-Agent-Memory 描述为关于 LLMs/MLLMs 记忆的精选系统、基准与论文集合，覆盖 long-term context、retrieval 与 reasoning [^17]。

---

## 五、矛盾与不确定性

### 5.1 产业界产品化动态覆盖度不足

q4 仅 Anthropic 一家有实质产品化证据（含具体数字与日期）[^18][^19]；Google 与 Microsoft 的条目均为平台概览页，检索正文中明确不含 Memory Bank / Foundry 的容量、配额或定价数字 [^20][^21][^22]；OpenAI 的记忆产品化动态完全缺失。**单一厂商证据无法支撑「产业界产品化动态」这一子问题，属覆盖度不足。**

### 5.2 Mem0 两套基准口径未对齐

同一系统存在两套口径不同的基准数字：论文报告 LOCOMO 上 LLM-as-a-Judge 相对 OpenAI 提升 26% [^5][^24]；README 自报 LoCoMo 92.5（旧 71.4）[^8][^23]。README 明确声明分数反映 Mem0 托管平台（managed platform），包含开源 SDK 中不具备的专有优化；开源用户应预期方向性相似但数值不完全相同的收益 [^8][^23]。两套数字未做口径对齐说明，读者无法判断哪个代表开源能力，**构成未解决的一致性问题**。

### 5.3 memvid 自报数字内部冲突且与 Mem0 冲突

memvid 自报 '+35% SOTA on LoCoMo'、'+76% multi-hop / +56% temporal vs. industry average'、0.025ms P50 延迟 [^9][^30]。但 README 中 'SOTA' 与 'standard' 均未定义具体对比基线 [^30]，且与 Mem0 自报的 LoCoMo 92.5 直接冲突——两者不可能同时为 SOTA。同时 0.025ms P50 与同 README 中 'Sub-5ms local memory access' 并存但未说明二者关系 [^30]。**该矛盾未解决，且无第三方验证。**

### 5.4 跨系统可比性缺口

Zep 与 Mem0 从未在同一基准上对比：Zep 未评 LOCOMO，Mem0 未评 DMR/LongMemEval [^24][^25]。所有对比均为厂商自评 [^24][^25]。q2/q3 中出现的多个第三方 benchmark harness 均仅 1–11 stars、未读取 README 正文、结果可信度无法确认，**无法作为独立验证来源**。跨系统可比性缺口未解决。

### 5.5 其他不确定性

- Mem0 论文声称对比了六类基线，包括『established memory-augmented systems』『an open-source memory solution』『a proprietary model system』『a dedicated memory management platform』，但摘要未点名 Zep、MemGPT/Letta 或 memvid [^24]。
- memvid README 未给出评测脚本路径、基线系统清单、评测配置或复现命令 [^30]。
- Google Memory Bank 与 Microsoft Foundry 的容量、配额、定价数字在检索正文中均未出现 [^20][^21][^22]。
- 来源 [^37] 主题为 LLM 的不确定性估计与校准，与 LLM Agent 记忆分类体系无关，不能作为本报告证据。

---

## 参考来源

[^1]: A Survey on the Memory Mechanism of Large Language Model based Agents — https://arxiv.org/abs/2404.13501 (arxiv, 2024-04-21)
[^2]: From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs — https://arxiv.org/abs/2504.15965 (arxiv, 2025-04-22)
[^3]: A Survey on the Memory Mechanism of Large Language Model based Agents (HTML full text, Introduction) — https://arxiv.org/html/2404.13501v1 (arxiv, 2024-04-21)
[^4]: A-MEM: Agentic Memory for LLM Agents — https://arxiv.org/abs/2502.12110v11 (arxiv, 2025-02-17)
[^5]: Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory — https://arxiv.org/abs/2504.19413v1 (arxiv, 2025-04-28)
[^6]: Zep: A Temporal Knowledge Graph Architecture for Agent Memory — https://arxiv.org/abs/2501.13956v1 (arxiv, 2025-01-20)
[^7]: Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents — https://arxiv.org/abs/2502.06975v1 (arxiv, 2025-02-10)
[^8]: mem0ai/mem0 — The Memory Layer for AI Agents — https://github.com/mem0ai/mem0 (github)
[^9]: memvid/memvid — Memory layer for AI Agents (single-file) — https://github.com/memvid/memvid (github)
[^10]: getzep/graphiti — A Framework for Building Temporal Knowledge Graphs — https://github.com/getzep/graphiti (github)
[^11]: getzep/zep — Zep Cloud Examples & Integrations — https://github.com/getzep/zep (github)
[^12]: letta-ai/letta (f.k.a. MemGPT) — stateful agents with memory — https://github.com/letta-ai/letta (github)
[^13]: tigerless-labs/agent-memory — long-term memory runtime for AI agents — https://github.com/tigerless-labs/agent-memory (github)
[^14]: Dataojitori/nocturne_memory — Long-Term Memory Server for MCP Agents — https://github.com/Dataojitori/nocturne_memory (github)
[^15]: matrixorigin/memoria — Git for AI Agent Memory — https://github.com/matrixorigin/memoria (github)
[^16]: IAAR-Shanghai/Awesome-AI-Memory — curated knowledge base on AI memory — https://github.com/IAAR-Shanghai/Awesome-AI-Memory (github)
[^17]: TeleAI-UAGI/Awesome-Agent-Memory — curated systems, benchmarks, and papers — https://github.com/TeleAI-UAGI/Awesome-Agent-Memory (github)
[^18]: Managing context on the Claude Developer Platform (context editing + memory tool) — https://claude.com/resources/articles/context-management (web)
[^19]: Bringing memory to teams (Claude app memory) — https://claude.com/resources/articles/memory (web, 2025-10-23)
[^20]: Scale your agents — Gemini Enterprise Agent Platform (Google Cloud, Memory Bank overview URL redirected here) — https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/overview (web)
[^21]: Agent Platform Pricing | Google Cloud (Vertex AI generative AI pricing) — https://cloud.google.com/vertex-ai/generative-ai/pricing (web)
[^22]: Agents in Microsoft Foundry — Foundry Agent Service overview — https://learn.microsoft.com/en-us/azure/ai-foundry/agents/overview (web)
[^23]: mem0ai/mem0 README (main branch) — https://raw.githubusercontent.com/mem0ai/mem0/main/README.md (github)
[^24]: Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory — https://arxiv.org/abs/2504.19413 (arxiv, 2025-04-28)
[^25]: Zep: A Temporal Knowledge Graph Architecture for Agent Memory — https://arxiv.org/abs/2501.13956 (arxiv, 2025-01-20)
[^26]: FerroxLabs/flux-memory-benchmark — https://github.com/FerroxLabs/flux-memory-benchmark (github)
[^27]: maximem-ai/memory_and_context_eval_harness — https://github.com/maximem-ai/memory_and_context_eval_harness (github)
[^28]: RudrenduPaul/memtrust — https://github.com/RudrenduPaul/memtrust (github)
[^29]: kannaka-labs/kannaka-bench — https://github.com/kannaka-labs/kannaka-bench (github)
[^30]: memvid/memvid README — Benchmark Highlights section — https://raw.githubusercontent.com/memvid/memvid/main/README.md (github)
[^31]: letta-ai/letta archive branch README — https://github.com/letta-ai/letta/tree/archive (github)
[^32]: letta-ai/letta archive branch AGENTS.md — https://raw.githubusercontent.com/letta-ai/letta/archive/AGENTS.md (github)
[^33]: letta-ai/letta-code README — https://github.com/letta-ai/letta-code (github)
[^34]: Letta Documentation (docs.letta.com) homepage — https://docs.letta.com/ (web)
[^35]: letta-ai/letta Releases page — https://github.com/letta-ai/letta/releases (github, 2026-05-14)
[^36]: Version Control System for Data with MatrixOne — https://arxiv.org/abs/2604.03927 (arxiv, 2026-04-05)
[^37]: Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach — https://arxiv.org/abs/2404.15993 (arxiv, 2024-04-24)
[^38]: MemGPT: Towards LLMs as Operating Systems — https://arxiv.org/abs/2310.08560v2 (arxiv, 2023-10-12)
[^39]: From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2) — https://arxiv.org/abs/2502.14802v2 (arxiv, 2025-02-20)
[^40]: OSU-NLP-Group/HippoRAG — https://github.com/OSU-NLP-Group/HippoRAG (github)