# Agentic RAG：定义、架构关键差异与工程现状

## 摘要

Agentic RAG 指把自主 AI 智能体嵌入 RAG 流水线，以突破传统 RAG 的静态工作流限制[^1][^5]。其架构关键差异不在检索组件本身，而在控制层：传统 RAG 是"检索一次、生成一次"的固定流水线[^32]，Agentic RAG 则把检索与生成组织为由模型自身引导的迭代循环，agent 自主决定执行哪些动作、何时执行、是否迭代[^29][^28]。这一转变带来三类结构性后果：一是检索调用次数与上下文精炼可被重复执行[^29]；二是失败模式从"检索质量不足"变为控制循环层面的 retrieval thrash、tool storms、context bloat[^32]；三是评估与可靠性问题被形式化为序列决策问题（有限时域 POMDP）[^31]。工程侧，RAGFlow、LangGraph、LlamaIndex、Haystack 等已提供 agentic 检索或编排能力[^8][^9][^10][^11]，但公开证据不足以支撑跨实现的默认参数横向对比，也缺乏 Agentic RAG 相对传统 RAG 的 token/调用次数倍数量化数据（见"矛盾与不确定性"）。

---

## 1. 定义：什么是 Agentic RAG

### 1.1 核心定义

Agentic RAG 的权威定义来自 arXiv:2501.09136 综述：**通过将自主 AI 智能体（autonomous AI agents）嵌入 RAG 流水线，超越传统 RAG 的静态工作流限制**[^1][^5]。

该定义包含三个可拆解的要素：

1. **被嵌入的对象是自主智能体**，而非固定函数模块；
2. **嵌入位置是 RAG 流水线**，即检索、生成、增强这一三部分基础结构之上[^3]；
3. **目的指向是突破静态工作流**，即获得多步推理与复杂任务管理所需的适应性（adaptability）[^1][^5]。

### 1.2 智能体在此语境下的标准定义

LlamaIndex 官方文档给出的 agent 定义可直接作为该语境下的操作性定义：**agent 是由 LLM 驱动的半自主软件，给定任务后执行一系列步骤，从工具集中选择最佳工具完成每一步，并判断任务是否完成或需继续循环**[^23]。

Haystack 官方文档从组件角度给出等价刻画：**Agent 组件是 tool-using agent，与 chat-based LLM 和工具交互，迭代式解决复杂查询；它是 loop-based 系统，迭代调用工具、更新状态、生成 prompt，直到满足可配置的 exit_conditions 之一**[^24]。

### 1.3 四类 agentic 设计模式

综述归纳出 Agentic RAG 中智能体运用的四类设计模式：**reflection（反思）、planning（规划）、tool use（工具使用）、multi-agent collaboration（多智能体协作）**[^1][^5]。这四类模式是后续架构差异分析的基本词汇。

### 1.4 架构分类法

该综述提出基于四个维度的 Agentic RAG 架构分类法（taxonomy）：**agent cardinality（智能体数量）、control structure（控制结构）、autonomy（自主性）、knowledge representation（知识表示）**；并指出其操作结构范围从 sequential steps（顺序步骤）到 adaptive collaboration（自适应协作）[^1][^5]。

---

## 2. 传统 RAG 的架构基线

要说明"关键差异"，必须先固定基线。

### 2.1 基本流程

传统 RAG 的基本流程被描述为：**先找到与查询语义匹配的文档，再把文档传给 LLM（如 ChatGPT）以抽取正确答案**[^4]。LangChain 官方博客给出同一流程的三步表述：embedding 用户查询、检索相关文档、将文档传给 LLM 生成有依据的答案[^27]。

### 2.2 设计目标与固有限制

RAG 系统的三个设计目标是：a) 减少 LLM 幻觉回答，b) 将来源/引用与生成回答关联，c) 免除为文档标注元数据的需求[^4]。同时，RAG 系统存在两类固有限制：**信息检索系统固有的限制，以及对 LLM 的依赖**[^4]。

RAG 被提出用于应对 LLM 的幻觉、知识过时、推理过程不透明不可追溯等问题[^3]。其框架基础被拆解为三部分：retrieval（检索）、generation（生成）、augmentation（增强）[^3]。范式演进被划分为 Naive RAG、Advanced RAG、Modular RAG 三类[^3]。

### 2.3 静态性的具体表现

传统 RAG 的静态性在两个层面被证据直接指出：

- **工作流层面**：传统 RAG 系统受限于 static workflows，缺乏多步推理与复杂任务管理所需的适应性[^1][^5]。
- **检索策略层面**：传统做法是 indiscriminately retrieving and incorporating a fixed number of retrieved passages，无论检索是否必要或段落是否相关[^2][^6]。Self-RAG 明确指出这一做法会削弱 LM 的通用性[^2]。

此外，Anthropic 指出传统 RAG 在编码信息时会移除上下文（或译作丢失上下文），导致系统常常无法从知识库检索到相关信息[^15][^26]。

---

## 3. 架构关键差异

### 3.1 差异一：从固定流水线到模型引导的迭代循环

这是最根本的差异。论文 arXiv:2601.07711 给出两个范式的对照定义：

- **Enhanced RAG（固定工作流）**：router 决定是否触发检索、rewriter 改写查询、retriever 选块、reranker 排序后交给 generator[^28]。
- **Agentic RAG**：由 LLM 作为编排者（orchestrator），自主决定执行哪些动作、何时执行、以及是否迭代[^28]。

该论文正文进一步指出，Agentic RAG 系统**不再是固定流水线，而是由模型自身引导的迭代循环（iterative loops guided by the model itself）**；检索与上下文精炼**可以被重复执行（Retrieval and context refinement can be repeated）**，因为 agent 依据任务演化状态自主选择操作——这是其相对固定流水线在调用次数上可能增加的结构性原因[^29]。

第三方分析给出同一结构的流程刻画：传统 RAG 是"检索一次、生成一次"的简单流水线，而 Agentic RAG 是一个控制循环（plan → retrieve → evaluate → decide → retrieve again）[^32]。

### 3.2 差异二：控制权的归属

在 Enhanced RAG 中，控制流由预先编排的组件序列决定（router → rewriter → retriever → reranker → generator）[^28]。在 Agentic RAG 中，控制权上移到模型：模型是编排者，决定动作集合、时序与迭代与否[^28]。

这一差异在官方教程中有直接体现。LangGraph 官方 agentic RAG 教程的目标是**构建一个 retrieval agent，由 agent 自行决定何时检索向量库、何时直接回答用户**[^22]。其步骤为：抓取并预处理文档 → 建立语义检索索引并创建 retriever tool → 构建可自主决定是否调用 retriever 的 agentic RAG 系统[^22]。

### 3.3 差异三：检索决策的粒度与条件性

传统 RAG 的检索是"无条件 + 固定数量"[^2][^6]。Self-RAG 的对照设计是训练单个 LM **按需自适应检索（adaptively retrieves passages on-demand）**，并通过特殊 token（reflection tokens）对检索到的段落及自身生成进行反思；reflection tokens 使 LM 在推理阶段可控，从而针对不同任务需求调整行为[^2][^6]。

需要区分的是：Self-RAG 属于检索策略层面的自适应，其机制是 reflection tokens 而非 agent 编排[^2][^6]。另一条独立路线 Adaptive-RAG 则用一个小型 LM 分类器预测查询复杂度等级，在迭代式检索增强 LLM、单步检索增强 LLM 以及无检索方法之间自适应切换[^33]。其动机是解决现有方法的两难：简单查询被施加不必要的计算开销，或复杂多步查询未被充分处理[^34]。

### 3.4 差异四：多智能体与工具编排成为一等公民

Agentic RAG 的架构分类法把 agent cardinality（智能体数量）与 control structure（控制结构）列为首要维度[^1][^5]，这在传统 RAG 的三部分框架（retrieval / generation / augmentation）中没有对应位置[^3]。

具体架构形态上，LlamaIndex 给出的 Agentic RAG 架构为：**大语料被拆分为小文档，每个文档创建一个 document agent（具备 embedding 搜索与摘要能力），再在其上建立 top-level / meta-agent；meta-agent 执行工具检索（tool retrieval），并使用 Chain-of-Thought 回答用户问题**[^16]。该示例中语料被限制为 100 个文档，系统提示要求每个 document agent 必须至少使用一个工具作答，不得依赖先验知识[^16]。

### 3.5 差异五：失败模式的位置发生迁移

传统 RAG 的失败点被归纳为 7 个，且其两条核心结论是：RAG 系统的验证只有在运行期间才可行；RAG 系统的鲁棒性是演化出来的，而非在起始阶段设计进去的[^4]。

Agentic RAG 的失败模式则位于控制循环内部。第三方分析列出三类在生产中反复出现的失败模式：**Retrieval Thrash（检索反复但不收敛）、Tool storms（工具调用过度、级联并重试直至预算耗尽）、Context bloat（上下文窗口被低信号内容填满，直到模型不再遵循自身指令）**[^32]。该文指出这些失败通常表现为"模型变差了"，但根因并非基础模型，而是缺乏预算控制（budgets）、弱停止规则（weak stopping rules）以及对 agent 决策循环零可观测性[^32]。

该文给出的 retrieval thrash 具体场景：用户询问加州远程员工报销政策，agent 反复改写查询（broadening→narrowing→broadening），多轮后耗尽检索预算，答案质量相比第一轮几乎没有提升[^32]。

### 3.6 差异六：评估对象从"输出质量"扩展到"轨迹与成本"

SoK 论文指出，当前研究缺乏将 Agentic RAG 作为序列决策系统的系统性理解，导致架构高度碎片化、评估方法不一致、可靠性风险未解决；该论文将 agentic 检索-生成循环形式化为**有限时域部分可观测马尔可夫决策过程（finite-horizon POMDP）**，显式建模其控制策略与状态转移[^31]。

该论文识别出自主循环固有的严重系统性风险：**复合式幻觉传播（compounding hallucination propagation）、记忆投毒（memory poisoning）、检索失配（retrieval misalignment）、级联式工具执行漏洞（cascading tool-execution vulnerabilities）**；并批评传统静态评估实践对 agentic 系统不适用，指出需要成本感知编排（cost-aware orchestration）、稳定自适应检索、形式化轨迹评估与监督机制等研究方向[^31]。

---

## 4. 工程实现现状

### 4.1 已明确提供 agentic 检索能力的实现

**RAGFlow**：自我定位为开源 RAG 引擎，将 RAG 与 Agent 能力融合，构建面向 LLM 的 context layer[^8]。README 的 Latest Updates 记录 2026-08-19 引入 Agentic RAG，提供 Low、Medium、High、Ultra 四档 thinking mode；2026-09-29 发布 RAGFlow 1.0.0-rc1[^8]。Key Features 中的 'Agentic Retrieval' 描述为多步检索：模型分析问题，必要时拆解问题、检索知识、验证证据（verify evidence），并通过多轮检索与推理（multiple rounds of retrieval and reasoning）收集更完整上下文以支撑有依据的回答；四档 thinking mode 用于按问题复杂度控制检索与推理深度[^8]。

**Azure AI Search**：Microsoft 提供明确命名为 'agentic retrieval' 的产品化能力，定位为面向 chat/copilot 应用中复杂问题的多查询（multi-query）检索管线，服务于 RAG 与 agent-to-agent 工作流[^25]。该能力可用基于 LLM 的查询规划（query planning，preview）把复杂查询拆解为更聚焦的子查询，并可利用聊天历史补充上下文；子查询并行执行，每个子查询经语义重排（semantically reranked）以提升最相关匹配；系统将最佳结果合并为统一响应供 LLM 生成有依据的答案，并可返回来源引用（source references）与活动日志（activity log）[^25]。该能力同时支撑 Microsoft Foundry 门户中的 Foundry IQ——面向 agent 的托管知识层（managed knowledge layer）[^25]。文档说明 agentic retrieval 相比单查询管线会增加延迟，但能处理单查询无法应对的复杂度（多重提问、依赖对话上下文、需改写/同义词扩展、拼写错误等）[^25]。

### 4.2 提供编排底座但自身不含检索组件的实现

**LangGraph**：自我定位为用于构建有状态 agent 的 low-level orchestration framework[^9][^17]。README 列出核心能力：durable execution（持久化执行、故障后从断点恢复）、human-in-the-loop（执行中检查与修改 agent state）、comprehensive memory（短期工作记忆 + 跨会话长期记忆）、LangSmith 调试、生产级部署[^9][^17]。README 明确 LangGraph 本身是低层编排框架，**不直接提供检索组件**；高层 agent 能力由构建在其上的 Deep Agents 包提供（可 plan、使用 subagents、利用文件系统）[^9]。README 提到被 Klarna、Replit、Elastic 等公司使用[^9]。

### 4.3 框架层的 agentic 转向

**LlamaIndex**：LlamaIndex Framework 自我定位为用于构建 agentic applications 的开源框架[^10]。README 声明公司重心已从 OSS 框架转向 LlamaParse（agentic OCR/解析/抽取/索引平台）、LiteParse 与 ParseBench/ExtractBench 基准[^10]。README 称该 OSS 框架自 2023 年发布以来一直是提供构建各类 RAG 与 agent 应用的编排工具集，并提出 'agents are the new consumers of documents' 的定位主张[^10]。构建方式分两种：llama-index 起步包（含 core 与部分集成）与 llama-index-core 定制包（按需添加集成）；README 称有超过 300 个 LlamaIndex 集成包可与 core 协同工作[^10]。

**Haystack**：自我定位为用于构建生产级 LLM 应用的开源 AI orchestration framework（Python）[^11][^19]。README 主张以模块化 pipeline 与 agent workflow 提供对 retrieval、routing、memory、generation 的显式控制（explicit control）[^11]。适用场景包括 scalable RAG systems、multimodal applications、semantic search、question answering、autonomous agents；强调 transparent architecture，便于实验、深度定制与部署[^11]。

**CrewAI**：自我定位为开源 Python 多 agent 自动化框架，提供高层抽象与低层 API 以构建生产级多 agent workflow[^13]。两种核心抽象：CrewAI Crews（基于角色的 agent 自主协作）与 CrewAI Flows（事件驱动自动化，结合精确 workflow 控制、单次 LLM 调用与对 Crews 的原生支持）[^13]。README 称已有超过 100,000 名开发者通过 learn.crewai.com 社区课程获得认证[^13]。

### 4.4 已进入维护模式的实现（重要时效性提示）

**AutoGen**：自我定位为用于创建可自主行动或与人类协作的多 agent AI 应用的框架[^12]。但 README 顶部标注 AutoGen **已进入 maintenance mode：不再接收新功能或增强，转为社区维护**；README 建议新用户改用 Microsoft Agent Framework（MAF），并称 MAF 是 AutoGen 的企业级继任者，已发布 production-ready 1.0 版本[^12]。MAF 宣称提供企业级多 agent 编排、多 provider 模型支持，以及通过 A2A 与 MCP 实现的跨运行时互操作[^12]。

### 4.5 生态热度参考

GitHub 仓库检索结果（抓取时点 2026-10-14）显示：langgenius/dify stars 158443、forks 25058，描述为构建 Agentic workflows 与 RAG pipelines 的协作工作区；langchain-ai/langchain stars 147842、forks 24795，描述为 'The agent engineering platform'；infiniflow/ragflow stars 92201、forks 10985；langchain-ai/langgraph stars 43254、forks 7354；bytedance/deer-flow stars 83852、forks 11680，描述为长时程 SuperAgent harness（sandboxes、memories、tools、skill、subagents、message gateway）；mem0ai/mem0 stars 67164、forks 7903，定位为 AI agent 的 memory layer；ruvnet/ruflo stars 74633、forks 8874，描述含 vector RAG integration 与多 agent swarm 编排；headroomlabs-ai/headroom stars 75321、forks 5846，描述为在 RAG chunks 等进入 LLM 前压缩以降低 token 消耗[^14]。

---

## 5. 矛盾与不确定性

本节显式声明证据不足之处。以下条目中，凡现有公开资料不足以确认的，均按要求标注。

### 5.1 量化成本/延迟数据完全缺失

Agentic RAG 相对传统 RAG 的 token 与调用次数倍数，**现有公开资料不足以确认**。

- 论文 arXiv:2601.07711 明确以 performance and costs 为评估维度[^28]，但可用证据中未提取到任何具体 token/调用次数倍数数字。
- Anthropic 给出的量化数据（检索失败率降低 49%，结合 reranking 后降低 67%；prompt caching 延迟降低 2 倍以上、成本降低最多 90%）属于 Contextual Retrieval 这一**检索增强技术**，而非 agentic 编排[^15][^26]。该发布命名为 'Contextual Retrieval'，并非 'Agentic RAG' 或 'agentic retrieval'[^26]。
- Microsoft Learn 仅定性称 agentic retrieval 相比单查询管线会增加延迟[^25]，未给出倍数。

因此，本报告不对"Agentic RAG 成本是传统 RAG 的 N 倍"作任何断言。

### 5.2 跨实现的默认参数横向对比不可行

各实现的默认参数配置（迭代轮数、top-k、模型选择）**现有公开资料不足以支撑横向对比表**。

可用证据仅覆盖：

- LangGraph 教程：`RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=50)`、InMemoryVectorStore、OpenAIEmbeddings()（未显式指定 embedding 模型名）[^22]。
- LlamaIndex 教程：默认模型 `gpt-4o-mini`；文档提示 agent 需要能力较强的模型，较小模型可能不太可靠[^23]。
- Haystack Agent：`max_agent_steps` 上限、可配置 `exit_conditions`；退出原因 `exit_reason` 取值包括 "text"、"length"、"content_filter"、满足工具退出条件的工具名、"max_agent_steps"、以及 hook 设置的自定义原因；运行元数据输出 step_count、token_usage、tool_call_counts、exit_reason；Agent 默认在 "length" 和 "content_filter" 时停止（原文句子被截断）[^24]。

RAGFlow 的 thinking mode 四档[^8]、CrewAI[^13]、AutoGen/MAF[^12]、Dify[^14] 均**无默认迭代轮数/top-k/模型配置证据**。

### 5.3 stars 元数据在不同证据条目间不一致

同一仓库的 stars 元数据在不同证据条目间存在不一致或缺失，**若引用具体 stars 数字将无法自洽**：

- run-llama/llama_index：一处为 stars 52492、forks 8336[^18]；另一处 README 证据未给出 stars 数[^10]。
- microsoft/autogen：一处为 stars 61397、forks 9280[^20]；另一处 README 证据未给出 stars 数[^12]。
- langchain-ai/langgraph：一处为 stars 43254[^14]，另一处为 stars 43255[^17]，差异说明数据抓取时点不同。

本报告在第 4.5 节引用 stars 时统一采用抓取时点为 2026-10-14 的检索结果[^14]，并在此声明：该组数字与其他条目间的差异源于抓取时点不同，不应混用。

### 5.4 AutoGen 作为"代表性实现"存在时效性风险

microsoft/autogen 的 README 自述已进入 maintenance mode、不再接收新功能或增强[^12]，last push 为 2026-04-15[^20]（距 2026-10-14 已约 6 个月无更新）。因此将其列为 Agentic RAG 的"代表性实现"存在时效性与误导风险。README 建议新用户改用 Microsoft Agent Framework（MAF）[^12]，但**现有证据中缺少 MAF 的仓库 stars 与官方文档证据**，无法在本报告中以同等证据强度替代介绍。

### 5.5 命名口径不一致的证据

LangChain 官方博客该文实际标题为 'Self-Reflective RAG with LangGraph'（作者 Ankush Gola，2024-02-07），URL 虽含 'agentic-rag' 但正文未以 'Agentic RAG' 命名该能力；文章介绍 'self-reflective RAG' 概念：用 LLM 自我纠正低质量检索与/或生成，并实现 CRAG 与 Self-RAG 论文思想[^27]。该来源仅作为 LangChain 官方在 agentic 检索编排方向的证据，**命名口径不符**，不应作为"Agentic RAG"术语的直接出处[^27]。

### 5.6 多智能体架构的失败风险有独立证据

多 agent LLM 系统（MAS）的性能提升往往微乎其微（performance gains on popular benchmarks are often minimal）[^30]。MAST 基于覆盖 7 个流行 MAS 框架、1600+ 条标注执行轨迹的数据集，通过对 150 条轨迹的严格分析建立失败分类学，识别出 14 种失败模式，归为 3 大类：系统设计问题、agent 间失配、任务验证；论文结论指出所识别的失败需要更复杂的解决方案，而非简单修补[^30]。这构成对"multi-agent collaboration"作为 Agentic RAG 设计模式[^1][^5]的重要反向证据。

### 5.7 参考实现本身存在缺陷的证据

第三方分析称 LangGraph 官方 agentic RAG 教程本身存在该 bug：一个无限检索循环，需要引入 rewrite_count 上限来修复；并评论"如果参考实现都能无限循环，生产系统肯定会"[^32]。该主张来自第三方技术博客而非 LangGraph 官方确认，**现有公开资料不足以确认该缺陷的当前状态**（是否已在官方教程中修复）。

---

## 参考来源

[^1]: Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG (arXiv:2501.09136v4) — https://arxiv.org/abs/2501.09136v4 (arxiv, 2025-01-15)
[^2]: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (arXiv:2310.11511v1) — https://arxiv.org/abs/2310.11511v1 (arxiv, 2023-10-17)
[^3]: Retrieval-Augmented Generation for Large Language Models: A Survey — https://arxiv.org/abs/2312.10997v5 (arxiv, 2023-12-18)
[^4]: Seven Failure Points When Engineering a Retrieval Augmented Generation System — https://arxiv.org/abs/2401.05856v1 (arxiv, 2024-01-11)
[^5]: Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG — https://arxiv.org/abs/2501.09136 (arxiv, 2025-01-15)
[^6]: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection — https://arxiv.org/abs/2310.11511 (arxiv, 2023-10-17)
[^7]: ReAct: Synergizing Reasoning and Acting in Language Models — https://arxiv.org/abs/2210.03629 (arxiv, 2022-10-06)
[^8]: infiniflow/ragflow — README (main branch) — https://raw.githubusercontent.com/infiniflow/ragflow/main/README.md (github, 2026-09-29)
[^9]: langchain-ai/langgraph — README (main branch) — https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md (github)
[^10]: run-llama/llama_index — README (main branch) — https://raw.githubusercontent.com/run-llama/llama_index/main/README.md (github)
[^11]: deepset-ai/haystack — README (main branch) — https://raw.githubusercontent.com/deepset-ai/haystack/main/README.md (github)
[^12]: microsoft/autogen — README (main branch) — https://raw.githubusercontent.com/microsoft/autogen/main/README.md (github)
[^13]: crewAIInc/crewAI — README (main branch) — https://raw.githubusercontent.com/crewAIInc/crewAI/main/README.md (github)
[^14]: GitHub 仓库检索结果（search_repos: 'agentic RAG' / 'RAGFlow' / 'langgraph agent'） — https://github.com/search?q=agentic+RAG&type=repositories (github, 2026-10-14)
[^15]: Introducing Contextual Retrieval (Anthropic Engineering) — https://www.anthropic.com/engineering/contextual-retrieval (web, 2024-09-19)
[^16]: Agentic RAG With LlamaIndex: Architecture Guide (LlamaIndex Blog) — https://www.llamaindex.ai/blog/agentic-rag-with-llamaindex-2721b8a49ff6 (web)
[^17]: langchain-ai/langgraph — https://github.com/langchain-ai/langgraph (github)
[^18]: run-llama/llama_index — https://github.com/run-llama/llama_index (github)
[^19]: deepset-ai/haystack — https://github.com/deepset-ai/haystack (github)
[^20]: microsoft/autogen — https://github.com/microsoft/autogen (github)
[^21]: crewAIInc/crewAI — https://github.com/crewAIInc/crewAI (github)
[^22]: LangGraph Agentic RAG tutorial (official docs) — https://docs.langchain.com/oss/python/langgraph/agentic-rag (web)
[^23]: LlamaIndex - Building an agent (official docs) — https://docs.llamaindex.ai/en/stable/understanding/agent/ (web)
[^24]: Haystack - Agent component (official docs, v3.3) — https://docs.haystack.deepset.ai/docs/agent (web)
[^25]: Agentic retrieval in Azure AI Search (Microsoft Learn) — https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview (web)
[^26]: Introducing Contextual Retrieval (Anthropic Engineering Blog) — https://www.anthropic.com/news/contextual-retrieval (web, 2024-09-19)
[^27]: Self-Reflective RAG with LangGraph (LangChain Blog) — https://blog.langchain.com/agentic-rag-with-langgraph/ (web, 2024-02-07)
[^28]: Is Agentic RAG worth it? An experimental comparison of RAG approaches — https://arxiv.org/abs/2601.07711 (arxiv, 2026-01-12)
[^29]: Is Agentic RAG worth it? An experimental comparison of RAG approaches (HTML full text) — https://arxiv.org/html/2601.07711v2 (arxiv, 2026-04-20)
[^30]: Why Do Multi-Agent LLM Systems Fail? (MAST) — https://arxiv.org/abs/2503.13657 (arxiv, 2025-03-17)
[^31]: SoK: Agentic Retrieval-Augmented Generation (RAG): Taxonomy, Architectures, Evaluation, and Research Directions — https://arxiv.org/abs/2603.07379 (arxiv, 2026-03-07)
[^32]: Agentic RAG Failure Modes: Retrieval Thrash, Tool Storms, and Context Bloat (Towards Data Science) — https://towardsdatascience.com/agentic-rag-failure-modes-retrieval-thrash-tool-storms-and-context-bloat-and-how-to-spot-them-early/ (web)
[^33]: Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity — https://arxiv.org/abs/2403.14403 (arxiv, 2024-03-21)
[^34]: Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity (HTML full text) — https://arxiv.org/html/2403.14403v2 (arxiv, 2024-03-28)
[^35]: The Effect of Text Chunk Size on Retrieval-Augmented Generation Performance — https://arxiv.org/abs/2607.24767 (arxiv, 2026-06-08)
[^36]: Chunking Methods on Retrieval-Augmented Generation - Effectiveness Evaluation Against Computational Cost and Limitations — https://arxiv.org/abs/2606.00881 (arxiv, 2026-05-30)