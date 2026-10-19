# 多智能体系统（Multi-Agent LLM）主流协作模式与适用场景

## 摘要

LLM 多智能体系统（LLM-based multi-agent systems）被定位为在「单个 LLM 作为规划/决策智能体」基础上的进一步发展，用于复杂问题求解与世界模拟；该领域的两大核心考察维度是「智能体如何被 profile（画像/角色设定）」与「智能体如何 communicate（通信）」[^1]。本报告基于 28 条证据，梳理出四类主流协作模式：**对等辩论（debate）**、**分层聚合（MoA 式）**、**角色化 SOP 流水线（MetaGPT/CrewAI 式）**、**编排器—子智能体委派（lead + subagents / handoffs）**，并归纳其通信结构与终止条件。在适用场景上，证据支持多智能体在**广度优先（breadth-first）查询**与**开放式自然语言生成**上占优，但在流行基准上的增益往往很小，且失败模式已被系统分类。**关键不确定性**：多智能体相对单智能体的算力放大倍数、同等预算下的公平对照结果，在现有证据中完全缺失；Anthropic 的内部评测数字（+90.2%）与 MAST、Cognition 的负面结论直接冲突且无第三方复现。

---

## 一、主流协作模式

### 1.1 对等辩论（Multiagent Debate）

**定义与机制**：该模式由多个语言模型实例在多轮中提出并辩论各自的回答与推理过程，最终收敛到一个共同答案[^2]。通信结构为「多个对等 LLM 实例之间多轮交换回答与推理」，**而非层级式**；终止条件为「到达一个共同最终答案（common final answer）」[^2]。该方法可直接应用于黑盒模型，且对所有任务使用相同的流程与提示（identical procedure and prompts）[^2]。作者将该思路称为 'society of minds' 方法[^2]。

**报告效果**：在数学与策略推理任务上显著提升，并提升生成内容的事实有效性、减少错误答案与幻觉[^2]。

### 1.2 分层聚合（Mixture-of-Agents, MoA）

**定义与机制**：MoA 构建分层架构，每层包含多个 LLM 智能体，每个智能体将**上一层所有智能体的输出**作为辅助信息来生成回复[^4]。与 debate 的对等结构不同，MoA 是逐层聚合的层级结构。

**报告效果**：在 AlpacaEval 2.0、MT-Bench 和 FLASK 三个基准上取得当时最优（state-of-art）表现，超过 GPT-4 Omni；仅使用开源 LLM 的 MoA 在 AlpacaEval 2.0 上以 65.1% 的得分领先，对比 GPT-4 Omni 的 57.5%[^4]。该证据支持：在开放式对话/指令跟随类（自然语言生成与理解）任务上，多智能体聚合可显著优于强单模型基线[^4]。

### 1.3 角色化 SOP 流水线（MetaGPT / CrewAI 式）

**MetaGPT**：核心哲学为 'Code = SOP(Team)'，即将标准作业流程（SOP）物化并应用于由 LLM 组成的团队[^12]。默认拓扑为模拟软件公司，内置角色包括产品经理、架构师、项目经理、工程师，提供软件公司全流程及精心编排的 SOP[^12]。以一行需求为输入，输出用户故事、竞品分析、需求、数据结构、API、文档等[^12]。其论文提出将 SOP 编码进 prompt 序列，用流水线范式为不同 agent 分配角色以分解复杂任务[^28]。

**CrewAI**：通过两种抽象提供协作——Crews（自治协作）与 Flows（事件驱动控制）[^10]。Crews 面向自治与协作智能，使用基于角色（role-based）的 AI 智能体；Flows 构建事件驱动自动化，结合精确工作流控制、单次 LLM 调用以及对 Crews 的原生支持[^10]。

### 1.4 编排器—子智能体委派（Lead + Subagents / Handoffs）

**Anthropic Research 架构**：一个 lead agent 规划研究流程，并通过工具创建**并行 subagents** 同时搜索信息[^18]。

**OpenAI Swarm / Agents SDK**：Swarm 通过两个原语实现协作——Agent 与 handoffs；Agent 包含 instructions 与 tools，可在任意时刻将对话移交给另一个 Agent[^15]。Swarm 完全由 Chat Completions API 驱动，调用之间无状态（stateless between calls），几乎完全在客户端运行[^15]。Swarm 已被 OpenAI Agents SDK 取代，后者被描述为 Swarm 的生产就绪演进版本[^15]。OpenAI Agents SDK 自我定位为构建多智能体工作流的轻量级框架，且 provider-agnostic，核心概念包含 Agents as tools / Handoffs，用于将任务委派给其他智能体[^16]。

**AG2 的 Network 抽象**：AG2 Classic（原 AutoGen 衍生框架）的多智能体模式包括 GroupChat / GroupChatManager、swarms、nested chats、sequential-chat；自 AG2 v1.0 起，多智能体抽象为 Network（hub + channels）[^17]。

### 1.5 框架层的拓扑中立性

值得注意的是，多个主流框架**不指定默认多智能体拓扑**：LangGraph 自我定位为低层编排框架（low-level orchestration framework），用于构建有状态（stateful）智能体，而非固定拓扑[^8]；其 README 未指定默认多智能体拓扑，而是推荐更高层的 Deep Agents 包，其智能体可规划、使用 subagents 并利用文件系统[^8]。CAMEL 的四项设计原则为 Evolvability、Scalability、Statefulness、Code-as-Prompt，其 Scalability 原则声称框架设计支持数百万（millions）智能体的系统，但 README 正文未指定单一默认拓扑[^14]。AutoGen README 的 Quickstart 展示的是单个 AssistantAgent 的 Hello World 与 MCP 工具接入，未在正文给出默认多智能体拓扑[^6]。

### 1.6 反思/进化机制（横切能力）

综述还考察「哪些机制促进智能体能力的增长」（mechanisms contributing to the growth of agents' capacities），即反思/进化类机制[^1]。代表性工作是 Reflexion：通过语言反馈（linguistic feedback）而非更新权重来强化语言 agent；Reflexion agent 对任务反馈信号进行口头反思（verbally reflect），并将反思文本保存在自身的 episodic memory buffer 中，以在后续试验（subsequent trials）中引导更好的决策[^26]。在 HumanEval 编码基准上，Reflexion 达到 91% pass@1，超过此前 SOTA 的 GPT-4 的 80%[^26]。

---

## 二、什么场景下多智能体优于单智能体

### 2.1 广度优先（breadth-first）查询

Anthropic 认为多智能体系统特别擅长 breadth-first 类查询，即需要**同时探索多个独立方向**的任务；举例：识别 S&P 500 信息技术板块所有公司董事会成员时，多智能体通过分解任务成功，而单智能体因缓慢的顺序搜索失败[^18]。

### 2.2 开放式自然语言生成与指令跟随

MoA 在 AlpacaEval 2.0、MT-Bench、FLASK 上超过 GPT-4 Omni，仅用开源 LLM 即在 AlpacaEval 2.0 上达到 65.1% vs 57.5%[^4]。这支持在开放式对话/指令跟随类任务上多智能体聚合的优势[^4]。

### 2.3 需要事实性与推理增强的任务

Multiagent Debate 在数学与策略推理任务上显著提升，并提升事实有效性、减少错误答案与幻觉[^2]。AgentVerse 实验覆盖文本理解、推理、编码、工具使用与具身 AI（embodied AI），声称多智能体群体表现优于单智能体[^25]。

### 2.4 任务可分解为子任务的复杂场景

AutoGen 论文将多智能体对话作为核心机制，理由包括：chat 优化的 LLM 能吸收反馈、单一 LLM 可配置出多样能力、复杂任务可分解为子任务[^24]。MetaGPT 论文提出用流水线范式为不同 agent 分配角色以分解复杂任务[^28]。

### 2.5 选型原则：先找最简方案

Anthropic 建议：构建 LLM 应用时先找最简单的方案，只在必要时才增加复杂度——这可能意味着根本不构建 agentic 系统[^22]。agentic 系统通常以延迟和成本换取更好的任务表现，开发者应判断这种权衡是否值得[^22]。选型建议：任务定义明确时，workflows 提供可预测性与一致性；需要灵活性、模型驱动决策且规模较大时，agents 更合适[^22]。对许多应用而言，用检索和 in-context 示例优化单次 LLM 调用通常就足够了[^22]。

---

## 三、多智能体何时失效或不如单智能体

### 3.1 流行基准上增益往往很小

MAST 论文指出：尽管多智能体 LLM 系统（MAS）受到广泛关注，但它们在主流基准上的性能提升往往是微小的（"their performance gains on popular benchmarks are often minimal"）[^3][^19]。

### 3.2 失败分类学 MAST

论文构建了 MAST-Data，一个包含 1600+ 条标注执行轨迹（annotated traces）的数据集，覆盖 7 个主流 MAS 框架[^3][^19]。提出首个多智能体系统失败分类法 MAST，基于 150 条轨迹分析、由专家人工标注者指导，标注者间一致性 kappa = 0.88[^3][^19]。MAST 识别出 14 种失败模式，归为 3 大类：(i) 系统设计问题（system design issues）、(ii) 智能体间错位（inter-agent misalignment）、(iii) 任务验证（task verification）[^3][^19]。分析覆盖 GPT4、Claude 3、Qwen2.5、CodeLlama 等模型，以及 coding、math、general agent 等任务[^3][^19]。

### 3.3 长时运行、多轮对话场景下的脆弱性

Cognition 明确主张多智能体架构是构建 agent 的错误方式，点名批评 OpenAI 的 swarm 与 Microsoft 的 AutoGen 库在推广多智能体概念[^21]。核心论点是可靠性依赖 Context Engineering，多智能体架构在长时运行、多轮对话场景下非常脆弱（very fragile）[^21]。给出的关键失败机制：任务被拆分为子任务后，各子智能体可能误解子任务（如把 Flappy Bird 背景做成 Super Mario 风格），最终合并智能体被迫整合这些误解[^21]。作者指出真实世界任务有大量细微语境，容易被误传；仅把原始任务复制给子智能体不足以解决，因为生产系统中对话是多轮的、包含工具调用产生的隐含决策[^21]。提出 Principle 1：共享上下文，且共享完整的 agent trace，而不只是单条消息[^21]。

### 3.4 框架抽象带来的调试负担

Anthropic 观察到最成功的实现使用简单、可组合的模式，而非复杂框架；框架常增加抽象层、掩盖底层 prompt 与响应，使调试更难，并诱使开发者添加不必要的复杂度[^22]。LangChain 的 Harrison Chase 认为构建可靠 agentic 系统的难点在于确保 LLM 在每一步都拥有恰当的上下文；该文将 agentic 系统区分为 workflows 与 agents，并指出多数 agent 框架只是 agent 抽象集合，容易掩盖上下文控制、增加调试难度[^23]。Chase 批评 OpenAI 的 building agents 指南包含 'misguided takes'，同时表示喜欢 Anthropic 的 building effective agents 指南[^23]。

### 3.5 设计空间优化：prompts 与 topologies 共同决定有效性

论文对 MAS 的设计空间做了深入分析，指出 prompts 与 topologies 共同决定 MAS 的有效性[^20]。提出 Multi-Agent System Search (MASS) 框架，分三阶段优化：block-level 局部 prompt 优化、workflow topology 优化、workflow-level 全局 prompt 优化[^20]。作者称 MASS 优化的多智能体系统大幅优于一系列现有替代方案（摘要级主张，未读到具体基准分数）[^20]。

---

## 四、矛盾与不确定性

### 4.1 Anthropic 内部评测数字与负面证据的直接冲突（未解决）

Anthropic 内部评测显示：以 Claude Opus 4 为 lead agent、Claude Sonnet 4 为 subagents 的多智能体系统，在其内部 research eval 上比单智能体 Claude Opus 4 高出 90.2%[^18]。Anthropic 指出多智能体系统奏效的主要原因在于它帮助系统消耗足够多的 token 来解决问题（'Multi-agent systems work mainly because they help spend enough tokens to solve the problem'）[^18]。在 BrowseComp 评测中，三个因素解释了 95% 的性能方差，其中 token 使用量单独解释了 80% 的方差，工具调用次数与模型选择解释其余部分[^18]。

**该量化结论为 Anthropic 自家产品的内部评测，无第三方复现或独立基准佐证；且与 MAST（arXiv:2503.13657）「MAS 在流行基准上增益往往很小」[^3][^19]、Cognition「多智能体架构非常脆弱」[^21] 直接冲突。矛盾未解决，不能作为本报告的核心量化依据。**

### 4.2 算力放大倍数与同等预算公平对照：证据完全缺失

多智能体相对单智能体的 token/调用次数与延迟放大倍数，以及在同等算力预算下与 self-consistency / 长 CoT / 更强单模型的公平对照结果，在现有 28 条证据中**完全没有直接数据**。仅有 Anthropic 的定性表述（'help spend enough tokens'）与 80% 方差归因[^18]，缺少可引用的倍数与受控对照实验。**现有公开资料不足以确认**该放大倍数与公平对照结论。

### 4.3 负面侧证据的选择性偏差风险

多智能体在哪些任务类型上反而不如单智能体或收益被成本抵消，现有负面侧证据仅来自失败分类学（MAST）[^3][^19] 与厂商博客（Cognition[^21]、Anthropic 选型建议[^22]），**缺少受控实验直接比较「同一任务上 MAS vs 单智能体」并报告 MAS 更差的结果**；AgentVerse[^25]、MoA[^4]、Reflexion[^26] 均为正面结果，存在选择性偏差风险。

### 4.4 代表性框架的单智能体基线对照：无证据

**该子问题在现有证据中无任何证据。** AutoGen 论文（2308.08155）与 MetaGPT 论文（2308.00352）均只与「此前基于对话的多智能体系统」对比，未给出单智能体基线[^27][^28]；其余框架 README 亦未提及基线对照[^6][^8][^10][^14][^15][^16][^17]。AutoGen 论文摘要称其经验性研究展示了框架在数学、编码、问答、运筹学、在线决策、娱乐等多个领域的有效性，但摘要未提及单智能体基线对照[^27]；其引言提出：由于单个 LLM 已能展现广泛能力，不同配置的 agent 之间的对话可以模块化、互补地组合这些能力——这暗示其设计动机是相对单 LLM 的增强，但正文未（在已读部分）给出单智能体基线实验[^27]。MetaGPT 论文摘要明确其对比对象为「previous chat-based multi-agent systems」（此前的基于对话的多智能体系统），而非单智能体基线[^28]。

### 4.5 框架生态的变动性

AutoGen 已进入维护模式（maintenance mode），不再接收新功能或增强，转为社区管理；官方推荐新用户改用 Microsoft Agent Framework（MAF），称其为 AutoGen 的企业级继任者[^6]。Swarm 已被 OpenAI Agents SDK 取代[^15]。自 AG2 v1.0 起，pip install ag2 不再包含 autogen 命名空间或经典 agent 类；经典框架迁至独立仓库 ag2ai/ag2-classic，仍维护且可安装[^17]。这些变动意味着基于特定框架拓扑的结论具有时效性。

### 4.6 自述营销语需谨慎对待

CrewAI README 自述称已有超过 100,000 名开发者通过其社区课程获得认证，并称 CrewAI 正迅速成为生产级 agentic 自动化的标准（属自述营销语，无第三方佐证）[^10]。CAMEL 仓库描述自称 'The first and the best multi-agent framework'（属自述营销语）[^13]。

---

## 参考来源

[^1]: Large Language Model based Multi-Agents: A Survey of Progress and Challenges — https://arxiv.org/abs/2402.01680 (arxiv, 2024-01-21)
[^2]: Improving Factuality and Reasoning in Language Models through Multiagent Debate — https://arxiv.org/abs/2305.14325 (arxiv, 2023-05-23)
[^3]: Why Do Multi-Agent LLM Systems Fail? — https://arxiv.org/abs/2503.13657 (arxiv, 2025-03-17)
[^4]: Mixture-of-Agents Enhances Large Language Model Capabilities — https://arxiv.org/abs/2406.04692 (arxiv, 2024-06-07)
[^5]: microsoft/autogen (GitHub 仓库元数据) — https://github.com/microsoft/autogen (github)
[^6]: AutoGen README (microsoft/autogen) — https://raw.githubusercontent.com/microsoft/autogen/main/README.md (github)
[^7]: langchain-ai/langgraph (GitHub 仓库元数据) — https://github.com/langchain-ai/langgraph (github)
[^8]: LangGraph README (langchain-ai/langgraph) — https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md (github)
[^9]: crewAIInc/crewAI (GitHub 仓库元数据) — https://github.com/crewAIInc/crewAI (github)
[^10]: CrewAI README (crewAIInc/crewAI) — https://raw.githubusercontent.com/crewAIInc/crewAI/main/README.md (github)
[^11]: FoundationAgents/MetaGPT (GitHub 仓库元数据) — https://github.com/FoundationAgents/MetaGPT (github)
[^12]: MetaGPT README (FoundationAgents/MetaGPT) — https://raw.githubusercontent.com/FoundationAgents/MetaGPT/main/README.md (github)
[^13]: camel-ai/camel (GitHub 仓库元数据) — https://github.com/camel-ai/camel (github)
[^14]: CAMEL README (camel-ai/camel) — https://raw.githubusercontent.com/camel-ai/camel/master/README.md (github)
[^15]: Swarm README (openai/swarm) — https://raw.githubusercontent.com/openai/swarm/main/README.md (github)
[^16]: OpenAI Agents SDK README (openai/openai-agents-python) — https://raw.githubusercontent.com/openai/openai-agents-python/main/README.md (github)
[^17]: AG2 README (ag2ai/ag2) — https://raw.githubusercontent.com/ag2ai/ag2/main/README.md (github)
[^18]: How we built our multi-agent research system (Engineering at Anthropic) — https://www.anthropic.com/engineering/multi-agent-research-system (web, 2025-06-13)
[^19]: Why Do Multi-Agent LLM Systems Fail? — https://arxiv.org/abs/2503.13657v3 (arxiv, 2025-03-17)
[^20]: Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies — https://arxiv.org/abs/2502.02533 (arxiv, 2025-02-04)
[^21]: Don't Build Multi-Agents (Cognition) — https://cognition.ai/blog/dont-build-multi-agents (web, 2025)
[^22]: Building effective agents (Anthropic Engineering) — https://www.anthropic.com/engineering/building-effective-agents (web, 2024-12-19)
[^23]: How to think about agent frameworks (LangChain / Harrison Chase) — https://blog.langchain.com/how-to-think-about-agent-frameworks/ (web, 2025-04-20)
[^24]: AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation — https://arxiv.org/abs/2308.08155v2 (arxiv, 2023-08-16)
[^25]: AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors — https://arxiv.org/abs/2308.10848v3 (arxiv, 2023-08-21)
[^26]: Reflexion: Language Agents with Verbal Reinforcement Learning — https://arxiv.org/abs/2303.11366v4 (arxiv, 2023-03-20)
[^27]: AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation — https://arxiv.org/abs/2308.08155 (arxiv, 2023-08-16)
[^28]: MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework — https://arxiv.org/abs/2308.00352 (arxiv, 2023-08-01)