# LLM Agent 系统评测：维度、方法与工具链现状

## 摘要

LLM Agent 的评测正在从「单轮输出打分」转向「多轮交互 + 过程轨迹 + 环境状态」的复合评估。现有证据支持如下判断：主流评测维度可归纳为**结果正确性、过程/轨迹质量、工具使用正确性、交互与协调能力、可靠性与一致性、成本效率、安全性、鲁棒性**八类；方法上形成「基准环境（execution-based）+ LLM-as-a-Judge / Agent-as-a-Judge + 可观测性平台」三层结构。多份一手来源一致指出当前评测对 agentic 系统「不充分」：只关注最终结果会忽略逐步推进的本质[^2]，多数框架只看最终成功率而缺乏过程洞见[^25]。同时，基准本身的可信度问题已被实证：web agent 的既有结果存在过度乐观，根源在基准缺陷[^17]；τ-bench 系列出现任务定义与评测标准不一致，需人工修正版本[^32]。工具链层面，开源与商业平台在 offline/online 评测、trace grading、数据集版本化上已形成较完整能力，但成本效率、安全性、鲁棒性的细粒度评测仍被明确列为未解决缺口[^1]。

---

## 一、为什么 agent 评测不同于模型评测

传统单元测试与集成测试不足以应对 LLM agent，因为模型具有概率性，确定性的 pass/fail 断言通常不适用[^37]。Agent 评测必须同时覆盖**最终输出**与 **trajectory**（agent 返回用户前所采取步骤的列表），后者可与期望步骤列表对比[^37]。

对现有做法的权威批评来自 Agent-as-a-Judge：当代评测技术对 agentic 系统是不充分的，要么只关注最终结果、忽略 agentic 系统逐步推进的本质，要么需要过多人工[^2]。AgentBoard 同样指出，当前评测框架多聚焦最终成功率，对过程揭示的洞见很少，无法深入理解模型能力[^25]。

Anthropic 在架构上区分 workflows（由预定义代码路径编排 LLM 与工具）与 agents（LLM 动态自主决定流程与工具使用），并明确 agentic 系统常以延迟与成本换取更好的任务表现，开发者需判断该权衡是否值得[^20]。这一区分直接决定评测对象：workflow 可用确定性断言，agent 需要过程性评估。

---

## 二、评测维度体系

### 2.1 结果正确性（outcome）

最基础维度。WebArena 的指标是任务完成的功能正确性[^6]；τ-bench 的评测方法是将对话结束时的数据库状态与标注的目标状态（annotated goal state）比较[^8]；OSWorld 为每个任务提供自定义的基于执行的评测脚本，用于可靠、可复现的评测[^27]。

### 2.2 过程 / 轨迹质量（trajectory）

Google ADK 把 agent 评测拆为两部分：评估 trajectory 与工具使用（工具选择、策略、方法效率）；评估最终响应（质量、相关性、正确性）[^37]。AgentBoard 提出 fine-grained progress rate（细粒度进展率）指标，用于捕捉增量式进展，而非仅看最终成功率[^25]。Agent-as-a-Judge 则提供覆盖整个任务求解过程的中间反馈（intermediate feedback for the entire task-solving process）[^2]。

### 2.3 工具使用正确性

DeepEval 的 Agentic Metrics 包含 Task Completion、Tool Correctness、Goal Accuracy、Step Efficiency、Plan Adherence、Plan Quality、Tool Use、Argument Correctness[^9]。ADK 的 tool_trajectory_avg_score 是工具调用 trajectory 的精确匹配（基于参考，非 LLM-judge）[^38]；rubric_based_tool_use_quality_v1 则基于自定义 rubrics 由 LLM 判定工具使用质量[^38]。

### 2.4 交互与协调能力（含用户模拟）

τ-bench 模拟用户（由语言模型模拟）与语言 agent 之间的动态对话，agent 配备领域特定的 API 工具与策略指南，论文指出既有基准未测试 agent 与人类用户的交互能力，也未测试其遵循领域特定规则的能力[^8]。τ²-bench 进一步指出既有对话 agent 基准只模拟 single-control 环境（仅 AI agent 可用工具，用户是被动信息提供者），并提出 Telecom dual-control 域，建模为 Dec-POMDP，agent 与用户都使用工具在共享动态环境中行动[^30]。实验显示 agent 从 no-user 切换到 dual-control 时性能显著下降[^30]。

### 2.5 可靠性与一致性

τ-bench 提出新指标 pass^k，用于评估 agent 在多次试验中的行为可靠性；结果显示即使是最先进的函数调用 agent（如 gpt-4o）任务成功率也 <50%，且表现相当不一致（retail 领域 pass^8 <25%）[^8]。

### 2.6 成本效率

被综述明确列为未来必须解决的关键缺口之一[^1]。可操作的度量示例见 τ²-Bench-Verified：README 报告 price/performance，Grok 4.1 Fast Reasoning 得分/成本比 2.24 最高，Claude Sonnet 4.5 仅 0.14[^32]。ADK 提供效率类指标：tool_call_count_v1、inference_call_count_v1、token_usage_v1、invocation_duration_v1[^38]。

### 2.7 安全性

同样被综述列为关键缺口[^1]。ADK 提供 safety_v1（agent 响应的安全性/无害性，LLM-as-a-Judge）与 hallucinations_v1（由 LLM 判定 agent 响应相对上下文的 groundedness）[^38]。NIST 表示正在建设 SI 测试与评估生态以评估前沿模型，并称改进的测试与评估方法与基准对验证 SI 系统至关重要，其产出为自愿性指南、工具与资源[^21]。

### 2.8 鲁棒性

被综述列为关键缺口[^1]。一个非模型层面的鲁棒性案例：Anthropic 公开披露 2025 年 8 月至 9 月初三个基础设施 bug 间歇性导致 Claude 响应质量下降，第一个 bug 于 8 月 5 日引入，影响约 0.8% 的 Sonnet 4 请求；8 月 31 日最严重时段 16% 的 Sonnet 4 请求受影响；约 30% 的 Claude Code 用户在该时段至少有一条消息被路由到错误服务器类型[^19]。该案例说明模型质量下降可能来自服务基础设施而非模型本身，且用户反馈难以与正常波动区分，导致检测与定位延迟[^19]。

### 2.9 多智能体特有维度

MAST（Multi-Agent System Failure Taxonomy）基于对 150 条轨迹的严格分析构建，标注者间一致性 kappa = 0.88，识别出 14 种独特失败模式，归为 3 大类：(i) 系统设计问题、(ii) 智能体间错位、(iii) 任务验证[^39]。论文指出多智能体 LLM 系统在流行基准上的性能增益往往很小，这一差距凸显了系统性理解 MAS 失败原因的必要性[^39]。

---

## 三、评测方法

### 3.1 基准环境（execution-based）

**通用 agent 能力**
- AgentBench：多维基准，由 8 个不同环境组成，评估 LLM-as-Agent 的推理与决策能力；覆盖 API 型与开源 LLM；顶级商业 LLM 在复杂环境中表现出较强 agent 能力，但与不超过 70B 的众多开源模型之间存在显著性能差距[^5]。识别出的主要失败原因：长程推理、决策与指令遵循能力薄弱[^5]。
- GAIA：面向 General AI Assistants，提出 real-world questions，需要 reasoning、multi-modality handling、web browsing 以及一般性的 tool-use proficiency[^23]。人类受访者得分 92%，而配备插件的 GPT-4 得分 15%[^23]。Level 3 样例题要求结合 NASA Astronomy Picture of the Day（2006-01-21）图像、NASA Astronaut Group 成员信息与在轨时长数据，输出格式为「姓氏; 分钟数（千分位逗号）」，ground truth 为「White; 5876」[^24]。GAIA 的答案被设计为 unambiguous，且刻意使其不太可能在训练数据中以纯文本形式直接找到[^24]。

**Web agent**
- WebArena：包含四个常见领域的全功能网站——电商、社交论坛、协作软件开发、内容管理，配备工具（如地图）与外部知识库（如用户手册）[^6]。最佳 GPT-4 基线 agent 端到端任务成功率仅 14.41%，显著低于人类 78.24%[^6]。
- Online-Mind2Web：含 300 个多样且真实的任务，覆盖 136 个网站，以接近真实用户使用方式评测 agent；论文提出新的 LLM-as-a-Judge 自动评测方法，报告其与人类判断达到约 85% 的一致性[^17]。
- BrowseComp：含 1,266 个问题，要求持续导航互联网以寻找难以发现、相互纠缠的信息；设计为预测答案为简短且可对照参考答案轻松验证[^22]。作者承认该基准回避了真实用户查询分布的挑战，如生成长答案或消解歧义[^22]。

**软件工程 agent**
- SWE-bench：包含 2,294 个来自真实 GitHub issue 及其对应 pull request 的问题，覆盖 12 个流行 Python 仓库；任务设定为给定代码库与 issue 描述，模型需编辑代码库以解决问题[^7]。解决 SWE-bench 问题常需跨多个函数、类甚至文件协调修改，要求模型与执行环境交互、处理超长上下文并进行复杂推理[^7]。当时最优模型 Claude 2 仅能解决 1.96% 的 issue[^7]。

**计算机操作 agent**
- OSWorld：被作者定位为 first-of-its-kind scalable, real computer environment for multimodal agents，支持跨操作系统（Ubuntu、Windows、macOS）的任务设置、基于执行的评测与交互式学习；包含 369 个计算机任务[^27]。人类可完成 72.36% 以上的任务，而最佳模型仅达到 12.24% 成功率；模型主要失败于 GUI grounding（界面元素定位）和 operational knowledge（操作知识）[^27]。项目页标注 8 个 Google Drive 任务可能需手动配置或因网络依赖可被排除，即 361 任务版本；2025-07-28 发布 OSWorld-Verified，2026-06-26 发布 OSWorld 2.0[^28]。

**对话 / 工具-用户交互**
- τ-bench 与 τ²-bench（见 2.4、2.5）。τ²-bench 提供 compositional task generator，从原子组件程序化生成多样、可验证的任务，保证域覆盖与可控复杂度；用户模拟器与环境紧耦合，其行为受工具与可观测状态约束，以提升模拟保真度[^30]。

### 3.2 LLM-as-a-Judge 与 Agent-as-a-Judge

LLM 作为评估者可提供可扩展、低成本、一致的评估，是传统专家驱动评估的有力替代方案；该综述将可靠性视为需要精心设计与标准化的重大挑战，提出提升可靠性的策略包括改善一致性、缓解偏见、适配多样评估场景，并提出评估 LLM-as-a-Judge 系统自身可靠性的方法论与配套新基准[^16]。

Agent-as-a-Judge 是 LLM-as-a-Judge 的有机扩展，加入 agentic 特性以提供覆盖整个任务求解过程的中间反馈；实验结论是它大幅优于 LLM-as-a-Judge，且与人工评测基线一样可靠[^2]。其概念验证测试床 DevAI 包含 55 个真实自动化 AI 开发任务，带有丰富人工标注，共 365 条层级化用户需求[^2]。

MAST 也开发了 LLM-as-a-Judge 标注流水线，与人工标注高度一致，用于规模化标注[^39]。

### 3.3 评测粒度

DeepEval 支持四档评测粒度：端到端黑盒、完整 agent 轨迹、单个 agent 步骤（LLM 调用、工具使用、检索、子 agent 交接）[^9]。其指标通过 LLM-as-a-judge 与本地运行的 NLP 模型实现，支持任意 LLM 作为裁判[^9]。

### 3.4 Offline 与 Online 评测闭环

LangSmith Evaluation 区分两类评测：offline evaluations（部署前测试，目标为数据集中的 examples，含 benchmarking、regression testing、unit testing、backtesting）与 online evaluations（生产监控，目标为 tracing 产生的 runs 和 threads，含实时监控、异常检测、生产反馈）[^15]。建议先为每个关键组件手工构造 5-10 个「good」示例作为 ground truth[^15]。针对不同系统给出评测对象示例：RAG 系统（好的检索与答案）、Agent（正确的工具选择、参数格式或轨迹）、Chatbot（有帮助、符合品牌的回复）[^15]。强调 offline 与 online 形成迭代反馈闭环：online 发现问题转为 offline 测试用例，offline 验证修复，online 确认生产改进[^15]。

OpenAI 的 trace grading 工作流：在 dashboard 打开 Logs > Traces；检视代表性 trace；创建 grader 并对所选 trace 运行；用结果改进 prompt、工具面、路由逻辑或 guardrails[^35]。一个 trace 捕获单次运行的端到端记录，包括 model calls、tool calls、guardrails 和 handoffs[^35]。当需要可重复性时，从单条 trace 转向可重复的 datasets 与 eval runs[^35]。

---

## 四、工具链与框架

### 4.1 开源评测框架

| 工具 | 定位与关键能力 | 证据 |
|---|---|---|
| DeepEval | 类 Pytest 的 LLM 单元测试；Agentic Metrics 八项；四档评测粒度；10,366 Commits | [^9] |
| RAGAS | LLM-based 与传统指标；测试数据生成；ragas quickstart CLI（agent_evals 等标注 Coming Soon）；1,147 Commits | [^10] |
| Inspect（UK AISI） | 内置 prompt engineering、工具使用、多轮对话与 model-graded evaluations；200+ 预置评测；8,005 Commits | [^12] |
| OpenAI Evals | eval registry；private evals；Completion Function Protocol 支持 prompt chains 或 tool-using agents；691 Commits | [^14] |
| Arize Phoenix | OpenTelemetry tracing；LLM 评测；Datasets/Experiments/Playground/Prompt Management；Remote MCP Server；10,288 Commits | [^11] |
| Langfuse | 可观测性 + LLM-as-a-judge/Code evaluators/人工标注；Datasets；自托管；10,139 Commits；自 2026 年 1 月起归属 ClickHouse | [^13] |
| LangSmith SDK | debug、evaluate、monitor；与 LangChain Python/JS 原生集成；3,175 Commits | [^33] |
| SAP agent-quality-inspect | 对来自不同来源与框架的 agentic AI 做基准测试，产出可跨用例与数据集比较的统计结果；Python；79 stars；最近推送 2026-06-26 | [^4] |

注：SAP/agent-quality-inspect 仅获取到仓库描述与元数据，未读取 README 正文，故不作为维度体系的结论依据[^4]。

### 4.2 厂商官方评测面

- **OpenAI**：建议在调试行为阶段从 traces 入手，trace grading 是识别工作流级问题最快的方式；graders 用结构化标准对 trace 打分[^35]。需注意 OpenAI 正在弃用 Evals 平台：现有内容在过渡期仍可用，2026-10-31 起对现有用户变为只读，平台计划于 2026-11-30 关停；建议评测新手或想要更迭代式实验环境的用户改用 Datasets[^36]。
- **Google ADK**：提供完整 criteria 表，覆盖工具使用 trajectory、响应质量与安全性，并标注每项是否基于参考、是否需要 rubrics、是否使用 LLM-as-a-Judge、是否支持用户模拟[^38]。具体包括 tool_trajectory_avg_score、response_match_score（ROUGE-1）、response_evaluation_score（Vertex AI coherence）、final_response_match_v2、rubric_based_final_response_quality_v1、rubric_based_tool_use_quality_v1、rubric_based_multi_turn_trajectory_quality_v1、hallucinations_v1、safety_v1、per_turn_user_simulator_quality_v1、multi_turn_task_success_v1、multi_turn_trajectory_quality_v1，以及四类效率指标[^38]。

### 4.3 评测框架的自我修正机制

τ-bench 系列的演进是重要信号。τ²-Bench-Verified 是原 τ²-bench 的修正与人工验证版本，修复了原数据集中任务定义、expected actions、评测标准与所述 policy 或数据库内容不一致的问题，对应论文 SABER: Small Actions, Big Errors — Safe-guarding Mutating Steps in LLM Agents[^32]。τ³-bench 包含 75+ 任务修复（移除错误 expected actions、澄清歧义指令、修复不可能约束），依据同一 SABER 分析；2026 年 7 月 v1.0.1 评分更新修复了 banking_knowledge 任务错误，导致该域分数变化，受影响 leaderboard 提交被重新评分[^31]。评测 agent 时应使用 base task split 以匹配原始 τ-bench 结构[^31]。

---

## 五、矛盾与不确定性

**1. 基准可信度与「进步幻觉」。** Online-Mind2Web 论文对当前 web agent 能力做了严格评估，结论与既有报告不同，指出此前结果存在过度乐观，并将这一差距归因于现有基准的缺陷[^17]。这与 τ-bench 系列发现的任务定义不一致问题[^32]相互印证：基准本身的质量是评测结论可信度的前提。

**2. 评测方法之间的一致性数据有限。** Agent-as-a-Judge 报告其与人工评测基线一样可靠[^2]；Online-Mind2Web 的 LLM-as-a-Judge 方法与人类判断达到约 85% 的一致性[^17]；MAST 的 LLM-as-a-Judge 标注流水线与人工标注高度一致[^39]。但三者使用不同的任务域与人工基线定义，现有公开资料不足以确认这些一致性数字之间可直接横向比较。

**3. 过程指标与结果指标的关系未量化。** AgentBoard 主张 fine-grained progress rate 优于仅看最终成功率[^25]，Agent-as-a-Judge 主张中间反馈优于仅看最终结果[^2]，但证据中未给出「过程指标提升在多大程度上预测结果指标提升」的定量结论。

**4. 成本效率评测缺乏统一口径。** 综述将成本效率列为关键缺口[^1]；τ²-Bench-Verified 报告 price/performance 比值[^32]；ADK 提供 token_usage_v1 等效率指标[^38]。三者的成本口径（是否含 user simulator 成本、是否含重试）在证据中未统一说明，现有公开资料不足以确认可跨平台比较。

**5. 工具链对比信息存在利益冲突。** Langfuse vs. LangSmith 对比页为 Langfuse（LangSmith 竞品）官方发布的对比指南，属竞品营销材料，其关于 LangSmith 的事实主张未经独立来源验证；页面自述所有 LangSmith 事实均于 2026 年 10 月对照 LangChain 公开来源核查，但非独立核实[^34]。该页面声称 LangSmith 为专有平台、仅客户端 SDK 开源、自托管/自定义 SSO/RBAC 均为 Enterprise 专属，并给出定价对比示例（50 万 traces/月、5 用户场景下 Langfuse Pro 为 $621，LangSmith Plus 为 $3,895）[^34]。这些数字不应作为独立结论使用。

**6. 非权威来源需剔除。** mightbesaad/llm-reliability-evals 仅 1 star、1 fork，为个人项目，其 8 模式失败分类法未经同行评审验证[^40]。该仓库自述其定位与能力评测、安全评测正交[^40]，但证据不足以支撑将其分类法纳入主流维度体系。

**7. 政策与标准语境的时间线异常。** NIST 页面显示依据 2026 年 9 月 29 日行政令 EO 14434「Inaugurating the Era of Super Intelligence」更新表述[^21]。该日期晚于本报告多数证据的抓取时点，其与当前评测实践的关系现有公开资料不足以确认。

**8. 综述层面的缺口陈述。** 综述明确指出未来研究必须解决的关键缺口：成本效率、安全性、鲁棒性的评估，以及开发细粒度、可扩展的评测方法[^1]；并指出当前趋势是转向更真实、更具挑战性且持续更新的评测[^1]。这意味着本报告归纳的维度体系中，2.6–2.8 三节的方法成熟度显著低于 2.1–2.5。

---

## 参考来源

[^1]: Survey on Evaluation of LLM-based Agents — https://arxiv.org/abs/2503.16416 (arxiv, 2025-03-20 (v1); v2 2026-04-23)
[^2]: Agent-as-a-Judge: Evaluate Agents with Agents — https://arxiv.org/abs/2410.10934 (arxiv, 2024-10-14 (v1); v2 2024-10-16)
[^3]: AgentBench: Evaluating LLMs as Agents — https://arxiv.org/abs/2308.03688 (arxiv, 2023-08-07 (v1); v3 2025-10-04)
[^4]: SAP/agent-quality-inspect — https://github.com/SAP/agent-quality-inspect (github, last push 2026-06-26)
[^5]: AgentBench: Evaluating LLMs as Agents — https://arxiv.org/abs/2308.03688v3 (arxiv, 2023-08-07)
[^6]: WebArena: A Realistic Web Environment for Building Autonomous Agents — https://arxiv.org/abs/2307.13854v4 (arxiv, 2023-07-25)
[^7]: SWE-bench: Can Language Models Resolve Real-World GitHub Issues? — https://arxiv.org/abs/2310.06770v3 (arxiv, 2023-10-10)
[^8]: τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains — https://arxiv.org/abs/2406.12045v1 (arxiv, 2024-06-17)
[^9]: confident-ai/deepeval — The LLM Evaluation Framework (README) — https://github.com/confident-ai/deepeval (github)
[^10]: explodinggradients/ragas — Supercharge Your LLM Application Evaluations (README) — https://github.com/explodinggradients/ragas (github)
[^11]: Arize-ai/phoenix — Arize Phoenix open-source AI observability platform (README) — https://github.com/Arize-ai/phoenix (github)
[^12]: UKGovernmentBEIS/inspect_ai — Inspect: LLM evaluation framework by UK AI Security Institute (README) — https://github.com/UKGovernmentBEIS/inspect_ai (github)
[^13]: langfuse/langfuse — Open source LLM engineering platform (README) — https://github.com/langfuse/langfuse (github)
[^14]: openai/evals — OpenAI Evals (README) — https://github.com/openai/evals (github)
[^15]: LangSmith Evaluation — LangChain official docs — https://docs.smith.langchain.com/evaluation (web)
[^16]: A Survey on LLM-as-a-Judge — https://arxiv.org/abs/2411.15594 (arxiv, 2024-11-23 (v1), last revised 2025-10-19 (v6))
[^17]: An Illusion of Progress? Assessing the Current State of Web Agents — https://arxiv.org/abs/2504.01382 (arxiv, 2025-04-02 (v1), last revised 2025-10-08 (v4))
[^18]: Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks — https://arxiv.org/abs/2503.09572 (arxiv, 2025-03-12 (v1), last revised 2025-04-22 (v3))
[^19]: A postmortem of three recent issues (Anthropic Engineering) — https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues (web, 2025-09-17)
[^20]: Building effective agents (Anthropic Engineering) — https://www.anthropic.com/engineering/building-effective-agents (web, 2024-12-19)
[^21]: Super intelligence (NIST) — https://www.nist.gov/artificial-intelligence (web)
[^22]: BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents — https://arxiv.org/abs/2504.12516 (arxiv, 2025-04-16)
[^23]: GAIA: a benchmark for General AI Assistants — https://arxiv.org/abs/2311.12983 (arxiv, 2023-11-21)
[^24]: GAIA: A Benchmark for General AI Assistants (HTML full text, Introduction) — https://arxiv.org/html/2311.12983v1 (arxiv, 2023-11-21)
[^25]: AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents — https://arxiv.org/abs/2401.13178v2 (arxiv, 2024-01-24)
[^26]: AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents (HTML full text, Introduction) — https://arxiv.org/html/2401.13178v2 (arxiv, 2024-01-24)
[^27]: OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments (arXiv:2404.07972v2) — https://arxiv.org/abs/2404.07972v2 (arxiv, 2024-04-11)
[^28]: OSWorld 官方项目页 — https://os-world.github.io/ (web)
[^29]: xlang-ai/OSWorld README — https://github.com/xlang-ai/OSWorld (github)
[^30]: τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment — https://arxiv.org/abs/2506.07982v1 (arxiv, 2025-06-09)
[^31]: sierra-research/tau2-bench (README) — https://github.com/sierra-research/tau2-bench (github)
[^32]: amazon-agi/tau2-bench-verified (README) — https://github.com/amazon-agi/tau2-bench-verified (github)
[^33]: langchain-ai/langsmith-sdk (README) — https://github.com/langchain-ai/langsmith-sdk (github)
[^34]: Langfuse vs. LangSmith (Langfuse 官方竞品对比页) — https://langfuse.com/faq/all/langsmith-alternative (web)
[^35]: OpenAI Platform Docs — Agent Evals (evaluation surfaces for agent workflows) — https://platform.openai.com/docs/guides/agent-evals (web)
[^36]: OpenAI Platform Docs — Evaluations (Evals API guide) — https://platform.openai.com/docs/guides/evals (web)
[^37]: Google Agent Development Kit (ADK) Docs — Why evaluate agents / What to evaluate — https://google.github.io/adk-docs/evaluate/ (web)
[^38]: Google ADK Docs — Evaluation Criteria (metrics table) — https://google.github.io/adk-docs/evaluate/criteria/ (web)
[^39]: Why Do Multi-Agent LLM Systems Fail? — https://arxiv.org/abs/2503.13657v3 (arxiv, 2025-03-17)
[^40]: mightbesaad/llm-reliability-evals — https://github.com/mightbesaad/llm-reliability-evals (github)