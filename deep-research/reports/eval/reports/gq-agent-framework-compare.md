## 摘要

三个框架的设计差异可以用一句话概括：**LangGraph 把 agent 工作流建模为显式状态图，AutoGen 把多 agent 协作建模为消息驱动的对话与团队，CrewAI 把协作建模为带角色分工的项目团队与流程**。LangGraph 的核心抽象是 State / Nodes / Edges，构建流程为定义 state → add_node/add_edge → compile，底层执行是受 Pregel 启发的 message passing 与 super-step 模型[^1]；AutoGen 分两层，AgentChat 是构建在 autogen-core 之上的高层多 agent API，提供预设 Agents 与 Teams（RoundRobin / Selector / Magentic-One / Swarm / GraphFlow）[^3][^6]；CrewAI 以 role / goal / backstory 定义的 Agent 与 Crew 为单位，Process 仅支持 sequential 与 hierarchical 两种[^7][^8][^9]，另以 Flows 提供事件驱动的跨 Crew 编排[^10]。适用场景上，LangGraph 偏向需要显式控制流、持久化与 human-in-the-loop 的长时运行有状态 agent[^1][^2][^19][^34]；AutoGen 偏向研究型多 agent 对话编排与灵活的消息/运行时抽象[^3][^4][^28]；CrewAI 偏向角色化团队协作的快速搭建与流程化任务执行[^7][^8][^40]。

需要前置声明两点不确定性：其一，现有证据中**没有任何针对三者的对比基准或实验数字**，唯一的 arxiv 论文 MacNet 研究的是自建 DAG 协作网络，并未评测这三个框架[^27]；其二，AutoGen 的长期定位存在张力——Microsoft 官方博客与迁移指南称其能力被整合进 Microsoft Agent Framework[^28][^30]，但 AutoGen 仓库在 2025-09-30 仍发布 0.7.5[^38]、2026-10 仍有 push[^12]，是否进入维护模式缺少一手证据。

---

## 一、LangGraph：以显式状态图为核心的底层编排框架

**核心定义。** LangGraph 官方表述为「At its core, LangGraph models agent workflows as graphs」，行为由三个组件定义[^1]：

- **State**：共享数据结构，表示应用当前快照，通常用共享 state schema 定义；State 由 schema 加 reducer 函数组成，reducer 规定更新如何应用到 state[^1]。
- **Nodes**：编码 agent 逻辑的函数，输入是当前 state，执行计算或副作用，返回更新后的 state[^1]。
- **Edges**：根据当前 state 决定下一个执行哪个 Node 的函数，可以是条件分支或固定转移[^1]。

官方口号是「nodes do the work, edges tell what to do next」，并强调 Nodes 和 Edges 本质上只是函数，可以包含 LLM 也可以只是普通代码[^1]。主图类是 StateGraph，由用户定义的 State 对象参数化；构建流程为定义 state → add_node/add_edge → **必须 compile**，compile 阶段做结构检查（如无孤立节点）并指定 checkpointers、breakpoints 等运行时参数[^1]。

**执行模型。** 底层图算法用 message passing，受 Google Pregel 启发，以离散的 super-step 推进；同一 super-step 内并行运行的节点属于同一步，顺序运行的属于不同步；所有节点 inactive 且无消息在途时终止[^1]。这一模型直接决定了它的并行语义是「图结构决定」而非「对话轮次决定」。

**持久化是双系统设计。** Checkpointer 把某个 thread 的图状态持久化为 checkpoints，用于短期、thread 作用域记忆（对话连续性、human-in-the-loop、time travel、容错）；Store 在图状态之外持久化应用自定义数据，用于长期、跨 thread 记忆（用户偏好、事实、共享知识）；两者在 compile 时传入，官方建议多数应用并用[^2]。实现包括 InMemorySaver、PostgresSaver、SqliteSaver[^2]。

**human-in-the-loop 是语言级原语。** `interrupt()` 可在节点任意位置调用，暂停执行并通过持久化层保存状态，无限期等待；恢复用 `Command(resume=...)`，且必须使用相同 thread ID；节点恢复时会从调用 interrupt 的节点开头重新执行[^19]。前提是 checkpointer、thread ID 与 JSON 可序列化 payload[^19]。这比「在节点前后设静态断点」更细粒度，官方明确区分了动态 interrupt 与静态 breakpoints[^19]。

**定位与工程状态。** PyPI 自述为「a low-level orchestration framework for building, managing, and deploying long-running, stateful agents」，提供 durable execution、streaming、human-in-the-loop、persistence、memory 等基础设施[^34]。许可证为 MIT[^33][^34]，Development Status 为 5 - Production/Stable[^34]。GitHub 仓库 43,859 stars、7,534 forks，主语言 Python，最近 push 2026-10-25[^11]；采用 monorepo 多包独立发版，1.2.17 于 2026-10-24 由 CI 机器人发布[^35]。

---

## 二、AutoGen：分层架构 + 消息驱动 + 团队预设

**分层定义。** 官方表述为「AgentChat is a high-level API for building multi-agent applications. It is built on top of the autogen-core package.」[^3]。初学者从 AgentChat 起步，高级用户用 autogen-core 的事件驱动编程模型获得更强控制[^3]。AgentChat 提供「直观的默认值」：带预设行为的 Agents 与带预定义多 agent 设计模式的 Teams[^3]。

**Agent 层。** 所有 agent 共享 name、description、run、run_stream 属性；官方明确 agent 被期望是**有状态的**，run 应以新消息调用，而不是传入完整历史[^5]。内置 AssistantAgent 被官方自我定性为用于原型与教学目的的 "kitchen sink" agent，非常通用，建议理解设计后自行实现[^5]。工具由同一个 agent 在同一次 run() 调用内直接执行[^5]。

**Team 层与协调模式。** Team 定义是「a team is a group of agents that work together to achieve a common goal」[^6]。预设包括 RoundRobinGroupChat（轮流发言、共享上下文、每回合广播给所有 agent）、SelectorGroupChat（由 ChatCompletion 模型基于共享上下文选下一个发言者）、MagenticOneGroupChat（面向开放式 web 与文件任务）、Swarm（用 HandoffMessage 表示切换）[^6]。SelectorGroupChat 的运行流程为：分析上下文（含参与者 name、description）→ 模型选发言者 → 被选中 agent 响应并广播 → 检查终止条件[^23]。官方使用建议是：简单任务先从单 agent 开始，待单 agent 不足时再转向多 agent team[^6]。

**终止条件是独立的一等抽象。** 基类 TerminationCondition 是 callable，接收自上次调用以来的消息序列，应终止则返回 StopMessage 否则 None；有状态但在每次 run/run_stream 后自动重置；可用 AND / OR 组合；内置 MaxMessageTermination、TextMentionTermination、TokenUsageTermination、TimeoutTermination、HandoffTermination、ExternalTermination 等[^20]。

**Core 层：runtime 是执行环境。** runtime 负责 agent 间通信、身份与生命周期管理、安全与隐私边界[^4]。支持 standalone（单进程，如 SingleThreadedAgentRuntime）与 distributed（多进程、跨语言跨机器）两种，二者同一套 API 可切换[^4]。distributed runtime 由 host servicer 与多个 worker 组成[^4]，但官方明确标注其为**实验性**，API 会有破坏性变更，需额外安装 `autogen-ext[grpc]`，跨语言场景所有消息类型须用共享 protobuf schema[^49]。注意 AgentChat 的 agent 由应用创建、不由 runtime 直接管理，要在 Core 中使用需写 RoutedAgent 包装器[^48]。

**状态与代码执行。** save_state()/load_state() 保存加载 agent、team 及终止条件状态，典型场景是无状态端点需从持久化存储加载状态的 Web 应用[^22]。代码执行提供 DockerCommandLineCodeExecutor（容器内执行，默认镜像 python:3-slim）与 LocalCommandLineCodeExecutor（宿主机执行，官方以 Attention 标注需谨慎）[^21]。

**工程状态与定位张力。** GitHub 61,554 stars、9,313 forks，最近 push 2026-10-16[^12]。仓库根 LICENSE 为 CC BY 4.0[^36]，而 PyPI 上 autogen-agentchat 声明 MIT，两者不一致，须区分「仓库根许可证」与「具体发行包许可证」[^36][^37]。PyPI 最新版 0.7.5 发布于 2025-09-30[^37][^38]。Microsoft 官方博客将 AutoGen 定位为「源自研究的实验性多智能体编排」，企业就绪度评为 "Minimal"，并称 Microsoft Agent Framework「doesn't replace Semantic Kernel and AutoGen — it builds on them」[^28]；迁移指南称 AutoGen 采用事件驱动核心加高层 Team 抽象，而 MAF 以类型化的、基于图的 Workflow 为中心[^30]。

---

## 三、CrewAI：角色化团队 + 双流程 + Flows 编排

**Agent 定义。** CrewAI 中 Agent 是自主单元，能执行特定任务、基于 role 与 goal 做决策、使用工具、与其他 agent 沟通协作、保持交互记忆、并在允许时委派任务[^7]。官方类比为具备特定技能与职责的专门团队成员（Researcher / Writer）[^7]。核心属性是 role（职能与专长）、goal（引导决策的个体目标）、backstory（上下文与人格设定）[^7]。关键行为参数包括 allow_delegation（默认 False）、max_iter（默认 20）、reasoning（默认 False）、respect_context_window（默认 True，通过摘要控制上下文窗口）、code_execution_mode（safe 用 Docker / unsafe 直接执行，默认 safe）[^7][^25]。

**Crew 与 Process。** Crew 定义是「a collaborative group of agents working together to achieve a set of tasks. Each crew defines the strategy for task execution, agent collaboration, and the overall workflow.」[^8]。核心属性为 tasks、agents、process（默认 sequential）[^8]。**Process 只有两种实现**：Sequential（按顺序执行，前一任务输出作为下一任务上下文）与 Hierarchical（模拟企业层级，manager 负责规划、委派与验证，任务不预先分配，需 manager_llm 或 manager_agent）[^9][^24]。Process 类实现为 Enum，把取值限定为这两种以保证类型安全[^9][^24]。这是三者中流程抽象最收敛的：LangGraph 的边是任意函数、AutoGen 的 Team 有五种预设，CrewAI 的 Process 是封闭枚举。

**Flows：与 Crew 互补的事件驱动层。** Flow 用于简化 AI 工作流创建与管理，官方称其允许创建「structured, event-driven workflows」，四特性为简化工作流创建、状态管理、事件驱动架构、灵活控制流（条件逻辑、循环、分支）[^10]。用 `@start()` 标记入口（可多个、常并行）、`@listen()` 标记监听另一任务输出[^10]。每个 Flow 实例在 state 中自动获得唯一 UUID，kickoff() 返回最后完成方法的输出[^10]。官方明确 Flow 与 Crew 是互补关系：Flow 负责跨 Crew/任务的结构化编排，Crew 负责一组 agent 围绕任务集合的协作[^10]。

**任务级控制。** Task 支持 human_input（默认 False）、async_execution（默认 False）、guardrail / guardrails（进入下一任务前校验输出，guardrail_max_retries 默认 3）、context（指定哪些任务输出作为上下文）、output_file / output_json / output_pydantic、callback[^26]。

**定位与工程状态。** PyPI 自述为「a lean, lightning-fast Python framework built entirely from scratch—completely independent of LangChain or other agent frameworks」，提供 Crews 与 Flows 两种编排范式[^40]。许可证 MIT[^39]。GitHub 60,081 stars、8,869 forks，最近 push 2026-10-24[^13]。版本 1.15.30，patch 高频递增，1.15.29 为功能密集版本、1.15.30 为极小修复版本[^41]。README 目录含 "CrewAI vs LangGraph" 对比章节，说明其将 LangGraph 视为主要对照对象[^40]。商业侧有 CrewAI AMP（Agent Management Platform），提供托管部署、REST API、可观测性、Tool Repository、Webhook Streaming、Crew Studio[^46]；定价两档 Basic（Free，50 workflow executions/月）与 Enterprise（Custom，含 SSO、RBAC、VPC 部署）[^47]。

---

## 四、设计理念的三维对比

**维度一：编排的「第一性抽象」不同。**

| 框架 | 第一性抽象 | 控制流载体 | 流程空间 |
|---|---|---|---|
| LangGraph | 状态图（State + Nodes + Edges） | 边函数 + super-step 调度 | 开放：边是任意函数，可条件可固定[^1] |
| AutoGen | 消息 + 团队 | Team 预设 + 终止条件 | 半开放：五种预设 Team，可用 Core API 自行实现[^3][^6][^23] |
| CrewAI | 角色 + 任务 + 流程 | Process 枚举 + Flows 装饰器 | 收敛：Process 仅 sequential / hierarchical[^9][^24] |

LangGraph 的抽象层级最低——官方自述为 "low-level orchestration framework"[^34]，Nodes 和 Edges 只是函数[^1]，控制权完全交给开发者。CrewAI 的抽象层级最高——开发者声明 role/goal/backstory 与任务列表，由框架按 Process 决定执行顺序[^7][^8][^9]。AutoGen 居中且分层：AgentChat 提供「直观的默认值」[^3]，需要时下沉到 autogen-core 的事件驱动模型[^3][^4]。

**维度二：并行与信息传播模型不同。** LangGraph 的并行由图的 super-step 结构决定，同一步内节点并行、不同步顺序执行[^1]。AutoGen 的 RoundRobinGroupChat 中每个 agent 在自己回合把响应**广播给其他所有 agent** 以保持团队上下文一致[^6]，SelectorGroupChat 同样广播[^23]。CrewAI 的 Sequential 流程中前一任务输出作为下一任务上下文，可用 Task 的 context 参数自定义[^9][^24]。值得注意的对照证据：arxiv 论文 MacNet 用 DAG 组织 agent 并按拓扑序编排，每轮只有精炼后的 artifact 沿边传播而非整段对话，作者称此举「prevents global broadcasting and suppresses context explosion」，即对群聊式全局广播成本的间接批评[^27]；该论文还报告不规则拓扑优于规则拓扑，并支持超千 agent 协作与「协作缩放律」[^27]。但需强调：**MacNet 并未评测这三个框架**，不能据此直接判定某一框架优劣。

**维度三：状态与持久化的作用域划分不同。** LangGraph 显式区分 thread 作用域的 Checkpointer 与跨 thread 的 Store，两者互补[^2]，并把 interrupt 建立在持久化层之上[^19]。AutoGen 用 save_state()/load_state() 保存 agent、team 与终止条件状态[^22]，终止条件在每次 run 后自动重置以支持续接[^20]。CrewAI 在 Flow 层提供 state（每个实例自动获 UUID，执行期间持久保存数据）[^10]，Crew 层提供 memory（短期/长期/实体记忆）与 checkpoint 属性[^8]。三者的差异在于：LangGraph 把持久化当作图执行的一等基础设施，AutoGen 当作可显式调用的序列化能力，CrewAI 则分散在 Flow state 与 Crew memory 两处。

---

## 五、适用场景的差异

**LangGraph 适合：需要显式控制流与长时运行的有状态 agent。** 判据来自其自身能力面：durable execution、streaming、human-in-the-loop、persistence、memory[^34]；compile 阶段可指定 checkpointer 与 breakpoints[^1]；interrupt 支持 response_schema 校验恢复值（需 langgraph>=1.2.12）[^19]；支持多 schema，可定义 private schema 供内部节点通信并显式约束 input_schema/output_schema[^1]。当工作流需要精确的暂停/恢复语义与跨会话记忆分层时，这套原语是三者中最完整的。

**AutoGen 适合：研究型多 agent 对话编排与需要运行时灵活性的场景。** AgentChat 的 Team 预设覆盖轮流、模型选择、handoff、开放式任务等多种协调模式[^6][^23]；终止条件体系（含 TokenUsageTermination、ExternalTermination 等）粒度细[^20]；Core 层提供 standalone 与 distributed 两种 runtime 且同一套 API 可切换[^4]。但需注意：distributed runtime 官方标注为实验性[^49]；AssistantAgent 被官方定性为教学用 "kitchen sink"[^5]；AutoGen Studio 被官方两次明确声明「not meant to be a production-ready app」与「research prototype」[^44][^45]，且 README 警告会有破坏性变更[^45]。官方自己也建议简单任务先从单 agent 开始[^6]。

**CrewAI 适合：角色化团队协作的快速搭建与流程化任务执行。** role/goal/backstory 的建模方式贴近人类团队分工[^7]；Process 只有两种、语义清晰[^9][^24]；Task 层内建 guardrail 校验、结构化输出、human_input、async_execution[^26]；需要跨 Crew 编排时用 Flows 的 @start/@listen 装饰器[^10]。官方自述「completely independent of LangChain or other agent frameworks」[^40]，对不希望引入 LangChain 技术栈的团队是一个明确卖点。企业侧有 AMP 提供托管部署与治理能力[^46][^47]。

**一个跨框架的观察：三者都在向「图/工作流」收敛。** LangGraph 本来就是图[^1]；AutoGen AgentChat 已列出 GraphFlow（通过 agent 的有向图构建多 agent 工作流）作为协调模式之一[^3]；CrewAI 的 Flows 提供条件逻辑、循环与分支[^10]。Microsoft 的迁移指南也把这一趋势讲得最直白：Agent Framework 以类型化的、基于图的 Workflow 为中心，沿边路由数据并在输入就绪时激活 executor[^30]。此外，生态上已出现混用：Azure 官方示例 azure-ai-travel-agents 由 Langchain.js、Llamaindex.TS 与 Microsoft Agent Framework 共同编排[^17][^32]；LangSmith Observability 官方声明框架无关，集成清单中明确列出 CrewAI[^42]。

---

## 矛盾与不确定性

**1. 缺少三者对比的基准评测（blocking）。** 49 条证据中仅 1 条 arxiv 论文（MacNet, 2406.07155），且该论文研究的是自建 DAG 拓扑协作网络，**并未评测 LangGraph / AutoGen / CrewAI 三者**，其抓取片段也未包含任何任务成功率或 token 成本的具体数字[^27]。因此本报告对「适用场景」的判断全部基于各框架官方文档的能力面与设计约束，而非实测性能。三者谁在何种任务上更优，**现有公开资料不足以确认**。

**2. AutoGen 的长期定位存在未澄清的张力。** Microsoft 官方博客（2025-10-01）将 AutoGen 定位为「源自研究的实验性多智能体编排」、企业就绪度 "Minimal"，并称 MAF 在二者基础上构建[^28][^29]；迁移指南称 MAF 是 AutoGen 所开创理念的「重要演进」[^30]。但 AutoGen 仓库 2025-09-30 仍发布 0.7.5[^38]、2026-10-16 仍有 push[^12]。AutoGen 是否进入维护模式、是否有后续版本路线图，**缺少一手证据**（如仓库 README 的 status/maintenance 声明、release note 中的 deprecation 说明）。此外，上述全部证据均来自 Microsoft 自家渠道，**无独立第三方来源交叉验证**。

**3. 许可证信息不一致。** AutoGen 仓库根 LICENSE 为 CC BY 4.0[^36]，而 PyPI 上 autogen-agentchat 声明 MIT[^37]。本报告按证据要求区分「仓库根许可证」与「具体 Python 发行包许可证」，但二者为何不同、各覆盖哪些产物，**现有公开资料不足以确认**。

**4. 厂商自述的营销性主张未采信。** CrewAI 声称「超过 100,000 名通过 learn.crewai.com 课程认证的开发者」[^40]、「Used by 65% of the Fortune 500」[^47]，均为厂商单页出处、无第三方核验，本报告未将其作为独立事实使用。同理，CrewAI 的 "lean, lightning-fast" 自述[^40]属定性主张，无基准支撑。

**5. 版本与发布节奏的可比性有限。** 三个仓库的最近 push 时间集中在 2026-10 月（LangGraph 10-25[^11]、CrewAI 10-24[^13]、AutoGen 10-16[^12]），但 PyPI 上最新版本日期差异很大：langgraph 1.2.17 为 2026-10-24[^34]，crewai 1.15.30 为 2026-10-24[^41]，而 autogen-agentchat 0.7.5 停留在 2025-09-30[^37][^38]。仓库活跃与发行包更新之间的这一落差，与第 2 点的定位张力相互印证，但**不足以单独证明** AutoGen 已停止功能迭代。

**6. 关于「图编排优于群聊广播」的判断需谨慎。** MacNet 报告不规则拓扑优于规则拓扑、并批评全局广播导致上下文膨胀[^27]，这与 LangGraph 的图模型方向一致、与 AutoGen 的广播式 GroupChat 形成对照[^6][^23]。但该论文既未评测这三个框架，其抓取片段也未给出具体基准分数与 token 成本数字[^27]，因此**不能据此推断框架层面的优劣**，只能作为设计取向差异的外部旁证。

## 参考来源

[^1]: LangGraph Graph API（官方文档：Graphs / StateGraph / State / Nodes / Edges / Compiling） — https://docs.langchain.com/oss/python/langgraph/graph-api（web）
[^2]: LangGraph Persistence（官方文档：Checkpointer vs Store） — https://docs.langchain.com/oss/python/langgraph/persistence（web）
[^3]: AutoGen AgentChat 概览（官方文档：AgentChat 与 autogen-core 的分层） — https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html（web）
[^4]: AutoGen Core 概念：Agent Runtime Environments（官方文档） — https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/core-concepts/architecture.html（web）
[^5]: AutoGen AgentChat Agents 教程（官方文档：Agent 基类属性与 AssistantAgent） — https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html（web）
[^6]: AutoGen AgentChat Teams 教程（官方文档：Team 定义与 GroupChat 预设） — https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/teams.html（web）
[^7]: CrewAI Agents 概念（官方文档） — https://docs.crewai.com/en/concepts/agents（web）
[^8]: CrewAI Crews 概念（官方文档） — https://docs.crewai.com/en/concepts/crews（web）
[^9]: CrewAI Processes 概念（官方文档：Sequential / Hierarchical） — https://docs.crewai.com/en/concepts/processes（web）
[^10]: CrewAI Flows 概念（官方文档：事件驱动工作流） — https://docs.crewai.com/en/concepts/flows（web）
[^11]: langchain-ai/langgraph (GitHub repository metadata) — https://github.com/langchain-ai/langgraph（github）
[^12]: microsoft/autogen (GitHub repository metadata) — https://github.com/microsoft/autogen（github）
[^13]: crewAIInc/crewAI (GitHub repository metadata) — https://github.com/crewAIInc/crewAI（github）
[^17]: Azure-Samples/azure-ai-travel-agents (Microsoft 官方示例，引用 Microsoft Agent Framework) — https://github.com/Azure-Samples/azure-ai-travel-agents（github）
[^19]: LangGraph Docs — Interrupts (human-in-the-loop) — https://docs.langchain.com/oss/python/langgraph/interrupts（web）
[^20]: AutoGen Docs — AgentChat Termination — https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/termination.html（web）
[^21]: AutoGen Docs — Command Line Code Executors (Docker / Local) — https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/command-line-code-executors.html（web）
[^22]: AutoGen Docs — Managing State (save_state / load_state) — https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/state.html（web）
[^23]: AutoGen Docs — Selector Group Chat — https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/selector-group-chat.html（web）
[^24]: CrewAI Docs — Processes (Sequential / Hierarchical) — https://docs.crewai.com/concepts/processes（web）
[^25]: CrewAI Docs — Agents (参数清单：max_iter / max_rpm / code_execution_mode 等) — https://docs.crewai.com/concepts/agents（web）
[^26]: CrewAI Docs — Tasks (human_input / async_execution / guardrail 等) — https://docs.crewai.com/concepts/tasks（web）
[^27]: Scaling Large Language Model-based Multi-Agent Collaboration (MacNet) — https://arxiv.org/abs/2406.07155（arxiv, 2024-06-11 (v1); v3 2025-03-17; accepted to ICLR-2025）
[^28]: Introducing Microsoft Agent Framework: The Open-Source Engine for Agentic AI Apps（Microsoft Foundry 官方博客） — https://devblogs.microsoft.com/foundry/introducing-microsoft-agent-framework-the-open-source-engine-for-agentic-ai-apps/（web, 2025-10-01）
[^29]: Microsoft Foundry 博客 2025 年 10 月归档页（用于确认公告发布日期） — https://devblogs.microsoft.com/foundry/2025/10/01/（web, 2025-10-01）
[^30]: AutoGen to Microsoft Agent Framework Migration Guide（Microsoft Learn 官方迁移指南） — https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/（web）
[^32]: GitHub 仓库搜索结果 — Microsoft Agent Framework 生态落地情况 — https://github.com/search?q=microsoft+agent-framework（github）
[^33]: langchain-ai/langgraph — LICENSE (MIT License) — https://github.com/langchain-ai/langgraph/blob/main/LICENSE（github）
[^34]: PyPI — langgraph 项目页（版本、许可证表达式、发布日） — https://pypi.org/project/langgraph/（web, 2026-10-24）
[^35]: langchain-ai/langgraph — Releases 页面（2026 年 10 月发布序列） — https://github.com/langchain-ai/langgraph/releases（github, 2026-10-24）
[^36]: microsoft/autogen — LICENSE (Creative Commons Attribution 4.0 International) — https://github.com/microsoft/autogen/blob/main/LICENSE（github）
[^37]: PyPI — autogen-agentchat 项目页（版本、许可证、发布日） — https://pypi.org/project/autogen-agentchat/（web, 2025-09-30）
[^38]: microsoft/autogen — Releases 页面（python-v0.7.x 发布序列） — https://github.com/microsoft/autogen/releases（github, 2025-09-30）
[^39]: crewAIInc/crewAI — LICENSE (MIT License) — https://raw.githubusercontent.com/crewAIInc/crewAI/main/LICENSE（github）
[^40]: PyPI — crewai 项目页（版本与框架自述） — https://pypi.org/project/crewai/（web, 2026-10-24）
[^41]: crewAIInc/crewAI — Releases 页面（1.15.x 发布序列） — https://github.com/crewAIInc/crewAI/releases（github, 2026-10-24）
[^42]: LangSmith Observability — 官方文档首页 — https://docs.smith.langchain.com/（web）
[^44]: AutoGen Studio — AutoGen 官方文档（user guide） — https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/index.html（web）
[^45]: microsoft/autogen — python/packages/autogen-studio/README.md — https://github.com/microsoft/autogen/tree/main/python/packages/autogen-studio（github）
[^46]: CrewAI AMP — 官方文档 enterprise/guides/getting-started — https://docs.crewai.com/en/enterprise/guides/getting-started（web）
[^47]: CrewAI 官方定价页 — https://www.crewai.com/pricing（web）
[^48]: AutoGen Core User Guide — Agent and Agent Runtime — https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/agent-and-agent-runtime.html（web）
[^49]: AutoGen Core User Guide — Distributed Agent Runtime — https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/distributed-agent-runtime.html（web）