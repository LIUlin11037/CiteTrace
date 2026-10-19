# LangGraph、AutoGen、CrewAI：设计理念与适用场景差异

## 摘要

三者的分歧不在功能清单，而在"谁来控制执行路径"。LangGraph 把 agent 工作流显式建模为图：State（共享状态快照）、Nodes（做事的函数）、Edges（决定下一步的函数），官方概括为"nodes do the work, edges tell what to do next"[^1]，控制权在开发者手里，代价是必须自己定义 state、连边、compile[^1]。CrewAI 把控制权交给角色与流程：agent 是"具有特定技能、专长和职责的专门团队成员"[^8]，协作由 Crew 的 process（sequential / hierarchical）决定[^6][^21]，开发者描述"谁做什么"，而非"下一步走哪条边"。AutoGen 把控制权交给消息：agent 是"通过消息通信、维护自身状态、并响应收到的消息或自身状态变化而执行动作的软件实体"[^3]，多 agent 协作通过共享消息线程的 group chat 模式实现，轮次由 Group Chat Manager 决定[^19]。

一个必须先声明的前提：AutoGen 已于 2025 年 10 月进入维护模式，不再接收新功能[^13][^29]，官方建议新用户改用 Microsoft Agent Framework[^13]。因此"三者并列对比"这一命题本身已经失真——真正活跃的三条线是 LangGraph、CrewAI、以及 AutoGen 的两个后继（MAF 与社区分叉 AG2）[^29][^34]。

## 一、设计理念：图、角色、消息

### LangGraph：低层编排，状态显式流转

LangGraph 自我定位为"Low-level orchestration framework for building stateful agents"（低层编排框架）[^12]。其核心抽象是图，由三个组件定义[^1]：

- **State**：表示应用当前快照的共享数据结构，通常用共享 state schema 定义；由 schema 加 reducer functions 组成，reducer 指定如何把更新应用到 state，schema 可为 TypedDict 或 Pydantic model[^1]。
- **Nodes**：编码 agent 逻辑的函数，接收当前 state、执行计算或副作用、返回更新后的 state[^1]。
- **Edges**：根据当前 state 决定下一个执行哪个 Node 的函数，可为条件分支或固定转移[^1]。

构建流程是：先定义 state、添加 nodes 和 edges，然后 compile；文档强调"You MUST compile your graph before you can use it"，编译提供结构检查（如无孤立节点），并可指定 checkpointers、breakpoints 等 runtime args[^1]。底层图算法使用 message passing 定义通用程序，受 Google Pregel 启发，程序以离散的 super-steps 推进；并行运行的节点属于同一 super-step，顺序运行的节点属于不同 super-step[^1]。PyPI 页面补充说明其设计灵感来自 Pregel 与 Apache Beam，公开接口借鉴 NetworkX[^15]。

值得注意的是其分层策略：README 明确建议，若想快速构建 agent，应改用更高层的 Deep Agents 包（构建在 LangGraph 之上）[^12]；PyPI 描述也建议，需要确定性+agentic 混合工作流、深度定制、精细控制延迟时使用 LangGraph，快速构建则用 LangChain[^15]。

### CrewAI：角色化团队 + 事件驱动流程

CrewAI 自我定位为"open-source Python framework with high-level abstractions and low-level APIs for building production-ready multi-agent workflows"[^14]，核心抽象为两类：Crews（角色化自主协作）与 Flows（事件驱动、精确控制）[^14][^17]。

Crew 的定义是"代表一组协作的 agent 共同工作以实现一组任务"，定义任务执行策略、agent 协作方式与整体工作流；关键属性包括 tasks、agents、process（sequential 或 hierarchical，默认 sequential）[^6]。Agent 的定义是"一个自主单元，能够执行特定任务、基于其 role 和 goal 做决策、使用工具达成目标、与其他 agent 沟通协作、维护交互记忆、在允许时委派任务"[^8]；官方比喻是"具有特定技能、专长和职责的专门团队成员"[^8]。核心属性为 role（职能与专长）、goal（引导决策的个体目标）、backstory（提供上下文与个性）[^8]，另有 llm、tools、allow_delegation（默认 False）、max_iter（默认 20）、reasoning（默认 False）等[^8]。

Flows 用于简化 AI 工作流的创建与管理，允许组合与协调编码任务和 Crews，创建结构化、事件驱动的工作流，可连接多个任务、管理状态并控制执行流；支持条件逻辑、循环、分支，机制上用 @start() 标记入口点、@listen() 监听上游方法输出，每个 Flow 实例自动在 state 中获得唯一 UUID[^7]。

### AutoGen：消息驱动的可组合 agent

AutoGen 对 agent 的定义是"一种通过消息通信、维护自身状态、并响应收到的消息或自身状态变化而执行动作的软件实体"；这些动作可修改 agent 状态并产生外部效果，如更新消息日志、发送新消息、执行代码或调用 API[^3]。由多个交互 agent 组成的系统称为 multi-agent applications[^3]。

其模块化主张很明确：每个 agent 是自包含单元（self-contained unit），可独立开发、测试和部署；agent 本质上是可组合的（inherently composable），简单 agent 可组合成复杂应用[^3]。多智能体应用中的 agent 可以在同一进程或同一机器上运行，也可跨不同机器或组织边界运行，可用不同编程语言实现并使用不同 AI 模型或指令[^3]。

运行时层面，AutoGen 提供 runtime environment 负责促进 agent 间通信、管理身份与生命周期、执行安全与隐私边界；支持 standalone（单进程，Python 示例为 SingleThreadedAgentRuntime）与 distributed（多进程，由 host servicer 与多个 workers 组成）两种类型，两者提供共同 API 集合，可在不改变 agent 实现的情况下切换[^4]。

分层上，AgentChat 是构建多 agent 应用的高层 API，构建在 autogen-core 之上，新手推荐从 AgentChat 入手；autogen-core 提供事件驱动编程模型，面向高级用户[^16]。AgentChat 中所有 agent 共享 name、description、run、run_stream 等属性方法，agent 被期望是有状态的，run() 应以新消息调用而非完整历史[^5]。AssistantAgent 是内置 agent，使用语言模型并具备使用工具的能力，文档称其为面向原型与教育用途的 "kitchen sink" agent[^5]。

## 二、多 agent 协作机制

**AutoGen 的 group chat**：一组 agent 共享同一条消息线程，全部订阅并发布到同一个 topic；参与者轮流发布消息，过程是顺序的——同一时刻只有一个 agent 在工作；轮次顺序由 Group Chat Manager agent 维护，它在收到消息后选择下一个发言的 agent，选择算法通常使用 round-robin 或带 LLM 的 selector[^19]。消息协议为：用户或外部 agent 向公共 topic 发布 GroupChatMessage → manager 选出下一个 speaker 并发送 RequestToSpeak → 该 agent 向公共 topic 发布 GroupChatMessage → 持续到 manager 处达到终止条件[^19]。group chat 可用于将复杂任务动态分解为更小的、由具有明确角色的专门 agent 处理的任务，也可嵌套成层级结构，每个参与者本身是一个递归的 group chat[^19]。AgentChat 内置了 selector group chat 实现，无需用 Core API 自行搭建[^19]。

SelectorGroupChat 的机制是：由生成式模型基于共享上下文选择下一个 speaker；默认情况下团队不会连续选择同一个 speaker，除非它是唯一可用的 agent，可通过 allow_repeated_speaker=True 改变；运行流程为"分析当前对话上下文（含对话历史与参与者的 name、description）→ 模型确定下一个 speaker → 提示该 speaker 回复并广播给所有其他参与者 → 检查终止条件"[^20]。对话结束时返回 TaskResult，包含该任务的对话历史；任务完成后对话上下文保留在团队及所有参与者中，下一个任务可延续之前的上下文，调用 reset() 可重置[^20]。

**CrewAI 的 process**：Process 用于编排 agent 执行任务，类比人类团队中的项目管理，按预定义策略分配与执行任务；实现有两种——Sequential（顺序执行，保证任务按有序推进）与 Hierarchical（按管理层级组织任务，基于结构化指挥链进行委派与执行）[^21]。Sequential 流程中任务执行遵循任务列表中的预定义顺序，一个任务的输出作为下一个任务的上下文，可通过 Task 类的 context 参数定制[^21]。Hierarchical 流程模拟企业层级：可指定自定义 manager agent 或自动创建一个，必须指定 manager 语言模型（manager_llm）；任务不是预先分配的，manager 根据 agent 的能力分配任务、审查输出并评估完成情况，负责规划、委派与验证[^21]。Process 类实现为枚举（Enum），保证类型安全[^21]。Crew 层面还支持 memory（短期、长期、实体记忆）、planning（激活后每次 Crew 迭代前将全部 Crew 数据发送给 AgentPlanner 规划任务）、manager_agent 等能力[^6]。

**LangGraph 的 subgraph**：官方文档提供 subgraph 机制，明确其用途之一是"Building multi-agent systems"；subgraph 可作为节点嵌入父图，支持两种通信模式——在节点内调用 subgraph（状态 schema 不同），或直接将编译后的 subgraph 传给 add_node（共享状态键）；文档强调 subgraph 接口（输入/输出 schema）被遵守时，父图无需了解 subgraph 内部细节，支持分布式开发[^41]。

## 三、状态与持久化：差异最实质的一层

**LangGraph 提供两套互补的持久化系统**[^2][^18]：

- **Checkpointers**：将某个 thread 的图状态持久化为 checkpoints，用于短期、thread 作用域的记忆，包括对话连续性、human-in-the-loop 工作流、time travel 和容错；作用域为单个 thread，访问方式是在 graph config 中传入 thread_id[^2][^18]。
- **Stores**：将应用自定义数据持久化在图状态之外，用于长期、跨 thread 的记忆，包括用户偏好、事实和共享知识；作用域跨 thread，可从节点或应用代码读写[^2][^18]。

挂载方式为 builder.compile(checkpointer=checkpointer, store=store)[^2][^18]。后端方面，MemorySaver/InMemorySaver 将 checkpoint 存在 RAM 中，进程重启后全部丢失；生产环境应使用 PostgresSaver（PostgreSQL，支持异步）或 SqliteSaver（本地文件）[^18]。使用 PostgresSaver 时 thread_id 存储在长度受限的列中，需保持在 255 字符以内；长对话中 checkpoint 会不断累积，可能增加延迟与存储成本，需定期裁剪或设置保留策略[^18]。子图各自管理自己的 checkpoint namespace，子图更新状态时父图可能不会立即看到变化；跨图边界的数据应通过 Store 共享或配置子图写入父 checkpoint[^18]。

**CrewAI 的持久化粒度更粗**。源码显示 @persist 装饰器叠加在 @start() 之上，同时支持同步与异步方法，持久化粒度为方法级：save_state 以 flow_uuid + method_name 为键，保存整个 Flow 状态 state_data[^45]。persist_state 从 flow 实例读取 state，要求 state 含 'id' 字段，否则抛 ValueError('Flow state must have an \'id\' field for persistence')[^45]。默认后端为 SQLiteFlowPersistence[^46]，默认 DB 路径为 db_storage_path()/flow_states.db，表结构为 id / flow_uuid / method_name / timestamp / state_json，状态以 JSON 序列化存储，使用 WAL 日志模式，SQLite busy timeout 为 30 秒[^47]。该模块自述定位为"适合开发与测试，或中等性能要求的生产场景"[^47]。另有单独的 pending_feedback 表用于 Flow 暂停/恢复，支持异步人工反馈[^47]。

关于后端可替换性：通过 set_flow_persistence_factory() 可注册进程级自定义后端，注释列举可为 database、remote service、in-memory fake，显式传入的 persistence= 实例优先级高于默认工厂[^46]。但需注意，persistence 目录仅包含 __init__.py、base.py、decorators.py、factory.py、sqlite.py 五个文件，不存在 postgres、redis 或其它数据库后端实现文件，表明官方未内置 PostgreSQL/Redis 持久化后端[^48]。这与某商业博客所称"CrewAI 通过 2025 年引入的 Flows 与 @persist 装饰器在方法级保存状态到 SQLite/PostgreSQL/Redis"存在出入[^25]——源码层面只见到 SQLite 内置实现，PostgreSQL/Redis 需自行通过工厂注册。

**AutoGen 的状态模型**：AgentChat 的 agent 被期望是有状态的，run() 会更新 agent 内部状态——将消息加入 agent 的消息历史[^5]。但需注意，关于"AutoGen 采用 ephemeral 状态模型、在长时运行工作流中表现吃力、易出现对话漂移"的说法来自商业博客[^25]，该来源含咨询导流、未披露基准方法学与样本量，应视为营销性主张。AutoGen 官方文档层面，其持久化能力的具体设计在给定证据中未获充分展开，现有公开资料不足以确认其 checkpoint 机制与 LangGraph 的对应关系。

## 四、依赖与工程结构

三者的依赖策略差异明显：

- **langgraph** 版本 1.2.15，requires-python >=3.10，运行时 dependencies 明确包含 'langchain-core>=1.4.7' 与 'pydantic>=2.7.4'，即直接依赖 langchain-core[^42]。不过 PyPI 页面说明 LangGraph 由 LangChain Inc 开发，但可不依赖 LangChain 独立使用[^15]。
- **autogen-agentchat** 版本 0.7.5，requires-python >=3.10，唯一运行时依赖是 'autogen-core==0.7.5'，即采用自研分层架构（agentchat 依赖 core），不依赖 LangChain 或 crewai[^44]。
- **crewai** requires-python >=3.10，作者为 Joao Moura，依赖列表中可见 'instructor>=1.3.3'、pydantic、openai 等第三方库（依赖数组在抓取中被截断，未能读取完整清单）[^43]。PyPI 描述宣称 CrewAI 是"built entirely from scratch—completely independent of LangChain or other agent frameworks"[^17]。

## 五、生态与活跃度

GitHub 元数据（截至抓取时点）[^9][^10][^11]：

| 框架 | stars | forks | 最近 push |
|---|---|---|---|
| langchain-ai/langgraph | 43,257 | 7,355 | 2026-10-14 |
| microsoft/autogen | 61,399 | 9,281 | 2026-04-15 |
| crewAIInc/crewAI | 59,685 | 8,723 | 2026-10-14 |

AutoGen 的 push 时间明显早于另两个框架[^10]。版本节奏上，langgraph 最新版 1.2.15 发布于 2026-10-12[^15]；autogen-agentchat 最新版 0.7.5 发布于 2025-09-30，距 2026 年已超过一年未发新版，与其 maintenance mode 状态一致[^16]；crewai 最新版 1.15.28[^17]。

商业与社区层面：LangGraph 宣称被 Klarna、Replit、Elastic 等公司使用[^12]，提供 JS/TS 等价库 LangGraph.js[^12]；CrewAI 宣称超过 100,000 名开发者通过 learn.crewai.com 社区课程获得认证[^14][^17]，提供商业控制平面 CrewAI AMP Suite（托管部署、可观测性、治理、安全、企业支持）[^14][^17]；AutoGen 提供无代码 GUI（pip install -U "autogenstudio"）[^13]，其项目历史数据为两年间 98 个 release、3,776 commits、2,488 个已解决 issue、50.4k GitHub stars、559 名贡献者[^29]。

## 六、AutoGen 的三条分叉线：对比前提已失真

这是本报告必须单独成节的问题。将 AutoGen 作为与 LangGraph/CrewAI 并列的活跃框架，前提已经不成立：

**microsoft/autogen 进入维护模式**。README 顶部标注 "Maintenance Mode"：不再接收新功能，转为社区维护，官方建议新用户改用 Microsoft Agent Framework (MAF)，并提供迁移指南[^13]。维护者 Eric Zhu 于 2025-10-02 发布的置顶公告宣布 AutoGen 与 Semantic Kernel 合并为单一统一框架 MAF，AutoGen 仍将被维护（stable API，继续接收 critical bug fixes 与 security patches），但不再添加重大新功能[^29]。LangChain 官方页面（2026-06-23）三方印证了这一点，并称 MAF 于 2026 年 4 月达到 1.0 GA（.NET 与 Python），AutoGen 不再是 Microsoft 推荐用于新项目的框架[^40]。PyPI 显示 autogen-agentchat 仅 2 名 maintainers[^16]。

**MAF 的设计取向与 AutoGen 不同**。官方博客对比表将 AutoGen 的 Focus 列为 "Experimental multi-agent orchestration from research"，编排方式为 "Dynamic LLM orchestration (debate, reflection, facilitator/worker, group chat)"，企业就绪度为 "Minimal"[^30]。迁移指南明确关键差异之一：编排风格上，AutoGen 将 event-driven core 与高层 Team 配对；MAF 则以 typed、graph-based Workflow 为中心，沿 edges 路由数据[^31]。公告称最重大的变化是新的 workflow API，允许用 graph-based 方式定义复杂的多步、多 agent 工作流，sequential、parallel、Magentic 等编排模式都构建在该 workflow API 之上[^29]。博客还指出 AutoGen 在互操作方面 "lacks standardized cross-runtime protocols"，而 MAF 内置 MCP + A2A + OpenAPI 连接器[^30]。换言之，MAF 在编排范式上向 LangGraph 式的图模型靠拢。

**AG2 是社区分叉线**。AG2 官方称原 AutoGen 项目于 2023 年 9 月发布首个公开预览版，AG2 于 2024 年 11 月从该代码库分叉，动因是原 AutoGen 架构难以采纳新协议与理念[^34]。自 AG2 v1.0 起，协议驱动框架成为顶层包（import ag2），经典框架迁出至独立仓库 ag2ai/ag2-classic（import autogen），pip install ag2 不再包含 autogen 导入名或经典 agent 类[^28][^35]。多智能体机制从经典 GroupChat/swarms/nested chats 演进为 Network（hub + channels）模型，核心 agent 类由 ConversableAgent 变为 Agent[^28][^35]。Network 将 Agent 组织为 hub-and-spoke 拓扑，由中心注册表协调基于 channel、协议驱动的命名 agent 间交换；Hub 持有 registry、audit log、channel table、write-ahead logs 等权威状态；Passport 提供稳定身份，Resume 提供能力声明，Rule 提供 per-agent 治理[^36]。默认传输为 LocalLink（同进程双工队列），可替换为 WsLink 以跨进程/跨主机部署[^36]。Network 为 fully opt-in：未导入该包时，裸 Agent 仍可独立运行且行为不变[^36]。AG2 由来自多个组织的志愿者维护，联系人为 Chi Wang 与 Qingyun Wu[^28][^35]。

## 七、互操作：三条不同的路径

**LangGraph 走协议标准化路线**。LangChain 于 2024-11-19 开源 Agent Protocol，定位为智能体通信的通用接口，围绕 Runs（执行 agent 的 API）、Threads（组织多轮执行的 API）、Store（长期记忆 API）三类概念定义框架无关的 API[^26]。仓库 README 自述为 "codifying the framework-agnostic APIs that are needed to serve LLM agents in production"，核心端点包括 POST /runs/wait、POST /runs/stream、POST /threads、GET /threads/{thread_id}/history 等；LangGraph Platform 实现该协议的 superset，并欢迎社区其他实现[^37]。LangChain 明确提供指南，说明如何将其他框架（如 AutoGen、CrewAI）作为 LangGraph agent 内的 sub-agent 使用，以及如何用 LangGraph Platform 部署其他框架构建的 agent[^26]。该博客将 AutoGen、OpenAI Assistant API、CrewAI、LlamaIndex 列为 LangGraph 之外用户常用的框架，并指出各框架 API 不统一是多 Agent 世界的障碍[^26]。

**AG2 走工具层协议对接**。AG2 官方文档提供教程，演示如何将 LangChain Tools 与 PydanticAI Tools 等不同框架的 LLM 工具集成进 AG2 框架，集成 LangChain 工具需安装 pip install -U ag2[openai,interop-langchain]；文档同时列出 MCP 会话工作流、PydanticAI Tools Integration 等教程[^27]。

**MAF 走内置多协议连接器**。博客称 MAF 通过 MCP、A2A、OpenAPI 提供开放标准与互操作性，使 agent 可移植且厂商中立[^30]。

协议层面的分工已较清晰：A2A 官方文档将 MCP 与 A2A 定位为 "distinct but highly complementary"——MCP 是 "vertical"，加深单个 agent 的能力，连接工具、资源、API；A2A 是 "horizontal"，跨边界连接 agent，用于发现、协商与信息交换；官方总结为 "MCP 给每个 agent 深度，A2A 给系统广度"[^38]。A2A Protocol v1.0 于 2026-03-12 发布，为 "first stable, production-ready version"，新增异构环境支持、多租户、签名 Agent Cards 等能力，基于 JSON+HTTP、gRPC、JSON-RPC 等成熟协议[^39]。官方明确 "Complementary to MCP, not a replacement"，实践中 "MCP inside agents, A2A between agents"[^39]。

## 八、适用场景：官方自述与第三方主张

**来自官方自述的选型信号**：

- LangGraph：README 建议快速构建 agent 应改用更高层的 Deep Agents 包[^12]；PyPI 建议需要确定性+agentic 混合工作流、深度定制、精细控制延迟时使用[^15]；核心能力列举为 durable execution、human-in-the-loop、comprehensive memory、LangSmith 调试、生产部署[^12]。
- CrewAI：Crews 面向自主协作，Flows 面向事件驱动、细粒度控制并原生支持 Crews[^17]；提供高层简洁性与低层精确控制两种使用方式[^17]。
- AutoGen：AgentChat 面向新手，autogen-core 面向高级用户[^16]；AssistantAgent 被定位为面向原型与教育用途的 "kitchen sink" agent[^5]。

**第三方选型主张（需谨慎对待）**：某商业博客给出"快速原型用 CrewAI，需要持久性与合规的生产系统用 LangGraph，实验/研究型项目用 AutoGen"的建议[^25]，并给出每任务平均成本与恢复率：LangGraph $0.08 / 96%，AutoGen $0.45 / 68%，CrewAI 未给出成本、恢复率 72%[^25]。该来源为商业博客（含咨询导流），未披露基准方法学与样本量，其成本/恢复率数字应视为营销性主张，需交叉验证[^25]。

学术层面，一篇 arXiv 综述对 CrewAI、LangGraph、AutoGen、Semantic Kernel、Agno、Google ADK、MetaGPT 做系统性对比，评估维度为架构原则、通信机制、记忆管理、安全护栏以及与面向服务计算范式的契合度——属于定性/架构层面，而非基准分数[^22][^32]。该论文将互操作性列为待解决的开放挑战与未来研究方向之一[^22]，但未提供 GAIA / SWE-bench / AgentBench 等基准上的成功率、成本或延迟定量对比数据[^32]。

## 矛盾与不确定性

**1. 两个基准结论方向完全相反，无法据此选型。** 一个个人基准（sweta2503，107 个数据工程任务，同一 LLM Groq Llama 3.3 70B）报告 LangGraph 在 SQL 生成 87.5%、Pipeline 调试 79.0%、Transformation 75.0%，平均约 2,700 tokens、约 12.7s 延迟；CrewAI 为 82.6% / 73.7% / 68.8%，约 5,005 tokens、约 20.0s；AutoGen 为 82.6% / 79.0% / 56.3%，约 5,678 tokens、约 17.9s；结论为 LangGraph 在准确率、token 成本与延迟三项上同时领先[^23]。但该仓库 stars 为 0、仅 4 次提交，属个人小规模基准，非权威第三方评测[^23]。另一份白皮书（Kurt Boden，2025 年 9 月，v2.0，50 个任务分 K=1/2/3 三个复杂度层级，采用基于 ChatGPT 的验证方法学）报告语义准确率排序为 CrewAI 最高 87.3%、SMOLAgents 80.0%、AutoGen 76.7%、LangGraph 垫底 68.7%，与前者方向完全相反[^24]。该白皮书还指出传统精确匹配评估方法可能将平台能力低估 20–40%[^24]。两个来源可信度均低，矛盾未解决，**无法据此给出选型建议**。

另有一篇标题高度相关的期刊论文（JATIR，paperid=140332）存在，但其 PDF 与文章页均两次抓取失败，正文未能读取，具体基准、指标与数值结果均未获证实，本条仅记录来源存在性线索，不作为结论依据[^33]。

**2. AutoGen 作为并列对比对象的前提已失真。** AutoGen 已于 2025-10 进入维护模式（README + 官方公告 + LangChain 页面三方印证）[^13][^29][^40]，AG2 于 2024-11 分叉[^34]，MAF 于 2026-04 GA[^40]。任何将 microsoft/autogen 视为活跃候选框架的对比都已过时。三者应明确区分为：microsoft/autogen（维护模式，仅 critical bug fixes 与 security patches）[^29]、ag2ai/ag2（社区分叉，协议驱动、Network 模型）[^28]、Microsoft Agent Framework（企业级后继，typed graph-based Workflow）[^30][^31]。此外，AutoGen 的 GitHub stars（61,399）高于另两者[^10]，但这一数字反映的是历史积累而非当前活跃度，其最近 push 时间为 2026-04-15，明显早于另两个框架的 2026-10-14[^9][^10][^11]。

**3. CrewAI 持久化后端的能力边界存疑。** 商业博客称 CrewAI 通过 Flows 与 @persist 装饰器在方法级保存状态到 SQLite/PostgreSQL/Redis[^25]，但源码目录中不存在 postgres、redis 或其它数据库后端实现文件，表明官方未内置 PostgreSQL/Redis 持久化后端[^48]，SQLiteFlowPersistence 自述定位为"适合开发与测试，或中等性能要求的生产场景"[^47]。PostgreSQL/Redis 需通过 set_flow_persistence_factory() 自行注册[^46]。该博客说法与源码证据不一致，其可信度进一步下降。

**4. AutoGen 状态模型的描述缺乏官方证据支撑。** "AutoGen 采用 ephemeral 状态模型、在长时运行工作流中表现吃力、资源消耗高、易出现对话漂移"的说法仅来自商业博客[^25]，未获官方文档印证。AutoGen 官方文档层面仅说明 agent 是有状态的、run() 更新消息历史[^5]，其 checkpoint 与恢复机制的具体设计在给定证据中未获充分展开，**现有公开资料不足以确认**其与 LangGraph checkpoint 机制的对应关系。

**5. 定量对比数据的整体缺失。** 除上述两个低可信度基准外，学术综述明确未提供 GAIA / SWE-bench / AgentBench 等标准基准上的成功率、成本或延迟定量对比[^32]。因此关于三者在真实生产负载下的性能差异，**现有公开资料不足以确认**。

## 参考来源

[^1]: LangGraph Graph API — Graphs, State, Nodes, Edges, StateGraph — https://docs.langchain.com/oss/python/langgraph/graph-api（web）
[^2]: LangGraph Persistence — Checkpointer vs. Store — https://docs.langchain.com/oss/python/langgraph/persistence（web）
[^3]: AutoGen Core — Agent and Multi-Agent Applications — https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/core-concepts/agent-and-multi-agent-application.html（web）
[^4]: AutoGen Core — Agent Runtime Environments (standalone / distributed) — https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/core-concepts/architecture.html（web）
[^5]: AutoGen AgentChat — Agents (AssistantAgent, run/run_stream) — https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html（web）
[^6]: CrewAI Concepts — Crews — https://docs.crewai.com/concepts/crews（web）
[^7]: CrewAI Concepts — Flows — https://docs.crewai.com/concepts/flows（web）
[^8]: CrewAI Concepts — Agents (role-based) — https://docs.crewai.com/concepts/agents（web）
[^9]: langchain-ai/langgraph (GitHub repo metadata via search) — https://github.com/langchain-ai/langgraph（github, 2026-10-14）
[^10]: microsoft/autogen (GitHub repo metadata via search) — https://github.com/microsoft/autogen（github, 2026-04-15）
[^11]: crewAIInc/crewAI (GitHub repo metadata via search) — https://github.com/crewAIInc/crewAI（github, 2026-10-14）
[^12]: LangGraph README (raw, main branch) — https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md（github）
[^13]: AutoGen README (raw, main branch) — https://raw.githubusercontent.com/microsoft/autogen/main/README.md（github）
[^14]: CrewAI README (raw, main branch) — https://raw.githubusercontent.com/crewAIInc/crewAI/main/README.md（github）
[^15]: PyPI: langgraph project page — https://pypi.org/project/langgraph/（web, 2026-10-12）
[^16]: PyPI: autogen-agentchat project page — https://pypi.org/project/autogen-agentchat/（web, 2025-09-30）
[^17]: PyPI: crewai project page — https://pypi.org/project/crewai/（web）
[^18]: LangGraph Persistence (Checkpointers & Stores) — https://docs.langchain.com/oss/python/langgraph/persistence#checkpoints（web）
[^19]: AutoGen Core — Group Chat design pattern — https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html（web）
[^20]: AutoGen AgentChat — SelectorGroupChat — https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/selector-group-chat.html（web）
[^21]: CrewAI — Processes (Sequential & Hierarchical) — https://docs.crewai.com/en/concepts/processes（web）
[^22]: Agentic AI Frameworks: Architectures, Protocols, and Design Challenges — https://arxiv.org/abs/2508.10146（arxiv, 2025-08-13）
[^23]: agent-framework-benchmark: LangGraph vs CrewAI vs AutoGen on 107 real data engineering tasks — https://github.com/sweta2503/agent-framework-benchmark（github）
[^24]: Agent Orchestration Benchmark: A Comprehensive Analysis of Multi-Agent Frameworks (White Paper v2.0) — https://github.com/kurtmb/agent-orchestration-benchmark/blob/main/docs/white_paper/Agent_Orchestration_Benchmark_White_Paper.md（github, 2025-09）
[^25]: LangGraph vs CrewAI vs AutoGen: State, Cost, and Reliability Compared (AlterSquare) — https://altersquare.io/blog/langgraph-vs-crewai-vs-autogen-review-recommend-production-deployment（web）
[^26]: LangGraph Agent Protocol: Interoperability for LLM agents (LangChain Blog) — https://www.langchain.com/blog/agent-protocol-interoperability-for-llm-agents（web, 2024-11-19）
[^27]: Cross-Framework LLM Tool Integration with AG2 (AG2 Classic Docs) — https://classic.docs.ag2.ai/latest/docs/use-cases/notebooks/notebooks/tools_interoperability/（web）
[^28]: ag2ai/ag2: AG2 (formerly AutoGen) — Open-Source AgentOS for AI Agents — https://github.com/ag2ai/ag2（github）
[^29]: AutoGen Update — Announcement (Discussion #7066) — https://github.com/microsoft/autogen/discussions/7066（github, 2025-10-02）
[^30]: Introducing Microsoft Agent Framework: The Open-Source Engine for Agentic AI Apps — https://devblogs.microsoft.com/foundry/introducing-microsoft-agent-framework-the-open-source-engine-for-agentic-ai-apps/（web）
[^31]: AutoGen to Microsoft Agent Framework Migration Guide (Microsoft Learn) — https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/（web）
[^32]: Agentic AI Frameworks: Architectures, Protocols, and Design Challenges — https://arxiv.org/abs/2508.10146v1（arxiv, 2025-08-13）
[^33]: Benchmarking Multi-agent Frameworks: Langgraph Vs. Crewai Vs. Autogen (JATIR, paperid=140332) — https://www.jatir.org/article.php?paperid=140332（web）
[^34]: Why did we create AG2? (AG2 official docs — motivation) — https://docs.ag2.ai/docs/user-guide/motivation/（web）
[^35]: ag2ai/ag2 README (AG2: Open-Source AgentOS for AI Agents) — https://raw.githubusercontent.com/ag2ai/ag2/main/README.md（github）
[^36]: AG2 Network Overview (hub & channels) — https://docs.ag2.ai/docs/user-guide/network/overview/（web）
[^37]: GitHub - langchain-ai/agent-protocol (README) — https://github.com/langchain-ai/agent-protocol（github）
[^38]: A2A and MCP: Detailed Comparison (A2A Protocol official docs) — https://a2a-protocol.org/latest/topics/a2a-and-mcp/（web）
[^39]: A2A Protocol Ships v1.0: Production-Ready Standard for Agent-to-Agent Communication (A2A official blog) — https://a2a-protocol.org/latest/blog/2026/03/12/a2a-protocol-ships-v10-production-ready-standard-for-agent-to-agent-communication/（web, 2026-03-12）
[^40]: LangChain vs. AutoGen (LangChain official resource page) — https://www.langchain.com/resources/langchain-vs-autogen（web, 2026-06-23）
[^41]: Use subgraphs (LangGraph official docs) — https://docs.langchain.com/oss/python/langgraph/use-subgraphs（web）
[^42]: langgraph pyproject.toml (libs/langgraph) — https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/pyproject.toml（github）
[^43]: crewai pyproject.toml (lib/crewai) — https://raw.githubusercontent.com/crewAIInc/crewAI/main/lib/crewai/pyproject.toml（github）
[^44]: autogen-agentchat pyproject.toml — https://raw.githubusercontent.com/microsoft/autogen/main/python/packages/autogen-agentchat/pyproject.toml（github）
[^45]: crewAI/lib/crewai/src/crewai/flow/persistence/decorators.py (main branch) — https://raw.githubusercontent.com/crewAIInc/crewAI/main/lib/crewai/src/crewai/flow/persistence/decorators.py（github）
[^46]: crewAI/lib/crewai/src/crewai/flow/persistence/factory.py (main branch) — https://raw.githubusercontent.com/crewAIInc/crewAI/main/lib/crewai/src/crewai/flow/persistence/factory.py（github）
[^47]: crewAI/lib/crewai/src/crewai/flow/persistence/sqlite.py (main branch) — https://raw.githubusercontent.com/crewAIInc/crewAI/main/lib/crewai/src/crewai/flow/persistence/sqlite.py（github）
[^48]: crewAI/lib/crewai/src/crewai/flow/persistence/ (directory listing, main branch) — https://github.com/crewAIInc/crewAI/tree/main/lib/crewai/src/crewai/flow/persistence（github）