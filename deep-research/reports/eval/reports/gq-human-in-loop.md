# Human-in-the-loop 在 LLM Agent 系统中的主流实现模式与适用场景

## 摘要

本报告基于 15 条证据，梳理 Human-in-the-loop（HITL）在 LLM Agent 系统中的实现模式。证据显示，当前工程实践中的 HITL 主要落地为四类可辨识的模式：**中断—恢复式（interrupt/resume）**、**工具调用前置审批式（approval gating）**、**代理参与者式（user-proxy agent）**、**任务级审查式（task-level review）**，另有**事件驱动式人工输入**与**策略防火墙式（action firewall）**两种变体。这些模式的差异集中在三个维度：暂停发生在何处、状态能否持久化并恢复、审批粒度是单次调用还是整轮 run。

需要预先声明三项证据缺口：其一，HITL 的**标准定义与分类框架**在现有来源中并未给出，HITL 与 human-on-the-loop / human-in-command / human oversight 的系统性区分缺乏直接支撑；其二，**arXiv 学术来源为零**，所有量化收益（成功率、成本、延迟、人工负担）均无证据；其三，**真实产业部署案例缺失**，唯一产业侧来源为工具自述仓库。这些缺口在正文与「矛盾与不确定性」一节中显式标注。

---

## 一、定义与合规背景：证据能说什么、不能说什么

### 1.1 现有定义

现有来源中唯一给出的 HITL 定义来自 Wikipedia，将其宽泛地表述为「a model requiring human interaction」（需要人类交互的模型）[^1]。该来源进一步指出，HITL 被用于多个语境：与建模与仿真（M&S）中的 live/virtual/constructive 分类法相关联；在机器学习语境中指人类协助计算机在构建模型时做出正确决策，通过选择最关键的、用于精炼模型的数据来改进机器学习（相较随机采样）；在对话式 AI 中用于管理需要人类共情的复杂交互[^1]。同一来源还提到，HITL 与相关的 human-on-the-loop 概念也被用于致命性自主武器语境[^1]。

**必须明确指出**：现有公开资料不足以确认 HITL 的标准分类框架，也不足以确认 HITL 与 human-on-the-loop、human-in-command、human oversight 之间的系统性区别。Wikipedia 仅给出泛化定义并提及 human-on-the-loop 的一个应用语境，未做系统区分[^1]；EU AI Act 第14条只规定 human oversight 的合规义务，不构成 HITL 分类学[^2]；Amershi 等的 18 条 guideline 明确被定位为设计准则而非自动化等级/监督模式分级框架[^9][^11]。

### 1.2 合规侧的强制要求

EU AI Act 第14条要求高风险 AI 系统应以包含适当人机界面工具的方式设计与开发，使其在使用期间能够被自然人有效监督（effectively overseen by natural persons）[^2]。人类监督的目标是预防或最小化高风险 AI 系统按预期用途使用时可能出现的对健康、安全或基本权利的风险[^2]。该条款将 human oversight 定位为对高风险 AI 系统的强制性设计与开发要求，而非可选的交互模式[^2]。

设计准则侧，Amershi 等（CHI 2019）提出 18 条「generally applicable design guidelines for human-AI interaction」[^9]，经多轮评估验证，包括一项有 49 名设计从业者参与、针对 20 款流行 AI 产品的用户研究[^9]。HAX Design Library 给出 18 条的完整标题，其中与人类监督/控制直接相关的条目包括 G7（支持高效调用）、G8（支持高效撤销）、G9（支持高效纠正）、G15（鼓励细粒度反馈）、G17（提供全局控制）、G18（变更通知）[^10]。该 guideline 集合按目标分类，包含 Appropriate reliance、Fairness、Personalization、Reliability、Transparency 等[^10]，并被描述为规定「an AI system should behave」的规范性行为准则，而非自动化程度的等级划分[^11]。

---

## 二、模式一：中断—恢复式（Interrupt / Resume）

### 2.1 机制

LangGraph 的 HITL 原语是 `interrupt()` 函数，可在图节点内任意位置调用，接受任意 JSON 可序列化值并暴露给调用方[^3]。该中断是动态的（dynamic），可放在代码任意位置并基于应用逻辑条件触发，区别于静态断点（static breakpoints，仅在特定节点前/后暂停）[^3]。

触发 interrupt 时，LangGraph 通过持久化层（checkpointer）保存图状态，并无限期等待直到恢复执行[^3]。恢复方式是用 `Command(resume=...)` 重新调用图，该值成为节点内 `interrupt()` 调用的返回值[^3]。持久化机制依赖 checkpointer，生产环境应使用持久化 checkpointer（如数据库支持）[^3]。`thread_id` 是持久游标：复用同一 `thread_id` 恢复同一 checkpoint，新值则开启全新线程与空状态[^3]。配置形态为 `config={"configurable": {"thread_id": ...}}`，告知 checkpointer 加载哪个状态[^3]。中断载荷通过事件流暴露：`graph.stream_events(..., version="v3")` 时出现在 `stream.interrupts`，`stream.interrupted` 为 True 表示因等待输入而暂停；默认 `invoke()` API 下出现在 `__interrupt__` 下[^3]。

### 2.2 适用场景

该模式的核心特征是**状态可持久化、可无限期等待、可跨进程恢复**，因此适合长时运行、人工响应可能延迟数小时甚至数天的流程。由于中断位置由应用逻辑动态决定，它也适合「仅在特定条件下才需要人工介入」的场景，而非固定节点审批。

---

## 三、模式二：工具调用前置审批式（Approval Gating）

### 3.1 机制

OpenAI Agents SDK 的 HITL 流程用于暂停 agent 执行，直到人工批准或拒绝敏感工具调用[^4]。工具通过声明 `needs_approval` 表明需要审批：设为 `True` 表示始终需要审批，或提供 async 函数按调用逐次判定；该 callable 接收 run context、解析后的工具参数与 tool call ID[^4]。

审批面是 run 级别的（run-wide），不限于当前顶层 agent：工具属于当前 agent、经 handoff 到达的 agent、或嵌套 `Agent.as_tool()` 执行时，同一模式均适用[^4]。嵌套 `Agent.as_tool()` 场景下，中断仍在外层 run 上浮现，需在外层 `RunState` 上批准/拒绝并恢复原始顶层 run[^4]。`Agent.as_tool()` 可在两层发生审批：agent 工具本身可通过 `Agent.as_tool(..., needs_approval=...)` 要求审批，嵌套 agent 内部的工具也可在嵌套 run 启动后自行发起审批，两者都通过同一外层 run 中断流处理[^4]。`RunState` 允许序列化被暂停的 run，并在决策做出后恢复[^4]。

一个关键的安全设计：callable 审批规则在 SDK 无法安全解析参数时 **fail closed**——参数缺失、为空、仅空白、JSON 畸形、合法 JSON 但非对象（如 null/list）、含 NaN/Infinity/-Infinity 等非标准常量时，callable 不被调用，该调用需人工审批[^4]。除手动中断审批外，部分工具类型还支持程序化审批回调（programmatic approval callbacks），使 run 无需暂停即可继续[^4]。

### 3.2 适用场景

该模式适合**高风险工具调用**（如写操作、外部副作用、资金相关动作）的逐次把关。`needs_approval` 支持静态 `True` 与动态 callable 两种形态，意味着既可对某类工具一律审批，也可按参数内容（如金额阈值、目标资源）条件审批。fail-closed 设计使其适合安全敏感场景——解析失败时默认不放行。

---

## 四、模式三：代理参与者式（User-Proxy Agent）

### 4.1 机制

AutoGen AgentChat 提供两种与团队交互的方式：(1) 在 `run()`/`run_stream()` 执行期间通过 `UserProxyAgent` 提供反馈；(2) run 终止后，通过下一次 `run()`/`run_stream()` 的输入提供反馈[^5]。`UserProxyAgent` 是内置的特殊 agent，作为用户的代理向团队提供反馈；团队决定何时调用它[^5]。在 `RoundRobinGroupChat` 中 `UserProxyAgent` 按传入团队的顺序被调用；在 `SelectorGroupChat` 中由 selector prompt 或 selector 函数决定何时调用[^5]。

关键约束：`UserProxyAgent` 在 run 期间被调用时会阻塞团队执行，直到用户提供反馈或报错；这会使团队处于无法保存或恢复的不稳定状态（cannot be saved or resumed）[^5]。由于阻塞特性，官方建议仅用于需要即时反馈的短交互，例如按钮点击式批准/拒绝，或需立即关注否则任务失败的告警[^5]。示例用 `TextMentionTermination("APPROVE")` 作为终止条件，用户说 APPROVE 时结束对话[^5]。

与状态持久化能力对照，AutoGen 另提供 `save_state()` 与 `load_state()` 方法，可保存/加载 agent、team 以及 termination condition 的状态；对 team 调用 `save_state` 时会一并保存团队中所有 agent 的状态[^15]。文档给出的动机场景是 Web 应用：无状态端点响应请求时需要从持久化存储中加载应用状态[^15]。`AssistantAgent` 的状态内容为其 model_context（LLM 消息历史）；示例中保存的状态包含 `type='AssistantAgentState'`、`version='1.0.0'` 与 `llm_messages` 列表[^15]。自定义 agent 可覆写 `save_state()`/`load_state()` 以定制行为，默认实现保存和加载空状态[^15]。

### 4.2 适用场景

该模式把人类建模为团队中的一个**参与者（agent）**，而非外部的审批门。适合需要人类持续参与对话、提供即时反馈的短交互。其阻塞且不可保存/恢复的特性，使其**不适合**长时间等待或需要跨会话恢复的流程[^5]。

---

## 五、模式四：任务级审查式（Task-Level Review）

### 5.1 机制

CrewAI 的 Task 提供 `human_input` 属性（`Optional[bool]`），表示该任务是否应由人工审查 agent 的最终答案，默认 `False`[^6]。Task 还提供 `guardrail` / `guardrails` 属性（`Callable` 或 `List[Callable]`），用于在进入下一任务前验证任务输出，`guardrail_max_retries` 控制验证失败时的最大重试次数（默认 3）[^6]。Task 提供 `callback` 属性（`Any`），为任务完成后执行的函数/对象[^6]。任务执行流程有两种：Sequential（按定义顺序执行）与 Hierarchical（按角色与专长分配 agent），在创建 Crew 时通过 `process` 参数定义[^6]。

### 5.2 适用场景

该模式的审批粒度是**任务级**而非调用级：人工审查的是 agent 完成某任务后的最终答案，而非单次工具调用。适合任务边界清晰、产出可独立评审的流水线（如 Sequential 流程中的关键节点）。`guardrail` 提供了**程序化验证**与**人工审查**的组合：guardrail 先自动校验输出，失败则重试至上限，`human_input` 则在需要时引入人工。

---

## 六、模式五：事件驱动式人工输入（Event-Driven Human Input）

### 6.1 机制

LlamaIndex Workflows 是事件驱动、基于 step 的执行流控制模型：step 接收一个 event，做工作，返回另一个 event，返回的 event 触发下一个类型注解可接受它的 step[^7]。文档明确列出 step 可执行的动作包括：调用 LLM、运行检索、请求人工输入（ask for human input）、更新共享状态、派发一批工作[^7]。Workflows 库可独立安装（`pip install llama-index-workflows`）；`llama-index-core` 自带 Workflows，可通过 `llama_index.core.workflow` 导入[^7]。Workflows 以 async 为一等公民，假设运行在 async 环境[^7]。分支是普通 if 语句返回不同 event 类型；循环是返回被更早 step 处理的 event；并发工作是返回 `list[Event]` 的 step 配合接受 `list[Event]` 的 step；需要动态流时可直接从 Context 发送 event[^7]。

官方模板展示了具体实现：从 `workflows.events` 导入 `StartEvent`、`StopEvent`、`InputRequiredEvent`、`HumanResponseEvent` 四个事件类[^14]。自定义事件通过继承实现：`FlightSearchEvent(InputRequiredEvent)` 携带 `candidate_flights` 字段；`FlightChoiceEvent(HumanResponseEvent)` 携带 `chosen_flight` 与 `continue_booking` 字段[^14]。步骤函数通过类型注解声明输入/输出事件，例如 `search_for_flight` 返回 `StopEvent | FlightSearchEvent`，`chosen_flight` 接收 `FlightChoiceEvent` 并返回 `StopEvent`[^14]。步骤可注入 `Context` 参数（`ctx: Context`）以及通过 `Annotated[..., Resource(get_llm)]` 注入资源[^14]。运行方式：`handler = w.run(message=message)`，然后 `async for ev in handler.stream_events()` 流式消费事件；当收到 `FlightSearchEvent` 时向用户提问[^14]。人工响应通过 `handler.ctx.send_event(FlightChoiceEvent(...))` 回送到工作流，而非直接返回值[^14]。Workflow 实例化时可传 `timeout` 与 `verbose` 参数，例如 `FlightSearchWorkflow(timeout=100, verbose=False)`[^14]。

### 6.2 适用场景

该模式将人工输入建模为**事件流中的一种事件类型**，与 LLM 调用、检索等动作同构。适合需要将人工介入编织进复杂异步流程、且介入点由事件类型而非固定节点决定的场景。`timeout` 参数的存在意味着该模式可对人工响应设置时限。

---

## 七、模式六：策略防火墙式（Action Firewall / Pre-Execution Gate）

### 7.1 机制

Cordum 自我定位为「The action firewall for AI agents」，在风险性工具调用、shell 命令、工作流和生产变更执行前强制策略与人工审批，并保留可审计证据（auditable evidence）[^8]。仓库描述明确其核心机制为「Enforce policy and human approval before risky tool calls, shell commands, workflows, and production changes」——即审批发生在动作执行之前（pre-execution gate）[^8]。README 正文说明其包含 Cordum Edge，定位为面向 Claude Code 及其他本地 AI-agent 动作的「Compliance Firewall」[^8]。

系统架构包含独立的 Safety Kernel（gRPC，端口 50051）与 Context Engine（gRPC，端口 50400），以及 API gateway、scheduler、workflow engine、dashboard、NATS、TLS-secured Redis 等组件[^8]。部署后内置 smoke test（`platform_smoke.sh`）会创建一个 approval-gated workflow、启动一次 run、批准该 gate，并校验 run 达到 succeeded 状态——证明审批门（approval gate）是工作流的一等原语[^8]。README 提供 Helm chart（cordum-helm）与 docker-compose 多环境配置（dev / enterprise override / HA / release），并支持 cosign keyless OIDC 镜像签名验证，面向生产部署[^8]。仓库包含 GOVERNANCE.md、SECURITY.md、ADOPTERS.md、BENCHMARKS.md 等治理与合规相关文档，暗示其面向受监管/企业环境[^8]。该仓库 stars 约 509、最近 push 时间 2026-10-06（来自检索元数据），表明 2024-2026 期间此类「审批防火墙」工具处于活跃开发状态[^8]。

### 7.2 适用场景

该模式将 HITL 从**框架内原语**提升为**独立的外部控制平面**，审批发生在动作执行之前，且强调可审计证据与合规文档[^8]。适合受监管/企业环境，需要跨工具、跨 agent 统一策略执行与审计留痕的场景。**但需注意**：以上均为仓库 README 自述主张，非独立验证的部署案例。

---

## 八、辅助机制：Filter 作为审批前置控制点

Semantic Kernel 提供三类 filter：Function Invocation Filter、Prompt Render Filter、Auto Function Invocation Filter[^12]。Function Invocation Filter 在每次 `KernelFunction` 被调用时执行，可访问函数及其参数信息、处理执行异常、在调用前或调用后覆盖函数结果（如缓存、负责任 AI 场景）、以及在失败时重试函数（如切换到备用 AI 模型）[^12]。Prompt Render Filter 在 prompt 渲染操作前触发，可查看/修改将发送给 AI 的 prompt（如 RAG、PII 脱敏），也可通过覆盖函数结果阻止 prompt 提交给 AI（如语义缓存）[^12]。Auto Function Invocation Filter 在自动函数调用范围内运行，提供额外上下文，包括 chat history、待执行的全部函数列表以及迭代计数器，并允许终止自动函数调用流程（例如在三个计划函数中的第二个就得到所需结果时）[^12]。每个 filter 都包含一个 context 对象，携带函数执行或 prompt 渲染的相关信息[^12]。

文档明确指出 filters 用于在审批流程开始前校验权限（'filters are leveraged to validate permissions before an approval flow begins'），即 filter 可作为审批流程的前置控制点[^12]。

理解该机制需先理解 Semantic Kernel 的 auto function calling 循环，共 6 步：(1) 用 JSON schema 序列化 kernel 中所有可用函数及其输入参数；(2) 将序列化函数与当前 chat history 作为输入发送给模型；(3) 模型处理输入并生成响应，响应可能是 chat message 或一个/多个 function call；(4) 若为 chat message 则返回调用方，若为 function call 则提取函数名与参数；(5) 用提取的函数名与参数在 kernel 中调用该函数；(6) 将函数结果作为 chat history 的一部分回传给模型[^13]。步骤 2-6 会重复执行，直到模型返回 chat message 或达到最大迭代次数（max iteration number）[^13]。文档以 `OrderPizzaPlugin` 为例，包含 `get_pizza_menu`、`add_pizza_to_cart`、`remove_pizza_from_cart`、`get_pizza_from_cart`、`get_cart`、`checkout` 等函数，C# 中用 `[KernelFunction("...")]` 与 `[Description(...)]` 标注[^13]。auto function calling 是默认行为，也可选择手动调用函数[^13]。

---

## 九、模式对比与场景选择

| 模式 | 暂停位置 | 状态可持久化/恢复 | 审批粒度 | 代表实现 | 适合场景 |
|---|---|---|---|---|---|
| 中断—恢复式 | 节点内任意位置（动态） | 是（依赖 checkpointer）[^3] | 由应用逻辑决定 | LangGraph `interrupt()`[^3] | 长时运行、人工响应延迟大、条件触发 |
| 工具调用前置审批式 | 工具调用前 | 是（`RunState` 可序列化）[^4] | 单次工具调用（可静态或动态） | OpenAI Agents SDK `needs_approval`[^4] | 高风险工具调用逐次把关，安全敏感 |
| 代理参与者式 | run 执行期间 | 否（阻塞时不可保存/恢复）[^5] | 对话轮次 | AutoGen `UserProxyAgent`[^5] | 短交互、即时反馈、按钮式批准 |
| 任务级审查式 | 任务完成后 | 未在证据中说明 | 任务级最终答案 | CrewAI `human_input`[^6] | 任务边界清晰的流水线关键节点 |
| 事件驱动式 | 事件流中 | 未在证据中说明 | 事件类型 | LlamaIndex `InputRequiredEvent`/`HumanResponseEvent`[^7][^14] | 异步复杂流程、介入点由事件类型决定 |
| 策略防火墙式 | 动作执行前（外部控制平面） | 未在证据中说明 | 策略级（跨工具/工作流） | Cordum[^8] | 受监管/企业环境、统一策略与审计 |

**组合使用**：证据显示这些模式并非互斥。例如 CrewAI 的 `guardrail`（程序化验证）+ `human_input`（人工审查）构成两级把关[^6]；Semantic Kernel 的 filter 可作为审批流程的前置权限校验[^12]；OpenAI Agents SDK 的程序化审批回调可使 run 无需暂停即继续，与手动中断审批并存[^4]。

---

## 十、矛盾与不确定性

### 10.1 HITL 定义与分类框架缺失

**现有公开资料不足以确认** HITL 的标准分类框架，也不足以确认 HITL 与 human-on-the-loop、human-in-command、human oversight 之间的系统性区别。三条相关来源均未提供分类框架：Wikipedia 仅给出泛化定义（'a model requiring human interaction'）并提及 human-on-the-loop 用于致命性自主武器语境，未做系统区分[^1]；EU AI Act 第14条只规定 human oversight 的合规义务，不构成 HITL 分类学[^2]；Amershi 等 18 条 guideline 明确被定位为设计准则而非自动化等级/监督模式分级框架[^9][^11]。本报告第二节所列的模式划分，是基于工程实现证据的归纳，而非引用某一既有分类学。

### 10.2 学术证据与量化收益完全缺失

15 条证据中**没有任何 arXiv 论文或学术评测来源**。human-approval gating、uncertainty-triggered escalation、interactive/co-pilot 协作、human feedback 在线策略修正等模式的成功率、成本、延迟、人工负担量化数据完全缺失。本报告无法就「哪种模式在何种指标上更优」给出有证据支撑的结论。

### 10.3 产业部署案例缺失

唯一产业侧来源是 cordum-io/cordum 仓库（约 509 stars），属工具自述/README 主张，非实际部署案例[^8]，且未涉及金融交易、医疗、代码执行、客服退款、法律文书等具体场景的审批粒度、介入频率、延迟预算或合规配置。产业侧证据严重不足，且无独立来源交叉验证。本报告第七节对 Cordum 的描述均转述其 README 主张，不应视为已验证的部署事实。

### 10.4 部分模式的状态持久化能力未在证据中说明

任务级审查式（CrewAI `human_input`）[^6]、事件驱动式（LlamaIndex）[^7][^14]、策略防火墙式（Cordum）[^8] 的暂停状态能否持久化并跨会话恢复，现有证据未明确说明。相比之下，中断—恢复式[^3]、工具调用前置审批式[^4] 有明确的状态持久化机制，代理参与者式则明确**不可**保存或恢复[^5]。

### 10.5 术语边界模糊

Wikipedia 指出 HITL 被用于多个语境（M&S 分类法、机器学习数据选择、对话式 AI、致命性自主武器）[^1]，说明该术语本身跨领域且边界不清。将其应用于 LLM Agent 系统时，不同框架对「HITL」的所指并不一致：LangGraph 指图执行中断[^3]，OpenAI Agents SDK 指工具调用审批[^4]，AutoGen 指用户代理参与[^5]，CrewAI 指任务输出审查[^6]。本报告按各来源自身用法转述，不做统一化。

---

## 参考来源

[^1]: Human-in-the-loop — Wikipedia — https://en.wikipedia.org/wiki/Human-in-the-loop (web)
[^2]: Article 14: Human Oversight — EU AI Act Explorer — https://artificialintelligenceact.eu/article/14/ (web)
[^3]: LangGraph — Interrupts (Human-in-the-loop) — https://docs.langchain.com/oss/python/langgraph/interrupts (web)
[^4]: OpenAI Agents SDK — Human-in-the-loop — https://openai.github.io/openai-agents-python/human_in_the_loop/ (web)
[^5]: AutoGen AgentChat — Human-in-the-Loop — https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/human-in-the-loop.html (web)
[^6]: CrewAI — Tasks (human_input attribute) — https://docs.crewai.com/en/concepts/tasks (web)
[^7]: LlamaIndex Workflows — Introduction (event-driven steps) — https://docs.llamaindex.ai/en/stable/module_guides/workflow/ (web)
[^8]: cordum-io/cordum — The action firewall for AI agents (Agent Control Plane) — https://github.com/cordum-io/cordum (github)
[^9]: Guidelines for Human-AI Interaction (Amershi et al., CHI 2019) — Microsoft Research publication page — https://www.microsoft.com/en-us/research/publication/guidelines-for-human-ai-interaction/ (web, 2019-05)
[^10]: HAX Design Library — the 18 Guidelines for Human-AI Interaction (Microsoft HAX Toolkit) — https://www.microsoft.com/en-us/haxtoolkit/library/guidelines/ (web)
[^11]: HAX Toolkit overview — positioning of the Guidelines within the human-centered AI design lifecycle — https://www.microsoft.com/en-us/haxtoolkit/ (web)
[^12]: What are Filters? — Semantic Kernel (Microsoft Learn) — https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/filters (web)
[^13]: Function calling with chat completion — Semantic Kernel (Microsoft Learn) — https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/chat-completion/function-calling/ (web, 2023-07-12)
[^14]: run-llama/template-workflow-human-in-the-loop — src/human_in_the_loop/workflow.py — https://raw.githubusercontent.com/run-llama/template-workflow-human-in-the-loop/main/src/human_in_the_loop/workflow.py (github)
[^15]: Managing State — AutoGen AgentChat User Guide (microsoft/autogen) — https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/state.html (web)