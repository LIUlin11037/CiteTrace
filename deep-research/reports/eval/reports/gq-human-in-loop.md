# Human-in-the-loop 在 LLM Agent 系统中的主流实现模式与适用场景

## 摘要

现有证据显示，主流 LLM Agent 框架的 HITL 实现可归为五类机制性模式：**中断-恢复式审批门控**（LangGraph 的 `interrupt()` / OpenAI Agents SDK 的 `needs_approval`）、**阻塞式人类代理**（AutoGen 的 `UserProxyAgent`、AG2 的 `human_input_mode`）、**任务级审阅开关**（CrewAI 的 `human_input` 布尔属性）、**事件驱动的步骤内人类输入**（LlamaIndex Workflow 的 `HumanChoiceEvent`）、以及**过滤器/回调式程序化介入**（Semantic Kernel 的 filter、OpenAI Agents SDK 的 programmatic approval callbacks）[^1][^2][^3][^9][^10][^11][^12][^13]。这些模式在「是否持久化状态」「是否阻塞执行」「介入粒度（run / task / step / 字段）」「触发条件是否可编程」四个维度上分化明显，直接决定其适用场景。合规侧，EU AI Act 第 14 条要求高风险系统可被自然人有效监督，且监督措施须与风险、自主性水平和情境相称，但该条对附件 III 类系统的义务生效时间为 2027 年 12 月 2 日[^5][^16]。量化证据极度稀缺：仅「按风险动态调节人工审计率」一类有理论保证与实验支撑[^18]，其余模式无对照实验数据；NIST 侧与部署侧介入率数据均空缺。

---

## 一、模式一：中断-恢复式审批门控（interrupt / resume）

### 机制定义

**LangGraph 的实现**：在图的任意节点调用 `interrupt()` 暂停执行，通过持久化层保存图状态，并无限期等待外部输入后恢复[^1]。恢复通过 `Command(resume=...)` 完成，其值成为节点内 `interrupt()` 调用的返回值，且必须使用与中断发生时相同的 `thread_id`[^1]。中断是**动态的**——可放在代码任意位置并依据应用逻辑条件触发，这与「静态断点（仅在特定节点前/后暂停）」形成对照[^1]。`interrupt` 支持 `response_schema`（Pydantic / TypedDict / dataclass）约束人类回复结构，客户端可据此渲染带类型的输入表单，即介入粒度可细化到**结构化字段级**[^1]。

LangGraph 将 HITL 列为四大核心能力之一，描述为「执行过程中任意点检查并修改 agent 状态」，并与 Durable execution（失败后从断点自动恢复）并列，说明 HITL 依赖持久化状态层实现[^7]。

**OpenAI Agents SDK 的实现**：HITL 流程定义为暂停 agent 执行，直到人类批准或拒绝敏感工具调用[^2]。审批由工具侧声明：`needs_approval` 可设为 `True`（总是需要审批）或一个异步函数（按调用逐次判定），即触发条件可配置为「全量」或「条件式」[^2]。暂停时 `RunResult.interruptions` 包含 `ToolApprovalItem`，携带 `agent.name`、`tool_name`、`arguments` 等细节；通过 `result.to_state()` 转为 `RunState`，调用 `state.approve(...)` / `state.reject(...)` 后以 `Runner.run(agent, state)` 恢复[^2]。

该实现的审批面是 **run 级别**的，不限于当前顶层 agent：经由 handoff 到达的 agent、或嵌套 `Agent.as_tool()` 执行中的工具，其中断仍浮现在外层 run 上，需在外层 `RunState` 上批准/拒绝并恢复原始顶层 run[^2]。此外支持「粘性决策」：`always_approve=True` / `always_reject=True` 可将同一决定持久化到该工具后续调用，`RunState` 也存储粘性决策，可经 `to_string()` / `to_json()` 序列化后跨会话恢复；对于函数工具，粘性决策按请求审批的 agent 作用域隔离，不同 agent 的同名工具不会继承该决定[^2]。

### 适用场景与约束

LangGraph 文档列出的常见 HITL 模式包括：审批工作流（在执行 API 调用、数据库变更、金融交易等关键动作前暂停）、审查与编辑（让人修改 LLM 输出或工具调用）、中断工具调用（执行前审查并编辑 tool call）、校验人类输入[^1]。这四类场景的共同前提是**动作不可逆或后果严重**，因此值得付出暂停成本。

关键实现约束：恢复时节点会从该节点开头重新执行，因此 `interrupt` 之前的代码会再次运行[^1]。这意味着中断点之前的副作用必须幂等，否则会产生重复执行问题——这是选型时容易被忽略的工程陷阱。

OpenAI Agents SDK 的 run 级审批面设计，使其适合**多 agent 委派链**场景：当顶层 agent 通过 handoff 把任务交给子 agent 后，子 agent 的敏感工具调用仍由外层统一把关，避免审批责任在委派链中丢失[^2]。粘性决策则适合**高频重复调用同一工具**的场景（如反复查询同一数据库），避免审批疲劳；但 agent 作用域隔离意味着不能假设「批准了 A agent 的 search 工具」就覆盖 B agent 的同名工具[^2]。

---

## 二、模式二：阻塞式人类代理（UserProxyAgent / human_input_mode）

### 机制定义

**AutoGen AgentChat**：提供两种与团队交互的方式——运行中（`run()` / `run_stream()` 执行期间）通过 `UserProxyAgent` 提供反馈；或在运行终止后，通过下一次 `run()` / `run_stream()` 的输入提供反馈[^9]。`UserProxyAgent` 是内置的特殊 agent，充当用户代理向团队提供反馈；将其加入团队后，由**团队决定何时调用它**征求用户反馈[^9]。调用时机由团队类型决定：`RoundRobinGroupChat` 中按传入团队的顺序调用，`SelectorGroupChat` 中由 selector prompt 或 selector function 决定[^9]。

**AG2（AutoGen v0.2 谱系）**：通过 `ConversableAgent` 的 `human_input_mode` 参数实现 HITL，由充当「人类」的 agent 承载，取值三档：`ALWAYS`（agent 使用人类输入作为其响应）、`TERMINATE`（仅在终止对话时请求输入）、`NEVER`（agent 从不请求人类输入）[^12]。

### 适用场景与约束

AutoGen 文档明确指出：`UserProxyAgent` 被调用时会**阻塞团队执行**，直到用户提供反馈或报错，这会拖住团队进度并使团队处于「无法保存或恢复」的不稳定状态[^9]。因此文档建议仅用于需要即时反馈的**短交互**，例如按钮点击式的批准/拒绝，或需要立即关注否则任务失败的告警[^9]。可通过自定义 `input_func` 适配不同前端：示例用 `input()` 读控制台，FastAPI 示例用 `websocket.receive_json()` 等待用户消息，另有 ChainLit、Streamlit 集成样例[^9]。运行中征求反馈会产生 `UserInputRequestedEvent`（含 `request_id`）事件，示例中配合 `TextMentionTermination("APPROVE")` 在用户回复 APPROVE 时终止[^9]。

「无法保存或恢复」这一约束是模式二与模式一的分水岭：阻塞式代理**不适合长时运行工作流**，因为进程中断即丢失全部上下文；而中断-恢复式依赖持久化层可无限期等待[^1][^9]。

AG2 文档给出的适用场景更为业务导向：需要细致判断的决策（金融合规、法律事务）、错误后果严重的场景（金融交易、安全关键系统）、受益于主观输入的流程（内容审批、设计选择）、监管要求人工监督的领域（金融服务、医疗）[^12]。示例中 `human_input_mode="ALWAYS"` 被描述为创建「检查点」，工作流继续前必须由人类提供输入，用于满足监管合规要求[^12]。示例模式为：finance_bot 依据系统消息判断交易是否可疑（如金额 > $10,000、供应商异常、备注含糊）并向人类 agent 请求批准，其余自动批准；人类给出总体批准则适用于所有待批交易[^12]。需注意这是**假设性示例而非部署数据**。

---

## 三、模式三：任务级审阅开关（human_input 布尔属性）

### 机制定义

CrewAI 的 Task 提供 `human_input` 可选属性（`Optional[bool]`），语义为「是否应由人工审阅 agent 的最终答案」，默认值为 `False`[^3]。这是**任务级布尔开关**，文档该节未描述中断/恢复 API、checkpoint 持久化或超时升级机制[^3]。

Task 还提供 `guardrail` / `guardrails`（在进入下一任务前校验任务输出）与 `guardrail_max_retries`（校验失败时的最大重试次数，默认 3），原 `max_retries` 属性已废弃并将在 v1.0.0 移除[^3]。任务执行流程由 Crew 的 `process` 决定，支持 sequential（按定义顺序执行）与 hierarchical（按角色与专长分配 agent）两种[^3]。CrewAI 文档导航将「Flows」描述为编排 start/listen/router 步骤、管理状态、持久化执行并恢复长时运行工作流；将「Tasks & Processes」描述为含 guardrails、callbacks 与 human-in-the-loop 触发器的顺序/层级/混合流程[^3]。

### 适用场景与约束

该模式的介入粒度最粗——仅审阅**任务最终答案**，而非中间工具调用或字段级输入，且触发条件是静态的（任务定义时确定）[^3]。它适合**产出物需要人工签核但过程无需干预**的场景，例如报告生成、内容定稿。与模式一的动态条件触发、模式二的团队调度触发相比，CrewAI 的开关不具备运行时按风险调节的能力[^1][^3][^9]。

值得注意的是，`guardrail` 提供了**程序化自动校验**路径，与人工审阅形成互补：能在进入下一任务前自动拦截不合格输出，减少需要人工介入的比例[^3]。但现有证据未说明 guardrail 失败后是否可升级为人工介入——「现有公开资料不足以确认」。

---

## 四、模式四：事件驱动的步骤内人类输入

### 机制定义

LlamaIndex Workflow 是事件驱动、基于 step 的执行流控制方式：step 接收事件、做工作、返回另一个事件，返回的事件触发类型注解可接受它的下一个 step[^10]。文档明确 step 可以调用 LLM、运行检索、**请求人类输入**、更新共享状态或分派批量工作[^10]。Workflow 用普通 Python 表达分支（返回不同事件类型的 if 语句）、循环（返回由更早 step 处理的事件）与并发（返回 `list[Event]` 并由接受 `list[Event]` 的 step 配对处理）[^10]。框架内置 `StartEvent` 与 `StopEvent` 两个特殊事件：`StartEvent` 承载初始输入，返回 `StopEvent` 时工作流立即停止并返回 result[^10]。

HITL 的具体形态见官方示例：某个 step 需要人类专业知识或输入才能运行，或需要人类校验 Workflow 的初始输出[^11]。示例用两个 step 循环实现 HITL：第一个 step 让 LLM 生成 Segment 并构造 Block，第二个 step（`prompt_human`）提示人类从 LLM 给出的动作列表中做选择，返回 `HumanChoiceEvent`[^11]。人类选择通过 `HumanChoiceEvent` 回传给 `create_segment` step，该 step 同时接受 `StartEvent | HumanChoiceEvent` 作为输入，从而形成「LLM 生成 → 人类选择 → 再生成」的循环[^11]。跨 step 的累积状态（`blocks` 列表）通过 Context 的 `ctx.store.get/set` 读写，而非依赖 step 间直接传参[^11]。工作流通过 `max_steps` 参数限制循环轮数，达到上限后生成收尾 Segment 并返回 `StopEvent`[^11]。

### 适用场景与约束

该模式的核心特征是**人类输入被建模为一等事件**，与 LLM 输出、检索结果走同一条事件流，因此天然支持多轮人机交替（而非一次性审批）[^10][^11]。适合**交互式创作、分步决策、需要人类在多个中间节点持续参与**的场景，如示例中的互动故事生成[^11]。`max_steps` 提供了循环上限保护，避免人机往复无限进行[^11]。Workflow 构造时可设置 `timeout`（秒）与 `verbose` 等参数用于调试[^10]。

与模式一相比，该模式未在证据中描述跨进程持久化与恢复语义（`ctx.store` 是工作流内共享状态，非 checkpoint 层），因此「现有公开资料不足以确认」其是否支持进程崩溃后的恢复。

---

## 五、模式五：过滤器 / 程序化回调式介入（不打断执行）

### 机制定义

**Semantic Kernel 的 filter**：用于在函数运行的方式与时机上提供控制与可见性；文档举例说明 filter 可用于「在审批流程开始前校验权限」，即只有特定人群能发起审批流程[^13]。三类 filter：Function Invocation Filter（每次 `KernelFunction` 调用时执行）、Prompt Render Filter（提示渲染前触发）、Auto Function Invocation Filter（在自动函数调用范围内运行）[^13]。Function Invocation Filter 可访问被执行的函数及其参数、处理执行中的异常、在执行前后覆盖函数结果（如缓存、负责任 AI 场景）、失败时重试（如切换备用模型）[^13]。Prompt Render Filter 可查看与修改将发送给 AI 的提示（如 RAG、PII 脱敏），或通过覆盖函数结果阻止提示提交给 AI（如语义缓存）[^13]。Auto Function Invocation Filter 提供额外上下文，包括聊天历史、待执行函数列表与迭代计数，并允许终止自动函数调用过程[^13]。

每个 filter 都有 context 对象与 next 委托/回调；在 filter 中**必须调用 next** 才能继续到下一个 filter 或原操作，不调用则操作不会执行[^13]。多个同类型 filter 可注册；使用依赖注入注册时顺序不保证，多 filter 时执行顺序可能不可预测[^13]。

**OpenAI Agents SDK 的程序化审批回调**：除暂停-恢复的人工审批外，部分工具类型还支持 programmatic approval callbacks，使 run 无需暂停即可继续——即存在「不打断执行」的替代路径[^2]。可调用审批规则在无法安全解析参数时采取 **fail closed**：参数缺失、为空、仅空白、JSON 畸形、JSON 合法但非对象（如 `null` 或 list）、或含 `NaN`/`Infinity` 等非标准常量时，不调用该 callable，而要求人工审批[^2]。

### 适用场景与约束

该模式的定位是**用程序化规则替代大部分人工介入，仅在规则无法判定时升级到人**。Semantic Kernel 的 filter 适合横切关注点（权限校验、PII 脱敏、缓存、备用模型重试）的统一拦截[^13]；OpenAI Agents SDK 的 fail-closed 设计则明确了「解析不确定即交人」的安全默认值[^2]。二者共同指向一个工程原则：**人工介入应是异常路径而非默认路径**，否则会产生审批疲劳。

风险点在于：Semantic Kernel 多 filter 的执行顺序在依赖注入注册下不可预测，若审批逻辑依赖顺序，行为可能不稳定[^13]。

---

## 六、量化证据：仅「动态审计率」一类有支撑

现有证据中，唯一直接对应 HITL 模式且带量化结果的是 arxiv 论文《Reliable AI-Assisted Decisions via Selective Auditing》[^18]。该论文把「何时自动接受 AI 决策、何时转交人类审计员」形式化为**在线约束优化问题**：目标是最小化专家审计频率，同时保证未被审计的 AI 错误比例低于用户设定的阈值[^18]。论文允许提出决策的 agent 与给出置信度/正确性分数的 verifier agent 是不同模型（proposer-verifier 分离架构）[^18]。论文提出一族算法，声称在无分布假设下具备可证明的错误控制保证，其中一种方法在对抗性与 i.i.d. 数据流下均渐近达到最优审计率[^18]。实验场景包括：对 LLM 有害请求的筛查、定性文本标注、数据标注；结果显示在保持有效错误控制的同时，相比基线显著降低了人工审计需求[^18]。

该工作直接对应「按风险动态调节人工介入比例」这一模式，可作为干预率（audit rate）可调且可保证的理论依据[^18]。但需注意：论文摘要层面未给出成功率、延迟、人类负担的具体数值，也未与其他四类模式做对照。

其余四类模式（审批门控、澄清式提问、纠错回退、协作式规划、人类作为工具/子 agent）**无任何实验数据**。METR 的 RCT 虽涉及人机协作，但测量对象是 16 名资深开源开发者在自有仓库上完成 246 个真实 issue 的效率，核心结果是允许使用 AI 工具时完成时间反而延长 19%，而开发者主观预期提速 24%、经历变慢后仍相信提速 20%[^19]。该结果与 HITL 模式设计无直接关系，且页面自标「结果已过时，2026 年 2 月已发布基于 late-2025 AI 工具的新数据」，作者亦明确不声称结论可外推到软件开发以外领域[^19]。其可借鉴之处仅在于提示：**人类对自身介入效果的判断可能系统性偏离实测**，这对设计审批界面与评估 HITL 收益有警示意义，但不能作为 HITL 模式有效性的证据。

---

## 七、合规要求与实现模式的对应关系（及证据缺口）

EU AI Act 第 14(1) 款要求高风险 AI 系统在设计开发时（包括配备适当的人机界面工具）确保其在使用期间可被自然人**有效监督**[^5]。第 14(2) 款规定人工监督的目标是预防或最小化系统按预期用途使用、以及可合理预见误用情形下对健康、安全或基本权利产生的风险，尤其针对其他条款要求已适用后仍残留的风险[^5]。第 14(3) 款规定监督措施应与高风险 AI 系统的风险、自主性水平和使用情境**相称**[^5]。Recital 66 将 human oversight 列为高风险系统六项并列要求之一（与风险管理、数据质量、技术文档与记录保存、透明度、鲁棒性/准确性/网络安全并列），关联 Articles 10–15[^15]。

时间线上，第 14 条对附件 III 类高风险系统的生效时间为 2027 年 12 月 2 日（依第 113(c) 条），对附件 I 类为 2028 年 8 月 2 日[^5][^16]。附件 III 覆盖生物识别、关键基础设施、教育与职业培训、就业与工人管理、基本服务获取（含信用评分、寿险与健康险风险定价、紧急呼叫分诊）、执法等领域[^14]。

**关键证据缺口**：Art.14 页面在第 3 款后被截断，第 4 款（关于自动化偏差、忽略/覆盖输出、中止按钮等能力）正文未取得[^5]。而第 4 款恰是「人类应能忽略、覆盖或反转输出、通过中止按钮中断系统」等最可操作、最直接对应 HITL 实现模式（审批门控、中止/回退）的条款。因此**合规要求无法精确落到具体实现模式上**——只能确认「有效监督」「与风险相称」两条原则，无法确认具体能力清单。

从原则到模式的映射可作如下谨慎推断（属分析而非证据直述）：第 14(3) 款的「相称性」原则与选择性审计模式的「按风险动态调节审计率、保证错误率低于阈值」在目标上一致[^5][^18]；第 14(2) 款「最小化残留风险」与 fail-closed 审批规则的设计取向一致[^2][^5]。但这是目标层面的对应，非条款级的实现要求。

**NIST 侧证据为零**：AI RMF 落地页仅返回一句概述性表述（培育信任、促进创新、降低风险），未返回任何关于 human-AI configuration、人工监督具体职能或 2025 年更新的正文内容[^6]。AI 专题页抓取内容通篇以 "super intelligence" 替代 AI 表述，并称依据 2026 年 9 月 29 日第 14434 号行政令更新措辞，与已知 NIST AI RMF / AI 600-1 语境不一致，判定为异常外部数据，不予采信[^17]。因此**无法就 NIST 对 HITL 的合规要求作出任何结论**。

---

## 八、矛盾与不确定性

**1. 框架量化维度完全空缺。** 现有 GitHub 证据全部是第三方示例/画像仓库：canvas-callback（★90，TypeScript，last push 2025-03-19）、langgraph-interrupt-workflow-template（★32，Python，last push 2026-10-22）、Deuz-SDK（★519，TypeScript，last push 2026-10-16）、spec-to-agents（★115，Python，last push 2026-06-12）、crewai-cloud（★0，第三方 API 画像）[^4]。检索明确未返回 langchain-ai/langgraph、crewAIInc/crewAI、openai/openai-agents-python、microsoft/semantic-kernel、run-llama/llama_index 等官方仓库条目，故不能据此得出官方框架的 stars 与活跃度[^4]。**框架选型比较缺少量化依据**，本报告的模式归纳仅基于官方文档的机制描述，不代表采用度或成熟度排序。

**2. 五类模式中四类无实验数据。** 审批门控、澄清式提问、纠错回退、协作式规划、人类作为工具/子 agent 这五类中，仅「按风险动态调节审计率」有量化支撑[^18]。成功率、成本、延迟、人类负担四个维度均无对照实验数据，无法给出模式间的定量优劣排序。

**3. 部署侧介入率与失效案例缺失。** 现有证据中没有任何部署侧介入率统计（如审批通过率、人工接管频率）或 HITL 失效案例（如审批疲劳导致橡皮图章、自动化偏差、审批绕过事故）。AG2 文档的金融合规示例属假设性示例，非部署数据[^12]。METR 的「主观感知与实测方向相反」结果提示自动化偏差可能存在，但该研究不测 HITL 模式，作者亦明确限制外推范围[^19]。

**4. EU AI Act 第 4 款缺失导致合规-实现映射不完整。** 如第七节所述，Art.14 第 4 款正文未取得，最可操作的监督能力要求（忽略/覆盖/反转输出、中止按钮）无法确认[^5]。因此本报告只能给出原则级对应，不能声称某实现模式「满足」或「不满足」第 14 条。

**5. NIST 证据不可用。** AI RMF 落地页内容过薄，AI 专题页被判定为异常数据，二者均无法支撑结论[^6][^17]。NIST 与 EU AI Act 在 HITL 要求上是否一致、是否存在冲突，**现有公开资料不足以确认**。

**6. 部分机制细节未覆盖。** CrewAI 的 `human_input` 是否可与 Flows 的持久化恢复组合使用、guardrail 失败后是否可升级为人工介入，文档该节未描述[^3]；LlamaIndex Workflow 的 `ctx.store` 是否具备跨进程持久化语义，证据未说明[^10][^11]；LangGraph 恢复时节点从头重执行所带来的副作用重复问题，文档仅陈述约束未给出通用缓解方案[^1]。这些均属「现有公开资料不足以确认」。

## 参考来源

[^1]: LangGraph Docs — Interrupts (human-in-the-loop patterns) — https://docs.langchain.com/oss/python/langgraph/interrupts（web）
[^2]: OpenAI Agents SDK — Human-in-the-loop — https://openai.github.io/openai-agents-python/human_in_the_loop/（web）
[^3]: CrewAI Tasks 文档（human_input 属性定义） — https://docs.crewai.com/en/concepts/tasks（web）
[^4]: GitHub 仓库检索结果（第三方 HITL 示例仓库，非官方框架仓库） — https://github.com/ahmad2b/canvas-callback（github）
[^5]: EU AI Act Article 14: Human Oversight (AI Act Explorer 正文转载) — https://artificialintelligenceact.eu/article/14/（web）
[^6]: NIST AI Risk Management Framework (官方页面) — https://www.nist.gov/itl/ai-risk-management-framework（web）
[^7]: langchain-ai/langgraph README（低层有状态编排框架，HITL 为核心特性） — https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md（github）
[^9]: AutoGen 文档：AgentChat Human-in-the-Loop（UserProxyAgent 阻塞式反馈） — https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/human-in-the-loop.html（web）
[^10]: LlamaIndex Docs — Workflows (event-driven steps, human input) — https://docs.llamaindex.ai/en/stable/module_guides/workflow/（web）
[^11]: LlamaIndex Docs — Choose Your Own Adventure Workflow (Human In The Loop) — https://docs.llamaindex.ai/en/stable/examples/workflow/human_in_the_loop_story_crafting/（web）
[^12]: AG2 (AutoGen v0.2 lineage) Docs — Human in the Loop: Adding Human Oversight — https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/human-in-the-loop/（web）
[^13]: Microsoft Learn — What are Filters? (Semantic Kernel) — https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/filters（web）
[^14]: Annex III: High-Risk AI Systems Referred to in Article 6(2) — EU AI Act Explorer — https://artificialintelligenceact.eu/annex/3/（web, 2024-07-12）
[^15]: Recital 66 — EU AI Act Explorer — https://artificialintelligenceact.eu/recital/66/（web, 2024-07-12）
[^16]: Implementation Timeline — EU AI Act Explorer (Future of Life Institute), last updated 31 August 2026 — https://artificialintelligenceact.eu/implementation-timeline/（web, 2026-08-31）
[^17]: NIST 人工智能专题页（抓取内容异常，不予采信） — https://www.nist.gov/artificial-intelligence（web）
[^18]: Reliable AI-Assisted Decisions via Selective Auditing — https://arxiv.org/abs/2610.31662（arxiv, 2026-10-22）
[^19]: Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity (METR RCT) — https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/（web, 2025-07-10）