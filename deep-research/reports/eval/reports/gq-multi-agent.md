## 摘要

多智能体 LLM 系统（LLM-MA）的主流协作模式可归为五类可辨识的拓扑：**层级式（orchestrator/leader–worker）**、**水平式群聊/共享线程**、**辩论式（对等批评–修正）**、**流水线/角色分工（SOP 驱动）**、**图/网络编排（handoff、channel、可优化图）**。学术综述给出的分类轴是 vertical 与 horizontal 两端的光谱，多数实际架构落在两者之间[^2]；工程框架则各自把某一端固化为原语：OpenAI Agents SDK 以 handoffs 与 agents-as-tools 为核心[^10][^19]，CrewAI 以 sequential/hierarchical process 加事件驱动 Flows 为核心[^36][^37][^38]，AG2 v1.0 以 hub-and-spoke 的 Network + typed channels 取代经典 GroupChat[^41][^42]，AutoGen 新版示例已改用 AgentTool 包装子 agent[^44]，MetaGPT 用角色分工 + SOP 流水线[^22]，ChatDev 用双 agent 通信 + chat chain[^23]。

「多智能体是否优于单智能体」的答案高度依赖任务结构与算力口径，且现有证据互相冲突：Anthropic 报告其 orchestrator-worker 研究系统比单智能体 Opus 4 高 90.2%[^16]；辩论式在算术、GSM8K、国际象棋上稳定超过单智能体与多数投票[^4]；但 MAST 对 7 个 SOTA MAS 的实证给出 41%–86.7% 的失败率，并指出相对单智能体与 best-of-N 的增益往往很小[^6]；算力受控的架构缩放研究给出 +80.8% 到 −70.0% 的区间，并发现 capability-saturation effect[^31]；面向用户的 τ²-bench 双控实验中，13 个有效单元格里 MAS 有 12 个更差[^32]。可辩护的结论是：**多智能体在高度可并行、信息量超出单上下文窗口、子任务可分解且相互依赖少的场景下有优势；在需要共享同一上下文、强顺序依赖、面向用户实时交互的场景下，单智能体是更优默认值。**

---

## 一、概念边界：什么算多智能体

综述给出的 agent 定义是「由语言模型驱动、能够规划并采取行动、在多次迭代中执行目标的实体」，并以 brain、perception、action 三组件为最低要求[^2]。单智能体架构由单个语言模型驱动，自行完成全部推理、规划与工具执行，其关键边界特征是**没有来自其他 AI agent 的反馈机制**（但可以有人类反馈）[^2]。多智能体架构则定义为涉及两个或更多 agent，各 agent 可使用相同或不同的语言模型、可访问相同或不同的工具，且通常各有自己的 persona[^2]。

另一条定义线来自 LLM-MA 综述：单智能体系统的构建聚焦内部机制与对外部环境的交互，而 LLM-MA 系统强调多样化的 agent profile、智能体间交互与集体决策过程[^1]。该综述把 LLM-MA 研究拆为四个分析维度——agents-environment interface、agent profiling、agent communication、agent capability acquisition[^1]，并把应用划分为 multi-agents for problem-solving 与 multi-agents for world simulation 两大主线[^1]。

需要区分的是「多智能体」与「多次采样」：Agent Forest 的方法是对同一 LLM 重复查询 N 次得到样本集，再用多数投票（按与其他样本的累计相似度取最大者）合成答案[^29]。这是单模型集成而非异构 agent 协作，其增益随 ensemble size 上升，且在更难任务、更弱模型上提升更大[^29]。把它与多智能体混为一谈会高估协作的贡献——MAST 正是把 best-of-N 采样列为应当对照的简单基线[^6]。

---

## 二、主流协作模式

### 2.1 层级式（vertical / orchestrator–worker）

**定义与步骤**：一个 agent 作为 leader，其他 agent 向其汇报；汇报 agent 可能只与 lead 通信，或 leader 与全体共享一段对话[^2]。典型实现是 lead agent 分析任务、制定策略、派生并行 subagents，subagents 各自在独立上下文窗口中探索不同方面后把要点压缩回传给 lead[^16]。

- **Magentic-One**：Orchestrator 指挥 WebSurfer、FileSurfer、Coder、ComputerTerminal 四个 agent，实现双循环——外循环维护 Task Ledger（事实、猜测、计划），内循环维护 Progress Ledger（当前进度、任务分配）；连续多步无进展则更新 Task Ledger 并重新规划[^20][^24]。团队共五个 agent，基于 AutoGen 实现[^24]。
- **AgentOrchestra**：central planner 把每个子任务路由到领域专用 sub-agent 或环境，后者只暴露精选工具集与局部上下文，从而把全局协调转化为一系列局部路由决策，支持树状扩展并限制 orchestrator 的决策范围[^25]。
- **Claude Code / Agent SDK 的 subagents**：subagent 是主 agent 可派生来处理聚焦子任务的独立 agent 实例，默认异步后台运行，Agent 工具结果为 `status: 'async_launched'`，报告在后续轮次到达；子代理上下文默认全新开始、不继承父对话（除非是 fork）[^49]。官方列出的收益为 context isolation、parallelization、specialized instructions、tool restrictions 四项[^49]。
- **CrewAI hierarchical process**：自动为 crew 分配一个 manager 协调任务的规划与执行，通过委派与结果验证完成；必须指定 manager_llm 或自定义 manager_agent 才能启用[^37]。任务不是预先分配的，manager 依据 agents 能力分配任务、审查输出并评估完成情况[^37]。

### 2.2 水平式（horizontal / 共享讨论线程）

**定义**：所有 agent 地位平等，处于同一共享讨论线程中，每个 agent 能看到其他所有消息，可自愿认领任务或调用工具，无需由 lead 指派[^2]。综述强调 vertical 与 horizontal 是光谱的两端，多数现有架构落在两者之间[^2]。

经典实现是 AutoGen 的 conversable agent 与 GroupChat：AutoGen 被定义为通过多个可互相对话的 agent 构建 LLM 应用的开源框架，agent 是 customizable、conversable 的，可在组合 LLM、人类输入与工具的各种模式下运行[^45]。但需注意版本断裂：microsoft/autogen 仓库已 archived，Microsoft 不再提供 bug 修复、安全修复、新特性或支持，README 明确建议不要用于生产，官方指定继任者为 Microsoft Agent Framework (MAF)[^44]；AG2 作为 AutoGen 血统的独立项目，自 v1.0 起顶层包改为协议驱动的 `ag2`，ConversableAgent、GroupChat、`import autogen` 已不在顶层包中，经典框架迁至 ag2ai/ag2-classic[^41]。

### 2.3 辩论式（multi-agent debate）

**定义与步骤**：多个同构 LLM 实例先各自独立作答，再在每轮读取并批评其他所有智能体的回答与推理过程，据此更新自身答案，重复若干轮后收敛到单一共同答案[^4]。实验设置为 3 个智能体 + 2 轮辩论，基线为单智能体、单智能体+反思、多智能体多数投票，全部任务使用同一套流程与提示模板，仅需黑盒模型访问[^4]。

一个变体是**对立辩手 + judge**：两个立场对立的辩手（affirmative/negative，即 angel 与 devil）以 tit for tat 状态轮流发言，另设 judge 管理并裁决辩论过程；辩手按固定顺序基于辩论历史 H 表达论点，judge 有判别模式（判断是否已得正确解则结束）与抽取模式（迭代上限内未识别出正确解则从完整历史抽取最终答案）[^5]。该文提出并定义了自我反思中的 Degeneration-of-Thought 问题：一旦 LLM 智能体对自身答案建立信心，即使初始立场错误，后续自我反思也无法产生新想法[^5]。

### 2.4 流水线 / 角色分工（SOP 驱动）

**定义与步骤**：为不同 agent 分配不同角色，把复杂任务拆解为多个 agent 协作的子任务，并把 SOP（标准作业程序）编码进 prompt 序列——MetaGPT 称之为 assembly line paradigm[^22]。MetaGPT 的 agent 通过文档与图表（结构化输出）而非对话进行通信，这是它与 ChatDev 的关键差异[^22]。

ChatDev 的拓扑是**双 agent 通信**：用 chat chain 规定 agent「沟通什么」、用 communicative dehallucination 规定「如何沟通」，覆盖 design、coding、testing 三阶段；论文称该设计通过避免复杂的多智能体拓扑来简化通信、加速达成共识[^23]。其 chat chain 只传递前一阶段的「解决方案」而非完整对话历史，以降低信息过载[^23]；上下文记忆按阶段切分为短期记忆（维持单阶段对话连续性）与长期记忆（跨阶段保留上下文感知）[^23]。

### 2.5 图 / 网络编排（handoff、channel、可优化图）

这一类不是「谁跟谁说话」的拓扑，而是把协作显式建模为可编排、可审计的结构。

- **OpenAI Agents SDK**：编排分两类——由 LLM 决策（自主规划、用工具取数、用 handoffs 委派）与由代码编排（更确定、在速度/成本/性能上更可预测）[^19]。两种最常用模式是 **Agents as tools**（manager agent 通过 `Agent.as_tool()` 调用专家 agent，自己保留对话控制权与最终答案）与 **Handoffs**（triage agent 把对话路由给专家，该专家在当轮剩余部分成为活跃 agent）[^19]。选择判据：专家只应协助有界子任务、不应接管面向用户对话时用 agents as tools；路由本身是工作流的一部分、且希望被选中专家拥有当轮剩余对话时用 handoffs；两者可组合[^19]。代码编排的常见模式包括结构化输出分类选下一 agent、链式串联（研究→大纲→写作→批评→改进）、while 循环中 task agent 与 evaluator agent 交替直到通过标准、以及用 `asyncio.gather` 并行跑互不依赖的任务[^19]。
- **AG2 Network**：hub-and-spoke 拓扑，由中心注册表协调基于 channel、协议驱动的交换，模块完全 opt-in[^42]。四种内置 channel adapter：conversation（恰好 2 方、自由发言、显式关闭或 TTL）、consulting（恰好 2 方、严格 1Q1R、回复后自动关闭）、discussion（2+ 方、round-robin）、workflow（2+ 方、声明式 TransitionGraph、由图终止）[^42]。Hub 持有注册表、write-ahead log 与审计轨迹[^41]。默认传输为 LocalLink（同进程双工队列），可换 WsLink 让每个 agent 运行在独立进程或主机而 HubClient API 不变[^42]。
- **CrewAI Flows**：事件驱动工作流，通过 `@start()` 标记入口点、`@listen()` 监听某任务输出，支持条件逻辑、循环、分支与 state 管理[^38]。官方定位 Flows 为应用 backbone（定义步骤、逻辑与数据流动），Crews 是 Flow 内执行具体工作的团队单元；对任何 production-ready 应用应从 Flow 开始，仅当某步骤需要一队 agents 完成特定复杂任务且需要自主性时才在该步骤内使用 Crew[^39]。
- **可优化图**：GPTSwarm 把 LLM-based agent 描述为计算图，节点实现处理多模态数据或查询 LLM 的函数，边描述操作之间的信息流；图可递归组合成更大的复合图表示层级协作，并提出 node optimization（精炼节点级 prompt）与 edge optimization（改变图连通性改进编排）[^28]。DyLAN 采用两阶段范式——Team Optimization 用基于 Agent Importance Score 的无监督 agent selection 挑选最优子集，Task Solving 阶段由 LLM-powered ranker 在中间动态停用低效 agent[^27]。
- **通信剪枝**：AgentPrune 把整个 LLM-MA 框架建模为时空通信图（同轮内为 spatial edges，跨对话为 temporal edges），通过 low-rank 原理引导的 graph mask 一次性剪枝得到稀疏拓扑并固定用于后续推理[^26]。它首次识别并形式化定义了 communication redundancy 问题，即相当一部分消息传递对协作智能没有实质贡献[^26]。

### 2.6 框架定位的元观察

LangGraph 明确自述为 low-level orchestration framework and runtime，「focused entirely on agent orchestration」，且「does not abstract prompts or architecture」——不预设多智能体架构，架构由使用者自行搭建[^35]；其 README 正文未出现 supervisor、swarm、hierarchical teams 等协作模式名称，也未主张任何特定拓扑[^34]。这与 CrewAI 的高层抽象形成对照：CrewAI 提供 high-level abstractions 与 low-level APIs，并把协作模式明确划分为 Crews（自主性、role-based）与 Flows（事件驱动、精确控制）两条互补路径[^36]。LangChain 的立场是：workflows 是由预定义代码路径编排 LLM 与工具的系统，agents 是 LLM 动态决定自身流程与工具使用的系统，生产中几乎所有的 agentic 系统都是两者的组合[^21]。

---

## 三、多智能体何时优于单智能体

### 3.1 支持多智能体的证据

**可并行、超出单上下文窗口的研究型任务。** Anthropic 内部评测中，以 Claude Opus 4 为 lead、Claude Sonnet 4 为 subagents 的多智能体系统在内部 research eval 上比单智能体 Claude Opus 4 高 90.2%[^16]。其归因是「花掉足够多的 token」：BrowseComp 上三个因素解释 95% 的性能方差，其中 token 使用量单独解释 80%，另两个是工具调用次数与模型选择[^16]。多智能体在 breadth-first 类查询上尤其出色——让系统找出标普 500 信息技术板块各公司全部董事会成员时，多智能体通过分解任务成功，单智能体因缓慢的串行搜索失败[^16]。

**需要多条不同执行路径、且子问题可独立处理。** 综述指出单智能体模式在需要长序列步骤时常常难以完成，而多智能体可通过让各 agent 独立处理子问题来应对并行任务与鲁棒性问题[^2]；多智能体在需要协作与多条不同执行路径时更有优势[^2]。使用多个 agent persona 的解法相比 Chain-of-Thought 提示有显著提升，并引用「shaped personality verifiably influences LLM behavior」的发现[^2]。

**需要异构能力组合。** LLM-MA 相对单 LLM 智能体的两点增益是：把 LLM 特化为多个能力不同的 agent；使这些异构 agent 之间能够交互，从而有效模拟复杂真实环境[^1]。Microsoft 主张模块化多智能体把不同技能封装在独立 agent 中简化开发与复用（类比面向对象编程），plug-and-play 设计支持增删 agent 而无需重做整个系统，而单智能体系统常受困于不灵活的工作流[^20]。

**辩论带来的推理增益。** 3 agent + 2 轮辩论的实测：算术任务单智能体 67.0±4.7%、单智能体+反思 72.1±4.5%、多数投票 69.0±4.6%、辩论 81.8±2.3%；GSM8K 分别为 77.0±4.2%、75.0±4.3%、81.0±3.9%、85.0±3.5%；国际象棋走子最优性（归一化 ΔPS）为 91.4±10.6、102.1±11.9、102.2±6.2、122.9±7.6[^4]。作者报告同时使用多个模型智能体与多轮辩论对最佳性能是重要的，并观察到辩论中模型会就不确定的事实产生分歧，从而把这些事实从答案中剔除，降低幻觉；还存在所有模型初始都答错、随辩论推进而得到正确答案的案例，说明辩论并非只是放大群体中已有的正确答案[^4]。

**模型异构性作为改进手段。** 对 MAD 的系统性评测（5 种方法 × 9 个基准 × 4 个基础模型）发现模型异构性是对现有 MAD 框架普遍有效的改进手段，能一致提升各框架表现，并主张把模型异构性作为核心设计原则[^7]。

**软件工程的角色化流程。** MetaGPT 自述在 HumanEval 与 MBPP 上取得 85.9% 与 87.7% 的 Pass@1，并称达到 100% 任务完成率[^22]（注意：这些数字取自引言与贡献列表的自述，未核对结果表格）。ChatDev 自述显著提升软件质量的完整性、可执行性与需求一致性[^23]（定性表述，未读到具体百分比）。

### 3.2 反对或限制多智能体的证据

**失败率高、增益小。** MAST 明确指出 MAS 相对单智能体框架、以及 best-of-N 采样这类简单基线的性能增益往往很小；对 7 个 SOTA 开源 MAS 的实证分析给出 41% 到 86.7% 的失败率区间[^6]。其失败分类法含 14 种失败模式，归为系统设计问题、智能体间失配、任务验证三大类[^6]。案例研究显示失败多源于系统设计而非单纯模型能力：ChatDev 中 CPO agent 出现「Disobey Role Specification」（未经 CEO agent 共识即终止对话），仅做工作流调整让 CEO 拥有最终决定权即带来整体任务成功率 +9.4%[^6]。作者强调孤立修补不足以获得稳健可靠性，往往需要根本性的 MAS 重设计[^6]。

**算力受控下的架构效应可正可负。** 覆盖 260 个配置、6 个 benchmark、5 种架构、3 个 LLM 家族的研究在标准化工具、prompt 与算力以隔离架构效应的条件下，给出相对单智能体基线 +80.8%（可分解的金融推理）到 −70.0%（顺序规划）的性能变化区间，即架构与任务结构的匹配决定协作成败[^31]。核心发现是 capability-saturation effect：当单智能体基线表现超过一定水平后，协调带来的收益递减；另两项发现是工具密集型任务会带来多智能体开销、缺少集中式验证的架构比集中式协调架构更容易传播错误[^31]。其预测模型交叉验证 R²=0.373（用 task-grounded capability 指标时为 0.413），可为 87% 的留出配置识别最佳架构[^31]。

**面向用户的场景中 MAS 系统性更差。** 在 τ²-bench 的 airline/retail/telecom 三个域上，用同一模型构建 stock 单智能体系统与集中式多智能体系统，覆盖 3 个模型家族 6 个模型、约 7,950 段模拟对话，并为每个 episode 记录 completion-token 账本用于算力匹配比较[^32]。在固定模拟用户下，13 个有效单元格中 MAS 有 12 个更差、1 个持平，没有任何单元格显示 MAS 收益；拟合增益为负，加权均值 −28 分，95% CI [−44, −12]；在规则预测 MAS 应获益的两个单元格上，MAS 反而分别落后 13 分和 22 分；无预算上限时最好也只能以 5.5 倍于 SAS 的 token 消耗打平[^32]。MAS 未能可检测地减少错误的数据库写入，而是把其他失败转化为预算耗尽与不作为[^32]。作者结论是面向用户的 agent 应以 SAS 为默认选择，多智能体设计必须从该场景本身给出 compute-matched 证据[^32]。

**MAD 常打不过简单基线。** 系统性评测的核心负面结论是 MAD 常常无法超越简单的单智能体基线（Chain-of-Thought、Self-Consistency），即使消耗显著更多的推理时算力；作者指出既有 MAD 研究存在基准覆盖有限、基线对比薄弱、实验设置不一致三类评测缺陷[^7]。

**上下文分裂导致误传。** Cognition 明确反对多智能体架构，点名 OpenAI 的 swarm 与 Microsoft 的 autogen 所推行的多智能体概念是「错误的构建 agent 方式」[^18]。核心失败机制：把任务拆给 subagents 时，subagent 会误读子任务（如把 Flappy Bird 背景画成 Super Mario 风格），最终 agent 被迫合并两个被误解的产物[^18]。其两条原则是共享上下文（且要共享完整的 agent trace，而不只是单条消息）、每个动作都携带隐含决策（相互冲突的决策会带来坏结果），并主张这两条原则极少值得违反，因此应默认排除不满足它们的架构[^18]。它推荐的最简合规架构是单线程线性 agent[^18]。

**成本量级。** agents 通常比 chat 交互多用约 4 倍 token，多智能体系统比 chat 多用约 15 倍 token，因此多智能体只适合任务价值足够高、能覆盖性能提升成本的任务[^16]。AgentPrune 的测量显示各类通信方法相比简单 chain 结构带来 2~11.8 倍的 token 消耗增长[^26]；其自身在 MMLU 上以 5.6 美元成本达到与 SOTA 通信拓扑 43.7 美元相当的结果，集成到现有框架可带来 28.1%~72.8% 的 token 消耗下降[^26]。

### 3.3 场景判据汇总

Anthropic 明确指出不适合多智能体的场景：需要所有 agent 共享同一上下文、或 agent 之间存在大量依赖的领域；并称多数编码任务的真正可并行部分少于研究任务，且 LLM agent 目前不擅长实时协调与委派[^16]。适合的是高度可并行化、信息量超出单上下文窗口、需要对接大量复杂工具的有价值任务[^16]。官方还给出按查询复杂度缩放投入的提示工程要点：简单事实查找只需 1 个 agent 配 3–10 次工具调用，直接对比可能需要 2–4 个 subagents 各 10–15 次调用，复杂研究可用 10 个以上 subagents；早期版本常见失败模式是对简单查询过度投入（如为简单查询派生 50 个 subagents）[^16]。

Claude Code 文档给出的 subagent 使用判据是上下文卫生：当某个附带任务会用搜索结果、日志或文件内容淹没主对话、而这些内容之后不会再被引用时，就用 subagent 在它自己的上下文里完成并只回传摘要[^17]。综述的对照判据是：单智能体在问题定义清晰、不需要其他 agent persona 或用户反馈时表现更好[^2]。

Cognition 对 Claude Code subagents 的观察（截至 2025 年 6 月）提供了一个折中读法：它虽会派生子任务，但从不与子任务 agent 并行工作，且子任务通常只负责回答问题而非写代码；其收益在于子 agent 的调研工作不必留在主 agent 历史中，从而在耗尽上下文前支持更长的 trace[^18]。这与 Anthropic 自己把 subagent 收益定位为「保留上下文」而非「并行加速」在部分场景下是一致的[^17]。

---

## 四、矛盾与不确定性

**（1）性能结论方向相反，且对算力口径敏感。** Anthropic 报告 +90.2%[^16]，MAST 报告增益往往很小且失败率 41%–86.7%[^6]，架构缩放研究给出 +80.8% 到 −70.0% 的区间并指出 capability-saturation[^31]，τ²-bench 双控实验给出加权均值 −28 分[^32]。这些结果并非简单互斥：它们分别落在研究型 breadth-first 检索、开放式 coding/math/general agent、可分解金融推理、面向用户的工具调用对话四类不同任务上，且只有后两者做了显式的算力匹配或标准化。但现有公开资料不足以给出一个跨任务的统一判据——架构缩放研究的预测模型 R² 仅 0.373（0.413），说明任务结构对架构收益的解释力仍有限[^31]。

**（2）辩论式的收益存在评测争议。** 辩论论文报告了稳定的正增益[^4]，MAD 系统性评测则称其常无法超越 CoT 与 Self-Consistency[^7]。两者对「基线」的定义与实验设置不一致，而后者正是把「基线对比薄弱、实验设置不一致」列为既有 MAD 研究的三类缺陷之一[^7]。此外辩论论文自述提示词设计影响辩论时长——鼓励模型更「固执」的提示带来更长辩论和更好最终解，并观察到 LLM 智能体整体较为 agreeable（可能源于指令微调/RLHF）[^4]；对立辩手方案则发现辩手使用不同 LLM 时，LLM judge 可能并不公正，会偏好与其自身 backbone 相同的一方[^5]。这意味着辩论收益对提示与模型配置高度敏感。

**（3）框架生态证据不完整，结论偏旧。** 本报告涉及 Microsoft 阵营时只能依赖已归档的 microsoft/autogen 仓库与迁移指引[^44]，**完全没有 MAF 仓库本身的 stars、最近提交、README 主张的协作模式与已知局限的证据**，因此第二节对 Microsoft 侧协作模式的描述停留在 AutoGen 时代，作为 2026-10 时点的框架清单不完整。langchain-ai/langgraph 与 crewAIInc/crewAI 的 README 正文因 403 rate limit 未读取[^12][^13]，其协作模式主张仅靠官方 docs 侧证[^34][^35][^36][^37][^38][^39][^40]，README 层面的主张与已知局限无直接证据。ag2ai/ag2 的元数据条目同样未读到正文[^14]，但另有 main 分支 README 抓取可用[^41]。

**（4）MetaGPT 活跃度判断存在事实风险。** 仓库元数据显示 stars 71010、最近 push 2026-01-21，比同批框架（2026-10-2x）早约 9 个月[^9]。这与 MetaGPT 作为头部框架的常识不符，可能是抓取字段口径问题（默认分支 push 时间 vs 全仓库活动）或仓库迁移（FoundationAgents 组织改名）。据此下「活跃度偏低」的结论不可靠，需复核。

**（5）部分关键数字为论文自述、未核对表格。** MetaGPT 的 85.9%/87.7% Pass@1、5.4% 绝对提升、100% 任务完成率均取自引言与贡献列表，未读到结果表格正文核对[^22]；ChatDev 的质量提升为定性表述，未读到给出具体百分比的结果表格[^23]；MAD 论文称 GPT-3.5-Turbo 配合 MAD 在 Common MT 上可超过 GPT-4，但以定性方式陈述，抓取到的正文片段未给出具体分数[^5]。GPTSwarm 仅读到 arXiv 摘要页，不含任何基准分数或与单智能体的定量对比[^28]。

**（6）框架厂商主张未经独立验证。** CrewAI README 含「rapidly becoming the standard for production-ready agentic automation」「over 100,000 developers certified」[^36]，官方文档含「the leading open-source framework」「the standard for enterprise-ready AI automation」「Cost-Efficient: Optimized to minimize token usage and API calls」[^39]，均为厂商自述，未经独立验证。相对地，AgentPrune 给出了可核对的成本数字（5.6 美元 vs 43.7 美元）[^26]，Anthropic 给出了 token 倍数（4×/15×）[^16]，这类量化主张的可验证性明显更高。

**（7）工程实现层面的多智能体缺陷仍在活跃修复中。** claude-agent-sdk-python 的 CHANGELOG 显示 0.2.160 修复 background subagents 之后 follow-up turns 失败的问题（子代理在本轮结果到达前刚结束时 stdin 被过早关闭，后续轮次报 'Stream closed' 且模型报告工具被拒绝），改为监听 CLI 的 session_state_changed 消息并保持 stdin 打开直到 CLI 报告 idle，设可配置等待上限默认 10 分钟[^47]；0.2.170 修复并行工具调用从 session messages 中丢失的问题（此前只跟随 forked transcript 的一个分支，导致部分并行调用与结果被丢弃）[^47]。此外官方文档明示 Python SDK 的 AgentDefinition 缺少 autoCompactWindow 与 omitClaudeMd 两个字段（仅 TypeScript SDK 支持）[^49]。这些说明多智能体的并行与后台执行在工程上仍非「开箱稳定」。

**（8）评测方法论本身不成熟。** AI Agents That Matter 指出当前 agent 评测只关注准确率而忽略其他指标，导致 SOTA agent 不必要地复杂且昂贵，并使社区对准确率增益的来源得出错误结论；许多 agent 基准缺乏充分的留出集甚至完全没有，导致 agent 通过走捷径过拟合基准而变得脆弱；评测实践缺乏标准化造成普遍的可复现性问题[^8]。LangChain 对 agent builders 的调查显示，阻碍把更多 agent 投入生产的最大限制因素（远超其它选项）是 performance quality[^21]。这意味着本报告引用的多数「多智能体优于单智能体」的百分比，其可比性与外推性都应打折。

**（9）一个未被现有证据回答的问题。** arXiv 检索显示「multi-agent single agent compute budget」主题在 2024–2026 有持续产出（72 条结果），其中 model slicing 工作在给定算力预算下用多个更小的异构模型替代一个大模型并学习跨模型协作，其动机是单模型推理的误差传播问题[^33]。但该方向与主流 MAS 框架（handoff、群聊、辩论）的关系，以及「算力匹配下多模型协作 vs 单大模型」的系统性结论，现有公开资料不足以确认。

## 参考来源

[^1]: Large Language Model based Multi-Agents: A Survey of Progress and Challenges (arXiv:2402.01680v2) — https://arxiv.org/html/2402.01680v2（arxiv, 2024-04-19）
[^2]: The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey (arXiv:2404.11584v1) — https://arxiv.org/html/2404.11584v1（arxiv, 2024-04-17）
[^4]: Improving Factuality and Reasoning in Language Models through Multiagent Debate — https://arxiv.org/html/2305.14325v1（arxiv, 2023-05-23）
[^5]: Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate — https://arxiv.org/html/2305.19118v4（arxiv, 2023-05-30）
[^6]: Why Do Multi-Agent LLM Systems Fail? (MAST, arXiv:2503.13657v3) — https://arxiv.org/html/2503.13657v3（arxiv, 2025-03-17 (v1); v3 2025-10-26）
[^7]: Stop Overvaluing Multi-Agent Debate — We Must Rethink Evaluation and Embrace Model Heterogeneity (arXiv:2502.08788v3) — https://arxiv.org/abs/2502.08788（arxiv, 2025-02-12 (v1); v3 2025-06-21）
[^8]: AI Agents That Matter (arXiv:2407.01502) — https://arxiv.org/abs/2407.01502（arxiv, 2024-07-01）
[^9]: FoundationAgents/MetaGPT — The Multi-Agent Framework — https://github.com/FoundationAgents/MetaGPT（github, 2026-01-21）
[^10]: openai/openai-agents-python — OpenAI Agents SDK — https://github.com/openai/openai-agents-python（github, 2026-10-22）
[^12]: langchain-ai/langgraph — Build resilient agents（仅元数据，正文未读取） — https://github.com/langchain-ai/langgraph（github, 2026-10-25）
[^13]: crewAIInc/crewAI — role-playing autonomous agents（仅元数据，正文未读取） — https://github.com/crewAIInc/crewAI（github, 2026-10-24）
[^14]: ag2ai/ag2 — AG2 (formerly AutoGen): The Open-Source AgentOS（仅元数据，正文未读取） — https://github.com/ag2ai/ag2（github, 2026-10-24）
[^16]: How we built our multi-agent research system (Anthropic Engineering) — https://www.anthropic.com/engineering/multi-agent-research-system（web）
[^17]: Create custom subagents — Claude Code Docs (Anthropic) — https://docs.anthropic.com/en/docs/claude-code/sub-agents（web）
[^18]: Don't Build Multi-Agents / Principles of Context Engineering (Cognition) — https://cognition.ai/blog/dont-build-multi-agents（web）
[^19]: Agent orchestration — OpenAI Agents SDK docs — https://openai.github.io/openai-agents-python/multi_agent/（web）
[^20]: Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks (Microsoft Research Blog) — https://www.microsoft.com/en-us/research/blog/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/（web）
[^21]: How to think about agent frameworks (LangChain / Harrison Chase) — https://blog.langchain.com/how-to-think-about-agent-frameworks/（web, 2025-04-20）
[^22]: MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework (arXiv:2308.00352v7) — https://arxiv.org/abs/2308.00352v7（arxiv, 2023-08-01 (v1); v7 revised 2024-11-01）
[^23]: ChatDev: Communicative Agents for Software Development (arXiv:2307.07924v5) — https://arxiv.org/html/2307.07924v5（arxiv, 2023-07-16 (v1); v5）
[^24]: Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks (arXiv:2411.04468v1) — https://arxiv.org/html/2411.04468v1（arxiv, 2024-11-07）
[^25]: AgentOrchestra: Orchestrating Multi-Agent Intelligence with the Tool-Environment-Agent (TEA) Protocol (arXiv:2506.12508v6) — https://arxiv.org/html/2506.12508v6（arxiv, 2025-06-14）
[^26]: Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems (AgentPrune) — https://arxiv.org/html/2410.02506v1（arxiv, 2024-10-03）
[^27]: A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration (DyLAN) — https://arxiv.org/html/2310.02170v2（arxiv, 2023-10-03）
[^28]: Language Agents as Optimizable Graphs (GPTSwarm) — https://arxiv.org/abs/2402.16823（arxiv, 2024-02-26）
[^29]: More Agents Is All You Need (Agent Forest), arXiv:2402.05120v2, TMLR — https://arxiv.org/html/2402.05120v2（arxiv, 2024-02-03 (v1); 2024-10-11 (v2)）
[^31]: Towards a Science of Scaling Agent Systems — https://arxiv.org/abs/2512.08296（arxiv, 2025-12-09 (v1); v2 2026-04-08）
[^32]: Does Multi-Agent Collaboration Pay When Agents Serve Users? Testing Agent Scaling in Dual-Control Settings — https://arxiv.org/abs/2610.27655（arxiv, 2026-10-20）
[^33]: arXiv search listing: multi-agent single agent compute budget (72 results, incl. Collaborative Learning and Reasoning over Model Slices, arXiv:2610.12927) — https://arxiv.org/search/?query=multi-agent+single+agent+compute+budget&searchtype=all&size=50（arxiv, 2026-10）
[^34]: langchain-ai/langgraph README (main branch) — https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md（github）
[^35]: LangGraph Overview — official docs (docs.langchain.com) — https://docs.langchain.com/oss/python/langgraph/overview（web）
[^36]: crewAIInc/crewAI — README (main branch) — https://raw.githubusercontent.com/crewAIInc/crewAI/main/README.md（github）
[^37]: CrewAI Docs — Processes — https://docs.crewai.com/en/concepts/processes（web）
[^38]: CrewAI Docs — Flows — https://docs.crewai.com/en/concepts/flows（web）
[^39]: CrewAI Docs — Introduction (What is CrewAI / When to Use Crews vs. Flows) — https://docs.crewai.com/en/introduction（web）
[^40]: CrewAI Docs — Collaboration (含 Troubleshooting Collaboration) — https://docs.crewai.com/en/concepts/collaboration（web）
[^41]: ag2ai/ag2 README (main branch) — https://raw.githubusercontent.com/ag2ai/ag2/main/README.md（github）
[^42]: AG2 Docs — Network Overview — https://docs.ag2.ai/docs/user-guide/network/overview/（web）
[^44]: microsoft/autogen README (main branch, archived) — https://raw.githubusercontent.com/microsoft/autogen/main/README.md（github）
[^45]: AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation (arXiv:2308.08155) — https://arxiv.org/abs/2308.08155（arxiv, 2023-08-16 (v1); revised 2023-10-03 (v2)）
[^47]: claude-agent-sdk-python CHANGELOG.md — 发布节奏与多智能体相关缺陷修复 — https://raw.githubusercontent.com/anthropics/claude-agent-sdk-python/main/CHANGELOG.md（github）
[^49]: Claude Agent SDK 官方文档 — Subagents（协作模式、收益与 Python SDK 字段缺失） — https://platform.claude.com/docs/en/agent-sdk/subagents（web）