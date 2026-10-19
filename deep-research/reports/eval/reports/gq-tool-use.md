# LLM 工具调用（Function Calling）的训练、对齐与评测

## 摘要

工具调用（function calling / tool calling）在官方术语体系中被定义为模型与外部系统交互、访问训练数据之外数据的机制，其交互单元由 Tools（以 JSON schema 定义的功能）、Tool calls（模型返回的特殊响应）与 Tool call outputs（应用侧执行后回传的输出）三层构成[^1][^29]。训练路径上，公开研究呈现三条主线：一是以 Toolformer 为代表的自监督自标注路线，让模型自行决定调用哪些 API、何时调用、传什么参数、如何把结果融入后续 token 预测[^2]；二是以 ToolLLM、ToolACE、APIGen、APIGen-MT、Magnet 为代表的数据合成与指令微调路线，通过自动化流水线生成大规模、可验证的工具调用轨迹[^7][^8][^50][^51][^52]；三是以 ToolRL、Nemotron-Research-Tool-N1、AgentGym-RL 为代表的强化学习路线，用规则化或原则化奖励替代对中间推理轨迹的模仿[^9][^10][^37]。对齐层面，DiaTool-DPO 把工具增强 LLM 的交互建模为 MDP 并做多轮 DPO，专门训练「追问澄清」与「拒绝工具调用」两类对话控制行为[^40][^41]。评测层面，Berkeley Function Calling Leaderboard（BFCL）从 V1 的 AST 匹配演进到 V4 的整体式 agentic 评估[^22][^23][^24]，τ-bench 系列则把「用户模拟 + 数据库状态比对 + pass^k 可靠性」引入工具-智能体-用户三方交互[^25][^48]。需要强调的是，本报告可用的证据中，BFCL V4 与 τ-bench 上主流模型的具体得分存在来源可靠性或口径不一致问题，详见「矛盾与不确定性」。

---

## 一、概念与交互协议：工具调用到底是什么

### 1.1 官方定义与术语分层

OpenAI 官方文档把 function calling（亦称 tool calling）定义为「让模型与外部系统交互、访问训练数据之外数据的机制」[^1][^29]。其术语体系包含三个核心概念[^1][^29]：

- **Tools**：以 JSON schema 定义、告知模型可用的功能。官方明确区分，function 是 tool 的一种特定类型，由 JSON schema 定义；另一类是 custom tools，输入输出为自由文本[^1][^29]。
- **Tool calls**：模型判断需要调用某工具时返回的特殊响应，其中包含参数[^1]。
- **Tool call outputs**：应用侧执行工具后返回给模型的输出，可为结构化 JSON 或纯文本，必须包含对特定模型工具调用的引用（示例中以 `call_id` 关联）[^1][^29]。

完整交互流程为：请求中附带工具列表 → 模型返回 tool call（含参数）→ 应用执行并回传 output → 连同原始 prompt、tool call、output 一起回传模型 → 模型生成最终文本响应[^1][^29]。当应用函数数量多或 schema 庞大时，可将 function calling 与 tool search 配合，延迟加载罕用工具、仅在模型需要时加载；文档注明仅 gpt-5.4 及之后模型支持 `tool_search`[^29]。

### 1.2 并行调用与执行语义

Anthropic 官方文档规定：默认情况下 Claude 可能在单次响应中调用多个工具，响应 `stop_reason` 为 `tool_use`，且单个 assistant turn 中可包含多个 `tool_use` block[^28]。API 不规定执行顺序，开发者可并发执行（`Promise.all`、`asyncio.gather`）、按出现顺序串行执行，或任意组合；官方建议按工具性质选择策略——独立、只读操作通常可安全并行以降低延迟，有副作用、共享状态或有顺序要求的工具更适合串行[^28]。对每个 `tool_use` block 必须返回一个 `tool_result`，全部放在下一条 user message 中，用 `tool_use_id` 匹配，且所有 `tool_result` block 必须位于该消息中任何文本内容之前；若某个调用未被执行（例如串行执行时前一个调用失败），仍必须为其返回 `tool_result` 并设置 `is_error: true` 及简要说明[^28]。computer use 与 browser use 工具规则更严格：当 Claude 在一个 turn 中返回多个成员工具调用（batch action）时，必须按出现顺序串行执行，并在第一个失败处停止[^28]。

### 1.3 标准化连接层：MCP

Model Context Protocol（MCP）是 Anthropic 于 2024-11-25 开源的开放标准，用于连接 AI 助手与数据所在系统（内容仓库、业务工具、开发环境），以单一协议替代碎片化的定制集成[^5]。官方类比为「MCP 之于 AI 应用，如同 USB-C 端口之于电子设备」[^4]。架构上，开发者可通过 MCP servers 暴露数据，或构建 MCP clients（AI 应用）连接这些服务器，支持安全双向连接[^5]。发布时同时提供 MCP 规范与 SDK、Claude Desktop 的本地 MCP server 支持，以及含 Google Drive、Slack、GitHub、Git、Postgres、Puppeteer 等预构建服务器的开源仓库[^5]。生态支持方包括 Claude、ChatGPT、Visual Studio Code、Cursor 等[^4]。参考实现仓库 `modelcontextprotocol/servers` 明确声明其仅存放 MCP steering group 维护的少量参考服务器，完整列表应查阅 MCP Registry，并警告这些服务器是用于演示 MCP 特性与 SDK 用法的参考实现，「not as production-ready」[^20]。规范仓库 `modelcontextprotocol/modelcontextprotocol` 包含 MCP specification、protocol schema 与官方文档，schema 以 TypeScript 定义并同时提供 JSON Schema 版本[^21]。

---

## 二、训练路径之一：自监督与早期工具学习范式

### 2.1 Toolformer：自监督自标注

Toolformer 是一个被训练来「决定调用哪些 API、何时调用、传什么参数、以及如何将结果融入后续 token 预测」的模型[^2]。其训练方式为自监督（self-supervised），每个 API 只需少量示例（a handful of demonstrations）[^2]。集成的工具包括计算器、问答系统、两个搜索引擎、翻译系统、日历[^2]。在多种下游任务上实现显著提升的零样本性能，常可与更大模型竞争，且不牺牲核心语言建模能力[^2]。

### 2.2 ReAct：推理与行动交织

ReAct 让 LLM 以交织（interleaved）方式同时生成推理轨迹（reasoning traces）和任务特定动作（actions）[^3]。推理轨迹帮助模型归纳、追踪、更新行动计划并处理异常；动作让模型与外部来源（知识库、环境）交互以获取额外信息[^3]。在 HotpotQA、Fever 上通过调用 Wikipedia API 缓解幻觉与错误传播；在 ALFWorld、WebShop 两个交互式决策基准上，分别以 34% 和 10% 的绝对成功率超越模仿学习与强化学习方法，且仅用 1–2 个 in-context 示例[^3]。

### 2.3 工具学习范式的系统化

Tool Learning with Foundation Models 提出「tool learning with foundation models」范式，将专用工具与基础模型结合，以提升问题求解的准确性、效率与自动化[^6]。该工作将现有研究归纳为两类：tool-augmented learning 与 tool-oriented learning；并提出通用工具学习框架——从理解用户指令出发，模型应学会将复杂任务分解为若干子任务、通过推理动态调整计划、并选择合适工具完成每个子任务[^6]。论文实验使用 18 个代表性工具评估当前基础模型的工具使用潜力[^6]。

---

## 三、训练路径之二：数据合成与指令微调

### 3.1 ToolLLM / ToolBench：真实 API 规模化

ToolLLM 指出当前指令微调主要关注基础语言任务而忽略工具使用领域，导致开源 LLM（如 LLaMA）工具使用能力显著受限[^7]。其构建的 ToolBench 指令微调数据集从 RapidAPI Hub 收集 16,464 个真实 RESTful API，覆盖 49 个类别，含单工具与多工具场景[^7]。数据构建分三阶段：API 收集、指令生成（由 ChatGPT 生成）、解路径标注（用 ChatGPT 搜索有效 API 调用链）[^7]。方法上提出基于深度优先搜索的决策树算法以增强 LLM 推理能力，并开发自动评估器 ToolEval；基于 ToolBench 微调 LLaMA 得到 ToolLLaMA，并配备神经 API 检索器，在未见 API 上展现零样本泛化能力[^7]。

仓库层面，`OpenBMB/ToolBench` 的 README 徽章声明数据集规模为 Tool_Num 3451、API_Num 16464、Current_Dataset_Size 126K、Total_API_Call 469K、Average_Reasoning_Traces 4.0，并声明已发布 Tool_LLaMA 模型；仓库 5751 stars、492 forks、168 open issues，Apache-2.0 许可，创建于 2023-05-28，最近一次 push 为 2025-05-21[^14]。

### 3.2 ToolACE：自演化合成 + 双层验证

ToolACE 提出一个自动化的 agentic 流水线，用于生成准确、复杂、多样的工具学习数据[^8]。它采用自演化合成（self-evolution synthesis）流程构建包含 26,507 个多样化 API 的 API 池；对话数据由多个 agent 交互生成，并由形式化的思考过程（formalized thinking process）引导[^8]。为保证数据准确性，实现规则与模型结合的双层验证系统（dual-layer verification）[^8]。仅 8B 参数的模型在合成数据上训练后，在 Berkeley Function-Calling Leaderboard 上达到 SOTA 性能，可与最新 GPT-4 模型媲美[^8]。

### 3.3 APIGen / APIGen-MT：可验证数据与多轮模拟

APIGen 收集了 3,673 个可执行 API，覆盖 21 个类别，用于可扩展、结构化地生成函数调用数据集[^50]。每条数据经三阶段分层验证：格式检查、实际函数执行、语义验证[^50]。论文发布了一个包含 60,000 条高质量条目的数据集（Salesforce/xlam-function-calling-60k）[^50]。仅 7B 参数的模型在 Berkeley Function-Calling Benchmark 上达到 SOTA，超过多个 GPT-4 模型；1B 模型超过 GPT-3.5-Turbo 和 Claude-3 Haiku[^50]。

APIGen-MT 是两阶段框架：第一阶段由 agentic pipeline 生成带 ground-truth 动作的任务蓝图（task blueprints），借助 LLM 评审委员会与迭代反馈；第二阶段通过模拟人-智能体交互将蓝图转化为完整交互轨迹[^51]。基于此训练了 xLAM-2-fc-r 系列模型，参数规模从 1B 到 70B；论文声称模型在 τ-bench 和 BFCL 基准上超过 GPT-4o 和 Claude 3.5 等前沿模型，且更小的模型在多轮设置中超过更大的对应模型，并在多次试验中保持更高一致性；同时开源了 5K 条合成数据轨迹（Salesforce/APIGen-MT-5k）及训练好的模型[^51]。

`SalesforceAIResearch/xLAM` 仓库的 README News 声明 xLAM-2-fc-r 在最新 BFCL Leaderboard 上取得 Top-1 性能（2025-04-15 条目）；声明 2024-08 发布完整 xLAM 家族（Large Action Models），在 Berkeley Function-Calling Leaderboard 上排名 #1 与 #6；声明 xLAM-1b-fc-r 与 xLAM-7b-fc-r 在 BFCL 上分别排名 #3 与 #25，「outperforming many significantly larger models」；声明数据集 xlam-function-calling-60k 在 HuggingFace 上位列 Top-3 trending datasets（截至 2024-07-04，173,670 个数据集中）；并开源 APIGen-MT-5k 数据集与 ActionStudio 训练框架[^17]。

### 3.4 Magnet：图翻译式多轮轨迹合成

Magnet 是 2025 年提出的多轮工具调用训练轨迹合成框架，基于从函数签名路径（function signature path）到查询与可执行函数调用序列的自动迭代翻译[^52]。它用图（graph）建模多轮场景中的复杂函数交互，并设计新的节点操作构建可靠的签名路径[^52]。受 context distillation 启发，用教师模型生成正/负轨迹：上下文中提供参考函数调用序列作为正提示，对比性的错误函数调用作为负提示[^52]。14B 模型 Magnet-14B-mDPO 在 BFCL-v3 上得 68.01，在 ToolQuery 上得 73.30，大幅超过教师模型 Gemini-1.5-pro-002[^52]。

### 3.5 ToolGen：把工具变成 token

ToolGen 将工具知识直接集成进 LLM 参数，把每个工具表示为一个唯一 token，使 LLM 能像下一 token 预测一样生成工具调用与参数，将工具调用与语言生成无缝融合[^11]。它无需额外检索步骤即可访问和使用大量工具，提升性能与可扩展性；在超过 47,000 个工具上的实验显示，ToolGen 在工具检索与自主任务完成两方面均取得更优结果[^11]。作者指出 ToolGen 支持端到端工具学习，并可与 chain-of-thought、强化学习等技术结合[^11]。

---

## 四、训练路径之三：强化学习与偏好优化

### 4.1 ToolRL：奖励设计是核心

ToolRL 指出 SFT 难以泛化到不熟悉或复杂的工具使用场景，而 RL（尤其 R1 类模型）展现推理与泛化潜力[^9]。该工作提出首个针对 RL 范式中工具选择与应用任务奖励设计的系统性研究，分析奖励的类型、尺度、粒度与时间动态；提出面向工具使用任务的原则化奖励设计，并用 Group Relative Policy Optimization（GRPO）训练 LLM[^9]。在多个基准上评测，报告相对基座模型提升 17%、相对 SFT 模型提升 15%，训练稳健、可扩展且稳定；代码已开源[^9]。

### 4.2 Nemotron-Research-Tool-N1：纯 RL 挑战 SFT-then-RL

Nemotron-Research-Tool-N1 指出以往提升工具调用能力主要依赖用更强模型蒸馏轨迹做 SFT，常导致模仿式推理、限制泛化[^10]。该工作提出 Nemotron-Research-Tool-N1 系列工具调用推理模型，采用基于规则的强化学习；训练使用二元 RL 奖励，仅评估工具调用的格式有效性与功能正确性，不对中间推理轨迹施加监督[^10]。在多个主要基准上，Tool-N1-7B/14B 明显优于 GPT-4o[^10]。值得注意的负面结论是：该工作使用 5,518 条蒸馏推理轨迹，系统比较 SFT、RL 与 SFT-then-RL 流水线，发现广泛采用的 SFT-then-RL 范式不一定优于纯 RL[^10]。

### 4.3 AgentGym / AgentEvol：跨环境自演化

AgentGym 是一个面向 LLM-based agent 的框架，覆盖多样环境下的评测与演化（训练）[^12]。论文提出 AgentGym 框架，包含多样环境与任务，支持 broad、real-time、uni-format、concurrent 的 agent 探索；并提出 AgentEvol 方法，用于研究 agent 在未见过的任务与环境上的自演化（self-evolution）潜力[^34]。论文识别出构建通用 LLM agent 的三要素（trinity）：1) 多样环境；2) 轨迹集（trajectory set）以赋予基础能力与先验知识；3) 有效且可扩展的演化方法[^34]。自演化流程为两阶段：agent 先根据人类监督做 behavioral cloning（行为克隆），随后跨环境与任务进行探索与学习以自我演化[^35]。论文将现有方法分为两类：一类依赖人类监督、逐步模仿专家轨迹（behavioral cloning），需熟练标注者与大量资金、难以扩展；另一类让 agent 在孤立环境中探索学习，得到泛化有限的专家型 agent[^35]。

AgentGym 包含 14 个环境，覆盖 web navigating、text games、house-holding tasks、digital games、embodied tasks、tool-using 与 programming 七大类；环境与任务采用统一格式（ReAct format），支持实时反馈与并发[^36][^37]。项目页环境表列出各环境的轨迹数（Traj）与评测数（Eval）：WebShop 3930/200、WebArena 0/20、MAZE 215/25、Wordle 955/25、ALFWorld 2420/200、SciWorld 2120/200、BabyAI 810/90、TextCraft 374/100、Weather 311/20（表格在 Weather 行后被截断）[^36]。套件包含高质量轨迹集 AgentTraj 与 benchmark suite AgentEval[^36]。模型 AgentEvol-7B 基于 Llama-2-Chat-7B，使用 AgentEvol 算法训练：先在 AgentTraj 集上做 behavioral cloning，随后从更广的指令集进行探索与学习；模型卡声称演化后其性能在多项任务上超过 SOTA 模型[^38]。AgentTraj-L 数据集在 HF 上显示为 14.5k 行（train split），字段含 conversations、item_id，样本为多轮对话格式[^39]。2025-09-10 发布 AgentGym-RL 框架与论文 AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning（arXiv:2509.08755）[^37]。

### 4.4 DiaTool-DPO：多轮偏好优化与对话控制

DiaTool-DPO 由 Kakao Corp. 团队提出，v1 提交于 2025-04-02，v2 修订于 2025-07-14，被 SIGDIAL 2025 接收[^40]。方法将 TA-LLM 交互建模为马尔可夫决策过程（MDP），定义 5 个不同的对话状态，并依据状态转移轨迹把用户 query 分为 3 类[^40]。偏好数据自动构造：通过将用户 query 与不匹配的对话轨迹配对，自动生成被拒绝（rejected）轨迹，形成成对的正确/错误对话流数据集[^40][^41]。论文还引入一个专门用于对话控制的 objective loss[^40]。

论文的动机是：TA-LLM 必须能控制对话流——决定 (1) 向用户追问澄清信息、(2) 发起工具调用、还是 (3) 在没有合适工具时拒绝工具调用；若不能生成追问或拒绝工具调用，会导致用幻觉信息发起工具调用的风险[^41]。论文明确其动机之一是仅靠 SFT 学习对话技能存在局限[^41]。摘要报告的核心数字：信息收集（information gathering）达 94.8%，工具调用拒绝（tool call rejection）达 91%，接近 GPT-4o 水平；相对基线分别提升至 44% 和 9.6%（即基线为 44% 与 9.6%）[^40]。论文声称方法无需额外专家示范或人工标注即可处理多样化真实场景[^40]。作者自述贡献为两点：为工具增强 LLM 自动生成 DPO 数据集；提出一个专门用于训练工具增强 LLM 多轮 DPO 的损失函数；并指出近期工具增强 LLM 工作多聚焦于训练数据生成或基准构建，针对训练目标（training objective）的工作不多[^42]。

---

## 五、评测体系（一）：Berkeley Function Calling Leaderboard

### 5.1 定位与版本演进

BFCL 自我定位为「the first comprehensive and executable function call evaluation dedicated to assessing Large Language Models' (LLMs) ability to invoke functions」[^16][^43]。官方 V4 榜单页将其定位为「From Tool Use to Agentic Evaluation of Large Language Models」，评估 LLM 准确调用函数（工具）的能力，使用真实世界数据并定期更新[^22]。版本演进为：V1 引入 AST 作为评估指标；V2 引入企业级与开源社区贡献的函数；V3 引入多轮交互（multi-turn interactions）；V4 引入整体式 agentic 评估（holistic agentic evaluation）[^22]。README 的表述为 v1（Simple/Parallel/Multiple Function Call eval with AST）、v2（Enterprise 与 OSS-contributed Live Data）、v3（Multi-Turn & Multi-Step）、V4（Part 1 Agentic Web Search、Part 2 Agentic Memory Management、Part 3 Agentic Format Sensitivity）[^16][^43]。

时间线上，BFCL V2 于 2024-08-19 发布，引入企业贡献数据，处理 bias 与数据污染问题，聚焦动态真实场景；BFCL V3 于 2024-09-19 发布，引入多轮与多步函数调用评估[^23]。

### 5.2 V1 方法论

BFCL 自称是「首个对 LLM 调用函数与工具能力进行全面评估」的基准[^23]。数据集由 2k 个 question-function-answer 三元组构成，覆盖多语言（python、java、javascript、restAPI）、多应用领域与复杂用例[^23]。覆盖的函数调用形式包括 parallel（一个函数输入、多次函数调用输出）与 multiple（多个函数输入、一个函数输出）[^23]。数据集具体构成：100 条 Java、50 条 JavaScript、70 条 REST API、100 条 SQL、1,680 条 Python，涵盖 simple、parallel、multiple、executable 场景以及 function relevance detection[^23]。其中 function relevance detection 指：当提供的函数不适合回答用户问题时，评估模型是否会拒绝选择任何函数（此时给出 Error Message）[^23]。基准会实际执行这些函数来评估模型，榜单同时包含各模型的 cost 与 latency[^23]。

### 5.3 V3 方法论：从 AST 匹配到状态验证

BFCL V3 引入 multi-turn 与 multi-step function call（工具使用）新类别，发布日期 2024-09-19，最后更新 2024-12-10[^44]。V3 仍包含 BFCL V1 的 Expert Curated (Non-live) 数据集与 BFCL V2 的 User Contributed (Live) 数据集[^44]。评测方式的改变是：不再仅用 AST 拆解参数对并匹配候选答案，而是验证模型运行函数后 API 系统（如文件系统、预订系统）的实际状态[^44]。

### 5.4 V4 方法论：整体式 agentic 评估

BFCL V4 将 function calling 视为 Agent 的构建模块，提出整体式 agentic 评估，包含三部分：web search（Part 1）、memory（Part 2）、format sensitivity（Part 3）[^24]。BFCL 全程依赖 AST（Abstract Syntax Tree）或状态转移（state-transition）验证，以保证评估的确定性与最小波动[^24]。V1 为专家策划的单轮 simple、parallel、multiple 函数调用；V2 引入社区爱好者与企业贡献的函数；V3 引入多轮多步函数调用，允许模型与用户来回交互、提出澄清问题并细化方案[^24]。

V4 总分（Overall Score）为百分比加权平均，共 5088 条计分条目，另有 5218 条非计分条目（Format Sensitivity，不影响分数）[^24]。权重构成为：Agentic 占 40%（665 条，未加权平均）、Multi-Turn 占 30%（800 条，未加权平均）、Live 占 10%（1351 条，加权平均）、Non-Live 占 10%（1150 条，未加权平均）、Hallucination Measurement 占 10%（1122 条，未加权平均）[^24]。Agentic 子类包含：Relevance（18 条）、Web Search（200 条，未加权平均）、Memory（465 条，未加权平均）[^24]。Web Search 使用 DuckDuckGo Search API，任务为多跳问题（multihop questions）[^24]。Multi-Turn 子类包含 Base Case（200 条）与 Augmented Cases（600 条，独立类别）[^24]。Non-Live 子类包含 Simple（258 条）、Multiple（1053 条）、Parallel（16 条）、Parallel Multiple（24 条）[^24]。

榜单页补充规定：区分 FC（模型原生支持 function/tool calling）与 Prompt（用模型普通文本生成能力绕过 function calling）两种模式；同时报告 Cost（整个基准的预估成本，美元）与 Latency（秒）；Overall Accuracy 定义为所有子类别的未加权平均（unweighted average）；Format sensitivity 测试用例仅对 prompt（非 FC）模型支持[^22]。榜单页标注 Last Updated: 2026-04-12，模型使用 commit f7cf735 评估，可通过 `pip install bfcl-eval==2025.12.17` 复现[^22]。官方引用为 ICML 2025 论文[^22]。

### 5.5 工程实现

评测包以 PyPI 名 `bfcl-eval` 发布，README 提醒勿与无关的 `bfcl` 项目混淆[^16]。评测支持自托管模型后端 vllm 与 sglang，README 称 sglang 在多轮场景下比 vllm 快得多但仅支持 SM 80+ 较新 GPU，T4/V100 应使用 vllm[^16]。官方评估代码库的 `model_config.py` 导入了 `LingAPIHandler`、`NovaHandler`、`QwenAPIHandler`、`QwenAgentThinkHandler`、`QwenAgentNoThinkHandler`、`Granite4FCHandler`、`Granite3FCHandler` 等，表明 Ling、Amazon Nova、Qwen、IBM Granite 4 系列被官方 BFCL 评估框架支持[^45]。README 指向实时榜单地址，但 README 正文本身不含任何模型分数表[^43]。

---

## 六、评测体系（二）：τ-bench 系列与工具-智能体-用户交互

### 6.1 τ-bench：用户模拟 + 数据库状态比对 + pass^k

τ-bench 模拟用户（由语言模型模拟）与语言 Agent 之间的动态对话，Agent 被提供领域特定的 API 工具与策略指南（policy guidelines）[^25]。评估方式为「高效且忠实」的过程：将对话结束时的数据库状态（database state）与标注的目标状态（annotated goal state）进行对比[^25]。论文提出新指标 pass^k，用于评估 Agent 行为在多次试验（multiple trials）中的可靠性[^25]。实验显示即使最先进的 function calling Agent（如 gpt-4o）在任务上的成功率也低于 50%；Agent 行为相当不一致：在 retail 领域 pass^8 低于 25%[^25]。论文指出需要能提升 Agent 一致行动与可靠遵守规则能力的方法[^25]。τ-bench 全称为 Tool-Agent-User Interaction Benchmark，用于衡量 Agent 与（模拟）人类用户及程序化 API 交互的能力[^26]。

论文提出部署真实世界系统对 Agent 的三项关键要求：(1) 在长时程中与人类及程序化 API 无缝交互，逐步收集信息并澄清意图；(2) 准确遵守任务或领域特定的复杂策略与规则；(3) 在大规模、数百万次交互中保持一致性与可靠性[^26]。论文以航空订票 Agent 为例：用户想改签到不同目的地机场时，Agent 需通过与用户交互收集信息、依据指南检查航空公司政策、使用复杂预订 API 查找新航班并（若可能）重新预订[^26]。论文指出既有语言 Agent 基准常采用简化的指令遵循设定：Agent 在信息全部预先给出的情况下自主与环境（网页、代码终端或 API）交互，没有人类在环交互，也无需查阅领域特定指南[^26]。τ-bench 包含 τ-airline 等领域；示例轨迹中 Agent 需依据领域政策拒绝用户请求（改签 basic economy 航班）并提出新方案（取消并重新预订），考验 Agent 对复杂数据库、规则与用户意图的长上下文零样本推理[^26]。

### 6.2 τ²-bench：双控制环境

τ²-bench 由 Sierra 提出，作者 Victor Barres、Honghua Dong、Soham Ray、Xujie Si、Karthik Narasimhan，2025-06-09 提交[^48]。它新增 Telecom 双控制（dual-control）域，建模为 Dec-POMDP，agent 与用户都可使用工具作用于共享动态环境[^48]。论文指出既有 benchmark（含 τ-bench 的 retail/airline）为单控制环境，用户仅被动提供信息；τ²-bench 引入用户主动参与修改世界状态[^48]。论文提出组合式任务生成器，从原子组件程序化生成多样、可验证的任务，保证领域覆盖与可控复杂度[^48]。实验显示：agent 从 no-user 切换到 dual-control 时性能显著下降，凸显引导用户的难度；论文提供细粒度消融，区分推理错误与沟通/协调错误[^48]。τ-bench 的 retail 与 airline 域测试 agent 解决约束满足任务的能力，约束来自领域策略与用户认知状态（信念、目标、偏好）的组合；τ²-bench 中用户由另一个 AI agent 模拟，给定基于场景的指令和一组作用于其自身数据库的工具[^49]。

### 6.3 τ³-bench 与实时榜单

官方代码与数据发布仓库 `sierra-research/tau2-bench` 为 τ-bench / τ²-bench / τ³-bench 的发布地，MIT 许可，Python 3.12+，stars 2235、forks 578，最近 push 2026-10-13；README 关联论文 arXiv:2506.07982，并指向实时榜单 https://taubench.com[^46]。README 说明 τ³-bench 新增 Knowledge 域（banking_knowledge，基于 RAG 的知识检索客服域）与 Voice 全双工（音频原生）评测，支持 OpenAI、Gemini、xAI 等 realtime 提供方[^46]。README 记录 2026 年 7 月 v1.0.1 评分更新：修复 banking_knowledge 任务错误，受影响榜单提交已重新评分；其他领域不受影响[^46]。README 说明 τ³-bench 基于 SABER（arXiv:2512.07850, Cuadron et al., 2025）的分析，在 airline、retail、banking 域修复 75+ 任务（删除错误期望动作、澄清歧义指令、修正不可行约束、补充缺失回退行为）[^46]。README 提示：评测 agent（非训练）时应使用 base 任务划分，以匹配原始 τ-bench 结构，此为默认设置[^46]。

榜单演进时间线为：τ-bench（2024-06，原始 benchmark，含 Retail、Airline，使用 pass^k 可靠性指标）；τ²-bench（2025-06，新增 Telecom 双控制域）；τ³-bench（2026-03，新增 voice 与 knowledge）[^47]。首页仅展示各榜单 Pass^1 的前三名摘要，未在首页展开分领域 pass^k 明细[^47]。

---

## 七、评测体系（三）：失败模式诊断与跨环境评测

### 7.1 ToolFailBench：分解「工具使用失败」

ToolFailBench 是一个诊断性基准，覆盖金融、医学、法律、网络安全、房地产五个领域的 1,000 个任务，用于测量 LLM agent 的工具使用失败[^33]。论文指出聚合基准分数会掩盖工具使用失败的具体位置：从不调用所需工具的模型与调用了工具但忽略结果的模型，在最终任务准确率上可能看起来相似[^33]。基准设计包含 tool-required 任务（返回值是模型无法猜到的，迫使模型信任工具）与 control 任务（挂载相同工具但应直接作答）[^33]。论文用规则分类器加两个 LLM 评判器多数投票，为每条轨迹打上四类标签：Tool-Skip、Result-Ignore、Output-Fabrication、Unnecessary-Tool-Use[^33]。在 19 个头部模型上，最佳模型达到 86.33% 的 Clean Tool-Use Rate，论文据此认为忠实的工具使用尚未饱和[^33]。论文发现聚合分数相近的模型失败方式不同：多数模型在无工具 control 任务上保持克制，而 Llama-3.1 系列表现出 Always-Call 模式；同参数规模下 Llama-3.1-70B 与 Qwen2.5-72B 在 control 任务准确率上相差 89 个百分点[^33]。论文结论主张工具使用评测不应只衡量 agent 是否调用工具，还应衡量其是否正确使用工具输出、以及在不需要工具时是否避免调用[^33]。

### 7.2 AgentBench 与 AgentGym 的评测侧

`THUDM/AgentBench` 仓库 README 声明 2025-10-10 引入基于 AgentRL 的 AgentBench FC (Function Calling)；仓库 3772 stars、290 forks、80 open issues，Apache-2.0 许可，创建于 2023-07-28，最近一次 push 为 2026-02-08[^19]。AgentGym 侧的 benchmark suite 为 AgentEval，与轨迹集 AgentTraj 一同发布[^36][^37]。

---

## 八、工程生态与安全面

### 8.1 客户端与训练框架生态

`QwenLM/Qwen-Agent` 为 Qwen 团队（Alibaba Group）维护的 agent 框架，17177 stars、1743 forks、567 open issues，Apache-2.0 许可，创建于 2023-09-22，最近一次 push 为 2026-03-04[^18]。`ShishirPatil/gorilla` 仓库标题为「Gorilla: Large Language Model Connected with Massive APIs」，关联论文 arXiv 2305.15334，topics 包含 api、api-documentation、openai-functions、llm，13051 stars、1413 forks、291 open issues，创建于 2023-05-19，最近一次 push 为 2026-04-13[^15]。

### 8.2 上下文成本：工具定义的隐性开销

Anthropic 于 2025-11-24 发布三项 beta 特性，使 Claude 能动态发现、学习与执行工具：Tool Search Tool、Programmatic Tool Calling、Tool Use Examples[^32]。Anthropic 指出工具定义会大量消耗上下文：五 server 示例中 GitHub 35 个工具约 26K tokens、Slack 11 个约 21K tokens、Sentry 5 个约 3K tokens、Grafana 5 个约 3K tokens、Splunk 2 个约 2K tokens，合计 58 个工具约 55K tokens；Anthropic 内部曾见到工具定义消耗 134K tokens[^32]。Anthropic 明确表示最常见的失败是「选错工具」与「参数不正确」，在工具名称相似时尤其严重（如 `notification-send-user` 与 `notification-send-channel`）[^32]。Tool Search Tool 按需发现工具，官方称相比传统方式保留 191,300 tokens 上下文（传统方式为 122,800）；传统方式下 50+ MCP 工具定义约 72K tokens，加上对话历史与系统提示后约 77K tokens 在任何工作开始前即被占用[^32]。Programmatic Tool Calling 允许 Claude 在代码执行环境中调用工具，以减少对模型上下文窗口的占用；自然语言工具调用每次调用都需要一次完整推理，中间结果无论是否有用都会堆积在上下文中[^32]。Tool Use Examples 旨在提供演示工具有效用法的通用标准，因为 JSON schema 只能定义结构上合法的内容，无法表达使用模式（何时包含可选参数、哪些参数组合合理、API 期望的约定）[^32]。Anthropic 称 Claude for Excel 使用 Program
Programmatic Tool Calling 读取和修改数千行电子表格而不使模型上下文窗口过载[^32]。

### 8.3 安全面：MCP 引入的新攻击面

MCP 官方规范设有专门的安全最佳实践章节，用于补充 MCP Authorization 规范，识别 MCP 实现特有的安全风险、攻击向量与最佳实践，目标读者为 MCP 授权流程实现者、MCP server 运营者与评估 MCP 系统的安全人员[^30]。规范明确描述 Confused Deputy Problem：攻击者可利用连接第三方 API 的 MCP proxy server 制造「混淆代理」漏洞，使恶意客户端在未获正当用户同意的情况下取得授权码[^30]。该攻击成立需同时满足四个条件：MCP proxy server 对第三方授权服务器使用静态 client ID；proxy 允许 MCP 客户端动态注册（各自获得 client_id）；第三方授权服务器在首次授权后设置 consent cookie；proxy 未在转发至第三方授权前实施按客户端区分的同意流程[^30]。规范给出完整攻击链：用户先正常认证使第三方授权服务器在 user agent 上设置 consent cookie，攻击者随后发送含恶意 redirect_uri 与新动态注册 client_id 的授权请求链接，浏览器携带既有 consent cookie 导致授权服务器跳过同意页面，授权码被重定向至攻击者服务器，攻击者据此换取 MCP server 的访问令牌[^30]。

OWASP 指出 MCP 由 Anthropic 于 2024 年 11 月提出，用于标准化 AI 应用（LLM 客户端）与外部工具、数据源、服务的连接[^31]。OWASP 认为 MCP 引入了全新的攻击面：AI agent 基于自然语言动态执行工具并访问敏感系统；与传统 API 由开发者控制每次调用不同，MCP 让 LLM 自行决定调用哪些工具、何时调用、用什么参数[^31]。OWASP 将 MCP 特有风险概括为 prompt injection、供应链攻击与 confused deputy 问题的组合[^31]。其给出的 MCP 架构为：User ↔ MCP Host（AI 应用，如 Claude Desktop、Cursor、IDE 插件）↔ MCP Client ↔ MCP Server(s) ↔ Tools/Data/APIs；传输方式为 stdio（本地）或 Streamable HTTP（远程），旧的 HTTP+SSE 传输已废弃[^31]。OWASP 强调 LLM 会在上下文中看到所有已连接 server 的全部工具描述，这是理解跨 server 攻击的关键[^31]。OWASP 列出的关键风险包括：Tool Poisoning（在工具描述、参数 schema 或返回值中隐藏恶意指令以操纵 LLM 行为）、Rug Pull Attacks（server 在用户初次批准后更改工具定义，使受信任工具转为恶意）、Tool Shadowing / Cross-Origin Escalation（恶意 server 的工具描述操纵 agent 行为）[^31]。

---

## 九、矛盾与不确定性

### 9.1 BFCL V4 上主流模型的具体得分与排名：无可靠证据

唯一提供 BFCL-V4 分数的来源是第三方聚合页 llm-stats.com，其列出 21 个模型的分数（Score 为 0-1 区间）[^27]。排名前列为：Qwen3.7 Max（Alibaba Cloud / Qwen Team）0.750 居首；Ling 3.0 Flash（InclusionAI）0.730 第二；Qwen3.5-397B-A17B 与 Qwen3.7-Plus 并列 0.729[^27]。其他条目包括 Qwen3.5-122B-A10B 0.722、Qwen3.5-27B 0.685、Qwen3 Max Thinking 0.677、Qwen3.5-35B-A3B 0.673、Qwen3.5-9B 0.661、Nova 2 Pro（Amazon）0.616、IBM Granite 4.2 30B 0.614、Nova 2 Lite 0.603、Nova 2 Omni 0.583、Seed 1.8（ByteDance）0.572、LFM2.5-2.6B（Liquid AI）0.569、IBM Granite 4.2 3B/8B 0.524、Qwen3.5-4B 0.503、Qwen3.5-2B 0.436、LFM2.5-VL-3B 0.325、Qwen3.5-0.8B 0.253[^27]。

但该来源存在多重可靠性问题：其模型命名（Qwen3.7 Max、Ling 3.0、Nova 2、Seed 1.8、LFM2.5 等）无法与官方 BFCL 榜单交叉验证，且完全不含 GPT/Claude/Gemini/Llama 条目；该页自身标注 Dataset 与 Code 均为「Soon」（尚未提供），并提示「Notice missing or incorrect data? Let us know」[^27]。官方榜单页（gorilla.cs.berkeley.edu/leaderboard.html）与 BFCL README 均只给方法论与版本演进，不含分数表[^22][^43]。因此，本报告无法给出 BFCL V4 上主流模型可靠的具体得分与排名，**现有公开资料不足以确认**。

需要区分的是，若干论文与仓库 README 报告了各自模型在 BFCL 上的成绩：ToolACE 称其 8B 模型在 BFCL 上达到 SOTA、可与最新 GPT-4 媲美[^8]；APIGen 称其 7B 模型在 Berkeley Function-Calling Benchmark 上达到 SOTA、超过多个 GPT-4 模型，1B 模型超过 GPT-3.5-Turbo 与 Claude-3 Haiku[^50]；xLAM README 称 xLAM-2-fc-r 在最新 BFCL Leaderboard 上取得 Top-1（2025-04-15 条目），xLAM-1b-fc-r 与 xLAM-7b-fc-r 分别排名 #3 与 #25[^17]；Magnet 报告 Magnet-14B-mDPO 在 BFCL-v3 上得 68.01[^52]。这些均为作者自报口径，且未标注对应的 BFCL 版本与评测 commit，不能直接横向比较，也不构成对「当前 BFCL V4 主流模型排名」的回答。

### 9.2 τ-bench 上主流模型的成功率水平：口径冲突未解决

τ-bench 原始论文（arXiv:2406.12045，2024-06）称最先进 function calling agent（如 gpt-4o）成功率低于 50%、retail 域 pass^8 低于 25%[^25]。而 taubench.com 榜单显示 τ²-bench Text 上 Qwen3.5-397B-A17B 87.9%、Gemini 3.0 Pro 85.4%、Claude Opus 4.5 85.3%（首页展示指标为 Pass^1）[^47]。两者相差近 40 个百分点。

该差异至少可能来自四个未澄清的因素：(1) 基准版本不同——前者为原始 τ-bench（Retail、Airline），后者为 τ²-bench Text（Retail、Airline、Telecom）[^47]；(2) 指标不同——前者报告的是任务成功率与 pass^8，后者首页展示的是 Pass^1[^25][^47]；(3) 时间跨度约两年，模型能力本身有提升；(4) τ³-bench 已基于 SABER 分析在 airline、retail、banking 域修复 75+ 任务（删除错误期望动作、澄清歧义指令、修正不可行约束、补充缺失回退行为），并记录 2026 年 7 月 v1.0.1 对 banking_knowledge 的评分修复，说明榜单口径本身在演进[^46]。榜单首页仅展示各榜单 Pass^1 的前三名摘要，未展开分领域 pass^k 明细，因此**现有公开资料不足以确认**两个数字是否可直接对比。本报告在引用时严格区分来源与口径，不做跨口径的合并陈述。

### 9.3 训练范式的未决问题

公开证据在「SFT 与 RL 的相对优劣」上存在明确张力。Nemotron-Research-Tool-N1 使用 5,518 条蒸馏推理轨迹系统比较 SFT、RL 与 SFT-then-RL，发现广泛采用的 SFT-then-RL 范式不一定优于纯 RL[^10]；ToolRL 则报告相对基座模型提升 17%、相对 SFT 模型提升 15%[^9]。两者并不直接矛盾（前者比较的是流水线组合，后者比较的是 RL 与 SFT 基线），但共同指向一个尚未收敛的问题：在工具调用任务上，监督信号应当施加在中间推理轨迹上，还是仅施加在最终调用的格式有效性与功能正确性上[^9][^10]。DiaTool-DPO 的作者亦指出，近期工具增强 LLM 工作多聚焦于训练数据生成或基准构建，针对训练目标（training objective）本身的工作不多[^42]。

### 9.4 证据覆盖的局限

本报告未覆盖的证据缺口包括：BFCL V4 官方榜单页与 README 均不含分数表，官方分数只能通过实时榜单地址获取[^43]；AgentGym 项目页的环境表在 Weather 行后被截断，无法给出完整的 14 环境轨迹数与评测数[^36]；AgentGym 论文声称演化后的 agent 可达到与 SOTA 模型相当的结果（原文「comparable to SOTA models」），AgentEvol-7B 模型卡则声称「outperforms SOTA models on many tasks」，两处表述强度不一致，且均未在可用证据中给出具体对比数字[^34][^38]。此外，关于「主流模型在工具使用上的评测怎么做」这一问题，本报告能给出的可靠结论是方法论层面的（BFCL 的 AST/状态转移验证与 V4 加权构成、τ-bench 系列的数据库状态比对与 pass^k、ToolFailBench 的四类失败标签），而非具体模型的横向排名。

## 参考来源

[^1]: Function calling - OpenAI Platform Documentation — https://platform.openai.com/docs/guides/function-calling（web）
[^2]: Toolformer: Language Models Can Teach Themselves to Use Tools — https://arxiv.org/abs/2302.04761v1（arxiv, 2023-02-09）
[^3]: ReAct: Synergizing Reasoning and Acting in Language Models — https://arxiv.org/abs/2210.03629v3（arxiv, 2022-10-06）
[^4]: What is the Model Context Protocol (MCP)? - MCP Official Documentation — https://modelcontextprotocol.io/docs/getting-started/intro（web）
[^5]: Introducing the Model Context Protocol - Anthropic — https://www.anthropic.com/news/model-context-protocol（web, 2024-11-25）
[^6]: Tool Learning with Foundation Models — https://arxiv.org/abs/2304.08354v3（arxiv, 2023-04-17）
[^7]: ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs — https://arxiv.org/abs/2307.16789v2（arxiv, 2023-07-31）
[^8]: ToolACE: Winning the Points of LLM Function Calling — https://arxiv.org/abs/2409.00920v2（arxiv, 2024-09-02）
[^9]: ToolRL: Reward is All Tool Learning Needs — https://arxiv.org/abs/2504.13958v1（arxiv, 2025-04-16）
[^10]: Nemotron-Research-Tool-N1: Exploring Tool-Using Language Models with Reinforced Reasoning — https://arxiv.org/abs/2505.00024v2（arxiv, 2025-04-25）
[^11]: ToolGen: Unified Tool Retrieval and Calling via Generation — https://arxiv.org/abs/2410.03439v3（arxiv, 2024-10-04）
[^12]: AgentGym: Evolving Large Language Model-based Agents across Diverse Environments — https://arxiv.org/abs/2406.04151（arxiv, 2024-06-06）
[^14]: OpenBMB/ToolBench — https://github.com/OpenBMB/ToolBench（github, 2023-05-28）
[^15]: ShishirPatil/gorilla — https://github.com/ShishirPatil/gorilla（github, 2023-05-19）
[^16]: Berkeley Function Calling Leaderboard (BFCL) — gorilla/berkeley-function-call-leaderboard/README.md — https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard（github）
[^17]: SalesforceAIResearch/xLAM — https://github.com/SalesforceAIResearch/xLAM（github, 2024-02-27）
[^18]: QwenLM/Qwen-Agent — https://github.com/QwenLM/Qwen-Agent（github, 2023-09-22）
[^19]: THUDM/AgentBench — https://github.com/THUDM/AgentBench（github, 2023-07-28）
[^20]: modelcontextprotocol/servers — https://github.com/modelcontextprotocol/servers（github, 2024-11-19）
[^21]: modelcontextprotocol/modelcontextprotocol — https://github.com/modelcontextprotocol/modelcontextprotocol（github, 2024-09-24）
[^22]: Berkeley Function Calling Leaderboard (BFCL) V4 — 官方榜单页 — https://gorilla.cs.berkeley.edu/leaderboard.html（web, 2026-04-12）
[^23]: Blog 8: Berkeley Function-Calling Leaderboard (BFCL V1 方法论) — https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html（web, 2024-08-19）
[^24]: BFCL V4: Web Search (Agentic 评估方法论与分数构成) — https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html（web, 2025-07-17）
[^25]: τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains — https://arxiv.org/abs/2406.12045（arxiv, 2024-06-17）
[^26]: τ-bench 论文正文（Introduction 部分） — https://arxiv.org/html/2406.12045v1（arxiv, 2024-06-17）
[^27]: BFCL-V4 Leaderboard (llm-stats.com 第三方聚合) — https://llm-stats.com/benchmarks/bfcl-v4（web）
[^28]: Parallel tool use - Claude Platform Docs — https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use（web）
[^29]: Function calling | OpenAI API — https://developers.openai.com/api/docs/guides/function-calling（web）
[^30]: MCP Security Best Practices - Model Context Protocol Specification (2025-06-18) — https://modelcontextprotocol.io/specification/2025-06-18/basic/security_best_practices（web, 2025-06-18）
[^31]: MCP (Model Context Protocol) Security Cheat Sheet - OWASP Cheat Sheet Series — https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html（web）
[^32]: Introducing advanced tool use on the Claude Developer Platform - Anthropic Engineering — https://www.anthropic.com/engineering/advanced-tool-use（web, 2025-11-24）
[^33]: ToolFailBench: Diagnosing Tool-Use Failures in LLM Agents — https://arxiv.org/abs/2607.04686（arxiv, 2026-07-06）
[^34]: AgentGym: Evolving Large Language Model-based Agents across Diverse Environments (arXiv:2406.04151v1) — https://arxiv.org/abs/2406.04151v1（arxiv, 2024-06-06）
[^35]: AgentGym (arXiv:2406.04151v1) — Introduction / self-evolution process — https://arxiv.org/html/2406.04151v1（arxiv, 2024-06-06）
[^36]: AgentGym Project Page — AgentGym Suite / Environment table — https://agentgym.github.io/（web）
[^37]: WooooDyy/AgentGym — README — https://github.com/WooooDyy/AgentGym（github, 2024-05-27）
[^38]: AgentGym/AgentEvol-7B — Hugging Face model card — https://huggingface.co/AgentGym/AgentEvol-7B（web, 2024-06-06）
[^39]: AgentGym/AgentTraj-L — Hugging Face dataset card — https://huggingface.co/datasets/AgentGym/AgentTraj-L（web, 2024-06-06）
[^40]: DiaTool-DPO: Multi-Turn Direct Preference Optimization for Tool-Augmented Large Language Models (arXiv:2504.02882v2) — https://arxiv.org/abs/2504.02882v2（arxiv, 2025-04-02）
[^41]: DiaTool-DPO 正文（ar5iv 渲染）— Introduction 与偏好对构造描述 — https://ar5iv.org/abs/2504.02882（arxiv, 2025-04-02）
[^42]: Hugging Face Papers 页面 2504.02882 — 作者自述（Sunghee Jung） — https://huggingface.co/papers/2504.02882（web, 2025-04-09）
[^43]: gorilla/berkeley-function-call-leaderboard README — https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/README.md（github）
[^44]: BFCL V3: Multi-Turn & Multi-Step Function Calling (官方博客) — https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html（web, 2024-09-19）
[^45]: ShishirPatil/gorilla — berkeley-function-call-leaderboard/bfcl_eval/constants/model_config.py — https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/constants/model_config.py（github）
[^46]: sierra-research/tau2-bench (GitHub repository) — https://github.com/sierra-research/tau2-bench（github, 2025-06-09）
[^47]: τ-bench Live Leaderboard (taubench.com) — https://taubench.com（web）
[^48]: τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment — https://arxiv.org/abs/2506.07982v1（arxiv, 2025-06-09）
[^49]: τ²-Bench (arXiv HTML full text, Introduction section) — https://arxiv.org/html/2506.07982v1（arxiv, 2025-06-09）
[^50]: APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets — https://arxiv.org/abs/2406.18518v1（arxiv, 2024-06-26）
[^51]: APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay — https://arxiv.org/abs/2504.03601v4（arxiv, 2025-04-04）
[^52]: Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation — https://arxiv.org/abs/2503.07826v1（arxiv, 2025-03-10）