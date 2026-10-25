# LLM 工具调用能力的训练、对齐与评测：基于一手证据的深度研究

## 摘要

本报告基于 58 条证据，梳理 LLM 工具调用（function calling）的三个层面：**接口契约层**（模型输出什么、由谁执行）、**训练与对齐层**（如何让模型学会调用、如何用 RL 奖励塑形）、**评测层**（如何判定调用"正确"）。核心发现：

1. **接口层已高度标准化但语义边界清晰**：主流 API（OpenAI、Anthropic、MCP）统一采用 JSON Schema 描述参数，且一致地把"执行"排除在模型职责之外——模型只生成参数，执行由宿主负责 [^1][^58][^3]。MCP 进一步把工具定性为"供 AI 模型执行的函数"，并规定 Hosts 在调用任何工具前必须获得用户明确同意 [^2]。
2. **训练路径存在两条清晰谱系**：自监督插入式（Toolformer 在文本序列中插入 API 调用 [^6]）与指令微调+数据合成式（ToolBench 用 ChatGPT 三阶段构造 16,464 个 API 的指令与解路径 [^7][^15]；CodeAct 用可执行 Python 统一动作空间 [^8]）。RL 路线的系统性研究出现较晚，ToolRL 称自己是首个针对通用工具选择与应用的 RL 训练系统性研究，相对 base model 提升 17% [^30]。
3. **评测正从"参数匹配"转向"状态验证"**：BFCL V3 起不再用 AST 拆解参数匹配候选答案，改为验证执行后 API 系统的实际状态 [^13]；τ-bench 比对对话结束时的数据库状态与标注目标状态 [^10][^11]；MCP-Universe 采用 execution-based evaluators 并明确拒绝 LLM-as-a-judge [^32][^33]。但评测阵营内部对 LLM-as-a-judge 是否可接受存在直接冲突（LiveMCPBench 采用之 [^35]，MCP-Universe 与 LiveMCP-101 批评之 [^32][^34]）。
4. **能力现状远未饱和**：τ-bench 上 gpt-4o 成功率 <50%、retail 域 pass^8 <25% [^10][^11]；τ²-Bench 双控制设定下 pass^1 为 gpt-4.1 34%、o4-mini 42%、claude-3.7-sonnet 49% [^12]；MCP-Universe 上 GPT-5 43.72% [^32]；LiveMCP-101 上 GPT-5 <60% [^34]。
5. **重大证据缺口**：GPT-5 / Claude 4.x / Gemini 2.5 的官方工具调用能力文档（并行调用、上下文长度、schema 特性）本次完全未取得一手证据 [^43][^41][^42]；BFCL 官方榜单页不含任何模型分数行 [^37]；Llama 4 无任何工具调用能力证据。

---

## 一、接口契约：模型生成什么，谁负责执行

### 1.1 标准定义：function calling 的输入、输出与执行边界

在 OpenAI Chat Completions 的规范中，`tools` 是可选参数，用于提供函数规格（function specifications），目的是让模型生成符合该规格的函数参数；**API 本身不执行任何函数调用，执行是开发者的责任，由模型输出驱动** [^1]。模型产出的是"一个包含调用这些函数所需参数的 JSON 对象"，而非执行结果 [^58]。

完整闭环由 messages 数组维护：角色包括 system、user、assistant、function；带 `tool_calls` 的 assistant 消息与 function 角色的结果消息构成闭环 [^1]。当模型决定调用时，响应含 `"finish_reason": "tool_calls"` 与一个 `tool_calls` 对象，其中含函数名与生成的参数 [^1]。

函数规格的字段级定义（OpenAI 官方 OpenAPI spec 生成）为：`name`（必填，仅允许 a-z、A-Z、0-9、下划线、连字符，最大长度 64）、`description`（规范明确其作用是"供模型决定何时以及如何调用该函数"）、`parameters`（JSON Schema 对象；省略即定义空参数列表）、`strict`（Optional[bool]）[^55]。`strict=true` 时模型严格遵循 parameters 中的 schema，但**仅支持 JSON Schema 的一个子集** [^55]。

调用强制程度由 `tool_choice` 控制，规范取值集合为字符串字面量 `"none"`、`"auto"`、`"required"` 三档，外加三种结构化形式：限定允许调用的工具集合、强制指定某个 function 工具、强制指定某个 custom 工具 [^56]。Cookbook 层面的对应表述是：默认由模型自行决定何时使用某个函数；可设为 `{"type": "function", "function": {"name": "my_function"}}` 强制使用特定函数；设为 `"none"` 则强制不使用 [^1]。

Anthropic 侧的对应定义使用 `input_schema` 字段承载 JSON Schema（type 必须为 `"object"`，draft 2020-12），与 `name` 并列必填；`description` 被建议尽可能详细，可用自然语言强化 schema 的重要方面以提升模型表现 [^57]。Anthropic 的 `ToolParam` 同样提供 `strict` 字段，语义为"为 true 时保证对工具名称与输入进行 schema 校验" [^57]。其 SDK 通过 `@beta_tool` 装饰器检查函数参数与 docstring 自动提取该 JSON Schema [^4]。

### 1.2 MCP：把工具接入标准化

MCP 是一个开放协议，用于在 LLM 应用与外部数据源、工具之间实现集成，使用 JSON-RPC 2.0 消息，在 Hosts（发起连接的 LLM 应用）、Clients（宿主应用内的连接器）、Servers（提供上下文与能力的服务）之间通信 [^2]。Server 向 client 提供的特性包括 Resources、Prompts、Tools（供 AI 模型执行的函数）[^2]。协议声明其受 Language Server Protocol 启发 [^2]。

工具定义的字段为：`name`（唯一标识）、`title`（可选展示名）、`description`、`inputSchema`（JSON Schema）、`outputSchema`（可选）、`annotations`（可选）[^3]。工具发现走 `tools/list`（支持 cursor/nextCursor 分页），调用走 `tools/call`，params 含 name 与 arguments 对象 [^3]。MCP 中的工具被设计为 **model-controlled**：语言模型可基于上下文理解与用户提示自动发现并调用工具，但协议本身不强制特定的用户交互模型 [^3]。

结果处理上，工具结果可含非结构化 `content`（text/image/audio/resource_link/embedded resource）与结构化 `structuredContent`；若提供 outputSchema，server 必须返回符合该 schema 的结构化结果，client 应当校验 [^3]。**规范特别澄清：structuredContent 是 server 产生的结果数据，与 LLM 的"structured outputs"（schema 约束的模型生成）无关** [^3]——这条区分对理解"约束解码"与"工具结果校验"是两件事至关重要。

错误处理分两类：协议错误用标准 JSON-RPC 错误（未知工具、无效参数、server 错误）；工具执行错误在结果中以 `isError: true` 报告 [^3]。

### 1.3 安全边界：协议层的显式约束

MCP 的安全原则写得相当强硬：**工具代表任意代码执行，必须谨慎对待；工具行为描述（如 annotations）除非来自可信 server，否则应视为不可信；Hosts 在调用任何工具前必须获得用户明确同意** [^2]。Server 侧必须校验所有工具输入、实施访问控制、对工具调用限流、清洗工具输出；client 应当对敏感操作提示用户确认、调用前向用户展示工具输入、校验工具结果、实现超时、记录审计日志 [^3]。

AutoGen README 给出同向的工程提示：只应连接可信的 MCP server，因为它们可能在本地环境执行命令或暴露敏感信息 [^52]。

---

## 二、训练与对齐：四条技术路线

### 2.1 路线一：自监督插入式（Toolformer）

Toolformer 被训练来决定"调用哪些 API、何时调用、传什么参数、以及如何最好地把结果并入后续 token 预测"，训练以自监督方式完成，每个 API 只需少量示例（a handful of demonstrations）[^6]。纳入的工具包括计算器、一个问答系统、两个不同的搜索引擎、一个翻译系统和一个日历 [^6]。

其形式化关注点是"在文本序列中插入 API 调用"并据其结果继续预测，**而非由外部宿主执行并回传结构化结果** [^6]。这一区别是理解早期路线与现代 function calling 分野的关键：Toolformer 把调用当作序列中的一段特殊 token，现代 API 把调用当作一个需要宿主执行并回填的协议事件 [^1][^3]。

效果上，Toolformer 在多种下游任务上取得显著提升的零样本性能，常与更大的模型相当，同时不牺牲其核心语言建模能力 [^6]。

### 2.2 路线二：指令微调 + 大规模数据合成（ToolBench / API-Bank）

ToolLLM 框架涵盖数据构造、模型训练与评测三部分 [^7]。其 ToolBench 指令微调数据集全程用 ChatGPT 自动构造，分三阶段：(i) API 收集——从 RapidAPI Hub 收集 16,464 个真实 RESTful API，覆盖 49 个类别；(ii) 指令生成——提示 ChatGPT 生成涉及这些 API 的多样指令，同时覆盖单工具与多工具场景；(iii) 解路径标注——用 ChatGPT 为每条指令搜索一条有效解路径（API 调用链）[^7][^15]。

为增强推理能力，作者提出基于深度优先搜索（DFS）的决策树算法，使模型能评估多条推理轨迹并扩展搜索空间 [^7][^15]。基于 ToolBench 微调 LLaMA 得到 ToolLLaMA，并配备神经 API 检索器为每条指令推荐合适的 API [^7][^15]。实验声称 ToolLLaMA 能执行复杂指令并泛化到未见过的 API，性能与 ChatGPT 可比，在分布外数据集 APIBench 上表现出较强零样本泛化 [^7][^15]。

API-Bank 提供了另一条数据规模参照：训练集含 1,888 条工具使用对话，来自 1,000 个不同领域的 2,138 个 API；基于该数据从 Alpaca 初始化训练了 Lynx [^16]。评测侧则构建了含 73 个 API 的可运行系统，标注 314 条工具使用对话、753 次 API 调用，用于评估规划、检索与调用能力 [^16]。结果显示 GPT-3.5 的工具使用优于 GPT-3，GPT-4 在规划方面表现突出；Lynx 比 Alpaca 提升超过 26 分，接近 GPT-3.5 [^16]。

### 2.3 路线三：动作空间统一化（CodeAct）

CodeAct 的出发点是对 JSON/文本动作格式的直接批评：现有 LLM agent 通常以预定义格式生成 JSON 或文本作为动作，受限于受限的动作空间（预定义工具范围）与灵活性不足（无法组合多个工具）[^8]。其方案是用可执行 Python 代码把动作整合为统一动作空间，配合 Python 解释器执行代码动作，并在多轮交互中依据新观测动态修正先前动作或发出新动作 [^8]。

在 API-Bank 与一个新构建的基准上对 17 个 LLM 的分析显示，CodeAct 优于广泛使用的替代方案，成功率最高提升 20% [^8]。配套的 CodeActInstruct 含 7k 条使用 CodeAct 的多轮交互，可与现有数据配合使用，在不损害模型通用能力的前提下提升 agent 类任务表现 [^8]。CodeActAgent 由 Llama2 与 Mistral 微调而来，集成 Python 解释器，可用现有库完成复杂任务并自主 self-debug [^8]。论文被 ICML 2024 接收 [^8]。

值得注意：Anthropic 的 `ToolParam` 中出现 `allowed_callers`（取值含 direct 与 code_execution_* 系列）与 `eager_input_streaming` 字段 [^57]，说明"代码执行作为工具调用载体"已进入商业 API 的类型系统，但证据未给出其语义细节。

### 2.4 路线四：强化学习与奖励设计（ToolRL）

ToolRL 把工具使用训练形式化为 Tool-Integrated Reasoning (TIR)：给定工具集 T={t1..tn} 与用户查询 Q，轨迹为多步 (r_i, T_i, o_i) 序列，其中 r_i 为自然语言推理、T_i 为该步调用的工具子集、o_i 为执行后的观测（含环境与用户反馈）[^30]。作者称这是首个针对通用工具选择与应用的 RL 训练系统性研究，区别于此前仅限搜索工具问答（Search-R1）或代码工具数学解题（ToRL）的窄范围工作 [^30]。

奖励设计沿四个维度系统探索：reward type（奖励什么方面）、reward scale（奖励多少）、reward granularity（信号多细）、reward dynamics（奖励随训练如何演化）[^30]。方法为算法无关的奖励设计，实证在 GRPO 与 PPO 两种算法上验证有效性 [^30]。报告的训练收益：相对 base model 提升 17%，相对 SFT model 提升 15%，跨多个工具使用与 QA 基准 [^30]。

三条奖励设计经验结论值得单列：(1) **更长的推理轨迹并非天然更好，长度奖励会降低性能**；(2) 动态奖励尺度有助于模型从简单到复杂行为平滑过渡；(3) 细粒度奖励分解带来更稳定有效的学习 [^30]。

ToolRL 对 SFT 的批评同样关键：SFT 在蒸馏的 deep-thinking 轨迹上训练会出现过度思考（overthinking）与泛化受限，模型会过度解读工具、无法拒绝不合适的工具，只是模仿 "but wait" 之类措辞而非真正深度思考 [^30]。这为"为什么需要 RL 而非纯 SFT"提供了机制性论据。

### 2.5 对齐的执行侧：谁在什么时机执行

训练之外，"对齐"在工程上还体现为执行时机与幂等性约束。Anthropic SDK 的 `client.beta.messages.tool_runner()` 返回迭代器，每次迭代产出一个 BetaMessage；迭代由每条消息的 `stop_reason` 驱动——为 `tool_use` 时 runner 执行所请求的工具并把结果发回，为 `pause_turn` 或 `compaction` 时原样回传该轮以便服务端恢复，其他 stop reason 则产出最终消息后停止且不运行工具 [^4]。

流式场景下可用 `run_tools_eagerly=True` 让工具在模型写完调用时立即执行，或用 `defer_tool_call()` 暂缓执行；**对不可安全重复执行的调用应延迟（defer），因为流式期间已执行的调用无法撤回，若回复被截断模型可能重复请求同一调用** [^4]。这是一条明确的"模型行为不确定性 → 执行侧必须做幂等防护"的对齐约束。

推理引擎侧的对齐则通过约束解码实现。vLLM 把 constrained decoding 行为拆成三件事分别处理：call obligation（是否必须调用）、structural-tag activation（语法是否激活）、argument-schema enforcement（参数 schema 是否强制）[^46]。structural tag 激活规则：named function 与 required 总是激活；auto 仅当至少一个 tool 设 `strict: true`，或 `--tool-strict-level` 为 function/parameter 时激活；none 禁用 [^46]。strict 模式分两层：tool-call envelope（标记与函数名）在 structural tag 生效时总被约束；单个 tool 的参数 schema 仅在该 tool 设 `strict: true`（或 `--tool-strict-level parameter`）时才被钉死 [^46]。

vLLM 文档对 named function calling 的保证措辞极为克制：**"保证生成可被正确解析的 function call，而不是高质量的 function call"** [^46]。文档还明确说明动机：多数 OpenAI 兼容客户端和 agent 框架从不设置 strict，导致 `tool_choice='auto'` 下模型无语法约束地生成 tool call，畸形标记可能泄漏进响应 [^46]。

SGLang 侧把 `--tool-call-parser` 的语义明确定义为"定义用于解释响应的 parser"，即输出侧解析器而非输入侧模板 [^49]；其 `tool_choice` 支持 required（强制至少调用一个工具），文档称该功能用 EBNF 文法实现以保证可靠的工具调用行为 [^49]。SGLang 还提供 `/parse_function_call` 原生端点，输入 text、tool_call_parser、tools，返回 normal_text 与 calls（含 name、parameters）[^49]。

一个重要的架构区分：vLLM 文档指出存在一类"response template"parser，它只从已生成文本中解析调用，解析失败的调用会被丢弃，且这类 parser 只提供解析、不提供格式相关的约束解码，因此会拒绝 strict tools、required/named tool choice 或 `parallel_tool_calls=false` 的请求 [^47]。这说明"解析"与"约束"是两个可分离的能力层。

---

## 三、评测：从参数匹配到状态验证

### 3.1 评测范式的三次迁移

**第一代：参数级匹配。** BFCL-v1 引入 AST 作为评估指标 [^37]；V3 之前用 AST 拆解函数参数对并在候选答案列表中匹配 [^13]。

**第二代：状态转移验证。** BFCL V3 于 2024-09-19 发布，改变校验方式：不再用 AST 匹配候选答案，而是验证模型执行函数后 API 系统（如文件系统、订票系统）的实际状态 [^13]。作者强调 BFCL 全程依赖 AST 或状态转移（state-transition）验证，以保证确定性与最小波动 [^13]。V3 新增多轮与多步类别，区分方式为：多步（multi-step）指用户只在最开始交互一次、模型与系统来回多次调用函数完成单个请求；多轮（multi-turn）指用户与助手多轮对话、每轮可能包含多步 [^13]。数据构成为 Base Multi-Turn 200 条、Augmented Multi-Turn 800 条（含模糊提示等额外复杂度）[^13]。

**第三代：整体 agentic 评测。** BFCL V4 于 2025-07-17 发布，定位为面向 LLM 的整体 agentic 评测，含 web search、memory、format sensitivity 三部分 [^14]。总分构成为：Overall = Agentic × 40% + Multi-Turn × 30% + Live × 10% + Non-Live × 10% + Hallucination × 10% [^14]。CHANGELOG 记录了这次权重再分配的方向：Live 由 33% 降至 10%，Non-Live 由 33% 降至 10%，Irrelevance 由 0% 升至 10%，Multi-Turn 由 33% 降至 30%，Agentic 由 0% 升至 40% [^38]——即权重向复杂多步 agentic 任务倾斜。V4 起 Non-Live Acc 与 Live Acc 的计算排除 Irrelevance/Relevance 类别分数 [^38]。

一个需要指出的口径冲突：BFCL 官方榜单页说明 Overall Accuracy 为所有子类别的**未加权平均** [^37]，而 V4 blog 给出的构成是**百分比加权**公式 [^14]。两者可能对应不同版本或不同汇总列，现有证据不足以判定孰为当前口径。

### 3.2 τ-bench 系列：把"用户"和"规则"纳入评测

τ-bench 的动机是已有基准不测试 agent 与人类用户的交互，也不测试其遵循领域特定规则的能力 [^11]。它模拟（由语言模型扮演的）用户与语言 agent 之间的动态对话，agent 被提供领域特定的 API 工具与政策指南；评测方式为对比对话结束时的数据库状态与标注的目标状态，而非仅比对文本输出 [^10][^11]。

其提出的 `pass^k` 指标用于评估 agent 在多次试验中的行为可靠性/一致性 [^10][^11]。实验结果：即使最先进的 function calling agent（如 gpt-4o）任务成功率也低于 50%；agent 行为相当不一致，retail 域 pass^8 <25% [^10][^11]。作者结论指向需要能提升 agent 一致行动与可靠遵循规则的方法 [^10]。

τ²-Bench 进一步指出已有对话 agent 基准都是"单控制"环境：只有 AI agent 能用工具改变世界，用户只是被动的信息提供者 [^12]。它引入 Telecom 双控制领域，形式化为 Dec-POMDP：agent 与用户都拥有各自的工具，作用于共享的动态环境 [^12]。任务由组合式生成器从少量原子基础场景（由 initialization、solution、assertion 函数定义）程序化组合，保证任务正确性、领域覆盖与复杂度可控 [^12]。用户模拟器与环境紧耦合，其行为受工具与可观测状态约束，以提升模拟保真度 [^12]。

关键结果：从 no-user 切换到 dual-control 时 agent 性能显著下降；新任务上 pass^1 为 gpt-4.1 34%、o4-mini 42%、claude-3.7-sonnet 49% [^12]。论文提供细粒度消融，区分推理错误与沟通/协调错误 [^12]。

### 3.3 MCP 生态评测：真实服务器 + 执行式评估

**MCP-Universe** 覆盖 6 个核心领域、11 个真实 MCP 服务器（Location Navigation、Repository Management、Financial Analysis、3D Design、Browser Automation、Web Searching），任务总数 231 个 [^32]。任务形式化为：S={s1..sk} 为 MCP 服务器集合，每个服务器暴露工具集 Ti；任务 τ 定义为三元组 (G, C, T_available) [^33]。

评分采用 execution-based evaluators，分三类：format evaluators（agent 格式合规）、static evaluators（时间无关内容匹配）、dynamic evaluators（对时间敏感任务自动获取实时 ground truth）[^32]。作者**明确拒绝 LLM-as-a-judge 路线**（点名 MCPEval 与 LiveMCPBench），理由是许多任务涉及实时数据，静态 LLM 知识无法正确评判 [^32]。

结果：GPT-5 成功率 43.72%，Grok-4 33.33%，Claude-4.0-Sonnet 29.44% [^32]。论文指出的挑战为长上下文（输入 token 随交互步数快速增长）与 unknown-tools（agent 不熟悉 MCP 服务器的精确用法）；企业级 agent（如 Cursor）表现未能超过标准 ReAct 框架 [^32]。论文还批评 MCP-RADAR 由 HumanEval/GSM8k 等既有数据集改造而来，存在数据泄漏与真实覆盖不足；批评 MCPWorld 过度依赖 GUI、对 MCP 工作流覆盖不足 [^32]。

**LiveMCP-101** 含 101 条真实世界查询，要求协同使用多个 MCP 工具，覆盖 41 个 MCP 服务器、260 个工具 [^34]。任务按难度分三层：Easy 30、Medium 30、Hard 41，难度由作者共识依据推理复杂度与工具链长度判定 [^34]。查询由 GPT-4.1 采样领域、OpenAI o3 生成，再经执行—LLM 辅助编辑—人工修订的迭代循环精炼与验证，约需 120 专家工时 [^34]。每条查询配一个已验证执行计划，指定 MCP 工具序列、显式参数化与后处理代码；计划平均 5.4 次工具调用，最多 15 次 [^34]。

评分采用并行评测：参考 agent 严格按计划执行产生同时刻参考输出，被评 agent 自主规划执行，再由 LLM judge 对比两者输出与轨迹打分 [^34]。18 个代表性 LLM 上，即使 GPT-5 任务成功率也低于 60% [^34]。错误分析归纳出 7 种失败模式，分属三类：工具规划与编排错误、参数错误、输出处理错误 [^34]。论文批评并发工作（LiveMCPBench、MCP-Bench）缺乏客观可验证 ground truth，仅依赖主观 LLM-as-a-judge rubric，损害评分可靠性 [^34]。

**LiveMCPBench** 含 95 个真实日常任务，提供 70 个服务器、527 个工具的可直接部署套件 [^35]。评分采用 LLM-as-a-Judge 框架直接验证任务结果，以处理动态数据源与多条有效解法路径 [^35]。评测 12 个 SOTA LLM：Claude-Sonnet-4 达 78.95% 任务成功率，多数模型仅 30–50% [^35]。分析结论：主动的工具组合与任务成功强相关；**检索错误占全部失败近一半，检索是主要瓶颈** [^35]。论文指出现有评测的缺陷：通常假设单服务器设置、直接把工具注入模型上下文，绕过了大规模检索与多工具组合的挑战 [^35]。

**MCPBench** 的定位不同：它评测 MCP 服务器本身而非 agent，指标为 accuracy、time consumption、token consumption，且只评测 data fetching 类任务（web search、database search），明确排除 world changing 类 MCP，理由是难以访问底层数据源状态 [^36]。Accuracy 由 LLM-based grader 判定（DeepSeek-v3 作为 grader）[^36]。部分结果：Brave Search 准确率 46.6%、耗时 13.98s；DuckDuckGo 13.62%、64.17s；Tavily 47.99%、95.52s；Exa 15.02%、231.24s [^36]。执行摘要结论值得注意：**MCP 服务器之间在有效性与效率上差异显著；使用 MCP 相比 function call 并未表现出明显提升** [^36]。

### 3.4 安全评测：间接提示注入

InjecAgent 构建了评估工具集成 LLM agent 对间接提示注入（IPI）脆弱性的基准，攻击者指令被嵌入 agent 所处理的外部内容（如邮件、网页）中 [^9]。基准含 1,054 个测试用例，覆盖 17 个用户工具与 62 个攻击者工具；攻击意图归为两类：对用户的直接伤害与隐私数据外泄 [^9]。共评估 30 个不同 LLM agent；ReAct 提示的 GPT-4 有 24% 的概率被攻击成功；在增强设置下（攻击者指令附加 hacking prompt 强化），攻击成功率近乎翻倍 [^9]。论文为 ACL 2024 Findings [^9]。

### 3.5 评测口径的横向对比

BFCL V3 blog 给出的对比表显示 BFCL-v3 在 Multi-Step、Multi-Turn、Implicit Action、Self-Correct、Irrelevancy、Long Ctx 各维度均为 ✓，而 BFCL-v2、AgentBoard、τ-bench、MMAU、ToolSandbox 各有缺失项 [^13]。MCP-Universe 的 Table 1 以 Real-World / Integration / Temporal Dynamics / Exec.Eval. 四维对比：MCPWorld 仅 Real-World+Temporal；MCP-RADAR 仅 Temporal；MCPEval 仅 Integration；LiveMCPBench 为 Real-World+Integration；MCP-Universe 三项全有 [^33]。LiveMCP-101 的 Table 1 则显示：LiveMCPBench 70 服务器、平均 2.8 步、有 Real Integration 与 Temporal Dynamics，但无 Verifiable Ground Truth；MCP-Bench 28 服务器；MCPEval 19 服务器；MCPWorld 10 服务器；MCP-RADAR 9 服务器 [^34]。

这些对比表由各论文自建维度、自评优劣，**不构成独立第三方评估**，引用时应视为作者主张而非共识结论。

---

## 四、生态与工程落地：可观测的热度信号

MCP 官方仓库群的元数据：`modelcontextprotocol/servers` stars 91,547、forks 11,864、主语言 TypeScript、last push 2026-10-25 [^17]；`python-sdk` stars 24,664、forks 4,074、last push 2026-10-25，仓库描述自称 official（未经组织页或 README 正文核验）[^18]；`typescript-sdk` stars 13,649、forks 2,337、last push 2026-10-23，同样自称 official 且未经正文核验 [^19]。

第三方 MCP server 的代表：`hangwin/mcp-chrome` stars 12,508、forks 1,147、last push 2026-01-06，描述称其为基于 Chrome 扩展的 MCP server，向 Claude 等 AI 助手暴露浏览器功能 [^20]；`mobile-next/mobile-mcp` stars 9,960、forks 891、last push 2026-10-25，描述称覆盖 iOS、Android、模拟器与真机 [^21]。两者均未读 README，功能实现细节未取证 [^20][^21]。

Agent 框架侧的定位值得注意：LangGraph README 自我定位为"低层编排框架"，核心能力列为 durable execution、human-in-the-loop、comprehensive memory、LangSmith 调试、production-ready deployment，**未把 tool/function calling 列为自身特性**，而将工具/agent 能力外推到生态组件（Deep Agents 与 LangChain）[^51]。AutoGen README 顶部标注"Project archived and no longer maintained"，明确建议不要用于生产环境，建议新用户改用 Microsoft Agent Framework [^52]；其 MCP 示例通过 McpWorkbench + StdioServerParams 接入 Playwright MCP server，并在 AssistantAgent 上设置 `max_tool_iterations = 10` [^52]。LlamaIndex README 声明公司重心已从 OSS 框架转向 LlamaParse，且抓取到的正文未出现 tool calling / function calling 的专门主张段落 [^53]。CrewAI README 把 tools 列为 agent 配置项之一（design-agent skill 的适用场景描述含 "role, goal, backstory, tools, LLMs, memory, guardrails"）[^54]。

推理引擎的 parser 生态：vLLM 文档可确认存在的 parser 名包括 hermes、mistral、llama3_json、glm47 [^47]；SGLang 文档列出的 parser 名为 apertus2509、deepseekv3、deepseekv31、deepseekv32、glm、gpt-oss、kimi_k2、llama3、llama4、mistral、pythonic、qwen、qwen3_coder、step3 [^49]。两侧 parser 命名体系不通用，说明"工具调用格式"目前仍是模型族私有约定而非跨引擎标准。

---

## 五、矛盾与不确定性

### 5.1 评测方法论的直接冲突

**LLM-as-a-judge 的正当性存在正面冲突。** MCP-Universe 明确拒绝该路线，理由是实时数据任务无法由静态 LLM 知识正确评判 [^32]；LiveMCP-101 批评并发工作缺乏客观可验证 ground truth、仅依赖主观 rubric 损害评分可靠性 [^34]。而 LiveMCPBench 恰恰采用 LLM-as-a-Judge 框架，理由是需处理动态数据源与多条有效解法路径 [^35]。MCPBench 则用 DeepSeek-v3 作 grader 判定 accuracy [^36]。三套立场各有其任务特性支撑，**目前不存在公认裁决**；跨基准比较分数时必须先确认其评分范式是否可比。

**分数不可直接横向比较。** 同为"工具使用成功率"，τ-bench 的 gpt-4o <50% [^10]、τ²-Bench 的 claude-3.7-sonnet 49% [^12]、MCP-Universe 的 Claude-4.0-Sonnet 29.44% [^32]、LiveMCPBench 的 Claude-Sonnet-4 78.95% [^35] 分属不同任务集、不同评分器、不同难度分布，**不构成同一模型的能力排序**。

**BFCL 汇总口径存疑。** 官方榜单页称 Overall Accuracy 为所有子类别未加权平均 [^37]，V4 blog 给出的是百分比加权公式 [^14]。现有证据不足以判定两者是否为同一指标的不同表述，或分属不同版本。

### 5.2 训练路线的证据强度不均

Toolformer 的自监督插入式 [^6]、ToolBench 的 ChatGPT 自动构造 [^7][^15]、CodeAct 的可执行代码动作 [^8] 均有论文级证据，但**均未提供与 RL 路线的同条件对照**。ToolRL 声称相对 SFT model 提升 15% [^30]，其对照基线是"蒸馏的 deep-thinking 轨迹"上的 SFT，不能外推为"RL 普遍优于 SFT"。

ToolRL 关于"长度奖励会降低性能"的结论 [^30] 与 ReAct 强调推理轨迹有助于归纳、跟踪、更新行动计划的立场 [^5] 并不直接矛盾（前者反对的是对长度的奖励，后者主张的是轨迹的结构性作用），但两者共同指向一个未解问题：**推理轨迹的长度与质量在工具调用任务中如何解耦度量**，现有证据不足以回答。

### 5.3 重大证据缺口（须显式声明）

**（a）主流闭源模型的官方工具调用能力文档完全缺失。** Anthropic 官方 tool use 文档两次抓取均返回区域限制页，未返回正文，因此**无法确认 Claude 4.x 的并行调用、strict schema、上下文长度与 tool_choice 选项** [^43]。OpenAI 侧仅有 openai-python 的 api.md 类型索引（含 parallel_tool_calls、tool_choice、tools 参数与多种 tool_choice 类型，但该文件仅为类型与方法索引，未包含并行调用语义、strict 模式行为或上下文长度的说明文字）[^40]，以及 README（示例模型标识为 gpt-5.5 而非课题设定的 GPT-5，且正文未涉及并行、strict schema、上下文长度或 tool_choice 选项）[^41]。Google 侧仅有 python-genai README（仅提及 Automatic Function Calling 将在下个大版本变更调用入口，未给出并行函数调用、responseSchema/strict 约束、上下文长度或 tool_choice 的具体说明）[^42]。**结论：GPT-5 / Claude 4.x / Gemini 2.5 在并行调用、上下文长度、schema 特性三个维度上，现有公开资料不足以确认。**

**（b）BFCL 官方榜单不含任何模型分数。** 榜单页抓取结果中不含任何具体模型的 Overall 分数或排名行（表格由前端 JS 渲染，未出现在返回正文中）[^37]；CHANGELOG 只证明模型被加入榜单（如 claude-opus-4-1-20250805、gpt-5-2025-08-07、gpt-5-mini、gpt-5-nano、Qwen3 系列等），不提供分数 [^38]。**结论：截至 2026 的 BFCL 公开分数与排名，现有公开资料不足以确认。**

**（c）开源模型的工具调用能力覆盖不足。** Qwen3 技术报告仅给出 BFCL v3 70.8 分与预训练序列长度（长上下文阶段 32768），未涉及其工具调用训练方式、并行调用支持或 schema 特性 [^44]；Llama 4 仅在 Qwen3 报告中作为对比基线出现（总参数 402B、激活 17B），无任何工具调用能力或榜单分数证据 [^44]。**结论：Qwen3、Llama 4 等开源模型的工具调用能力细节，现有公开资料不足以确认。**

**（d）τ-bench 系列缺新一代模型数据。** 仅有论文中的 gpt-4o、gpt-4.1、o4-mini、claude-3.7-sonnet 旧数据 [^10][^11][^12]，无 GPT-5 / Claude 4.x / Gemini 2.5 的 τ-bench 分数。

**（e）部分仓库证据仅到元数据层。** `modelcontextprotocol/servers`、`python-sdk`、`typescript-sdk` 均未读取 README 正文，其 server 清单、传输方式支持、工具注册 API 等具体主张未取证 [^17][^18][^19]；两个第三方 MCP server 同样未读 README [^20][^21]。Gorilla 官方仓库的 stars、活跃度与 README 主张未取得，仅有第三方封装仓库间接提及 gorilla-llm/gorilla-openfunctions-v2 这一模型名 [^22]。BFCL 官方库（gorilla-llm/berkeley-function-calling-leaderboard）的 stars
、活跃度与 README 主张未取得，检索到的 BFCL 相关仓库均为非官方衍生项目（中文评测版 [^23]、LlamaFactory 集成 [^24]、服务化封装 [^25]、v4 多模型协作实验 [^26]），**不能作为 BFCL 官方实现细节的证据**。

**（f）工具调用训练数据合成的一手证据薄弱。** 检索到的 `aaronlyt/toolsynth` 声称复现 ToolACE、ACEBench、AgentInstruct、Self-Instruct、Kimi K2 agentic 与 K3 KG-guided 等流水线，但该仓库 stars 仅 6，且 README 未核验，**其"faithful reproductions"主张不得作为这些数据集或流水线存在性与质量的证据** [^28]。同理，`puppetm4st3r/local_function_calling`（stars 17）仅为第三方封装，不能代表 Gorilla 官方实现 [^22]。

**（g）主流 agent 框架的检索覆盖不全。** 检索 'function calling agent framework tool use' 时未返回 LangChain/LangGraph/LlamaIndex/AutoGen/CrewAI 等主流框架，后续单独取得的 LangGraph、AutoGen、LlamaIndex、CrewAI README 中，**没有一个把 tool/function calling 作为自身核心特性段落来主张** [^51][^52][^53][^54]；`AIwithhassan/ai-agents-from-scratch`（stars 6）为教学项目，不代表主流框架 [^29]。因此"主流 agent 框架如何实现工具调用"这一子问题，**现有证据只到 README 定位层，未到实现层**。

### 5.4 概念边界上的易混点（须显式澄清）

**structuredContent ≠ structured outputs。** MCP 规范明确声明：`structuredContent` 是 server 产生的结果数据，与 LLM 的"structured outputs"（schema 约束的模型生成）无关 [^3]。前者是工具返回值，后者是约束解码产物（vLLM 用 xgrammar 或 guidance 作为后端实现 [^48]）。混淆二者会导致对"谁在约束谁"的误判。

**tool_choice 的强制 ≠ 参数质量。** vLLM 对 named function calling 的保证措辞是"保证生成可被正确解析的 function call，而不是高质量的 function call" [^46]。即 `tool_choice` 与 strict schema 解决的是**格式合法性**，不解决**语义正确性**；后者仍需评测层（状态验证、pass^k）来度量 [^13][^10]。

**MCP 的 model-controlled ≠ 无约束。** 协议把工具设计为 model-controlled（模型可自动发现并调用）[^3]，但同时规定 Hosts 在调用任何工具前必须获得用户明确同意 [^2]，且 server 必须校验输入、限流、清洗输出 [^3]。**"模型自主决定"与"宿主强制确认"在协议中是并存的，不是替代关系。**

**ReAct 的 action ≠ JSON schema 式工具调用。** ReAct 把动作定义为与外部来源/环境交互的接口，用于收集额外信息——这与 API 规范中"工具调用"的定位一致，但 **ReAct 未规定 JSON schema 式的参数约束** [^5]。因此 ReAct 是提示范式层面的工作，不能直接作为"function calling 训练方法"的证据。

**Toolformer 的 API 调用 ≠ 宿主执行回填。** Toolformer 的形式化关注点是"在文本序列中插入 API 调用"并据其结果继续预测，而非由外部宿主执行并回传结构化结果 [^6]。这与现代 function calling 的协议事件模型 [^1][^3] 是两种不同的抽象层级。

### 5.5 安全结论的适用范围限制

InjecAgent 的 24% 攻击成功率（ReAct-prompted GPT-4）与增强设置下近乎翻倍 [^9]，是**特定基准、特定提示范式、特定模型版本**下的测量值，不能外推为"所有工具集成 agent 的 IPI 风险水平"。MCP 协议层的安全要求（用户明确同意、输入校验、限流、输出清洗、审计日志）[^2][^3] 是规范性要求（MUST/SHOULD），**与实测攻击成功率之间不存在已被证据建立的因果链**——即"实施了这些要求能把 24% 降到多少"，现有公开资料不足以确认。

### 5.6 时间线与版本的不确定性

- BFCL 官方榜单页标注 Last Updated: 2026-04-12，页面标题与正文均标注 V4，未出现 V5 字样 [^37]；CHANGELOG 最新条目为 2025-10-01，文件中不存在任何 V5 发布记录 [^38]；README 列出的官方 blog 版本序列止于 BFCL V4 三篇 [^39]。**三条证据一致指向"截至取证时 BFCL 最新为 V4"，但榜单页日期（2026-04-12）晚于 CHANGELOG 最新条目（2025-10-01），说明榜单在 CHANGELOG 停止更新后仍有维护，二者更新节奏不同步。**
- 榜单模型结果基于 commit f7cf735 评估，复现需使用 `pip install bfcl-eval==2025.12.17` 或对应代码 checkpoint [^37]；2025-09-27 引入 `--partial-eval` 标志允许对结果文件子集做部分评估 [^38]。**这意味着同一模型在不同 commit 下的分数可能不可比，跨论文引用 BFCL 分数时须核对 commit 与包版本。**
- Qwen3 技术报告为 v1（2025-05-14 提交）[^45]，其 BFCL v3 70.8 分对应的是 BFCL v3 口径 [^44]，**不能与 BFCL V4 的加权总分直接比较**（V4 权重已重分配，Agentic 占 40% [^38]）。
- `Small Language Models are the Future of Agentic AI` 为立场论文，且存在 v1（2025-06-02）、v2（2025-09-15）、v3（2026-09-22）三个版本 [^31]；该文不涉及工具调用的偏好优化（DPO/ORPO）、过程奖励模型或可验证奖励 RL 的训练目标与数据构造，**对本课题的训练与对齐子问题无直接证据价值，仅作检索排除项记录** [^31]。ToolRL 虽把 DPO、SimPO、GRPO、DAPO、VAPO 列为 LLM RL 方法演进脉络中的相关方法，但仅作背景引用，**未在工具使用任务上评测 DPO/SimPO** [^30]。

### 5.7 需要后续补证的关键问题

综合上述缺口，以下问题在本报告证据范围内**无法给出确定答案**，需补充一手来源：

1. GPT-5、Claude 4.x、Gemini 2.5 的原生并行工具调用（parallel tool calls）语义、上下文长度上限、strict schema 支持的 JSON Schema 子集范围——需 OpenAI/Anthropic/Google 官方文档正文 [^40][^41][^42][^43]。
2. 上述模型在 BFCL V4、τ-bench、τ²-Bench 上的公开分数与排名——需榜单渲染后的表格数据或官方发布的分数文件 [^37]。
3. Qwen3、Llama 4 的工具调用训练数据构造方式、并行调用支持、schema 特性——需技术报告的训练章节或模型卡 [^44]。
4. 各推理引擎 parser（vLLM 的 hermes/mistral/llama3_json/glm47 [^47] 与 SGLang 的 14 种 parser [^49]）在相同模型上的解析成功率与误解析率的横向对比——现有证据仅列出 parser 名与适用模型族，**无性能对照数据**。
5. MCP 协议安全要求（用户确认、输入校验、限流、输出清洗）实施前后的 IPI 攻击成功率变化——需针对 MCP 场景的受控实验，InjecAgent 的测量不覆盖 MCP [^9]。

---

## 结语（证据范围内的收敛判断）

在现有 58 条证据可支撑的范围内，可以收敛出三点判断：

**第一，工具调用的"接口层"已经稳定，且稳定得相当彻底。** OpenAI、Anthropic、MCP 三方在"JSON Schema 描述参数 + 模型只生成参数不执行 + 宿主负责执行与回填"这一基本契约上高度一致 [^1][^55][^57][^3][^58]，差异主要在字段命名（`parameters` vs `input_schema`）与强制程度选项的丰富度（OpenAI 的 none/auto/required 加两种具名形式 [^56]，vLLM 的 named/auto/required/none [^46]，SGLang 的 required 等 [^49]）。这意味着**训练与评测的接口假设是共享的**，跨模型比较在接口层是可行的。

**第二，"训练"与"对齐"在工具调用场景下不是两个阶段，而是三层叠加。** 数据层（ToolBench 的三阶段合成 [^7][^15]、API-Bank 的 1,888 条对话 [^16]、CodeActInstruct 的 7k 条多轮交互 [^8]）、奖励层（ToolRL 的四维奖励设计与三条经验结论 [^30]）、执行层（Anthropic 的 defer 机制 [^4]、vLLM 的 structural tag 与 strict 分层 [^46]）各自解决不同问题：数据层解决"见过"，奖励层解决"选对"，执行层解决"不出错"。**三者缺一，模型都可能在实际部署中失败**——τ-bench 的 pass^8 <25% [^10][^11] 与 τ²-Bench 双控制下的性能骤降 [^12] 正是"数据层与奖励层做对了、执行层与协调层没做对"的表现。

**第三，评测正在从"模型会不会调"转向"系统做没做对"。** BFCL V3 的状态转移验证 [^13]、τ-bench 的数据库状态比对 [^10][^11]、MCP-Universe 的 execution-based evaluators [^32][^33]、LiveMCP-101 的参考计划并行执行 [^34]，共同指向一个方法论转向：**判定对象从模型输出文本，变为环境终态**。这一转向的代价是评测成本上升（LiveMCP-101 约 120 专家工时 [^34]）与评分范式分歧（LLM-as-a-judge 的正当性冲突 [^32][^34][^35]），收益则是分数与真实部署表现的相关性更可辩护。

**但必须强调：上述三点判断的证据基础在闭源旗舰模型一侧是残缺的。** Anthropic 官方文档不可达 [^43]、OpenAI 仅有类型索引与 README [^40][^41]、Google 仅有 AFC 变更提示 [^42]、BFCL 榜单无分数行 [^37]——**本报告对"主流模型"的描述，实际上主要由开源侧论文（ToolLLM、CodeAct、ToolRL、Qwen3）与评测侧论文（τ-bench、MCP-Universe、LiveMCP-101、LiveMCPBench）支撑，而非由 GPT-5 / Claude 4.x / Gemini 2.5 的官方能力披露支撑。** 这一结构性偏差应在引用本报告结论时一并考虑。

## 参考来源

[^1]: How to call functions with chat models (OpenAI Cookbook) — https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/How_to_call_functions_with_chat_models.ipynb（web）
[^2]: Model Context Protocol Specification (2025-06-18) — Overview — https://modelcontextprotocol.io/specification/2025-06-18（web, 2025-06-18）
[^3]: MCP Specification (2025-06-18) — Server / Tools — https://modelcontextprotocol.io/specification/2025-06-18/server/tools（web, 2025-06-18）
[^4]: Claude SDK for Python — Tools helpers (tools.md) — https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/tools.md（github）
[^5]: ReAct: Synergizing Reasoning and Acting in Language Models (arXiv:2210.03629v3) — https://arxiv.org/abs/2210.03629v3（arxiv, 2022-10-06）
[^6]: Toolformer: Language Models Can Teach Themselves to Use Tools (arXiv:2302.04761v1) — https://arxiv.org/abs/2302.04761v1（arxiv, 2023-02-09）
[^7]: ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs — https://arxiv.org/abs/2307.16789v2（arxiv, 2023-07-31）
[^8]: Executable Code Actions Elicit Better LLM Agents (CodeAct) — https://arxiv.org/abs/2402.01030（arxiv, 2024-02-01）
[^9]: InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents — https://arxiv.org/abs/2403.02691（arxiv, 2024-03-05 (v1); v3 2024-08-04）
[^10]: τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains — https://arxiv.org/abs/2406.12045（arxiv, 2024-06-17）
[^11]: τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains — https://arxiv.org/abs/2406.12045v1（arxiv, 2024-06-17）
[^12]: τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment — https://arxiv.org/abs/2506.07982（arxiv, 2025-06-09）
[^13]: BFCL V3: Multi-Turn & Multi-Step Function Calling (Gorilla / Berkeley) — https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html（web, 2024-09-19）
[^14]: BFCL V4: Web Search (BFCL V4 Agentic Part 1) — https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html（web, 2025-07-17）
[^15]: ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs (ToolBench / ToolEval) — https://arxiv.org/abs/2307.16789（arxiv, 2023-07-31）
[^16]: API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs — https://arxiv.org/abs/2304.08244（arxiv, 2023-04-14）
[^17]: modelcontextprotocol/servers — Model Context Protocol Servers — https://github.com/modelcontextprotocol/servers（github）
[^18]: modelcontextprotocol/python-sdk — official Python SDK for MCP servers and clients — https://github.com/modelcontextprotocol/python-sdk（github）
[^19]: modelcontextprotocol/typescript-sdk — official TypeScript SDK for MCP servers and clients — https://github.com/modelcontextprotocol/typescript-sdk（github）
[^20]: hangwin/mcp-chrome — Chrome MCP Server（第三方） — https://github.com/hangwin/mcp-chrome（github）
[^21]: mobile-next/mobile-mcp — MCP Server for Mobile Automation（第三方） — https://github.com/mobile-next/mobile-mcp（github）
[^22]: puppetm4st3r/local_function_calling — 第三方 Gorilla OpenFunctions 封装（非官方仓库） — https://github.com/puppetm4st3r/local_function_calling（github）
[^23]: aiopsplus/BFCL-CN — BFCL 中文评测衍生项目（非官方仓库） — https://github.com/aiopsplus/BFCL-CN（github）
[^24]: XueyiC/Llama-Factory-BFCL — BFCL 与 LlamaFactory 集成（非官方仓库） — https://github.com/XueyiC/Llama-Factory-BFCL（github）
[^25]: Shawn-Guo-CN/BFCL_server — BFCL 服务化封装（非官方仓库） — https://github.com/Shawn-Guo-CN/BFCL_server（github）
[^26]: OpenMLRL/LLM_Collab_BFCL — BFCL v4 多模型协作实验（非官方仓库） — https://github.com/OpenMLRL/LLM_Collab_BFCL（github）
[^28]: aaronlyt/toolsynth — 工具调用训练数据合成框架（第三方） — https://github.com/aaronlyt/toolsynth（github）
[^29]: AIwithhassan/ai-agents-from-scratch — 教学型 agent 循环示例（第三方，小型） — https://github.com/AIwithhassan/ai-agents-from-scratch（github）
[^30]: ToolRL: Reward is All Tool Learning Needs — https://arxiv.org/abs/2504.13958（arxiv, 2025-04-16）
[^31]: Small Language Models are the Future of Agentic AI — https://arxiv.org/abs/2506.02153（arxiv, 2025-06-02）
[^32]: MCP-Universe: Benchmarking Large Language Models with Real-World Model Context Protocol Servers — https://arxiv.org/abs/2508.14704（arxiv, 2025-08-20）
[^33]: MCP-Universe (HTML 正文) — 任务形式化与基准对比表 — https://arxiv.org/html/2508.14704v1（arxiv, 2025-08-20）
[^34]: LiveMCP-101: Stress Testing and Diagnosing MCP-enabled Agents on Challenging Queries — https://arxiv.org/abs/2508.15760（arxiv, 2025-08-21）
[^35]: LiveMCPBench: Can Agents Navigate an Ocean of MCP Tools? — https://arxiv.org/abs/2508.01780（arxiv, 2025-08-03）
[^36]: Evaluation Report on MCP Servers (MCPBench) — https://arxiv.org/abs/2504.11094（arxiv, 2025-04-15）
[^37]: Berkeley Function Calling Leaderboard (BFCL) V4 — 官方榜单页面 — https://gorilla.cs.berkeley.edu/leaderboard.html（web, 2026-04-12）
[^38]: gorilla / berkeley-function-call-leaderboard — CHANGELOG.md — https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/CHANGELOG.md（github）
[^39]: gorilla / berkeley-function-call-leaderboard — README.md — https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/README.md（github）
[^40]: openai-python — api.md (API 类型与方法索引) — https://raw.githubusercontent.com/openai/openai-python/main/api.md（github）
[^41]: openai-python — README (Responses API 与 Chat Completions 用法) — https://raw.githubusercontent.com/openai/openai-python/main/README.md（github）
[^42]: googleapis/python-genai — README (Google Gen AI SDK) — https://raw.githubusercontent.com/googleapis/python-genai/main/README.md（github）
[^43]: Anthropic 官方文档站点访问结果（docs.anthropic.com / docs.claude.com） — https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview（web）
[^44]: Qwen3 Technical Report (arXiv:2505.09388v1) — https://arxiv.org/html/2505.09388v1（arxiv, 2025-05-14）
[^45]: Qwen3 Technical Report — arXiv 摘要页（元数据） — https://arxiv.org/abs/2505.09388v1（arxiv, 2025-05-14）
[^46]: vLLM 官方文档：Tool Calling（渲染页） — https://docs.vllm.ai/en/latest/features/tool_calling.html（web）
[^47]: vLLM 仓库文档源文件 docs/features/tool_calling.md（main 分支） — https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/tool_calling.md（github）
[^48]: vLLM 官方文档：Structured Outputs — https://docs.vllm.ai/en/latest/features/structured_outputs.html（web）
[^49]: SGLang 官方文档：Tool Parser（Function Calling） — https://docs.sglang.ai/advanced_features/tool_parser.html（web）
[^51]: langchain-ai/langgraph — README（Low-level orchestration framework for building stateful agents） — https://github.com/langchain-ai/langgraph（github）
[^52]: microsoft/autogen — README（A framework for creating multi-agent AI applications） — https://github.com/microsoft/autogen（github）
[^53]: run-llama/llama_index — README（LlamaIndex Framework） — https://github.com/run-llama/llama_index（github）
[^54]: crewAIInc/crewAI — README（Fast and Flexible Multi-Agent Automation Framework） — https://github.com/crewAIInc/crewAI（github）
[^55]: openai-python: FunctionDefinition TypedDict (generated from OpenAI OpenAPI spec) — https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/shared_params/function_definition.py（github）
[^56]: openai-python: ChatCompletionToolChoiceOptionParam type alias — https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/chat/chat_completion_tool_choice_option_param.py（github）
[^57]: anthropic-sdk-python: ToolParam TypedDict (tool definition schema) — https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/src/anthropic/types/tool_param.py（github）
[^58]: openai-cookbook: Function-calling with an OpenAPI specification — https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/Function_calling_with_an_OpenAPI_spec.ipynb（github）