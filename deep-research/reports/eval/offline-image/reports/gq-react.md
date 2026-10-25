## 摘要

ReAct 的核心思想是让同一个语言模型以交错（interleaved）方式同时生成 reasoning traces（推理轨迹）与 task-specific actions（任务特定动作），使推理与行动互为支撑[^1]。它与纯 Chain-of-Thought 的关键区别不在"是否推理"，而在三处结构性差异：输出空间（单条思维链 vs. thought–action–observation 交错轨迹）、是否接地于外部环境（参数内部知识 vs. 显式调用外部来源）、以及失败模式的方向（未接地导致幻觉传播 vs. 缺乏推理导致无法收敛出答案）[^4][^5][^6]。需要前置声明的是：本报告所依据的知识库对 ReAct 的量化对比覆盖不完整——HotpotQA/Fever 上 ReAct vs CoT 的具体 EM/准确率数字缺失，ALFWorld/WebShop 的 34%/10% 提升仅有摘要级数字、缺基线与模型设定[^3][^9][^13]；ReAct 自身的失败模式（循环、无法回溯）在库内只有定性描述，无错误率统计[^6][^14]。

---

## 一、ReAct 的核心思想

### 1.1 出发点：reasoning 与 acting 的领域分工

ReAct 论文的出发点是一个领域分工现象：LLM 的 reasoning 能力（以 chain-of-thought prompting 为代表）与 acting 能力（以 action plan generation 为代表）此前被作为两个独立课题研究[^1]。ReAct 的问题意识即来自这一割裂——能否让两者在同一条生成轨迹中互相支撑。

### 1.2 核心机制：交错生成 reasoning traces 与 actions

ReAct 的核心思想是让同一个语言模型以交错（interleaved）的方式同时生成 reasoning traces（推理轨迹）与 task-specific actions（任务特定动作）[^1][^10]。两类输出的分工是明确的：

- **reasoning traces 的作用**：帮助模型诱导（induce）、跟踪（track）、更新（update）行动计划，并处理异常情况[^1][^10]。
- **actions 的作用**：让模型与外部来源（知识库或环境）交互，以收集额外信息[^1][^10]。

官方项目页给出 ReAct 轨迹的结构定义：reasoning trace（thought）与 action 交错，并由环境返回 observation 响应 action[^1]。因此 ReAct 的单步不是"想一步"，而是"想一步 → 做一个动作 → 收到环境反馈"的完整闭环[^1]。LangChain 官方维护的 ReAct agent 模板把这一循环写为：接收用户 query → 推理并决定动作 → 用可用工具执行动作 → 观察动作结果 → 重复第 2–4 步直到能给出最终答案[^21]。

### 1.3 Prompt 构成与"何时想、何时做"的自主权

ReAct prompt 由 few-shot task-solving trajectories 组成，其中包含人工书写的 text reasoning traces 与 actions，以及针对 actions 的环境 observations[^2]。关键设计选择在于：针对决策类任务，作者设计的是带稀疏 reasoning traces 的人类轨迹，让 LM 自行决定何时 think、何时 act[^2]。也就是说，推理的发生时机由模型在轨迹中自行判断，而非由提示强制每步都推理[^2]。这与"每步必须输出思维链"的 CoT 提示在控制流上有本质不同。

### 1.4 实验证据及其强度边界

ReAct 论文报告两组实验：知识密集型任务与交互式决策基准[^12]。

- **知识密集型任务**：在 HotpotQA 与 Fever 上，ReAct 通过与简单 Wikipedia API 交互，克服了 chain-of-thought reasoning 中普遍存在的幻觉与错误传播问题[^3][^12]。
- **交互式决策基准**：在 ALFWorld 与 WebShop 上，ReAct 相比模仿学习与强化学习方法分别取得 34% 与 10% 的绝对成功率提升[^3][^12]。
- 上述实验仅用一到两个 in-context 示例进行提示[^3][^12]。

证据强度必须如实标注：HotpotQA/Fever 部分缺少原始论文 Table 1/2 的具体 EM/准确率数字（PaLM-540B 下 ReAct vs CoT vs CoT-SC 的对比）；ALFWorld/WebShop 部分缺少具体成功率数值与基线名称，也未说明是 PaLM-540B 还是 GPT-3 设定[^3][^12]。34% 与 10% 来自论文摘要，在给定证据中无法被独立验证[^9]。

---

## 二、纯 Chain-of-Thought 的定义与边界

标准 CoT 只生成一条思维链——一系列中间推理步骤——作为 few-shot 提示中的示范，模型全程不与外部环境交互[^4]。其解码方式是朴素贪心解码（naive greedy decoding），即单条推理路径、单次前向生成[^15]。

CoT 家族已有多个变体，但边界清晰：Self-Consistency 采样一组多样化的推理路径而非只取贪心那一条，再对答案做边缘化投票，其直觉是复杂推理问题通常存在多条通往唯一正确答案的思考路径[^15]；相对标准 CoT 的提升为 GSM8K +17.9%、SVAMP +11.0%、AQuA +12.2%、StrategyQA +6.4%、ARC-challenge +3.9%[^15]。Plan-and-Solve 则指出 Zero-shot-CoT 存在三类缺陷——计算错误、缺失步骤错误、语义误解错误[^15]。

关键结论：所有 CoT 变体（zero-shot、self-consistency、plan-and-solve、ToT）的输出空间始终是文本推理步骤，没有任何一个变体引入"动作触发环境观测"这一环节[^4]。这是 ReAct 与整个 CoT 家族的分界线，而非仅与原始 CoT 的分界线。

---

## 三、ReAct 与纯 CoT 的四个关键区别

### 3.1 输出空间与循环结构

CoT 的输出是一条思维链，一次性生成后即给出答案；ReAct 的输出是 reasoning trace 与 action 交错、并由环境返回 observation 响应 action 的结构[^4]。工程化表述：CoT 是"单次前向的文本生成"，ReAct 是"生成 → 执行 → 观测 → 再生成"的循环[^4]。LangChain 官方对 agent 的定义与此同构——agent 是一个在循环中调用工具直到任务完成的模型（a model calling tools in a loop until a given task is complete）[^22]；其文档把该循环拆为 Model call（带提示词与可用工具调用 LLM，返回响应或工具执行请求）与 Tool execution（执行工具并返回结果）两步[^24]。

### 3.2 是否接地于外部环境

项目页的三方对比给出了最清晰的定位：reason-only baseline（即 chain-of-thought）因未接地于外部环境而受错误信息影响；act-only baseline 因缺乏推理而无法综合出最终答案；ReAct 以可解释且事实性的轨迹解决任务[^5]。论文侧对应表述为 ReAct 通过与简单 Wikipedia API 交互，克服了 chain-of-thought reasoning 中普遍存在的幻觉与错误传播问题[^5]。

机制层面：CoT 的知识来源完全在模型参数内部，推理链条再长也不引入新信息；ReAct 的 action 显式地把模型接到外部来源（知识库或环境）上以收集额外信息[^5]。这是"链条更长"与"链条更接地"的区别，不是同一维度上的程度差异。

### 3.3 失败模式的方向相反

CoT 侧失败模式：未接地导致幻觉与错误传播，且推理链条中的错误会沿链条传播[^6][^14]。act-only 侧失败模式：缺乏推理导致无法综合出最终答案[^6][^14]。ReAct 侧：项目页给出 ALFWorld 上的失败示例，承认 ReAct 并非完美[^6][^14]。

必须声明：关于 ReAct 自身的失败模式（推理灵活性不足、陷入循环、无法回溯），给定证据中只有项目页"并非完美 + 一个 ALFWorld 失败示例"这一句定性描述，缺少论文中的具体失败案例与错误率统计[^6][^14]。因此"失败模式方向相反"这一结论在 CoT 与 act-only 两侧有据，在 ReAct 一侧只能表述为"并非无失败"，无法给出实证量化对比[^14]。

### 3.4 可解释性与可纠正性的形态不同

CoT 的思维链本身可读，但它是纯文本的推理叙述，缺乏与外部世界交互的锚点[^7]。ReAct 的轨迹包含 thought、action、observation 三类元素，人工检查时不仅能读到模型"怎么想"，还能读到模型"做了什么"以及环境"返回了什么"[^7]。项目页主张 ReAct 格式便于人工检查，并可通过修改少量 model thoughts 进行行为纠正[^7]。注意这是作者主张；"可解释且事实性的轨迹"中的"事实性"部分，证据基础是 HotpotQA/Fever 上的定性结论，缺少细粒度的事实性错误率统计，只能视为作者主张而非已量化的结论[^13]。

---

## 四、工程生态中的位置：命名与循环语义的分离

ReAct 的"推理-行动交错 + 环境观测"循环结构在当代 agent 框架中普遍存在：LangChain 的 basic agent loop、LlamaIndex 的 agent 循环、OpenAI Agents SDK 的 agent loop、AutoGen 的工具调用序列都是同构的[^39]。但最直接以 ReAct 命名的预构建组件 create_react_agent 已被官方弃用：LangGraph v1 发布说明设有专门小节「Deprecation of create_react_agent」，官方原文称该预构建组件已被弃用，转而推荐 LangChain 的 create_agent，理由是 create_agent 提供更简单的接口，并通过引入 middleware 提供更强的定制能力[^23][^39]。LlamaIndex 默认 agent 类型为 FunctionAgent（使用 LLM provider 的 function/tool calling 能力执行工具），ReActAgent 与 CodeActAgent 被列为"使用不同 prompting 策略执行工具的"其他类型 agent，而非默认类型[^25]；OpenAI Agents SDK 与 AutoGen 的 agent loop 文档均未将其命名为 ReAct[^39]。

这两组证据构成一个未调和的矛盾：报告一方面记录循环结构普遍同构，另一方面记录 ReAct 命名的组件被弃用[^9][^39]。可接受的调和方向是——**循环结构同构 ≠ ReAct 命名或 ReAct 提示范式仍是默认**；弃用针对的是预构建组件 API，而非循环语义本身。据此不能断言 ReAct 仍是当前框架的默认命名范式或默认实现[^39]。

模型侧文档进一步显示，与 ReAct 循环同构的交互正以 tool call / tool output 的术语体系被规范化，而不使用 ReAct 命名：OpenAI 官方文档把 function calling（亦称 tool calling）定义为"让模型与外部系统交互、访问训练数据之外数据的机制"，术语分三层——Tools（以 JSON schema 定义的功能）、Tool calls（模型判断需调用工具时返回的特殊响应，含参数）、Tool call outputs（应用侧执行后回传，须含对特定工具调用的引用）[^36]。Anthropic 官方文档规定 Claude 可能在单次响应中调用多个工具，stop_reason 为 tool_use，单个 assistant turn 中可包含多个 tool_use block，且对每个 tool_use block 必须返回一个 tool_result、用 tool_use_id 匹配[^37]。该官方文档片段未出现 ReAct、thought 或 observation 字样[^37]。

官方仓库侧：ysymyth/ReAct 明确标注为 ICLR 2023 论文的 GPT-3 prompting 代码，覆盖 hotpotqa、fever、alfworld、webshop 四个 notebook，元数据为 4224 stars、404 forks、MIT 许可[^8][^27]；README 建议用户如需将 ReAct 用于更多任务，可尝试 LangChain 的 zero-shot ReAct Agent[^27]。作为量级参照，langchain-ai/langchain 为 147842 stars、langchain-ai/langgraph 为 43254 stars、LlamaIndex 为 52639 stars（抓取时点 2026-10-14 / 2026-10-23）[^32][^26]。这些是零散快照，不构成框架间 ReAct 主张的横向对比——各实现的默认参数配置（迭代轮数、top-k、模型选择）现有公开资料不足以支撑横向对比表[^33]。

---

## 五、ReAct 之后：训练侧改进与循环的成本问题

库内可确认的后续工作方向是训练侧改进，而非提示侧改良。Search-R1 指出，用提示让具备推理能力的 LLM 在推理时使用搜索引擎往往次优，因为 LLM 可能并不完全具备"如何最优地与搜索引擎交互"的能力；它通过 RL 让 LLM 在逐步推理过程中自主学习生成搜索查询并进行实时检索，在 7 个 QA 数据集上相比同设定 RAG 基线，Qwen2.5-7B 提升 41%、Qwen2.5-3B 提升 20%[^20]。ToolRL 指出当前 LLM 常通过 SFT 获取工具使用能力，但 SFT 难以泛化到不熟悉或复杂的工具使用场景，转而用 GRPO 训练并设计工具使用任务的奖励，实验显示相比基座模型提升 17%、相比 SFT 模型提升 15%[^20]。这些工作可视为对 ReAct 式"推理中调用工具"范式的训练侧改进，但它们本身不再以 ReAct 命名[^20]；给定证据未说明二者与 ReAct prompting 范式之间的直接继承或替代关系，也未给出在相同基准上的对照数字[^20]。

循环结构的成本侧压力有独立证据。Anthropic 明确指出"自然语言工具调用每次调用都需要一次完整推理，中间结果无论是否有用都会堆积在上下文中"，并以此作为 Programmatic Tool Calling 的动机[^38]；工具定义本身也消耗大量上下文——五 server 示例中 58 个工具约 55K tokens，Anthropic 内部曾见到工具定义消耗 134K tokens[^38]。Anthropic 的选型建议是：构建 LLM 应用时先找最简单的方案，只在必要时才增加复杂度——这可能意味着根本不构建 agentic 系统；agentic 系统通常以延迟和成本换取更好的任务表现；对许多应用而言，用检索和 in-context 示例优化单次 LLM 调用通常就足够了[^41]。该建议未点名 ReAct，体现的是对多轮 agent 循环的成本审慎态度[^41]。

---

## 矛盾与不确定性

**1. 量化对比缺失。** HotpotQA/Fever 上 ReAct 与 CoT 的量化对比在库内缺失，论文侧仅有定性转述，因此本报告只能陈述定性结论[^9][^13]。ALFWorld/WebShop 的 34% 与 10% 绝对成功率提升来自论文摘要，缺少具体成功率数值与基线名称，在给定证据中无法被独立验证[^9][^13]。

**2. ReAct 失败模式缺乏实证。** 关于推理灵活性不足、陷入循环、无法回溯，给定证据中只有项目页"并非完美 + 一个 ALFWorld 失败示例"这一句定性描述，缺少论文中的具体失败案例与错误率统计[^6][^14]。因此"失败模式方向相反"在 ReAct 一侧无法给出实证量化对比[^14]。

**3. ReAct 后续实证研究（2023–2026）在库内无覆盖。** 针对 ReAct 的长程轨迹错误累积率、循环率、回溯失败率，以及 ReAct vs CoT-SC vs Reflexion 的同基准对照数字，当前知识库未收录该信息。库内邻近证据均不能直接替代：Agentic RAG 的 POMDP 形式化与 compounding hallucination propagation、retrieval thrash、tool storms 等失败模式，其对象是 agentic RAG 而非 ReAct 论文的后续研究[^17][^18]；上下文工程的四类失效（Context Poisoning / Distraction / Confusion / Clash）对象是上下文长度[^19]；Reflexion 仅有 HumanEval 上 91% pass@1 对 GPT-4 80% 的数字，无与 ReAct 的对照[^16]。因此本报告不对"ReAct 长程轨迹的收益来源与失效模式"给出结论性判断——基于邻近领域证据的推断不构成 ReAct 后续研究的结论。

**4. 命名与循环语义的矛盾已按调和方向处理。** 循环结构同构与 create_react_agent 被弃用两条证据，本报告按"循环结构同构 ≠ ReAct 命名/提示范式仍是默认；弃用针对预构建组件 API 而非循环语义"处理，并据此得出"不能断言 ReAct 仍是当前框架的默认命名范式或默认实现"的结论[^39]。这一调和是结论陈述，不是开放问题。

**5. 生态热度对比不可行。** AutoGPT 与 OpenHands 的 stars/forks/最近提交在库内明确缺失[^32]；OpenHands README 已改版为 Agent Canvas 产品介绍，不再含可对比的 ReAct 主张[^34]；跨实现的默认参数横向对比被明确判定为证据不足，CrewAI、AutoGen/MAF、Dify 均无默认迭代轮数/top-k/模型配置证据[^33]。库内可用的只有零散单点参数，如 Haystack 的 max_agent_steps 与 exit_conditions[^29]、Semantic Kernel auto function calling 的最大迭代次数[^30]，不足以支撑框架间差异对比。

**6. 模型侧官方文档未对 ReAct 表态。** OpenAI o1 System Card 描述 o1 系列为"用大规模强化学习训练以使用思维链推理"，回答前会"produce a long chain of thought before responding to the user"，但该片段未提及 ReAct，也未对 ReAct 式提示作出任何推荐或不推荐的表态[^35]。Anthropic 把 context engineering 定位为 prompt engineering 的"自然演进"而非替代，该定位针对提示工程整体，不能外推为"Anthropic 仍推荐 ReAct 提示"[^40]。

## 参考来源

[^1]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（报告正文：摘要 / 一、核心思想 / 1.2 核心机制） — reports/eval/reports/gq-react.md · ## 摘要 L3-8；## 一、ReAct 的核心思想 L9-10；### 1.2 核心机制 L15-22（kb）
[^2]: gq-react.md · 1.3 Prompt 构成与「何时想、何时做」的自主权 — reports/eval/reports/gq-react.md · ### 1.3 Prompt 构成与「何时想、何时做」的自主权 L24-27（kb）
[^3]: gq-react.md · 1.4 实验证据（HotpotQA/Fever 与 ALFWorld/WebShop 两组实验） — reports/eval/reports/gq-react.md · ### 1.4 实验证据 L28-36（kb）
[^4]: gq-react.md · 三、ReAct 与纯 CoT 的关键区别（3.1 输出空间与循环结构） — reports/eval/reports/gq-react.md · ## 三、ReAct 与纯 CoT 的关键区别 L60-66（kb）
[^5]: gq-react.md · 3.2 区别二：是否接地于外部环境（含 reason-only / act-only / ReAct 三方对比） — reports/eval/reports/gq-react.md · ### 3.2 区别二：是否接地于外部环境 L68-73（kb）
[^6]: gq-react.md · 3.3 区别三：失败模式的方向相反 — reports/eval/reports/gq-react.md · ### 3.3 区别三：失败模式的方向相反 L74-82（kb）
[^7]: gq-react.md · 3.4 区别四：可解释性与可纠正性的形态不同 — reports/eval/reports/gq-react.md · ### 3.4 区别四：可解释性与可纠正性的形态不同 L84-87（kb）
[^8]: gq-react.md · 四、ReAct 在工程生态中的位置（官方代码仓库范围与元数据） — reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置 L101-107（kb）
[^9]: gq-react.md · 五、矛盾与不确定性（量化对比缺失的自述） — reports/eval/reports/gq-react.md · ## 五、矛盾与不确定性 L120-126（kb）
[^10]: LLM 工具调用（Function Calling）的训练、对齐与评测 · 2.3 工具学习范式的系统化（ReAct 数字的独立复述） — reports/eval/reports/gq-tool-use.md · ### 2.3 工具学习范式的系统化 L41-46（kb）
[^12]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（1.4 实验证据） — reports/eval/reports/gq-react.md · ### 1.4 实验证据 · L28-35（kb）
[^13]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（五、矛盾与不确定性） — reports/eval/reports/gq-react.md · ## 五、矛盾与不确定性 · L120-130（kb）
[^14]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（3.3 区别三：失败模式的方向相反） — reports/eval/reports/gq-react.md · ### 3.3 区别三：失败模式的方向相反 · L74-83（kb）
[^15]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（二、纯 Chain-of-Thought 的定义与边界） — reports/eval/reports/gq-react.md · ## 二、纯 Chain-of-Thought 的定义与边界 · L43-53（kb）
[^16]: 多智能体系统（Multi-Agent LLM）主流协作模式与适用场景（1.6 反思/进化机制） — reports/eval/reports/gq-multi-agent.md · ### 1.6 反思/进化机制（横切能力） · L41-46（kb）
[^17]: Agentic RAG：定义、架构关键差异与工程现状（3.6 评估对象扩展到轨迹与成本） — reports/eval/reports/gq-agentic-rag.md · ### 3.6 差异六：评估对象从"输出质量"扩展到"轨迹与成本" · L101-108（kb）
[^18]: Agentic RAG：定义、架构关键差异与工程现状（3.5 失败模式的位置发生迁移） — reports/eval/reports/gq-agentic-rag.md · ### 3.5 差异五：失败模式的位置发生迁移 · L93-100（kb）
[^19]: 上下文工程（Context Engineering）：定义、与提示词工程的关系与差异（1.3 上下文是有限资源） — reports/eval/reports/gq-context-eng.md · ### 1.3 为什么需要它：上下文是有限资源 · L28-46（kb）
[^20]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（五、矛盾与不确定性 + 参考来源） — reports/eval/reports/gq-react.md · ## 五、矛盾与不确定性 · L120-122 与 ## 参考来源 · L134-139（kb）
[^21]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（LangChain ReAct agent 模板循环结构） — reports/eval/reports/gq-react.md · ### 3.2 区别二：是否接地于外部环境 · L68-70（kb）
[^22]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（LangChain 对 agent 的定义） — reports/eval/reports/gq-react.md · ### 3.1 区别一：输出空间与循环结构 · L62-67（kb）
[^23]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（LangGraph v1 弃用 create_react_agent） — reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置：一个未调和的矛盾 · L101-110（kb）
[^24]: 上下文工程（Context Engineering）：定义、与提示词工程的关系与差异（LangChain agent 循环两步拆解） — reports/eval/reports/gq-context-eng.md · ### 3.2 时间性：离散任务 vs 迭代过程 · L87-93（kb）
[^25]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（LlamaIndex 默认 FunctionAgent） — reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置：一个未调和的矛盾 · L101-114（kb）
[^26]: 长上下文是否让 RAG 过时（LlamaIndex 仓库热度元数据） — reports/eval/reports/gq-long-context.md · ## 四、长上下文是否让 RAG 过时 · L39-49（kb）
[^27]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（ysymyth/ReAct 官方仓库元数据与 README 建议） — reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置：一个未调和的矛盾 · L101-107（kb）
[^29]: Agentic RAG：定义、架构关键差异与工程现状（Haystack Agent 的 max_agent_steps 与退出原因） — reports/eval/reports/gq-agentic-rag.md · ### 5.2 跨实现的默认参数横向对比不可行 · L153-161（kb）
[^30]: Human-in-the-loop 在 LLM Agent 系统中的主流实现模式与适用场景（Semantic Kernel auto function calling 最大迭代次数） — reports/eval/reports/gq-human-in-loop.md · ## 八、辅助机制：Filter 作为审批前置控制点 · L113-122（kb）
[^32]: Agentic RAG：定义、架构关键差异与工程现状（LangChain / LangGraph 仓库 stars 与活跃度） — reports/eval/reports/gq-agentic-rag.md · ### 4.5 生态热度参考 · L133-138（kb）
[^33]: Agentic RAG：定义、架构关键差异与工程现状（跨实现默认参数横向对比不可行） — reports/eval/reports/gq-agentic-rag.md · ### 5.2 跨实现的默认参数横向对比不可行 · L153-164（kb）
[^34]: 自动修 Bug 的 Coding Agent 在 SWE-bench 上的技术水平（OpenHands 仓库 README 已改版） — reports/eval/reports/gq-swe-agent.md · ### 5.3 OpenHands 成绩缺失 · L99-101（kb）
[^35]: 推理模型的 RL Scaling 与预训练 Scaling 的关系，及 Test-Time Compute 的重要性（含 OpenAI o1 System Card 官方表述转述） — reports/eval/reports/gq-rl-scaling.md · ### 1.2 推理模型的 RL 路线：官方表述 · L26-28（kb, 2024-12-05）
[^36]: LLM 工具调用（Function Calling）的训练、对齐与评测（含 OpenAI 官方 function calling 文档转述） — reports/eval/reports/gq-tool-use.md · ### 1.1 官方定义与术语分层 · L11-19（kb）
[^37]: LLM 工具调用（Function Calling）的训练、对齐与评测（含 Anthropic 官方工具调用文档转述） — reports/eval/reports/gq-tool-use.md · ### 1.2 并行调用与执行语义 · L21-24（kb）
[^38]: LLM 工具调用（Function Calling）的训练、对齐与评测（含 Anthropic 2025-11-24 三项 beta 特性转述） — reports/eval/reports/gq-tool-use.md · ### 8.2 上下文成本：工具定义的隐性开销 · L169-173（kb, 2025-11-24）
[^39]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（第四节：ReAct 在工程生态中的位置） — reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置：一个未调和的矛盾 · L101-116（kb）
[^40]: 上下文工程（Context Engineering）：定义、与提示词工程的关系与差异（Anthropic 官方「演进而非替代」定位） — reports/eval/reports/gq-context-eng.md · ### 2.1 官方定位：演进，而非替代 · L49-53（kb）
[^41]: 多智能体系统（Multi-Agent LLM）主流协作模式与适用场景（Anthropic 官方 agent 选型建议） — reports/eval/reports/gq-multi-agent.md · ## 三、多智能体何时失效或不如单智能体 · L71-72（kb）