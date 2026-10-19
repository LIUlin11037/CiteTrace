# ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别

## 摘要

ReAct 的核心思想是让同一个语言模型以**交错（interleaved）**的方式同时生成 reasoning traces（推理轨迹）与 task-specific actions（任务特定动作），使推理与行动互为支撑：推理帮助模型诱导、跟踪、更新行动计划并处理异常，动作则让模型与外部来源（知识库或环境）交互以收集额外信息[^1]。纯 Chain-of-Thought（CoT）则只生成一条思维链——一系列中间推理步骤——作为 few-shot 提示中的示范，模型全程不与外部环境交互[^3]。二者的关键区别可归纳为三点：(1) **输出空间**：CoT 只产出自然语言推理步骤，ReAct 产出「thought → action → observation」循环，动作会触发环境返回观测[^2]；(2) **接地方式**：CoT 完全依赖模型参数内的知识，ReAct 通过动作把推理锚定到外部知识库或环境[^1][^2]；(3) **失败模式**：CoT 因未接地而受幻觉与错误传播影响，纯 act-only 基线则因缺乏推理而无法综合出最终答案[^2]。需要指出的是，ReAct 是否仍是当前主流框架的默认 agent 范式存在明确矛盾：LangGraph v1 已弃用 `create_react_agent`，LangChain v1 以 `create_agent` 为新标准[^23][^24]，LlamaIndex 默认 agent 类型为 FunctionAgent 而非 ReActAgent[^7]，OpenAI Agents SDK 与 AutoGen 的 agent loop 文档均未将其命名为 ReAct[^27][^29]。

---

## 一、ReAct 的核心思想

### 1.1 问题起点：reasoning 与 acting 的割裂

ReAct 论文的出发点是一个领域分工现象：LLM 的 reasoning 能力（以 chain-of-thought prompting 为代表）与 acting 能力（以 action plan generation 为代表）此前被作为两个独立课题研究[^1]。官方项目页把这一割裂描述得更具体：语言模型在 reasoning（如 chain-of-thought prompting）与 acting（如 WebGPT、SayCan、ACT-1）两方面都在进步，但这两个方向此前保持分离，ReAct 提出将这两种基础能力结合[^2]。

### 1.2 核心机制：交错生成 reasoning traces 与 actions

ReAct 的做法是让 LLM 以交错方式同时生成 reasoning traces 与 task-specific actions[^1]。其协同机制是双向的：

- **reasoning traces 的作用**：帮助模型诱导（induce）、跟踪（track）、更新（update）行动计划，并处理异常情况[^1]。
- **actions 的作用**：让模型与外部来源（知识库或环境）交互，以收集额外信息[^1]。

项目页进一步给出了 ReAct 轨迹的结构定义：reasoning trace（thought）与 action 交错，并由环境返回 observation 响应 action[^2]。也就是说，ReAct 的单步不是「想一步」，而是「想一步 → 做一个动作 → 收到环境反馈」的完整闭环。

### 1.3 Prompt 构成与「何时想、何时做」的自主权

ReAct prompt 由 few-shot task-solving trajectories 组成，其中包含人工书写的 text reasoning traces 与 actions，以及针对 actions 的环境 observations[^2]。针对决策类任务，作者设计的是**带稀疏 reasoning traces 的人类轨迹**，让 LM 自行决定何时 think、何时 act[^2]。这一点是 ReAct 与「固定模板式推理」的重要分野：推理的发生时机由模型在轨迹中自行判断，而非由提示强制每步都推理。

### 1.4 实验证据

论文报告了两组实验：

- **知识密集型任务**：在 HotpotQA 与 Fever 上，ReAct 通过与简单 Wikipedia API 交互，克服了 chain-of-thought reasoning 中普遍存在的幻觉与错误传播问题[^1]。
- **交互式决策基准**：在两个交互式决策基准 ALFWorld 与 WebShop 上，ReAct 相比模仿学习与强化学习方法分别取得 34% 与 10% 的绝对成功率提升，且仅用一到两个 in-context 示例进行提示[^1]。

需要说明的是，上述两组结论在给定证据中均为定性转述或摘要级数字：HotpotQA/Fever 部分缺少原始论文 Table 1/2 的具体 EM/准确率数字（PaLM-540B 下 ReAct vs CoT vs CoT-SC 的对比），ALFWorld/WebShop 部分缺少具体成功率数值与基线名称，也未说明是 PaLM-540B 还是 GPT-3 设定，因此无法在本文中给出可交叉验证的量化对比（详见「矛盾与不确定性」）。

### 1.5 可解释性与可纠正性

项目页对 ReAct 的一个附加主张是：ReAct 以可解释且事实性的轨迹解决任务[^2]。同时项目页明确承认 ReAct 并非完美，给出了 ALFWorld 上的失败示例，但指出 ReAct 格式便于人工检查，并可通过修改少量 model thoughts 进行行为纠正[^2]。这构成了 ReAct 相对纯端到端方法的一个工程性优势：轨迹本身是可读的中间产物。

---

## 二、纯 Chain-of-Thought 的定义与边界

要谈「关键区别」，必须先完整写出 CoT 的标准定义。

**Chain-of-Thought（CoT）的定义**：CoT 是生成一条思维链，即**一系列中间推理步骤**（a series of intermediate reasoning steps）[^3]。其实现方式是 few-shot prompting——在提示中提供少量思维链示范作为示例（a few chain of thought demonstrations are provided as exemplars in prompting）[^3]。该推理能力在足够大的语言模型中自然涌现（emerge naturally in sufficiently large language models），即存在模型规模门槛[^3]。实验覆盖算术（arithmetic）、常识（commonsense）、符号推理（symbolic reasoning）三类任务；用 8 条 CoT 示例提示 540B 参数模型，在 GSM8K 数学应用题基准上达到 SOTA 准确率，超过带 verifier 的微调 GPT-3[^3]。

CoT 的后续变体进一步刻画了它的边界：

- **Zero-shot CoT**：在每次回答前简单加上 "Let's think step by step"，无需任何手工构造的 few-shot 示例；单一提示模板即可跨任务泛化[^4]。在 text-davinci-002 上，MultiArith 准确率从 17.7% 提升到 78.7%，GSM8K 从 10.4% 提升到 40.7%[^4]。该论文把 CoT 所针对的任务描述为不遵循标准 scaling law 的困难 System-2 任务[^4]。
- **Self-Consistency**：标准 CoT 使用的是朴素贪心解码（naive greedy decoding），即单条推理路径、单次前向生成；self-consistency 采样一组多样化的推理路径而非只取贪心那一条，再对答案做边缘化投票[^5]。相对标准 CoT 的提升为 GSM8K +17.9%、SVAMP +11.0%、AQuA +12.2%、StrategyQA +6.4%、ARC-challenge +3.9%[^5]。其直觉是复杂推理问题通常存在多条通往唯一正确答案的思考路径[^5]。
- **Plan-and-Solve**：指出 Zero-shot-CoT 存在三类缺陷——计算错误、缺失步骤错误、语义误解错误；PS Prompting 分两步，先制定计划将整个任务分解为更小的子任务，再按计划执行子任务[^16]。
- **Tree of Thoughts**：批评 LM 推理被限制在 token 级、从左到右的决策过程，在需要探索、策略性前瞻或初始决策关键的任务上表现不足；ToT 泛化 CoT，允许在「连贯的文本单元（thoughts）」上进行探索，支持多路径推理、自我评估、前瞻与回溯[^15]。

**关键观察**：以上所有 CoT 变体——无论 zero-shot、self-consistency、plan-and-solve 还是 ToT——其输出空间始终是**文本推理步骤**，没有任何一个变体引入「动作触发环境观测」这一环节。这正是 ReAct 与整个 CoT 家族的分界线。

---

## 三、ReAct 与纯 CoT 的关键区别

### 3.1 区别一：输出空间与循环结构

CoT 的输出是一条思维链，即一系列中间推理步骤，一次性生成后即给出答案[^3]。ReAct 的输出是 reasoning trace 与 action 交错、并由环境返回 observation 响应 action 的结构[^2]。用工程语言表述：CoT 是「单次前向的文本生成」，ReAct 是「生成 → 执行 → 观测 → 再生成」的循环。

这一区别在框架实现中被反复复现。LangChain 官方维护的 ReAct agent 模板把循环写为：接收用户 query → 推理并决定动作 → 用可用工具执行动作 → 观察动作结果 → 重复第 2-4 步直到能给出最终答案[^6]。LlamaIndex 对 agent 的描述同样是「给定任务与一组工具，选择最佳可用工具完成每一步；每步完成后 agent 判断任务是否完成——完成则返回结果，未完成则循环回到起点」[^28]。LangChain 官方对 agent 的定义更为凝练：agent 是一个在循环中调用工具直到任务完成的模型（a model calling tools in a loop until a given task is complete）[^25]。这些循环结构与 CoT 的单次生成形成结构性对照。

### 3.2 区别二：是否接地于外部环境

这是 ReAct 论文与项目页都明确强调的区别。项目页的对比说明指出：reason-only baseline（即 chain-of-thought）因未接地于外部环境而受错误信息影响；act-only baseline 因缺乏推理而无法综合出最终答案；ReAct 以可解释且事实性的轨迹解决任务[^2]。论文侧的对应表述是：ReAct 通过与简单 Wikipedia API 交互，克服了 chain-of-thought reasoning 中普遍存在的幻觉与错误传播问题[^1]。

换言之，CoT 的知识来源完全在模型参数内部，推理链条再长也不引入新信息；ReAct 的 action 显式地把模型接到外部来源（知识库或环境）上以收集额外信息[^1]。这也解释了为什么 ReAct 在 HotpotQA、Fever 这类需要事实落地的任务上具有结构性优势。

### 3.3 区别三：失败模式的方向相反

两类方法的失败模式方向相反，这是理解二者互补性的关键：

- **CoT 侧**：未接地导致幻觉与错误传播，且推理链条中的错误会沿链条传播[^1][^2]。此外 Zero-shot-CoT 还存在计算错误、缺失步骤错误、语义误解错误三类缺陷[^16]；标准 CoT 的单路径贪心解码存在明显性能上限，self-consistency 的增益（GSM8K +17.9% 等）正是这一上限的量化体现[^5]。
- **act-only 侧**：缺乏推理导致无法综合出最终答案[^2]。
- **ReAct 侧**：项目页给出 ALFWorld 上的失败示例，承认 ReAct 并非完美[^2]。

需要明确指出：关于 ReAct 自身的失败模式（推理灵活性不足、陷入循环、无法回溯），给定证据中只有项目页「并非完美 + 一个 ALFWorld 失败示例」这一句定性描述，缺少论文中的具体失败案例与错误率统计，因此无法在本文中给出 ReAct 失败模式的实证量化对比（详见「矛盾与不确定性」）。

### 3.4 区别四：可解释性与可纠正性的形态不同

CoT 的思维链本身也是可读的，但它是纯文本的推理叙述，缺乏与外部世界交互的锚点。ReAct 的轨迹包含 thought、action、observation 三类元素[^2]，因此人工检查时不仅能读到模型「怎么想」，还能读到模型「做了什么」以及环境「返回了什么」。项目页据此主张 ReAct 格式便于人工检查，并可通过修改少量 model thoughts 进行行为纠正[^2]。这是 CoT 轨迹所不具备的干预接口。

### 3.5 一张对照表

| 维度 | 纯 Chain-of-Thought | ReAct |
|---|---|---|
| 输出内容 | 一系列中间推理步骤（思维链）[^3] | reasoning trace 与 action 交错，环境返回 observation[^2] |
| 生成结构 | 单次前向、单条推理路径（标准 CoT 为贪心解码）[^5] | 生成 → 执行动作 → 观测 → 再生成的循环[^6][^28] |
| 知识来源 | 模型参数内部 | 通过 action 与外部来源（知识库或环境）交互[^1] |
| 接地性 | 未接地于外部环境[^2] | 接地于外部环境，轨迹被主张为事实性[^2] |
| 主要失败模式 | 幻觉与错误传播；计算/缺步/语义误解错误[^1][^16] | 项目页承认并非完美，给出 ALFWorld 失败示例[^2] |
| 人工干预方式 | 修改提示或采样多条路径投票[^5] | 修改少量 model thoughts 进行行为纠正[^2] |

---

## 四、ReAct 在工程生态中的位置：一个未调和的矛盾

ReAct 论文附带项目站点与代码[^1]，官方 GitHub 仓库 ysymyth/ReAct 明确标注为 ICLR 2023 论文的 GPT-3 prompting 代码，覆盖 hotpotqa、fever、alfworld、webshop 四个 notebook，元数据为 4224 stars、404 forks、MIT 许可[^22]。该仓库 README 建议用户如需将 ReAct 用于更多任务，可尝试 LangChain 的 zero-shot ReAct Agent，这常被解读为「ReAct 已被主流框架吸收为标准 agent 类型」[^22]。

但这一解读与当前官方文档存在直接矛盾：

- **LangGraph v1 发布说明**设有专门小节「Deprecation of create_react_agent」，官方原文为：LangGraph 的 `create_react_agent` 预构建组件已被弃用，转而推荐 LangChain 的 `create_agent`；理由是 `create_agent` 提供更简单的接口，并通过引入 middleware 提供更强的定制能力[^23]。
- **LangChain v1 发布说明**明确：`create_agent` 是 LangChain 中构建 agent 的新标准，取代 `langgraph.prebuilt.create_react_agent`；其底层机制被描述为「basic agent loop」——调用模型 → 让模型选择要执行的工具 → 当模型不再调用工具时结束[^24]。
- **LangChain Agents 文档**当前推荐的构造入口为 `from langchain.agents import create_agent`，文档中未出现 `create_react_agent` 作为推荐用法[^25]。
- **LlamaIndex** 的默认 agent 类型是 FunctionAgent，它使用 LLM provider 的 function/tool calling 能力执行工具；ReActAgent 与 CodeActAgent 被列为「使用不同 prompting 策略执行工具的」其他类型 agent，而非默认类型[^7]。
- **OpenAI Agents SDK** 文档描述的 agent loop 以 LLM 原生 tool calling 与 handoff 为核心控制流，文档未将其命名为 ReAct[^27]。
- **AutoGen AgentChat** 文档示例中 `agent.run()` 返回的消息序列为 TextMessage(user) → ToolCallRequestEvent → ToolCallExecutionEvent → ToolCallSummaryMessage，同样未使用 ReAct 命名[^29]。

因此，可以确认的是：ReAct 的「推理-行动交错 + 环境观测」循环结构在当代 agent 框架中普遍存在（LangChain 的 basic agent loop[^24]、LlamaIndex 的 agent 循环[^28]、OpenAI Agents SDK 的 agent loop[^27]、AutoGen 的工具调用序列[^29] 都是同构的）。但**不能据此断言 ReAct 仍是当前框架的默认命名范式或默认实现**——恰恰相反，最直接以 ReAct 命名的预构建组件 `create_react_agent` 已被官方弃用[^23][^24]。这两条证据之间的矛盾在给定材料中未被调和。

与此同时，ReAct 的思想仍在被后续研究扩展。例如 Search-R1 指出，用提示让具备推理能力的 LLM 在推理时使用搜索引擎往往次优，因为 LLM 可能并不完全具备「如何最优地与搜索引擎交互」的能力；Search-R1 通过 RL 让 LLM 在逐步推理过程中自主学习生成搜索查询并进行实时检索，在 7 个 QA 数据集上相比同设定 RAG 基线，Qwen2.5-7B 提升 41%、Qwen2.5-3B 提升 20%[^17]。ToolRL 则指出当前 LLM 常通过 SFT 获取工具使用能力，但 SFT 难以泛化到不熟悉或复杂的工具使用场景，转而用 GRPO 训练并设计工具使用任务的奖励，实验显示相比基座模型提升 17%、相比 SFT 模型提升 15%[^18]。这些工作可视为对 ReAct 式「推理中调用工具」范式的训练侧改进，但它们本身不再以 ReAct 命名。

---

## 五、矛盾与不确定性

**1. HotpotQA / Fever 上的量化对比缺失。** 论文侧仅有「ReAct 通过与简单 Wikipedia API 交互，克服了 chain-of-thought reasoning 中普遍存在的幻觉与错误传播问题」这一句定性转述[^1]，缺少原始论文 Table 1/2 的具体 EM/准确率数字（PaLM-540B 下 ReAct vs CoT vs CoT-SC 的对比）。因此本文无法给出 ReAct 与 CoT 在知识密集型任务上的量化对比，只能陈述定性结论。

**2. ALFWorld / WebShop 的提升数字无法交叉验证。** 「相比模仿学习与强化学习方法分别取得 34% 与 10% 的绝对成功率提升」来自论文摘要[^1]，但缺少具体成功率数值（如 ALFWorld 71% vs 45%、WebShop 40% vs 30% 之类）与基线名称，且未说明是 PaLM-540B 还是 GPT-3 设定。该数字在给定证据中无法被独立验证。

**3. 「ReAct 已被主流框架吸收为标准 agent 类型」与官方弃用公告直接矛盾。** 该说法依据的是 ysymyth/ReAct README 建议使用 LangChain zero-shot ReAct Agent[^22]，但与以下证据冲突：LangGraph v1 已弃用 `create_react_agent`[^23]，LangChain v1 以 `create_agent` 为新标准[^24]，LlamaIndex 默认 FunctionAgent[^7]，OpenAI Agents SDK 与 AutoGen 的循环均未命名为 ReAct[^27][^29]。矛盾未调和，不能据此断言 ReAct 仍是默认范式。更准确的表述是：ReAct 的循环**结构**被广泛继承，但 ReAct 的**命名与预构建组件**正在被更高层抽象取代。

**4. ReAct 失败模式缺乏实证。** 关于 ReAct 的失败模式（推理灵活性不足、陷入循环、无法回溯），给定证据中只有项目页「并非完美 + 一个 ALFWorld 失败示例」这一句定性描述[^2]，缺少论文中的具体失败案例与错误率统计。因此本文无法给出「CoT 的幻觉/事实错误 vs ReAct 的推理不灵活」这一两侧失败模式的实证对比。

**5. ReAct 的「事实性」主张强度有限。** 项目页主张 ReAct 以「可解释且事实性的轨迹」解决任务[^2]，但这一主张的证据基础是 HotpotQA/Fever 上的定性结论[^1]，缺少细粒度的事实性错误率统计。在给定材料范围内，只能将其视为作者主张而非已量化的结论。

**6. 关于 ReAct 与后续 RL 训练路线的定位关系，现有公开资料不足以确认。** Search-R1[^17] 与 ToolRL[^18] 都在改进「推理中调用工具」的能力，但给定证据未说明它们与 ReAct prompting 范式之间的直接继承或替代关系，也未给出二者在相同基准上的对照数字。

## 参考来源

[^1]: ReAct: Synergizing Reasoning and Acting in Language Models (arXiv:2210.03629v3) — https://arxiv.org/abs/2210.03629v3（arxiv, 2022-10-06）
[^2]: ReAct: Synergizing Reasoning and Acting in Language Models — 官方项目页 — https://react-lm.github.io/（web）
[^3]: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022) — https://arxiv.org/abs/2201.11903v6（arxiv, 2022-01-28）
[^4]: Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022) — https://arxiv.org/abs/2205.11916v4（arxiv, 2022-05-24）
[^5]: Self-Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al., 2022) — https://arxiv.org/abs/2203.11171v4（arxiv, 2022-03-21）
[^6]: langchain-ai/react-agent — LangGraph ReAct Agent Template — https://github.com/langchain-ai/react-agent（github, 2024-08-21）
[^7]: LlamaIndex 官方文档 — Agents (Framework Component Guides / Deploying) — https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/（web）
[^15]: Tree of Thoughts: Deliberate Problem Solving with Large Language Models — https://arxiv.org/abs/2305.10601v2（arxiv, 2023-05-17）
[^16]: Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models — https://arxiv.org/abs/2305.04091v3（arxiv, 2023-05-06）
[^17]: Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning — https://arxiv.org/abs/2503.09516v5（arxiv, 2025-03-12）
[^18]: ToolRL: Reward is All Tool Learning Needs — https://arxiv.org/abs/2504.13958v1（arxiv, 2025-04-16）
[^22]: ysymyth/ReAct (GitHub repository) — https://github.com/ysymyth/ReAct（github, 2022-11-13）
[^23]: LangGraph v1 release notes — Deprecation of create_react_agent — https://docs.langchain.com/oss/python/releases/langgraph-v1（web）
[^24]: LangChain v1 release notes — create_agent — https://docs.langchain.com/oss/python/releases/langchain-v1（web）
[^25]: LangChain Agents documentation — create_agent as the agent harness — https://docs.langchain.com/oss/python/langchain/agents（web）
[^27]: OpenAI Agents SDK — Running agents (official docs) — https://openai.github.io/openai-agents-python/running_agents/（web）
[^28]: LlamaIndex — Building an agent (official docs) — https://docs.llamaindex.ai/en/stable/understanding/agent/（web）
[^29]: AutoGen AgentChat — Agents tutorial (official docs) — https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html（web）