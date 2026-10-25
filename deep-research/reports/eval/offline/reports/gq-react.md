## 摘要

ReAct 的核心思想是让同一个语言模型以**交错（interleaved）**方式同时生成 reasoning traces 与 task-specific actions，并让两者双向协同：reasoning traces 帮助模型诱导、跟踪、更新行动计划并处理异常，actions 让模型与外部来源（知识库或环境）交互以收集额外信息[^1][^10]。它与纯 Chain-of-Thought 的关键区别可归纳为三点：输出空间不同（CoT 只产出自然语言推理步骤，ReAct 产出「thought → action → observation」循环）[^9]；接地方式不同（CoT 的知识来源完全在模型参数内部，ReAct 通过动作把推理锚定到外部环境）[^11]；失败模式方向相反（CoT 因未接地而受幻觉与错误传播影响，act-only 因缺乏推理而无法综合出最终答案）[^19]。需要前置声明的是：本报告所依据的知识库对 ReAct 的表述多为**转述而非论文原文引文**[^1]，且定量对比数据存在明显缺口[^8]。

---

## 一、ReAct 的核心思想

### 1.1 问题起点：reasoning 与 acting 的领域割裂

ReAct 论文的出发点是领域分工现象：LLM 的 reasoning 能力（以 chain-of-thought prompting 为代表）与 acting 能力（以 action plan generation 为代表）此前被作为两个独立课题研究[^4]。官方项目页补充 acting 侧的代表工作为 WebGPT、SayCan、ACT-1，并称这两个方向此前保持分离，ReAct 提出将两种基础能力结合[^4]。

### 1.2 核心机制：交错生成 reasoning traces 与 actions

ReAct 的轨迹结构有明确定义：reasoning trace（thought）与 action 交错，并由环境返回 observation 响应 action[^2][^10]。因此 ReAct 的**单步不是「想一步」，而是「想一步 → 做一个动作 → 收到环境反馈」的完整闭环**[^2][^10]。

两种成分各自承担明确职能：

- **reasoning traces**：帮助模型诱导（induce）、跟踪（track）、更新（update）行动计划并处理异常情况[^10]；
- **actions**：让模型与外部来源（知识库或环境）交互以收集额外信息[^10]。

协同是双向的：推理帮助模型维护计划，动作帮助模型获取参数内不存在的信息[^1]。

### 1.3 Prompt 构成与「何时想、何时做」的自主权

ReAct prompt 由 few-shot task-solving trajectories 组成，其中包含人工书写的 text reasoning traces 与 actions，以及针对 actions 的环境 observations[^3]。针对决策类任务，作者设计的是**带稀疏 reasoning traces 的人类轨迹**，让 LM 自行决定何时 think、何时 act[^3]。也就是说，推理的发生时机由模型在轨迹中自行判断，而非由提示强制每步都推理[^3]。这一点常被忽略：ReAct 不是「每步必须写 thought」的模板，而是把推理时机的裁量权交给模型。

### 1.4 实验证据（定性层面）

论文报告两组实验：知识密集型任务（HotpotQA、Fever）与交互式决策基准（ALFWorld、WebShop）[^5]。在 HotpotQA 与 Fever 上，ReAct 通过与简单 Wikipedia API 交互，克服了 chain-of-thought reasoning 中普遍存在的幻觉与错误传播问题[^5][^11]。在 ALFWorld 与 WebShop 上，ReAct 相比模仿学习与强化学习方法分别取得 34% 与 10% 的绝对成功率提升，且仅用一到两个 in-context 示例进行提示[^5]。上述数字的可靠性边界见第五节。

### 1.5 可解释性与可纠正性

项目页主张 ReAct 以可解释且事实性的轨迹解决任务[^6]。ReAct 格式便于人工检查，并可通过修改少量 model thoughts 进行行为纠正[^6]。同时项目页承认 ReAct 并非完美，给出 ALFWorld 上的失败示例[^6]。

---

## 二、与纯 Chain-of-Thought 的关键区别

先明确 CoT 的定义：纯 Chain-of-Thought 只生成一条思维链——一系列中间推理步骤——作为 few-shot 提示中的示范，模型**全程不与外部环境交互**[^9]。以这个定义为基准，差异体现在三个层面。

### 2.1 区别一：输出空间与循环结构

- **CoT**：只产出自然语言推理步骤[^9]。
- **ReAct**：产出「thought → action → observation」循环，动作会触发环境返回观测[^9]。

工程实现侧对同一循环的复述可见于 LangChain 官方维护的 ReAct agent 模板：接收用户 query → 推理并决定动作 → 用可用工具执行动作 → 观察动作结果 → 重复第 2-4 步直到能给出最终答案[^7][^21]。LlamaIndex 的描述同构：给定任务与一组工具，选择最佳可用工具完成每一步；每步完成后 agent 判断任务是否完成——完成则返回结果，未完成则循环回到起点[^21]。LangChain 官方对 agent 的定义为「一个在循环中调用工具直到任务完成的模型」[^21]。需注意这些是工程侧复述，不是论文原文[^7]。

### 2.2 区别二：是否接地于外部环境

这是两者最本质的分野。CoT 的知识来源完全在模型参数内部，**推理链条再长也不引入新信息**；ReAct 的 action 显式地把模型接到外部来源上以收集额外信息[^11]。

项目页的对比说明把两侧失败都点了出来：reason-only baseline（即 chain-of-thought）因未接地于外部环境而受错误信息影响；act-only baseline 因缺乏推理而无法综合出最终答案[^11]。论文侧对应表述为：ReAct 通过与简单 Wikipedia API 交互，克服了 chain-of-thought reasoning 中普遍存在的幻觉与错误传播问题[^11]。

### 2.3 区别三：失败模式方向相反

- **CoT 侧**：未接地导致幻觉与错误传播，推理链条中的错误会沿链条传播；Zero-shot-CoT 存在计算错误、缺失步骤错误、语义误解错误三类缺陷[^19]。
- **act-only 侧**：缺乏推理导致无法综合出最终答案[^19]。
- **ReAct 侧**：推理灵活性不足、陷入循环、无法回溯[^19]。

值得注意的是，ReAct 的失败模式与 CoT 并非同一维度上的强弱关系，而是「接地引入的新失败」对「不接地固有的旧失败」——这是范式取舍，不是纯改进。但 ReAct 侧的失败描述在库内只有项目页的定性陈述，缺少论文中的具体失败案例与错误率统计[^19]，因此无法给出实证量化对比。

### 2.4 区别小结

| 维度 | 纯 CoT | ReAct |
|---|---|---|
| 输出空间 | 自然语言推理步骤[^9] | thought → action → observation 循环[^9] |
| 知识来源 | 模型参数内部，链条再长不引入新信息[^11] | 通过 action 接外部知识库/环境[^11] |
| 推理时机 | 提示示范整条思维链[^9] | 模型自行决定何时 think、何时 act[^3] |
| 主要失败模式 | 幻觉与错误沿链传播[^19] | 灵活性不足、循环、无法回溯[^19] |

---

## 三、ReAct 在工程生态中的位置（与范式判断相关）

ReAct 的「推理-行动交错 + 环境观测」循环结构在当代 agent 框架中普遍存在：LangChain 的 basic agent loop、LlamaIndex 的 agent 循环、OpenAI Agents SDK 的 agent loop、AutoGen 的工具调用序列都是同构的[^23][^29]。但**不能据此断言 ReAct 仍是当前框架的默认命名范式或默认实现**：

- LangGraph v1 发布说明设有专门小节「Deprecation of create_react_agent」，该预构建组件已被弃用，转而推荐 LangChain 的 create_agent[^22]；create_agent 被明确为构建 agent 的新标准，底层机制被描述为「basic agent loop」——调用模型 → 让模型选择要执行的工具 → 当模型不再调用工具时结束[^22]。
- LlamaIndex 的默认 agent 类型是 FunctionAgent，使用 LLM provider 的 function/tool calling 能力执行工具；ReActAgent 与 CodeActAgent 被列为「使用不同 prompting 策略执行工具的」其他类型 agent，而非默认类型[^22]。
- OpenAI Agents SDK 文档以 LLM 原生 tool calling 与 handoff 为核心控制流，未将其命名为 ReAct[^23][^29]；AutoGen AgentChat 文档示例的消息序列同样未使用 ReAct 命名[^23][^29]。

原始仓库方面：官方 GitHub 仓库 ysymyth/ReAct 明确标注为 ICLR 2023 论文的 GPT-3 prompting 代码，覆盖 hotpotqa、fever、alfworld、webshop 四个 notebook，元数据为 4224 stars、404 forks、MIT 许可；README 建议用户如需将 ReAct 用于更多任务，可尝试 LangChain 的 zero-shot ReAct Agent[^24]。

训练侧出现了对 ReAct 式「推理中调用工具」范式的替代路径：Search-R1 指出用提示让具备推理能力的 LLM 在推理时使用搜索引擎往往次优，因为 LLM 可能并不完全具备「如何最优地与搜索引擎交互」的能力，转而用 RL 让模型自主学习生成搜索查询并实时检索，在 7 个 QA 数据集上相比同设定 RAG 基线，Qwen2.5-7B 提升 41%、Qwen2.5-3B 提升 20%[^13][^29]；ToolRL 指出 SFT 难以泛化到不熟悉或复杂的工具使用场景，改用 GRPO 训练并设计工具使用任务的奖励，实验显示相比基座模型提升 17%、相比 SFT 模型提升 15%[^13][^29]。这些工作可视为对 ReAct 式范式的训练侧改进，但它们本身不再以 ReAct 命名[^13][^29]。

---

## 四、矛盾与不确定性

**（1）定量对比数据缺失，这是本课题最实质的缺口。** HotpotQA / Fever 上论文侧仅有「通过与简单 Wikipedia API 交互克服幻觉与错误传播」的定性转述，缺少原始论文 Table 1/2 的具体 EM/准确率数字（PaLM-540B 下 ReAct vs CoT vs CoT-SC 的对比）[^8]。ALFWorld / WebShop 的提升数字（34% / 10%）来自论文摘要，缺少具体成功率数值与基线名称，且未说明是 PaLM-540B 还是 GPT-3 设定，无法被独立验证[^8]。因此「ReAct 比 CoT 好多少」在本知识库中**没有可核验的量化答案**。

**（2）ReAct 自身失败模式缺乏实证。** 关于推理灵活性不足、陷入循环、无法回溯，库内只有项目页「并非完美 + 一个 ALFWorld 失败示例」这一句定性描述，缺少论文中的具体失败案例与错误率统计[^19]。

**（3）证据形态本身是转述。** ReAct 核心思想的表述在知识库中标注为论文侧证据（arXiv:2210.03629v3），但呈现形式为转述而非论文原文引文[^1]；轨迹结构定义与 prompt 构成标注来源为官方项目页而非论文正文[^2][^3]。引用这些内容时应理解为「知识库对论文的复述」，而非逐字原文。

**（4）工程生态存在未调和的矛盾。** 一方面 ReAct 循环结构在主流框架中普遍同构存在[^23][^29]，另一方面最直接以 ReAct 命名的预构建组件 create_react_agent 已被官方弃用[^22]。该矛盾在给定材料中未被调和[^29]：可以确认的是「循环结构存活」，不能确认的是「ReAct 作为命名范式仍居默认地位」。

**（5）长程任务中的错误累积与上下文膨胀：当前知识库未收录针对 ReAct 轨迹的实证研究。** 库内只有间接证据——上下文工程归纳的四类失效模式（Context Poisoning / Distraction / Confusion / Clash）[^14]、DeepMind 记录的 Gemini 2.5 Pokémon agent 幻觉污染上下文导致执着于不可能目标[^14]、Databricks 发现 Llama 3.1 405b 正确率在约 32k 上下文时开始下降[^14]、LangChain 关于 agent 交错调用 LLM 与工具导致 token 累积的文档[^15]、以及 agentic RAG 论文识别的复合式幻觉传播与记忆投毒等系统性风险[^16]。这些均非针对 ReAct 轨迹的实验结论，无法支撑量化判断。

**（6）ReAct 的扩展工作覆盖缺口。** 关于 Tree-of-Thought、ReWOO、CodeAct 等 ReAct 扩展工作，当前知识库未收录其论文内容与结论（CodeAct 仅作为 OpenHands 的 SWE-bench 提交条目名 `20241029_OpenHands-CodeAct-2.1-sonnet-20241022` 出现，无机制描述）[^28]。离线模式下无法补齐该分支。

**（7）ReAct 与后续反思机制的边界。** 库内 Reflexion 相关证据显示其通过语言反馈而非更新权重来强化 agent，将反思文本保存在 episodic memory buffer 中以引导后续试验，在 HumanEval 上达到 91% pass@1[^12]。该机制与 ReAct 的关系（是叠加在 ReAct 循环之上，还是替代其推理成分）在给定证据中未被说明，现有公开资料不足以确认。

## 参考来源

[^1]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（§一 核心思想 / 摘要） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ## 一、ReAct 的核心思想 · L9-10（同文 ## 摘要 · L3-8）（kb）
[^2]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（§1.2 核心机制：交错生成 reasoning traces 与 actions） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ### 1.2 核心机制：交错生成 reasoning traces 与 actions · L15-23（kb）
[^3]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（§1.3 Prompt 构成与「何时想、何时做」的自主权） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ### 1.3 Prompt 构成与「何时想、何时做」的自主权 · L24-27（kb）
[^4]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（§1.1 问题起点：reasoning 与 acting 的割裂） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ### 1.1 问题起点：reasoning 与 acting 的割裂 · L11-14（kb）
[^5]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（§1.4 实验证据） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ### 1.4 实验证据 · L28-35（kb）
[^6]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（§1.5 可解释性与可纠正性 / §3.4） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ### 1.5 可解释性与可纠正性 · L37-42（同文 ### 3.4 · L84-87）（kb）
[^7]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（§3.1 区别一：输出空间与循环结构） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ### 3.1 区别一：输出空间与循环结构 · L62-66（kb）
[^8]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（§五 矛盾与不确定性） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ## 五、矛盾与不确定性 · L120-126（kb）
[^9]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（摘要节） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ## 摘要 · L3-8（kb）
[^10]: ReAct 核心机制：交错生成 reasoning traces 与 actions（1.2 节） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ### 1.2 核心机制：交错生成 reasoning traces 与 actions · L15-22（kb）
[^11]: ReAct 与 CoT 区别二：是否接地于外部环境（3.2 节） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ### 3.2 区别二：是否接地于外部环境 · L68-73（kb）
[^12]: 多智能体系统（Multi-Agent LLM）主流协作模式与适用场景 — 1.6 反思/进化机制（横切能力） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-multi-agent.md · ### 1.6 反思/进化机制（横切能力） · L41-46（kb）
[^13]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 四、ReAct 在工程生态中的位置 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置：一个未调和的矛盾 · L101-119（kb）
[^14]: 上下文工程（Context Engineering）：定义、与提示词工程的关系与差异 — 1.3 为什么需要它：上下文是有限资源 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-context-eng.md · ### 1.3 为什么需要它：上下文是有限资源 · L28-46（kb）
[^15]: 上下文工程（Context Engineering）：定义、与提示词工程的关系与差异 — 四、支撑上下文工程的基础设施 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-context-eng.md · ## 四、支撑上下文工程的基础设施 · L107-108（kb）
[^16]: Agentic RAG：定义、架构关键差异与工程现状 — 4.1 已明确提供 agentic 检索能力的实现 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-agentic-rag.md · ### 4.1 已明确提供 agentic 检索能力的实现 · L111-115（kb）
[^19]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 3.3 区别三：失败模式的方向相反 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ### 3.3 区别三：失败模式的方向相反 · L74-82（kb）
[^21]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（gq-react.md） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ### 3.1 区别一：输出空间与循环结构 · L62-67（kb）
[^22]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（gq-react.md）· 第四节 工程生态位置 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置：一个未调和的矛盾 · L101-110（kb）
[^23]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（gq-react.md）· 循环同构性与原始仓库 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置：一个未调和的矛盾 · L101-114（kb）
[^24]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（gq-react.md）· 原始 ReAct 仓库元数据 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置：一个未调和的矛盾 · L101-107（kb）
[^28]: 自动修 Bug 的 Coding Agent 在 SWE-bench 上的技术水平（gq-swe-agent.md）· 官方 experiments 仓库条目 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-swe-agent.md · ### 2.5 官方 experiments 仓库可见条目 · L51-56（kb）
[^29]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（第四节：OpenAI Agents SDK / AutoGen 与未调和矛盾） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置：一个未调和的矛盾 · L101-116（kb）