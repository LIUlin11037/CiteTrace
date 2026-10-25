## 摘要

ReAct 的核心思想是让同一个语言模型以交错（interleaved）方式同时生成 reasoning traces（推理轨迹）与 task-specific actions（任务特定动作）[^1][^10]。推理轨迹用于诱导、跟踪、更新行动计划并处理异常；动作用于与外部来源（知识库或环境）交互以收集额外信息[^1][^10]。它与纯 Chain-of-Thought 的关键区别不在于「是否推理」，而在于三点结构性差异：输出空间（文本思维链 vs. thought→action→observation 循环）、知识来源（模型参数内部 vs. 外部环境接地）、以及失败模式的方向（未接地导致幻觉传播 vs. 缺推理导致无法综合答案）[^6][^15][^16][^17]。需要前置声明的是：本知识库对 HotpotQA/FEVER 上 ReAct 相对 CoT/CoT-SC 的量化增益无覆盖，仅有定性转述[^5][^9][^20]；因此下文关于「事实性增益」的表述只能停留在作者主张层面。

---

## 一、ReAct 的核心思想

### 1.1 问题起点：reasoning 与 acting 的割裂

ReAct 论文的出发点是：LLM 的 reasoning 能力（以 chain-of-thought prompting 为代表）与 acting 能力（以 action plan generation 为代表）此前被作为两个独立课题研究[^4]。官方项目页把这一割裂描述得更具体：语言模型在 reasoning（如 chain-of-thought prompting）与 acting（如 WebGPT、SayCan、ACT-1）两方面都在进步，但这两个方向此前保持分离，ReAct 提出将这两种基础能力结合[^4]。

### 1.2 核心机制：交错生成与单步闭环

ReAct 轨迹的结构定义是：reasoning trace（thought）与 action 交错，并由环境返回 observation 响应 action[^2]。因此 ReAct 的单步不是「想一步」，而是「想一步 → 做一个动作 → 收到环境反馈」的完整闭环[^2]。用工程语言表述：CoT 是「单次前向的文本生成」，ReAct 是「生成 → 执行 → 观测 → 再生成」的循环[^15]。

这一循环在框架文档中有同构的落地写法。LangChain 官方维护的 ReAct agent 模板把循环写为：接收用户 query → 推理并决定动作 → 用可用工具执行动作 → 观察动作结果 → 重复第 2–4 步直到能给出最终答案，停止条件由模型产出最终答案触发[^25]。LlamaIndex 对 agent 的描述为：给定任务与一组工具，选择最佳可用工具完成每一步；每步完成后 agent 判断任务是否完成——完成则返回结果，未完成则循环回到起点[^26]。Semantic Kernel 的 auto function calling 循环则给出 6 步流程：序列化可用函数 → 连同 chat history 发给模型 → 模型返回 chat message 或 function call → 若为 function call 则提取函数名与参数 → 在 kernel 中调用该函数 → 将结果回传模型，步骤 2–6 重复直到模型返回 chat message 或达到最大迭代次数[^30]。

### 1.3 Prompt 构成与「何时想、何时做」的自主权

ReAct prompt 由 few-shot task-solving trajectories 组成，其中包含人工书写的 text reasoning traces 与 actions，以及针对 actions 的环境 observations[^3]。针对决策类任务，作者设计的是带稀疏 reasoning traces 的人类轨迹，让 LM 自行决定何时 think、何时 act[^3]。推理的发生时机由模型在轨迹中自行判断，而非由提示强制每步都推理——这是 ReAct 与「固定模板式推理」的重要分野[^3]。

### 1.4 实验证据（含证据强度限制）

知识密集型任务：在 HotpotQA 与 Fever 上，ReAct 通过与简单 Wikipedia API 交互，克服了 chain-of-thought reasoning 中普遍存在的幻觉与错误传播问题（定性结论，无具体 EM/准确率数字）[^5][^10][^11]。交互式决策基准：在 ALFWorld 与 WebShop 上，ReAct 相比模仿学习与强化学习方法分别取得 34% 与 10% 的绝对成功率提升，且仅用一到两个 in-context 示例进行提示即取得[^5][^10]。

---

## 二、纯 Chain-of-Thought 的定义与边界

CoT 被定义为生成一条思维链，即「一系列中间推理步骤」（a series of intermediate reasoning steps）[^12]。其实现方式是 few-shot prompting：在提示中提供少量思维链示范作为示例（a few chain of thought demonstrations are provided as exemplars in prompting）[^12]。该推理能力在足够大的语言模型中自然涌现（emerge naturally in sufficiently large language models），即存在模型规模门槛[^12]。实验覆盖算术、常识、符号推理三类任务[^12]。用 8 条 CoT 示例提示 540B 参数模型，在 GSM8K 数学应用题基准上达到 SOTA 准确率，超过带 verifier 的微调 GPT-3（此为论文主张，经报告转述）[^12][^11]。

CoT 家族内部还有变体，但输出空间边界一致：

- **Zero-shot CoT**：在每次回答前简单加上 "Let's think step by step"，无需任何手工构造的 few-shot 示例，单一提示模板即可跨任务泛化[^13]。在 text-davinci-002 上使 MultiArith 准确率从 17.7% 提升到 78.7%，GSM8K 从 10.4% 提升到 40.7%[^13]。该论文把 CoT 所针对的任务描述为不遵循标准 scaling law 的困难 System-2 任务[^13]。
- **Self-consistency**：标准 CoT 使用朴素贪心解码（单条推理路径、单次前向生成）；self-consistency 采样一组多样化推理路径后对答案做边缘化投票[^13][^23]。相对标准 CoT 的提升为 GSM8K +17.9%、SVAMP +11.0%、AQuA +12.2%、StrategyQA +6.4%、ARC-challenge +3.9%[^13][^23]。
- **边界**：所有 CoT 变体（zero-shot、self-consistency、plan-and-solve、ToT）的输出空间始终是文本推理步骤，没有任何一个变体引入「动作触发环境观测」这一环节[^14]。Plan-and-Solve 指出 Zero-shot-CoT 存在三类缺陷：计算错误、缺失步骤错误、语义误解错误[^14]；Tree of Thoughts 批评 LM 推理被限制在 token 级、从左到右的决策过程，并泛化 CoT 以允许在「连贯的文本单元（thoughts）」上探索，支持多路径推理、自我评估、前瞻与回溯[^14]。

---

## 三、关键区别

### 3.1 区别一：输出空间与循环结构

CoT 的输出是一条思维链，即一系列中间推理步骤，一次性生成后即给出答案[^15]。ReAct 的输出是 reasoning trace 与 action 交错、并由环境返回 observation 响应 action 的结构[^2][^15]。对照表将纯 CoT 的生成结构记为「单次前向、单条推理路径（标准 CoT 为贪心解码）」[^16]。纯 CoT 模型全程不与外部环境交互[^17]。

### 3.2 区别二：是否接地于外部环境

CoT 的知识来源完全在模型参数内部，推理链条再长也不引入新信息；ReAct 的 action 显式地把模型接到外部来源（知识库或环境）上以收集额外信息[^6][^16]。项目页的对比说明指出：reason-only baseline（即 chain-of-thought）因未接地于外部环境而受错误信息影响；act-only baseline 因缺乏推理而无法综合出最终答案；ReAct 以可解释且事实性的轨迹解决任务[^6]。对照表将纯 CoT 的接地性记为「未接地于外部环境」[^16]。

### 3.3 区别三：失败模式方向相反

CoT 侧失败模式：未接地导致幻觉与错误传播，且推理链条中的错误会沿链条传播[^7][^17]；对照表另记为「计算/缺步/语义误解错误」[^16]。act-only 侧失败模式：缺乏推理导致无法综合出最终答案[^7]。ReAct 侧：项目页给出 ALFWorld 上的失败示例，承认 ReAct 并非完美[^7]。

### 3.4 区别四：可解释性与可纠正性的形态不同

ReAct 的轨迹包含 thought、action、observation 三类元素[^8]。人工检查 ReAct 轨迹时不仅能读到模型「怎么想」，还能读到模型「做了什么」以及环境「返回了什么」[^8]。项目页主张 ReAct 格式便于人工检查，并可通过修改少量 model thoughts 进行行为纠正；这是 CoT 轨迹所不具备的干预接口[^8]。

### 3.5 一张对照表

| 维度 | 纯 CoT | ReAct |
|---|---|---|
| 输出内容 | 一系列中间推理步骤（思维链）[^16] | thought 与 action 交错，环境返回 observation[^2][^15] |
| 生成结构 | 单次前向、单条推理路径（标准 CoT 贪心解码）[^16] | 生成 → 执行 → 观测 → 再生成的循环[^15] |
| 知识来源 | 模型参数内部[^16] | 外部知识库或环境[^6][^16] |
| 接地性 | 未接地于外部环境[^16] | 通过 action 显式接地[^6] |
| 主要失败模式 | 幻觉与错误传播；计算/缺步/语义误解错误[^16] | 项目页承认并非完美，给出 ALFWorld 失败示例[^7] |
| 人工干预接口 | 仅可读推理文本[^8] | 可读 thought/action/observation，并可修改 model thoughts 纠正行为[^8] |

---

## 四、工程生态位置：一个未调和的矛盾

ReAct 的「推理-行动交错 + 环境观测」循环结构在当代 agent 框架中普遍存在：LangChain 的 basic agent loop、LlamaIndex 的 agent 循环、OpenAI Agents SDK 的 agent loop、AutoGen 的工具调用序列都是同构的[^21]。但命名与预构建组件层面出现了相反信号：

- LangGraph v1 发布说明设有专门小节「Deprecation of create_react_agent」：create_react_agent 预构建组件已被弃用，转而推荐 LangChain 的 create_agent；官方理由是 create_agent 提供更简单的接口，并通过引入 middleware 提供更强的定制能力[^27]。LangChain v1 发布说明明确 create_agent 是构建 agent 的新标准，取代 langgraph.prebuilt.create_react_agent；其底层机制被描述为「basic agent loop」：调用模型 → 让模型选择要执行的工具 → 当模型不再调用工具时结束[^27]。
- LlamaIndex 的默认 agent 类型是 FunctionAgent，它使用 LLM provider 的 function/tool calling 能力执行工具；ReActAgent 与 CodeActAgent 被列为「使用不同 prompting 策略执行工具的」其他类型 agent，而非默认类型[^26]。
- OpenAI Agents SDK 文档描述的 agent loop 以 LLM 原生 tool calling 与 handoff 为核心控制流，文档未将其命名为 ReAct[^29]；AutoGen AgentChat 示例中 agent.run() 返回的消息序列为 TextMessage(user) → ToolCallRequestEvent → ToolCallExecutionEvent → ToolCallSummaryMessage，同样未使用 ReAct 命名[^29]。
- LangChain 官方对框架分工的表述是 Agent = Model + Harness；create_agent 被定位为最小、高度可配置的 harness，harness 指模型循环之外的一切——prompt、tools 以及任何塑造行为的 middleware[^28]。

训练侧也出现了不再以 ReAct 命名的改进路径：Search-R1 指出用提示让具备推理能力的 LLM 在推理时使用搜索引擎往往次优，因为 LLM 可能并不完全具备「如何最优地与搜索引擎交互」的能力，转而通过 RL 让 LLM 在逐步推理中自主学习生成搜索查询并实时检索，在 7 个 QA 数据集上相比同设定 RAG 基线，Qwen2.5-7B 提升 41%、Qwen2.5-3B 提升 20%[^21][^35]。ToolRL 指出 SFT 难以泛化到不熟悉或复杂的工具使用场景，转而用 GRPO 训练并设计工具使用任务的奖励，实验显示相比基座模型提升 17%、相比 SFT 模型提升 15%[^21][^35]。这些工作可视为对 ReAct 式「推理中调用工具」范式的训练侧改进，但它们本身不再以 ReAct 命名[^21][^35]。

原始资产方面：官方 GitHub 仓库 ysymyth/ReAct 明确标注为 ICLR 2023 论文的 GPT-3 prompting 代码，覆盖 hotpotqa、fever、alfworld、webshop 四个 notebook，元数据为 4224 stars、404 forks、MIT 许可，README 建议用户如需将 ReAct 用于更多任务可尝试 LangChain 的 zero-shot ReAct Agent[^33]。论文出处条目为 arXiv 2210.03629v3（2022-10-06），与 CoT 论文 arXiv 2201.11903v6（2022-01-28）分列两条[^18]。

**结论性表述须降级**：证据实际支持的是「ReAct 的循环结构被广泛继承，但命名与预构建组件正在被更高层抽象取代」，不支持「ReAct 已被主流框架吸收为标准 agent 类型」这一强主张[^9][^20]。

---

## 五、矛盾与不确定性

1. **HotpotQA / FEVER 的量化对比缺失。** 所有相关证据（gq-react.md、gq-tool-use.md、gq-context-eng.md）均只转述「ReAct 通过与简单 Wikipedia API 交互，克服了 chain-of-thought reasoning 中普遍存在的幻觉与错误传播问题」这一句定性结论，缺少原始论文 Table 1/2 的具体 EM/准确率数字（PaLM-540B 下 ReAct vs CoT vs CoT-SC 的对比）[^5][^9][^20]。而「ReAct 相对纯 CoT 在事实性上的增益」正是本课题关键区别的核心实证依据，仅凭定性转述无法支撑。当前知识库未收录该信息。

2. **CoT-SC 与 ReAct 的混合方案库内无覆盖。** 库内 self-consistency 证据只描述其独立于 CoT 的增益（GSM8K +17.9% 等）[^13][^23]，没有任何一条涉及它与 ReAct 的组合或「ReAct 失败时回退到 CoT-SC」的 back-off 机制。因此「互补性」这一侧目前只有 ReAct 与 CoT 的对立描述，缺少互补侧的实证。当前知识库未收录该信息。

3. **ReAct 自身失败模式缺乏实证。** 关于 ReAct 的失败模式（推理灵活性不足、陷入循环、无法回溯），给定证据中只有项目页「并非完美 + 一个 ALFWorld 失败示例」这一句定性描述，缺少论文中的具体失败案例与错误率统计[^7][^20]。

4. **「事实性」主张强度有限。** 项目页主张 ReAct 以「可解释且事实性的轨迹」解决任务，但证据基础是 HotpotQA/Fever 上的定性结论，缺少细粒度的事实性错误率统计；在给定材料范围内只能视为作者主张而非已量化的结论[^20]。

5. **ALFWorld/WebShop 数字不可独立验证。** 「相比模仿学习与强化学习方法分别取得 34% 与 10% 的绝对成功率提升」来自论文摘要，但缺少具体成功率数值与基线名称，且未说明是 PaLM-540B 还是 GPT-3 设定[^5][^9]。

6. **框架生态主张与官方弃用公告直接矛盾。** 该矛盾在给定材料中未被调和[^9][^20][^27]。

7. **参考文献列表不完整。** gq-react.md 的参考文献列表可见 [^4][^5][^6][^7][^15][^16][^17][^18][^22]，未包含正文引用的 [^3] 条目[^19]。

8. **自主循环的系统性风险已有独立批评。** 有论文识别出自主循环固有的严重系统性风险：复合式幻觉传播、记忆投毒、检索失配、级联式工具执行漏洞，并批评传统静态评估实践对 agentic 系统不适用，指出需要成本感知编排、稳定自适应检索、形式化轨迹评估与监督机制等方向[^24]。这构成对 ReAct 式循环在生产环境适用性的外部约束，但该批评并非针对 ReAct 本身，本知识库未收录将其直接归因于 ReAct 的证据。

## 参考来源

[^1]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（报告：核心思想章节） — reports/eval/reports/gq-react.md · ## 一、ReAct 的核心思想 · L9-10（kb）
[^2]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（报告：核心机制章节） — reports/eval/reports/gq-react.md · ### 1.2 核心机制：交错生成 reasoning traces 与 actions · L15-23（kb）
[^3]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（报告：Prompt 构成章节） — reports/eval/reports/gq-react.md · ### 1.3 Prompt 构成与「何时想、何时做」的自主权 · L24-27（kb）
[^4]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（报告：问题起点章节） — reports/eval/reports/gq-react.md · ### 1.1 问题起点：reasoning 与 acting 的割裂 · L11-14（kb）
[^5]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（报告：实验证据章节） — reports/eval/reports/gq-react.md · ### 1.4 实验证据 · L28-35（kb）
[^6]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（报告：区别二 接地方式） — reports/eval/reports/gq-react.md · ### 3.2 区别二：是否接地于外部环境 · L68-73（kb）
[^7]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（报告：区别三 失败模式） — reports/eval/reports/gq-react.md · ### 3.3 区别三：失败模式的方向相反 · L74-82（kb）
[^8]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（报告：区别四 可解释性） — reports/eval/reports/gq-react.md · ### 3.4 区别四：可解释性与可纠正性的形态不同 · L84-87（kb）
[^9]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（报告：矛盾与不确定性） — reports/eval/reports/gq-react.md · ## 五、矛盾与不确定性 · L120-126（kb）
[^10]: LLM 工具调用（Function Calling）的训练、对齐与评测（报告：工具学习范式章节，含 ReAct 摘要级转述） — reports/eval/reports/gq-tool-use.md · ### 2.3 工具学习范式的系统化 · L41-46（kb）
[^11]: 上下文工程（Context Engineering）：定义、与提示词工程的关系与差异（报告：历史脉络章节，含 ReAct 定位） — reports/eval/reports/gq-context-eng.md · ### 2.3 一条清晰的历史脉络 · L69-73（kb）
[^12]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（第二节：纯 Chain-of-Thought 的定义与边界） — reports/eval/reports/gq-react.md · ## 二、纯 Chain-of-Thought 的定义与边界 · L43-51（kb）
[^13]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（Zero-shot CoT 与 few-shot CoT 的区分） — reports/eval/reports/gq-react.md · ## 二、纯 Chain-of-Thought 的定义与边界 · L43-53（kb）
[^14]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（CoT 家族输出空间边界） — reports/eval/reports/gq-react.md · ## 二、纯 Chain-of-Thought 的定义与边界 · L43-59（kb）
[^15]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（3.1 输出空间与循环结构） — reports/eval/reports/gq-react.md · ### 3.1 区别一：输出空间与循环结构 · L62-66（kb）
[^16]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（3.5 对照表） — reports/eval/reports/gq-react.md · ### 3.5 一张对照表 · L88-96（kb）
[^17]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（摘要：CoT 不与外部环境交互） — reports/eval/reports/gq-react.md · ## 摘要 · L3-8（kb）
[^18]: 上下文工程（Context Engineering）：定义、与提示词工程的关系与差异（参考来源：CoT 原始论文条目） — reports/eval/reports/gq-context-eng.md · ## 参考来源 · L204-213（kb, 2022-01-28）
[^19]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（参考来源：Zero-shot CoT / Self-Consistency 条目） — reports/eval/reports/gq-react.md · ## 参考来源 · L134-143（kb, 2022-05-24）
[^20]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（五、矛盾与不确定性：事实性主张强度有限） — reports/eval/reports/gq-react.md · ## 五、矛盾与不确定性 · L120-130（kb）
[^21]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（四、工程生态位置：Search-R1 与 ToolRL 的训练侧改进） — reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置：一个未调和的矛盾 · L101-116（kb）
[^22]: 多智能体系统（Multi-Agent LLM）主流协作模式与适用场景（1.6 反思/进化机制：Reflexion） — reports/eval/reports/gq-multi-agent.md · ### 1.6 反思/进化机制（横切能力） · L41-46（kb）
[^23]: DeepSeek-R1 核心技术路线与推理模型训练范式影响（2.6 测试时计算：Self-consistency） — reports/eval/reports/gq-deepseek-r1.md · ### 2.6 测试时计算：与 RL 路线并行的一条轴 · L102-107（kb）
[^24]: Agentic RAG：定义、架构关键差异与工程现状（4.1 自主循环的系统性风险） — reports/eval/reports/gq-agentic-rag.md · ### 4.1 已明确提供 agentic 检索能力的实现 · L111-113（kb）
[^25]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（3.1 区别一：输出空间与循环结构） — reports/eval/reports/gq-react.md · ### 3.1 区别一：输出空间与循环结构 · L62-67（kb）
[^26]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（LlamaIndex agent 循环与默认 agent 类型） — reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置 · L101-110（kb）
[^27]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（create_react_agent 弃用与 create_agent 新标准） — reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置：一个未调和的矛盾 · L101-110（kb）
[^28]: 上下文工程（Context Engineering）：定义、与提示词工程的关系与差异（4.4 框架层的定位分化） — reports/eval/reports/gq-context-eng.md · ### 4.4 框架层的定位分化 · L135-142（kb）
[^29]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（OpenAI Agents SDK 与 AutoGen 的循环命名） — reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置 · L101-114（kb）
[^30]: Human-in-the-loop 在 LLM Agent 系统中的主流实现模式与适用场景（Semantic Kernel auto function calling 循环） — reports/eval/reports/gq-human-in-loop.md · ## 八、辅助机制：Filter 作为审批前置控制点 · L113-122（kb）
[^33]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（官方仓库 ysymyth/ReAct 元数据） — reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置 · L101-107（kb）
[^35]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（训练侧替代路径 Search-R1 / ToolRL） — reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置 · L101-116（kb）