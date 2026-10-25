## 摘要

ReAct 的核心思想是把此前被分开研究的两种 LLM 能力——以 chain-of-thought 为代表的 reasoning 与以 action plan generation 为代表的 acting——合并到同一条生成轨迹里：模型交错（interleaved）生成 reasoning traces 与 task-specific actions，动作作用于外部来源（知识库或环境），环境返回 observation 作为下一步输入[^1][^2]。它与纯 CoT 的关键区别不在「想得更细」，而在输出空间多了一类元素：CoT 及其全部变体的输出始终是文本推理步骤，没有任何变体引入「动作触发环境观测」这一环节[^15]；ReAct 的单步是「想一步 → 做一个动作 → 收到环境反馈」的完整闭环[^2]。由此派生出接地性差异（CoT 知识来源完全在参数内部，ReAct 通过 action 与外部来源交互）[^6][^16]与失败模式差异（CoT 侧是未接地导致的幻觉与错误传播，act-only 侧是缺乏推理无法综合答案）[^5][^16]。需要提前声明：本报告所依据的知识库在定量对比、代价维度、工程热度三个方向上均无覆盖，且「ReAct 已成为主流框架标准 agent 类型」这一流行说法与库内记录的官方弃用公告直接冲突且未被调和[^11][^22][^23]。

---

## 一、问题起点：reasoning 与 acting 的割裂

ReAct 论文的出发点是一个学科分工事实：LLM 的 reasoning 能力（以 chain-of-thought prompting 为代表）与 acting 能力（以 action plan generation 为代表）此前被作为两个独立课题研究[^1]。官方项目页对这一割裂的表述是：语言模型在 reasoning（如 chain-of-thought prompting）与 acting（如 WebGPT、SayCan、ACT-1）两方面都在进步，但两个方向此前保持分离，ReAct 提出将这两种基础能力结合[^1]。

这一起点决定了 ReAct 的贡献性质：它不是提出一种新的推理算法，而是提出一种把已有推理与已有动作生成缝进同一条轨迹的组织方式。

## 二、核心机制：交错生成 reasoning traces 与 actions

### 2.1 定义与轨迹结构

ReAct 的做法是让 LLM 以交错（interleaved）方式同时生成 reasoning traces 与 task-specific actions[^2][^7]。两类元素的分工被明确界定：

- **reasoning traces** 的作用是帮助模型诱导（induce）、跟踪（track）、更新（update）行动计划，并处理异常情况[^2][^7]；
- **actions** 的作用是让模型与外部来源（知识库或环境）交互，以收集额外信息[^2][^7]。

项目页给出的轨迹结构定义是：reasoning trace（thought）与 action 交错，并由环境返回 observation 响应 action[^2]。因此一条完整轨迹包含 thought、action、observation 三类元素[^6]。用一句话概括其单步语义：ReAct 的单步不是「想一步」，而是「想一步 → 做一个动作 → 收到环境反馈」的完整闭环[^2]。

LangChain 官方维护的 ReAct agent 模板把这一循环写成工程化的五步：接收用户 query → 推理并决定动作 → 用可用工具执行动作 → 观察动作结果 → 重复第 2–4 步直到能给出最终答案[^21]。该循环结构与 CoT 的单次生成形成结构性对照[^21]。

### 2.2 Prompt 构成与「何时想、何时做」的自主权

ReAct prompt 由 few-shot task-solving trajectories 组成，其中包含人工书写的 text reasoning traces 与 actions，以及针对 actions 的环境 observations[^3]。关键设计选择在于：针对决策类任务，作者设计的是带稀疏 reasoning traces 的人类轨迹，让 LM 自行决定何时 think、何时 act[^3]。也就是说，推理的发生时机由模型在轨迹中自行判断，而非由提示强制每步都推理[^3]。

这一点常被忽略但很重要：ReAct 并不要求「每步必想」，它要求的是「想与做共享同一条轨迹、同一套上下文」。这与 CoT 的「必须先把推理链写完再给答案」是两种不同的控制结构。

### 2.3 实验证据及其边界

论文侧报告的定性结论是：在 HotpotQA 与 Fever 上，ReAct 通过与简单 Wikipedia API 交互，克服了 chain-of-thought reasoning 中普遍存在的幻觉与错误传播问题[^4][^5][^7][^12][^20]。在两个交互式决策基准 ALFWorld 与 WebShop 上，ReAct 相比模仿学习与强化学习方法分别取得 34% 与 10% 的绝对成功率提升[^4][^7][^20]，且上述决策任务结果仅用一到两个 in-context 示例进行提示即取得[^4][^7][^20]。

必须同时记录这些数字的证据强度。知识库内的报告自述：HotpotQA/Fever 部分缺少原始论文 Table 1/2 的具体 EM/准确率数字，ALFWorld/WebShop 部分缺少具体成功率数值与基线名称，也未说明是 PaLM-540B 还是 GPT-3 设定[^4][^11][^20]。因此 34% / 10% 只能作为摘要级数字引用，无法在库内独立验证，也无法判断其原始表述是相对提升还是绝对提升[^11]。

## 三、纯 Chain-of-Thought 的定义与边界

要做区别，先要把 CoT 的定义写全。

**CoT 的标准定义**：生成一条思维链，即「一系列中间推理步骤」（a series of intermediate reasoning steps）[^13]。**实现方式**：few-shot prompting，在提示中提供少量思维链示范作为示例（a few chain of thought demonstrations are provided as exemplars in prompting）[^13]。**能力来源**：该推理能力在足够大的语言模型中自然涌现（emerge naturally in sufficiently large language models），即存在模型规模门槛[^13]。**任务覆盖**：算术（arithmetic）、常识（commonsense）、符号推理（symbolic reasoning）三类[^13]。**标志性结果**：用 8 条 CoT 示例提示 540B 参数模型，在 GSM8K 数学应用题基准上达到 SOTA 准确率，超过带 verifier 的微调 GPT-3[^13]。

**Zero-shot CoT** 是同一思路的极简形式：只需在每次回答前加上 "Let's think step by step"，无需任何手工构造的 few-shot 示例，单一提示模板即可跨任务泛化[^14]。在 text-davinci-002 上，MultiArith 准确率从 17.7% 提升到 78.7%，GSM8K 从 10.4% 提升到 40.7%[^14]。该论文把 CoT 所针对的任务描述为不遵循标准 scaling law 的困难 System-2 任务[^14]。

**变体谱系**：Plan-and-Solve 指出 Zero-shot-CoT 存在三类缺陷——计算错误、缺失步骤错误、语义误解错误；PS Prompting 分两步，先制定计划将整个任务分解为更小的子任务，再按计划执行子任务[^15]。Tree of Thoughts 批评 LM 推理被限制在 token 级、从左到右的决策过程，在需要探索、策略性前瞻或初始决策关键的任务上表现不足；ToT 泛化 CoT，允许在「连贯的文本单元（thoughts）」上进行探索，支持多路径推理、自我评估、前瞻与回溯[^15]。

**共同边界（这是回答本题的关键）**：所有 CoT 变体（zero-shot、self-consistency、plan-and-solve、ToT）的输出空间始终是文本推理步骤，没有任何一个变体引入「动作触发环境观测」这一环节[^15]。

## 四、与纯 CoT 的关键区别

### 4.1 区别一：输出空间与循环结构

CoT 的输出空间是文本推理步骤，生成是一次性的、单向的：推理链写完，答案给出[^15]。ReAct 的输出空间包含 thought、action、observation 三类元素，生成结构是「生成 → 执行动作 → 观测 → 再生成的循环」[^6]。LangChain 模板把这一循环具体化为可执行的控制流[^21]。

这一区别不是「更长 vs 更短」，而是「开环 vs 闭环」：CoT 的推理链一旦开始生成，就没有外部信号可以改变它；ReAct 的每一步 action 都会带回一个模型无法自行编造的 observation，从而改变后续生成的条件。

### 4.2 区别二：是否接地于外部环境

项目页的对比说明把两侧的失效方向讲得很清楚：reason-only baseline（即 chain-of-thought）因未接地于外部环境而受错误信息影响；act-only baseline 因缺乏推理而无法综合出最终答案；ReAct 以可解释且事实性的轨迹解决任务[^5]。论文侧对应表述为：ReAct 通过与简单 Wikipedia API 交互，克服了 chain-of-thought reasoning 中普遍存在的幻觉与错误传播问题[^5][^7][^12]。

对照表把这一维度形式化为：ReAct 的知识来源是「通过 action 与外部来源（知识库或环境）交互」，接地性记为「接地于外部环境，轨迹被主张为事实性」[^6]。反过来说，CoT 的知识来源完全在模型参数内部，推理链条再长也不引入新信息[^16]。这也是 CoT 侧幻觉问题的结构性根源——幻觉在此语境下被定义为「与来源冲突、或无法被事实知识验证的内容」，即「生成看似合理但不符合事实的内容」[^17]。

### 4.3 区别三：失败模式的方向相反

CoT 侧的失败模式：未接地导致幻觉与错误传播，且推理链条中的错误会沿链条传播；Zero-shot-CoT 还存在计算错误、缺失步骤错误、语义误解错误三类缺陷[^16]。标准 CoT 的单路径贪心解码存在明显性能上限，self-consistency 的增益（GSM8K +17.9% 等）正是这一上限的量化体现[^16]。需要补充 self-consistency 的适用前提边界：其前提是「存在唯一正确答案」，对开放式生成或事实性无唯一答案的任务，其边缘化机制的有效性在现有证据中未获确认；且现有证据覆盖的是推理基准的准确率提升，而非幻觉率或事实性基准的直接降低[^18]。

ReAct 侧的失败模式：项目页承认并非完美，给出 ALFWorld 失败示例[^6]。但知识库内关于 ReAct 自身失败模式（推理灵活性不足、陷入循环、无法回溯）只有这一句定性描述，缺少论文中的具体失败案例与错误率统计，因此无法给出「CoT 的幻觉/事实错误 vs ReAct 的推理不灵活」这一两侧失败模式的实证对比[^24]。项目页主张 ReAct 以「可解释且事实性的轨迹」解决任务，但证据基础是 HotpotQA/Fever 上的定性结论，缺少细粒度事实性错误率统计，只能视为作者主张而非已量化结论[^24]。

### 4.4 区别四：可检查性与纠错接口

由于轨迹包含 thought、action、observation 三类元素，人工检查时不仅能读到模型「怎么想」，还能读到模型「做了什么」以及环境「返回了什么」[^6]。项目页进一步主张 ReAct 格式便于人工检查，并可通过修改少量 model thoughts 进行行为纠正[^6]。CoT 的推理链虽然也可读，但缺少 action 与 observation 这两个可外部核对的锚点。

### 4.5 一张对照表（据库内证据重建）

| 维度 | 纯 CoT（含变体） | ReAct |
|---|---|---|
| 输出元素 | 文本推理步骤[^15] | thought / action / observation[^6] |
| 生成结构 | 单次生成、单向推理链[^15][^21] | 生成 → 执行动作 → 观测 → 再生成的循环[^6][^21] |
| 知识来源 | 模型参数内部[^16] | 通过 action 与外部来源（知识库或环境）交互[^6] |
| 接地性 | 未接地，受错误信息影响[^5] | 接地于外部环境，轨迹被主张为事实性[^6] |
| 推理时机 | 提示强制先推理再作答[^13][^14] | 模型自行决定何时 think、何时 act[^3] |
| 主要失败模式 | 幻觉与错误传播、计算/缺失步骤/语义误解错误[^16] | 项目页承认并非完美，给出 ALFWorld 失败示例[^6]；缺实证统计[^24] |

## 五、ReAct 在工程生态中的位置：一个未调和的矛盾

库内存在两条方向相反的记录，必须并列呈现。

一侧是「已被吸收」的说法：知识库登记的 ReAct 官方代码仓库条目为 ysymyth/ReAct（GitHub repository），日期 2022-11-13[^10]；该仓库条目在报告中被用于支撑「ReAct 已被主流框架吸收为标准 agent 类型」的说法[^10]。同时，LangChain 的 basic agent loop、LlamaIndex 的 agent 循环、OpenAI Agents SDK 的 agent loop、AutoGen 的工具调用序列被描述为同构的循环结构[^23]。

另一侧是命名层面的取代：LangGraph v1 发布说明设有专门小节「Deprecation of create_react_agent」，该预构建组件已被弃用，转而推荐 LangChain 的 create_agent，官方理由是 create_agent 提供更简单的接口，并通过引入 middleware 提供更强的定制能力[^22]。LangChain v1 以 create_agent 为新标准，LlamaIndex 默认 FunctionAgent，OpenAI Agents SDK 与 AutoGen 的循环均未命名为 ReAct[^11]。

结论只能是受限的：循环结构同构，但不能据此断言 ReAct 仍是当前框架的默认命名范式或默认实现；最直接以 ReAct 命名的预构建组件 create_react_agent 已被官方弃用，而这两条证据之间的矛盾在给定材料中未被调和[^23][^11]。

## 六、矛盾与不确定性

1. **HotpotQA / Fever 的定量对比缺失**。论文侧仅有「通过与简单 Wikipedia API 交互克服幻觉与错误传播」这一句定性转述，缺少原始论文 Table 1/2 的具体 EM/准确率数字（PaLM-540B 下 ReAct vs CoT vs CoT-SC 的对比）[^11][^20]。当前知识库未收录该信息。

2. **ALFWorld / WebShop 的 34% 与 10% 无法独立验证**。数字来自论文摘要，但缺少具体成功率数值与基线名称，且未说明是 PaLM-540B 还是 GPT-3 设定[^11][^20]。当前知识库未收录该信息。

3. **ReAct 相对 CoT 的代价完全无覆盖**。token 消耗、平均交互步数、推理延迟、成本，在全部证据中均无量化或定性数据。当前知识库未收录该信息。这意味着「ReAct 用更多交互换接地性」这一直觉判断在本报告中无法被证据支撑。

4. **ReAct 失败模式缺乏实证**。仅有项目页「并非完美 + 一个 ALFWorld 失败示例」这一句定性描述，缺少具体失败案例与错误率统计[^24]。因此 4.3 节中两侧失败模式的对比只能停留在机制层面，不能升级为实证结论。

5. **「已被主流框架吸收为标准 agent 类型」与官方弃用公告直接矛盾**，且该矛盾在库内被明确标注为未调和[^11][^22][^23]。这是本课题的 blocking 级不确定性：任何关于 ReAct 当前工程地位的强断言都缺乏一致证据。

6. **工程热度维度无证据**。参考来源列表未包含任何仓库的 stars、forks 或最近提交日期[^25]；ysymyth/ReAct 条目仅有 2022-11-13 的登记日期[^10]，langchain-ai/react-agent 仅有 2024-08-21 日期[^25]。当前知识库未收录该信息。

7. **论文版本标注不一致**。知识库一处登记为 arXiv:2210.03629v3（2022-10-06）[^8]，另一处登记为 arXiv:2210.03629 无版本号（2022-10-06）[^9]，两者并存。本报告引用时未区分版本差异，涉及具体数字时以摘要级表述为准。

8. **学界对 ReAct 的后续批判与改进无覆盖**。Reflexion、ReWOO 在全部证据中零出现；Plan-and-Solve 仅作为 CoT 变体被提及（针对 zero-shot CoT 的三类缺陷），未涉及对 ReAct 的批判[^15]。当前知识库未收录该信息。库内两条相邻证据不能替代：Cognition 关于长时运行、多轮对话脆弱性的论点明确归属为多智能体架构，不是对单 agent ReAct 循环的实证分析[^26]；Agentic RAG 的三类控制循环失败模式（Retrieval Thrash、Tool storms、Context bloat）来源为第三方分析、非 arXiv 论文，且针对 Agentic RAG 而非 ReAct 本身[^27]。

## 参考来源

[^1]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 1.1 问题起点：reasoning 与 acting 的割裂 — reports/eval/reports/gq-react.md · ### 1.1 问题起点：reasoning 与 acting 的割裂 · L11-14（kb）
[^2]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 1.2 核心机制：交错生成 reasoning traces 与 actions — reports/eval/reports/gq-react.md · ### 1.2 核心机制：交错生成 reasoning traces 与 actions · L15-23（kb）
[^3]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 1.3 Prompt 构成与「何时想、何时做」的自主权 — reports/eval/reports/gq-react.md · ### 1.3 Prompt 构成与「何时想、何时做」的自主权 · L24-27（kb）
[^4]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 1.4 实验证据 — reports/eval/reports/gq-react.md · ### 1.4 实验证据 · L28-35（kb）
[^5]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 3.2 区别二：是否接地于外部环境 — reports/eval/reports/gq-react.md · ### 3.2 区别二：是否接地于外部环境 · L68-73（kb）
[^6]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 3.5 一张对照表 — reports/eval/reports/gq-react.md · ### 3.5 一张对照表 · L88-100（kb）
[^7]: LLM 工具调用（Function Calling）的训练、对齐与评测 — 2.3 工具学习范式的系统化（ReAct 段落） — reports/eval/reports/gq-tool-use.md · ### 2.3 工具学习范式的系统化 · L41-46（kb）
[^8]: LLM 工具调用（Function Calling）的训练、对齐与评测 — 参考来源（ReAct 论文条目） — reports/eval/reports/gq-tool-use.md · ## 参考来源 · L206-213（kb, 2022-10-06）
[^9]: Agentic RAG：定义、架构关键差异与工程现状 — 参考来源（ReAct 论文条目） — reports/eval/reports/gq-agentic-rag.md · ## 参考来源 · L193-205（kb, 2022-10-06）
[^10]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 参考来源（ysymyth/ReAct 仓库条目） — reports/eval/reports/gq-react.md · ## 参考来源 · L134-151（kb, 2022-11-13）
[^11]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 五、矛盾与不确定性 — reports/eval/reports/gq-react.md · ## 五、矛盾与不确定性 · L120-126（kb）
[^12]: 上下文工程（Context Engineering）：定义、与提示词工程的关系与差异 — 2.3 一条清晰的历史脉络 — reports/eval/reports/gq-context-eng.md · ### 2.3 一条清晰的历史脉络 · L69-78（kb）
[^13]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 二、纯 Chain-of-Thought 的定义与边界（CoT 原始定义部分） — reports/eval/reports/gq-react.md · ## 二、纯 Chain-of-Thought 的定义与边界 · L43-51（kb）
[^14]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 二、纯 Chain-of-Thought 的定义与边界（Zero-shot CoT 部分） — reports/eval/reports/gq-react.md · ## 二、纯 Chain-of-Thought 的定义与边界 · L43-53（kb）
[^15]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 二、纯 Chain-of-Thought 的定义与边界（CoT 变体与共同边界） — reports/eval/reports/gq-react.md · ## 二、纯 Chain-of-Thought 的定义与边界 · L43-59（kb）
[^16]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 3.3 区别三：失败模式的方向相反 — reports/eval/reports/gq-react.md · ### 3.3 区别三：失败模式的方向相反 · L74-82（kb）
[^17]: LLM 幻觉的主流缓解手段及其有效性边界 — 一、幻觉的定义与度量基线 — reports/eval/reports/gq-hallucination.md · ## 一、幻觉的定义与度量基线 · L9-10（kb）
[^18]: LLM 幻觉的主流缓解手段及其有效性边界 — 3.3 Self-Consistency（适用前提边界） — reports/eval/reports/gq-hallucination.md · ### 3.3 Self-Consistency · L69-74（kb）
[^20]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别（实验证据章节） — reports/eval/reports/gq-react.md · ### 1.4 实验证据 · L28-36（kb）
[^21]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — LangChain 官方 ReAct agent 模板的循环结构 — reports/eval/reports/gq-react.md · ### 3.1 区别一：输出空间与循环结构 · L62-67（kb）
[^22]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — LangGraph v1 弃用 create_react_agent — reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置：一个未调和的矛盾 · L101-110（kb）
[^23]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 循环结构同构但 ReAct 命名被取代（未调和矛盾） — reports/eval/reports/gq-react.md · ## 四、ReAct 在工程生态中的位置：一个未调和的矛盾 · L101-116（kb）
[^24]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 矛盾与不确定性：ReAct 失败模式缺乏实证 — reports/eval/reports/gq-react.md · ## 五、矛盾与不确定性 · L120-130（kb）
[^25]: ReAct 范式的核心思想及其与纯 Chain-of-Thought 的关键区别 — 参考来源：langchain-ai/react-agent 与 LlamaIndex Agents 文档 — reports/eval/reports/gq-react.md · ## 参考来源 · L134-143（kb）
[^26]: 多智能体系统（Multi-Agent LLM）主流协作模式与适用场景 — 长时运行、多轮对话场景下的脆弱性（相邻证据） — reports/eval/reports/gq-multi-agent.md · ### 3.3 长时运行、多轮对话场景下的脆弱性 · L81-83（kb）
[^27]: Agentic RAG：定义、架构关键差异与工程现状 — 控制循环内的三类失败模式（相邻证据） — reports/eval/reports/gq-agentic-rag.md · ### 3.5 差异五：失败模式的位置发生迁移 · L93-100（kb）