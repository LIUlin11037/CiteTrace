## 摘要

ReAct 的核心思想是把「语言推理」本身当作一种**动作**，塞进 agent 的动作空间里，让模型在同一个自回归循环中交错生成推理轨迹（thought）与任务动作（action），从而形成「推理以行动、行动以推理」的双向闭环[^1]。它与纯 Chain-of-Thought 的关键区别不在于「是否分步思考」，而在于**思考是否与外部环境 grounding**：CoT 的中间步骤完全在模型内部表示中产生、不改变外部世界、不接收任何 observation 反馈，ReAct 论文直接将这种形态批评为 static black box，认为它限制了反应式推理与知识更新能力，导致事实幻觉与推理过程中的错误传播[^1]。形式化上，这一区别表现为动作空间的增广：ReAct 定义 Â = A ∪ L，其中 L 是语言空间，â_t ∈ L 即 thought，它不改变外部环境、因而不产生 observation，只把 context 更新为 c_{t+1} = (c_t, â_t)[^1]。需要提前声明：本课题最关键的量化对比（ReAct vs 纯 CoT 在 HotpotQA/FEVER 上的具体数值、消融逐项结果）在现有证据中缺失，下文结论的强度受此限制（见「矛盾与不确定性」）。

## 一、ReAct 的形式化定义：把 thought 纳入动作空间

ReAct 的设定是一个标准的 agent-环境交互循环：t 时刻 agent 从环境收到 observation o_t ∈ O，按策略 π(a_t|c_t) 采取 action a_t ∈ A，context 为 c_t = (o_1, a_1, …, o_{t-1}, a_{t-1}, o_t)[^1]。

其核心形式化操作是**增广动作空间**：Â = A ∪ L，L 为语言空间[^1]。语言空间中的动作 â_t ∈ L 被称为 thought 或 reasoning trace，它不改变外部环境，因此不产生 observation 反馈；它的作用是对当前 context c_t 做推理以组织有用信息，并把 context 更新为 c_{t+1} = (c_t, â_t)，以支撑后续推理或行动[^1]。这一形式化在论文初版（v1）中即已完整存在，v3 未作实质改动[^2]；v3 相对前版的差异主要是 ICLR camera-ready 的笔误修正与排版扩充（文件体积从 538 KB 增至 1,256 KB）[^3]。

论文列举的 thought 类型包括：分解任务目标并制定行动计划、注入任务相关常识知识、从 observation 中抽取关键部分、跟踪进度并切换行动计划、处理异常并调整计划[^1]。

值得注意的是，由于语言空间 L 无界，在增广动作空间中「学习」很困难、需要强语言先验；论文因此主要采用冻结的 PaLM-540B，用少样本 in-context 示例提示模型同时生成领域动作与自由形式语言 thought，每个示例都是一条包含 actions、thoughts 与 observations 的人类轨迹[^1]。这决定了 ReAct 在原始论文中首先是一个**提示范式**而非训练方法。

## 二、ReAct 作为通用范式：reason to act 与 act to reason

论文把 ReAct 定义为一种通用范式：提示 LLM 以交错方式（interleaved manner）生成语言推理轨迹与任务相关动作，实现两个方向的双向协同——reason to act（动态推理以创建、维护、调整高层计划）与 act to reason（与 Wikipedia 等外部环境交互，把额外信息纳入推理）[^1]。官方项目页对 ReAct prompt 的定义与此一致：一个 ReAct prompt 由少样本任务求解轨迹构成，其中包含人工撰写的文本推理轨迹与动作，以及针对动作的环境 observation 反馈[^4]。

循环的具体形态**因任务而异**，这一点常被二次传播简化为「Thought-Action-Observation 三步固定循环」，与原文不符：

- 推理为主的任务（HotpotQA/FEVER）：交替生成 thought 与 action，求解轨迹由多个 thought-action-observation 步骤组成[^1]。
- 可能涉及大量动作的决策任务（ALFWorld/WebShop）：thought 只需稀疏地出现在轨迹中最相关的位置，由语言模型自行决定 thought 与 action 的异步发生[^1]；项目页亦确认 ALFWorld 上设计的是带稀疏推理轨迹的人类轨迹，让模型自行决定何时思考、何时行动[^4]。

动作空间方面，Wikipedia API 含三类动作：search[entity]（返回实体 wiki 页前 5 句，否则返回 top-5 相似实体）、lookup[string]（返回页面中包含该字符串的下一句，模拟浏览器 Ctrl+F）、finish[answer]（返回最终答案）[^1]。提示构造上，HotpotQA 与 FEVER 的 ReAct prompt 分别从训练集随机抽取 6 个和 3 个案例人工构造，论文脚注称更多示例并未提升性能[^1]。

## 三、与纯 Chain-of-Thought 的关键区别

### 3.1 CoT 的权威定义（作为对照基准）

CoT 的定义是「a series of intermediate natural language reasoning steps that lead to the final output」，即通向最终输出的一系列中间自然语言推理步骤[^5]。其 prompting 输入形式是三元组 ⟨input, chain of thought, output⟩，即在 few-shot 示例中把答案替换为分步推理过程[^5]。作者明确区分 CoT 与「解释」：解释通常出现在最终答案之后，而 CoT 在答案之前[^5]。CoT 是模型规模的涌现能力，对小模型无正向作用，仅在约 100B 参数以上模型才带来增益；小模型会生成流畅但不合逻辑的推理链，性能反而低于标准 prompting[^5]。

关键点在于：CoT 的推理链是**在输出最终答案之前、在模型内部完成的连续语言序列**，其定义中不含任何动作、环境或 observation 的概念[^5]。Zero-shot-CoT 进一步把这一形态压缩为一句触发句（如「Let's think step by step」）加两阶段 prompting，同样不涉及外部交互[^6]；Self-Consistency 则在 CoT 之上做 sample-and-marginalize 解码，采样多条推理路径后对最终答案多数投票，仍是单模型上的「self-ensemble」，无需额外训练或辅助模型[^7]。这三者构成「纯 CoT 家族」的完整谱系：**推理路径的数量可以增加，但与外部世界的接口始终为零**。

### 3.2 区别的四个层面

**（1）动作空间：是否把语言纳入 A。** CoT 的推理步骤不是动作空间中的元素，模型只产生一条通向答案的语言序列[^5]；ReAct 显式定义 Â = A ∪ L，把 thought 变成与 search/lookup/finish 并列的可选动作[^1]。这是两者在形式化层面的根本差异，也是其他差异的来源。

**（2）反馈通道：是否接收 observation。** ReAct 的 thought 不改变环境、不产生 observation，但**action 会**——action 的结果作为 observation 回流进 context，构成 c_t 的组成部分[^1]。CoT 的定义与输入形式中不存在 observation 这一项[^5]。ReAct 论文据此批评 CoT 是 static black box：模型用自身内部表示生成 thought，未与外部世界 grounding，限制了反应式推理与知识更新能力，导致事实幻觉与推理过程中的错误传播[^1]。项目页对 CoT（reason-only）基线的定性相同：因未与外部环境 grounding 以获取和更新知识，只能依赖有限内部知识，因而受错误信息影响[^4]。

**（3）知识来源：内部参数 vs 外部检索。** HotpotQA/FEVER 实验采用 question-only 设置，模型只收到问题/声明，不提供支持段落，须依赖内部知识或与外部环境交互检索[^1]。这使 ReAct 与 CoT 的对比成为「内部知识 vs 外部检索信息」的对比。论文报告：在 HotpotQA/FEVER 上 ReAct 优于纯动作生成模型、与 CoT 有竞争力，而整体最佳方法是 ReAct 与 CoT 的组合，兼顾内部知识与外部检索信息[^1]；ar5iv 交叉核对版本给出同样的引言级表述[^15]。这一「组合最优」的结论本身即说明二者不是替代关系，而是互补的信息源。

**（4）错误结构：单链传播 vs 可外部纠正。** CoT 的错误传播问题在后续文献中被明确刻画：Self-Consistency 指出模型并非完美推理者，可能在某个推理步骤中出错，而错误路径较不可能收敛到同一答案[^7]；CoT 原论文的错误分析显示，50 个答错样本中 46% 的推理链「几乎正确」（仅计算器错误、符号映射错误或漏一步），另 54% 存在语义理解或连贯性上的重大错误[^5]。ReAct 的应对不是采样投票，而是让错误暴露在可观察的轨迹上：论文报告 ReAct 在可解释性、可信度与可诊断性上的收益——人类可区分模型内部知识与外部环境信息，并可通过 thought editing 在过程中控制或纠正 agent 行为[^1]；项目页亦承认 ReAct 并不完美并给出 ALFWorld 失败案例，同时指出 ReAct 格式便于人工检查、可通过修改少量模型 thought 来纠正行为[^4]。

**（5）性能归因的边界。** 论文自述四项贡献：提出 ReAct 提示范式；在多基准上展示 few-shot 下优于单独推理或单独动作生成；给出系统消融分析（推理任务中 acting 的重要性、交互任务中 reasoning 的重要性）；分析 prompting 设置的局限并给出初步 finetuning 实验[^1]。在 ALFWorld 与 WebShop 上，1-shot 或 2-shot 的 ReAct prompting 即超过用 10³~10⁵ 个任务实例训练的模仿学习或强化学习方法，成功率绝对提升分别为 34% 与 10%[^1]；v1 摘要已含同样表述[^2]。项目页补充初步 finetuning 结果（HotpotQA，使用 ReAct prompting 轨迹）：ReAct 是跨模型尺寸最佳的 finetuning 格式，且 ReAct finetuned 的较小模型可超过 prompted 的较大模型[^4]。但需注意，这些数字**不能**直接用来回答「ReAct 与纯 CoT 的区别」——它们对照的是模仿学习/RL 与 act-only，而非 CoT。

### 3.3 一句话概括区别

纯 CoT 是「在封闭语言空间内把答案想出来」，ReAct 是「把想的过程也当作动作，边想边在外部世界里查、试、改」——前者优化的是单条推理链的质量，后者优化的是推理与外部信息之间的耦合[^1][^5]。

## 四、范式定位：ReAct 在后续工作中的位置

ReAct 被后续 agent 研究普遍视为可继承、可批评、可插拔的基础范式，这从侧面印证了它与 CoT 的谱系关系：

- **Reflexion** 把 ReAct 列为所继承的 LLM agent 范式之一（与 SayCan、Toolformer、HuggingGPT、WebGPT 并列），并指出这类方法「一直局限于用 in-context examples 来教 agent」[^16]；在架构上把 ReAct 当作可插拔的 Actor 生成模型之一（与 Chain of Thought 并列），在其上增加 memory 组件[^16]。Reflexion 的批评落点是「无法从试错中快速有效地学习」，解法是用语言反馈作为「semantic gradient signal」跨 episode 强化[^16]。它以 ReAct 为基线：在 AlfWorld 134 个任务上 ReAct + Reflexion 显著优于 ReAct，完成 130/134，且 ReAct-only 的性能在 trial 6 与 7 之间停止增长[^16]。
- **Tree of Thoughts** 的批评更彻底，指向自回归的 token 级、从左到右决策机制本身，把包括 CoT、CoT-SC 在内的既有方法概括为「沿单一连续语言序列采样」，转而维护一棵 thought 树并支持前瞻与回溯[^17]。ToT 的 thought 定义是「a coherent language sequence that serves as an intermediate step」，与 ReAct 的 thought 同属语言空间，但 ToT 不引入环境动作[^17]。
- **ToolLLM** 把已有工具学习工作的推理方式概括为采用 CoT 或 ReAct，并批评其无法充分激发 LLM 能力、难以处理复杂指令，提出 DFSDT 作为「our improved reasoning strategy over ReACT」，使模型能评估多条推理路径并回退[^18][^20]；其 ToolEval 以 ChatGPT-ReACT 作为 win rate 的对照基线[^18]。
- **Toolformer** 走的是另一条路：把 API 调用表示为可插入普通文本流的特殊 token 序列，通过自监督过滤（调用是否降低后续 token 的 loss）微调模型自身，使工具调用内化为语言建模序列的一部分而非外部提示模板[^19][^21]。

工程侧，ReAct 循环已被模板化、图结构化：LangChain 官方把 ReAct agent 作为 LangGraph 模板提供，README 提供「Open in LangGraph Studio」入口，表明循环被表达为可在图形化 Studio 中调试的图结构而非纯 prompt 字符串[^8]；亦有仓库把 LangGraph ReAct agent 与 MCP 工具协议组合[^10]；教学仓库把 ReAct 与 reflection 工作流并列讲授[^11]。但生态正在迁移：LlamaIndex 文档明确区分 FunctionAgent（使用 LLM provider 的 function/tool calling 能力执行工具）与 ReActAgent/CodeActAgent（使用「different prompting strategies」执行工具），当前主推的是 FunctionAgent[^22]；LangChain 当前 agent 入口是 `create_agent`，文档中未出现 `create_react_agent` 的用法示例，并给出「Agent = Model + Harness」的定义式表述[^23]。这说明 ReAct 作为**思想**已被吸收，而作为**显式 Thought/Action/Observation 文本循环**的实现形态正在被原生 tool calling 取代——低热度的 LlamaIndex ReAct 示例仓库（3 stars）只能证明该形态仍存在，不能作为生态主流形态的证据[^13]。AutoGPT 的 README 定位已从早期「自主 ReAct 循环」转向多专家 agent 团队产品形态[^9]，AgentGPT 最近 push 停留在 2025-04-29，明显早于同期 LangGraph 相关仓库[^14]，可作自主 agent 热度回落的旁证。

## 矛盾与不确定性

**（1）课题核心结论缺乏量化支撑。** ReAct 在 HotpotQA/FEVER 上「优于纯动作生成模型、与 CoT 有竞争力，整体最佳为 ReAct+CoT 组合」这一判断，在证据中仅有引言/摘要级定性表述[^1][^15]，**没有任何具体数值**（EM、精确匹配、幻觉率）。因此本报告无法给出「ReAct 相对纯 CoT 提升/下降多少」的量化结论，第三节的区别论述全部是机制层面的，而非效果层面的。这是本课题最大的证据缺口。

**（2）关键实验表格两次抓取均缺失。** ALFWorld 与 WebShop 上 34% 与 10% 的绝对成功率提升来自摘要[^1][^2]，但支撑它的第 4 节表格（含 Act-only、ReAct、ReAct+CoT 等对照与具体成功率）在两次抓取中均缺失，无法核对基线设置与对照条件[^15]。因此这两个数字只能作为「作者自述」引用，不能用于与 CoT 做条件对等的比较。

**（3）消融归因链条未闭合。** 论文自述有系统消融（推理任务中 acting 的重要性、交互任务中 reasoning 的重要性）[^1]，但证据中没有任何消融表格的具体结果（如 ReAct vs CoT vs CoT-SC vs Act-only 在各基准上的逐项数字）。因此「收益来自交错推理与行动这一结构本身，而非示例数量、模型规模或任务难度」这一归因**现有公开资料不足以确认**。

**（4）项目页主张的营销性质。** 项目页称 ReAct 在从问答到在线购物等多种任务上取得 state-of-the-art 的 few-shot 表现，这是作者自述的营销性主张，需以论文表格交叉验证[^4]——而如上所述，验证所需的表格在证据中缺失。

**（5）版本差异不影响定义，但影响引用精度。** v1 与 v3 在第 2 节形式化表述、摘要措辞、四项贡献列表上实质一致，差异集中在作者脚注与 Figure 1 caption 措辞[^2]；v3 的 arXiv comments 明确为「ICLR camera ready version with some typos fixed」[^3]。因此引用形式化定义时 v1/v3 可互换，但引用图表编号时应以 v3 为准。

**（6）工程侧证据强度不均。** ReAct 在工程生态中的「主流性」判断受证据质量限制：LangChain/LlamaIndex 官方文档显示的是**迁移趋势**（转向 function calling 与 create_agent）[^22][^23]，而仍以 ReAct 命名的仓库中，LangGraph 模板有官方背书且持续维护[^8]，MCP 集成仓库与教学仓库热度中等[^10][^11]，LlamaIndex ReAct 示例仓库热度极低[^13]。三者不能等权视为「ReAct 仍是主流实现形态」的证据。

## 参考来源

[^1]: ReAct: Synergizing Reasoning and Acting in Language Models (arXiv:2210.03629v3, ICLR camera-ready) — https://arxiv.org/html/2210.03629v3（arxiv, 2023-03-10）
[^2]: ReAct: Synergizing Reasoning and Acting in Language Models (arXiv:2210.03629v1, 初版) — https://arxiv.org/html/2210.03629v1（arxiv, 2022-10-06）
[^3]: arXiv:2210.03629 版本元数据与提交历史（v1/v2/v3） — https://arxiv.org/abs/2210.03629v3（arxiv, 2023-03-10）
[^4]: ReAct 官方项目页（react-lm.github.io） — https://react-lm.github.io/（web）
[^5]: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022) — https://arxiv.org/html/2201.11903v6（arxiv, 2022-01-28）
[^6]: Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022) — https://arxiv.org/html/2205.11916v4（arxiv, 2022-05-24）
[^7]: Self-Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al., 2022) — https://arxiv.org/html/2203.11171v4（arxiv, 2022-03-21）
[^8]: langchain-ai/react-agent — LangGraph template for a simple ReAct agent — https://github.com/langchain-ai/react-agent（github, 2024-08-21）
[^9]: Significant-Gravitas/AutoGPT — https://github.com/Significant-Gravitas/AutoGPT（github, 2023-03-16）
[^10]: braincrew-lab/langgraph-mcp-agents — LangGraph ReAct agent with MCP integration — https://github.com/braincrew-lab/langgraph-mcp-agents（github）
[^11]: emarco177/langgraph-course — ReAct 与 reflection 工作流的教学仓库 — https://github.com/emarco177/langgraph-course（github）
[^13]: fuyuantan/LlamaIndex-ReAct-Example — 打印 ReAct 系统提示的示例 — https://github.com/fuyuantan/LlamaIndex-ReAct-Example（github）
[^14]: reworkd/AgentGPT — 浏览器端自主 agent — https://github.com/reworkd/AgentGPT（github）
[^15]: ReAct 论文 ar5iv HTML 全文（第 1–3 节，交叉核对） — https://ar5iv.labs.arxiv.org/html/2210.03629（arxiv）
[^16]: Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366v4) — https://arxiv.org/html/2303.11366v4（arxiv, 2023-03-20）
[^17]: Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601v2) — https://arxiv.org/html/2305.10601v2（arxiv, 2023-05-17）
[^18]: ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs (arXiv:2307.16789v2) — https://arxiv.org/html/2307.16789v2（arxiv, 2023-07-31 (v1); v2 2023-10-03）
[^19]: Toolformer: Language Models Can Teach Themselves to Use Tools (arXiv:2302.04761) — https://arxiv.org/html/2302.04761v1（arxiv, 2023-02-09）
[^20]: ToolLLM 摘要页（arXiv:2307.16789v2 元数据与摘要） — https://arxiv.org/abs/2307.16789v2（arxiv, 2023-10-03 (v2)）
[^21]: Toolformer 摘要页（arXiv:2302.04761 元数据与摘要） — https://arxiv.org/abs/2302.04761（arxiv, 2023-02-09）
[^22]: LlamaIndex Docs — Deploying Agents（Agents 总览） — https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/（web）
[^23]: LangChain Docs — Agents（create_agent 主文档，原 migrate_agent 路径） — https://python.langchain.com/docs/how_to/migrate_agent/（web）