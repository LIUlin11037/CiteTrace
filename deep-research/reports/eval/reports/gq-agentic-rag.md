# Agentic RAG：定义、架构差异与工程落地证据

## 摘要

Agentic RAG 指把自主 AI agent 嵌入 RAG 流水线，使系统运用 reflection、planning、tool use、multi-agent collaboration 等 agentic design patterns，动态管理检索策略、迭代精炼上下文理解，并以从顺序步骤到自适应协作的运作结构适配工作流[^1]。其对照面是传统 RAG：后者被描述为静态且线性（static and linear）的流水线，限制了复杂多步推理、深层上下文理解与迭代式回答精炼[^1][^2]。架构上的关键差异可归纳为四点：(1) 引入显式控制层，由系统自行决定何时检索、如何改写查询、何时证据已足够[^2]；(2) 检索从一次性前置步骤变为可循环、可分支、可中断的图结构执行[^29][^32]；(3) 质量判定从外部离线指标内化为运行时的反思/评估动作（reflection token、retrieval evaluator、复杂度分类器）[^17][^19][^9]；(4) 状态与记忆成为一等公民，跨轮次、跨 agent 持久化[^28][^30]。需要说明的是，本报告所依据的证据在量化对比（延迟、token、检索次数倍数）与失败案例方面覆盖不足，相关结论显式标注为不确定。

---

## 1. 定义：Agentic RAG 是什么

### 1.1 权威定义与构成要素

arXiv:2501.09136 综述给出的定义性表述是：将自主 AI agent 嵌入 RAG 流水线，agent 运用 reflection、planning、tool use、multi-agent collaboration 等 agentic design patterns，动态管理检索策略、迭代精炼上下文理解，并通过从顺序步骤到自适应协作的运作结构来适配工作流[^1]。该定义可拆解为四个可检验要素：

- **输入**：用户查询（可能含对话历史）+ 可访问的外部知识源与工具集；
- **控制主体**：一个或多个具备自主决策能力的 agent，而非固定的代码路径；
- **运作机制**：agentic design patterns（reflection / planning / tool use / multi-agent collaboration）[^1][^7]，在系统层面进一步由 workflow patterns 组织——prompt chaining、routing、parallelization、orchestrator–worker、evaluator–optimizer 循环[^2][^7]；
- **产出**：经迭代精炼、可被外部证据支撑的回答，以及（在工程实现中）可检视的执行轨迹与状态[^31][^35]。

该综述同时指出领域现状是架构多样、术语不一致、缺乏比较性理解，此前没有统一分类法沿原则性设计维度组织 Agentic RAG 系统，这是其立论动机[^2]。它提出的分类法基于四个维度：agent cardinality（智能体数量）、control structure（控制结构）、autonomy（自主性）、knowledge representation（知识表示）[^1][^7]；第 5 节按 single-agent、multi-agent 与 graph-based 框架展开[^2]。

### 1.2 与传统 RAG 的定义性对照

传统 RAG 的经典形式有明确的形式化定义：由检索器 p_η(z|x)（参数 η，给定查询 x 返回文本段落的 top-K 截断分布）与生成器 p_θ(y_i|x,z,y_{1:i-1})（参数 θ，基于前 i-1 个 token、原始输入 x 与检索段落 z 生成当前 token）两个组件构成[^4]。检索器采用 DPR 的 bi-encoder 架构，p_η(z|x) ∝ exp(d(z)^T q(x))，其中 d(z)=BERT_d(z) 为文档编码器输出、q(x)=BERT_q(x) 为查询编码器输出；top-k 检索是 Maximum Inner Product Search 问题，可亚线性近似求解，文档索引被称为 non-parametric memory，BART-large 生成器参数 θ 被称为 parametric memory[^4]。检索内容与输入 x 的结合方式是直接拼接[^4]。

这一形式化的关键含义是：**检索发生在生成之前，且只发生一次（或固定次数）**。综述将这一形态概括为 Naive RAG 的三步流程 indexing → retrieval → generation，亦称 "Retrieve-Read" 框架[^5]；Agentic RAG 综述则把演进链条表述为 Naïve RAG（TF-IDF/BM25 关键词检索 + 静态数据集）→ Advanced RAG（dense retrieval 如 DPR、neural re-ranking、iterative/multi-hop retrieval）→ Modular RAG（pipeline 分解为独立可复用组件、支持 hybrid retrieval 与外部工具集成）[^7]，与 Gao 等的 Naive / Advanced / Modular 三范式划分一致[^3][^5]。

---

## 2. 架构关键差异（一）：从静态线性流水线到显式控制层

传统 RAG 的静态性不是实现缺陷而是架构属性：检索器与生成器在没有任何「应检索哪个文档」直接监督的情况下联合端到端训练，目标是最小化负边际对数似然[^4]；检索文档被当作隐变量，通过 RAG-Sequence（同一文档预测所有目标 token）或 RAG-Token（每个 token 可用不同文档）边缘化[^4]。整条链路没有「是否该检索」「检索够了没有」的判定位置。

Agentic RAG 的核心架构增量是**显式控制层（explicit control layer）**，用于指导系统如何对外部证据进行推理——协调检索、验证信息、调用工具、精炼回答[^2][^7]。控制层把原先由调用方硬编码的决策（检索与否、查询怎么写、结果够不够）移入系统内部运行时决策[^2]。

这一转变在工程框架中有直接对应物。LangGraph 把 agent 工作流建模为图：State（共享数据结构，表示应用当前快照）、Nodes（接收当前状态、执行计算或副作用、返回更新后状态的函数）、Edges（根据当前状态决定下一个执行哪个 Node 的函数，可为条件分支或固定转移），官方原文强调 "nodes do the work, edges tell what to do next"，通过组合 Nodes 与 Edges 可构建循环的（looping）工作流并随时间演进状态[^29]。底层图算法采用消息传递，灵感来自 Google Pregel，程序以离散 super-step 推进，节点在任一入边收到新消息时被激活，所有节点 inactive 且无消息在途时执行终止[^29]。Haystack 的 pipeline 同样被定义为有向多重图，官方明确「loops 与 branches 使构建 agent 这类复杂应用成为可能」，并给出 Loop-Based Auto-Correction、Agentic RAG with Fallback to Websearch 等教程[^32]。

**差异要点**：传统 RAG 的控制流是编译期固定的；Agentic RAG 的控制流是运行期由状态与条件边决定的，因而具备循环与回退能力[^29][^32]。

---

## 3. 架构关键差异（二）：检索时机的自适应化

传统 RAG 的固定次数检索被多处文献点名为失效根源。Self-RAG 指出传统 RAG 不加区分地检索并纳入固定数量的检索段落（indiscriminately retrieving and incorporating a fixed number of retrieved passages），无论检索是否必要、段落是否相关，都会削弱 LM 的通用性，或导致无用的回答生成[^6]；其项目页进一步说明标准 RAG 只检索一次或固定次数，且不评估段落相关性、也不检查输出是否真被段落支持[^18]。Gao 等综述把「面对复杂问题，基于原始查询的单次检索可能不足以获取充分上下文」列为 Naive RAG 的独立失效模式[^5]。

Agentic RAG 侧的对应机制有三条清晰的技术路线，均已在证据中出现：

**（a）反思 token 驱动的按需检索。** Self-RAG 训练单一任意 LM，通过特殊 token（reflection tokens）按需自适应检索，并对检索段落与自身生成进行反思[^6]。四类 token 为：Retrieve（是否需要检索）、IsRel（段落是否相关）、IsSup（生成是否被段落支持）、IsUse（回答是否有用，1–5 分）[^17]。推理算法在每个 segment（实验中以一句话为一段）先预测 Retrieve token，仅当判定需要检索时才调用检索器取回 top passages，否则直接生成；检索到的多个段落并行处理，并通过 reflection token 过滤不相关段落，推理时不需要额外外部模型（critic 只在训练时使用）[^17]。critic 由 GPT-4 生成的 reflection token 数据蒸馏而来，与 GPT-4 预测在多数 token 类别上一致率超过 90%，critic 与 generator 均以 Llama 2-7B 初始化[^17]。

**（b）复杂度分类器路由。** Adaptive-RAG 依据 query complexity 动态选择从最简单到最复杂的检索增强 LLM 策略，选择由一个较小的 LM 分类器实现，用自动收集的标签训练以预测 incoming query 的复杂度等级，在 no-retrieval、单步检索增强与迭代式检索增强之间切换[^9]。其形式化区分了三种策略：Non Retrieval（LLM 直接生成）、Single-step Retrieval（一次检索后生成）、Multi-step Retrieval（LLM 与 Retriever 多轮交互，每步 d_i = Retriever(q, c_i; D)，上下文 c_i 可含此前文档与答案）[^23]。官方实现中三种策略对应 ircot_qa（multi-step）、oner_qa（single-step）、nor_qa（zero/无检索），分类器为 t5-large，针对不同 LLM 各训练一个[^25]。

**（c）检索评估器触发纠正动作。** CRAG 设计轻量级 retrieval evaluator 评估检索文档整体质量，输出置信度并触发 {Correct, Incorrect, Ambiguous} 三种动作之一：Correct 走 Knowledge_Refine（decompose-then-recompose 过滤无关信息）；Incorrect 触发大规模网络搜索（先由 query rewriter 改写查询）；Ambiguous 则内部知识精炼与网络搜索两者并用[^11][^19]。阈值定义为：至少一篇文档 confidence 高于上阈值判 Correct；所有文档均低于下阈值判 Incorrect；其余为 Ambiguous[^20]。evaluator 由 T5-large 初始化并微调，参数量 0.77B，论文明确对比 Self-RAG 的 critic（指令微调的 LLaMA-2 7B）以主张自身更轻量[^11][^19]。CRAG 为 plug-and-play，论文中分别接入标准 RAG 与 Self-RAG 两种基线验证[^11]。

**（d）检索与推理交错。** IRCoT 指出 one-step retrieve-and-read 对多步 QA 不足，因为「what to retrieve」取决于「what has already been derived」，而后者又可能取决于先前检索到的内容；其做法是把检索与 CoT 中的推理步（句子）交错进行，用 CoT 引导检索、再用检索结果改进 CoT[^10]。关键超参：每步检索段落数 K ∈ {2,4,6,8}，推理步数上限 8 步，收集段落总数上限 15 段[^26]。

**差异要点**：传统 RAG 的检索次数是超参数；Agentic RAG 的检索次数是运行期输出，由 token 预测[^17]、分类器[^9]或评估器置信度[^19]决定。

---

## 4. 架构关键差异（三）：多 agent 编排与状态/记忆

传统 RAG 是两组件系统[^4]；Agentic RAG 的 taxonomy 把 agent cardinality 与 control structure 作为独立设计维度[^1][^7]，工程框架则提供了可复用的编排原语。

**编排模式。** LlamaIndex 提供三种多 agent 编排模式：内置 AgentWorkflow（声明一组 agent，由它管理交接）、内置 orchestrator 模式（一个 orchestrator agent 决定下一个调用哪个子 agent，子 agent 以 tools 形式暴露）、自定义 planner（自写 LLM 提示做规划并在代码中命令式调用 agent）[^30]。AgentWorkflow 的运行循环为：把用户消息交给 root agent → 执行该 agent 选择的工具 → 允许 agent 在它决定时把控制权 handoff 给另一个 agent → 重复直到某个 agent 返回最终答案；任意时刻当前活跃 agent 都可选择把控制权交还用户[^30]。FunctionAgent 通过 tools 列表声明可调用工具、通过 can_handoff_to 声明可交接目标、通过 system_prompt 描述角色与交接时机[^30]。Haystack 的 Agent 组件则是基于循环的系统：使用支持工具的 chat-based LLM 与外部工具，迭代解决复杂查询直到满足可配置的 exit_conditions，迭代过程为调用工具、更新状态、生成提示循环往复[^31]；其 tools 参数支持 Tool、ComponentTool、PipelineTool、AgentTool、MCPTool、Toolset、MCPToolset、SearchableToolset 等类型，即 pipeline 与 agent 都可被包装成工具供 agent 调用[^31]。

**状态与记忆。** 这是传统 RAG 架构中不存在的一层。LangGraph 提供两套互补持久化系统：checkpointer 持久化单个 thread 的图状态快照（短期、thread 级记忆，用于对话连续性、human-in-the-loop、time travel 与容错），store 持久化图状态之外的应用自定义数据（长期、跨 thread 记忆，用于用户偏好、事实与共享知识）；官方建议多数应用同时使用两者[^28]。实现细节上，checkpointer 通过 graph config 中的 thread_id 访问，store 由节点或应用代码读写条目，启用方式为 builder.compile(checkpointer=checkpointer, store=store)[^28]。子图各自管理自己的 checkpoint namespace，父图可能不会立即看到子图状态更新，跨图边界的数据应通过 Store 共享[^28]。LlamaIndex 的 AgentWorkflow 支持 initial_state 传入共享状态字典（如 research_notes、report_content、review），供各 agent 在循环中读写；orchestrator 模式下工具函数通过 ctx.store.edit_state() 读写 workflow 共享状态[^30]。Haystack 的 Agent 通过 state_schema 定义运行时状态，工具可通过 inputs_from_state 与 outputs_to_state 读写状态键[^31]。

**可观测性作为架构组件。** Haystack Agent 返回字典包含 messages、last_message、step_count、token_usage、tool_call_counts、exit_reason 及基于 state_schema 的动态键；exit_reason 取值包括 "text"、"length"、"content_filter"、满足退出条件的工具名、"max_agent_steps" 或 hook 通过 stop_run 设置的自定义原因，官方指出它便于用 ConditionalRouter 把 agent 输出路由到下游[^31]。Dify 的 Agent 节点输出包括 Final Answer、Tool Outputs、Reasoning Trace（逐步决策过程，ReAct 策略下尤其详细）、Iteration Count、Success Status、Agent Logs[^34]。这些字段在传统 RAG 的「检索→拼接→生成」链路中没有对应物。

**差异要点**：传统 RAG 的状态是单次请求内的 prompt 内容；Agentic RAG 的状态是跨步骤、跨 agent、跨会话持久化的图状态，且其演化过程本身是可观测输出[^28][^31][^34]。

---

## 5. 架构关键差异（四）：检索层本身被 agentic 化

除「agent 调用检索」外，还出现了「检索服务内部 agentic 化」的形态。Azure AI Search 把 agentic retrieval 定义为面向聊天/copilot 复杂问题的 multi-query pipeline，用于 RAG 模式与 agent-to-agent 工作流；可用基于 LLM 的 query planning（preview）把复杂查询拆解为更小、更聚焦的 subqueries，query planning 可利用 chat history 作为额外上下文[^35]。各 subquery 并行执行，每个都经过语义 rerank，随后把最佳结果合并为统一响应供 LLM 生成有依据的答案；除合并内容外还可返回 source references 与 activity log[^35]。架构分四阶段：workflow initiation（应用带 query 与 conversation history 调用 knowledge base 的 retrieve action）、query planning、query execution、result synthesis[^35]。检索推理强度可配置：minimal 跳过 query planning 直接发查询，low 与 medium 由 LLM 生成聚焦 subqueries，默认 low[^35]。文档明确说明 agentic retrieval 相比单查询管线会增加延迟（adds latency），但能处理单查询无法处理的复杂度，例如包含多个诉求的问题、依赖对话前文的问题、需要改写/同义词扩展的问题以及拼写错误[^35]。

对照 Dify 的 Knowledge Retrieval 节点可见传统形态：节点需配置检索什么（query）、在哪检索（知识库）、如何处理检索结果（节点级 retrieval settings），检索设置分知识库级与节点级两层，节点级 rerank 包括 Weighted Score、Rerank Model、Top K、Score Threshold[^33]。这是参数化配置而非运行期规划。

**差异要点**：传统 RAG 的查询改写属于 Advanced RAG 的 pre-retrieval 策略（query rewriting / transformation / expansion）[^5]；agentic retrieval 把改写升级为可配置强度的 LLM 规划阶段，并输出可审计的 activity log[^35]。

---

## 6. 工程落地热度（GitHub 证据）

框架侧的定位变化可作为架构差异的间接证据：

- **LangGraph**：43,859 stars、7,534 forks、MIT、Python，创建于 2023-08-09，README 自我定位为「Low-level orchestration framework for building stateful agents」，topics 含 rag、agents、multiagent[^12]。
- **LlamaIndex**：52,654 stars、8,424 forks、MIT，README 将 LlamaIndex Framework 描述为「an open-source framework to build agentic applications」，并称其历史上是一套支持构建各类 RAG 与 agent 应用的编排工具集；README 顶部 NOTE 声明公司重心已转向 LlamaParse（agentic OCR / 文档解析与抽取平台）等产品，OSS 框架仍作为开放工具包提供；称有超过 300 个集成包[^13]。
- **Haystack**：26,834 stars、3,335 forks、Apache-2.0，topics 显式包含 agentic-rag、agentic-ai、agent-framework、rag、context-engineering；README 自我定位为「an open-source AI orchestration framework for building production-ready LLM applications in Python」，强调对 retrieval、routing、memory、generation 的显式控制，并提供 before_llm、before_tool、on_exit 等生命周期钩子与 step_count、token_usage、工具调用跟踪[^14]。
- **Dify**：159,471 stars、25,204 forks，license 字段为 NOASSERTION，topics 含 agent、agentic-ai、agentic-framework、agentic-workflow、workflow、low-code、no-code、mcp、rag[^15]。

注意：以上 star 数与 push 时间均为工具返回时点快照[^12][^13][^14][^15]，不宜作为趋势推断依据；且 LlamaIndex README 自述更新频率低于官方文档[^13]。

---

## 7. 矛盾与不确定性

**（1）量化对比缺一手证据。** 本应支撑「Agentic RAG 相比传统 RAG 好在哪、代价多少」的核心数字，在现有证据中系统性缺失：Self-RAG 正文该部分未出现 PopQA/TriviaQA/PubHealth/ARC-Challenge 具体分数，也未给出检索调用次数或延迟/token 开销倍数[^17]；CRAG 官方 README 未含分数表、触发比例或延迟数据[^21]；Adaptive-RAG README 未列 F1/EM 结果表[^25]，ACL Anthology 落地页仅摘要无表格[^24]。唯一带具体数字的是 alphaXiv 的 AI 生成概览（如称 Self-CRAG 相对独立 Self-RAG 提升 PopQA +20.0% 准确率、Biography FactScore +36.9%；T5-based evaluator 相关性判定准确率 84.3% 对 ChatGPT 58.0%），但该页明确标注为二手 AI 生成概览、需交叉验证，且未给出各基线绝对分数与 Incorrect/Ambiguous 分支实际触发比例[^22]，不能作为本报告的数据源。**因此「Agentic RAG 的收益幅度与开销倍数」目前无一手证据支撑，现有公开资料不足以确认。**

**（2）作者自述的增益口径需谨慎。** 若干增益为作者自述且本报告未读到结果表格：Adaptive-RAG 称在覆盖多种 query 复杂度的开放域 QA 数据集上提升整体效率与准确率[^9]；IRCoT 称配合 GPT3 使检索提升最多 21 点、下游 QA 提升最多 15 点[^10]，其正文给出的精确口径为「11–21 recall points」（fixed-budget optimal recall 设置下相对 one-step、基于问题的检索 OneR 测得）与「up to 15 F1 points」，同时 CoT 中事实性错误最多减少 50%[^26]。Self-RAG 侧，论文引言给出定性结论称在四个任务上优于 retrieval-augmented ChatGPT、在所有任务上优于 Llama2-chat 与 Alpaca[^17]，项目页称在六个任务上优于 vanilla ChatGPT 或 Llama2-chat[^18]；自适应检索消融仅有定性表述（减少检索使 PopQA 相对下降约 40%，PubHealth 仅约 2% 轻微下降）[^18]，项目页正文未列出各数据集准确率数值表[^18]。

**（3）失败案例与「过度 agentic 化」批评覆盖不足。** 现有证据中仅一条相关来源：Adobe Express 的生产证据论文指出 agentic 助手下传统检索指标（CTR、MRR、behavioral NDCG）失效，原因是这些指标假设用户直接看到并作用于排序结果，而 agentic 助手中由生成层先消费检索输出；作者将失效归纳为 who / when / what 三类并称为 attribution gap（归因缺口），提出 Layered Agentic Retrieval Assessment (LARA) 分三层度量[^16]。该文报告同一套检索基础设施同时服务 gallery 搜索与 agentic 助手，后者会话量约为前者八分之一，chat 用户比 gallery 用户更频繁改写查询（44.4% vs. 33.1%）[^16]。但该文为 5 页短文、投稿 CHIIR 2027 Industry Track 而非已录用[^16]，单一来源且非同行评审，**不满足多来源交叉验证要求，相关结论应视为待验证**。

**（4）评测基准与成本-准确率指标缺失。** 现有基准证据只有 BrowseComp（1,266 道题，要求 agent 持续浏览互联网寻找难以发现、相互纠缠的信息；答案简短可自动校验，评分用 AI 模型判断语义等价；题目构造采用「倒置」策略，训练者需确认 GPT-4o、OpenAI o1 及早期 deep research 模型都无法解出；论文明确将其定位为衡量 agent 持久性与搜索创造性，而非真实用户查询分布，不覆盖长答案生成与歧义消解）[^36]，其人类基线为 1,255 道被尝试题目中 367 道（29.2%）被解出、888 道（70.8%）两小时后放弃，解出题中答案一致率 86.4%（317/367）[^37]；以及 Search-R1（让 LLM 在逐步推理中自主生成多次搜索查询并实时检索，用多轮搜索交互轨迹做 RL 优化，采用 retrieved token masking 与基于结果的简单奖励；在七个问答数据集上相比多种 RAG 基线在同设置下提升 41%（Qwen2.5-7B）与 20%（Qwen2.5-3B）；论文提供关于 RL 优化方法、LLM 选择与「响应长度动态」的实证分析）[^38]。**两者均未定义「检索轮次效率」或「成本-准确率权衡」的标准指标，专门针对 agentic 检索的评测基准在现有证据中不足以确认存在。**

**（5）术语与分类尚未收敛。** Agentic RAG 综述自述该领域架构多样、术语不一致、缺乏比较性理解，此前没有统一分类法[^2]；其四维 taxonomy（agent cardinality、control structure、autonomy、knowledge representation）[^1][^7] 是提出而非共识。同时 memory management 被该综述列为开放研究挑战之一（与 evaluation、coordination、efficiency、governance 并列）[^7]，说明状态/记忆层的工程实践（如 LangGraph 的 checkpointer/store 分工[^28]）尚未形成稳定范式。

**（6）延迟代价的方向性证据存在但幅度不明。** Azure 文档明确 agentic retrieval 相比单查询管线会增加延迟[^35]，这与多步检索「因反复访问 Retriever 与 LLM 而计算开销大」的论述方向一致[^23]；但具体倍率在现有证据中无一手数据，现有公开资料不足以确认。

## 参考来源

[^1]: Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG (arXiv:2501.09136v4) — https://arxiv.org/abs/2501.09136（arxiv, 2025-01-15）
[^2]: Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG — 正文 HTML 版（第 1 节 Introduction） — https://arxiv.org/html/2501.09136v4（arxiv, 2026-04-01）
[^3]: Retrieval-Augmented Generation for Large Language Models: A Survey (arXiv:2312.10997v5) — https://arxiv.org/abs/2312.10997（arxiv, 2023-12-18）
[^4]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., NeurIPS 2020) — https://arxiv.org/html/2005.11401v4（arxiv, 2020-05-22）
[^5]: Retrieval-Augmented Generation for Large Language Models: A Survey (Gao et al., arXiv:2312.10997v5) — https://arxiv.org/html/2312.10997v5（arxiv, 2023-12-18）
[^6]: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al., arXiv:2310.11511v1) — https://arxiv.org/abs/2310.11511v1（arxiv, 2023-10-17）
[^7]: Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG (v4) — https://arxiv.org/abs/2501.09136v4（arxiv, 2025-01-15）
[^9]: Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity — https://arxiv.org/abs/2403.14403v2（arxiv, 2024-03-21）
[^10]: Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT) — https://arxiv.org/abs/2212.10509v2（arxiv, 2022-12-20）
[^11]: Corrective Retrieval Augmented Generation (CRAG), arXiv:2401.15884v3 — https://arxiv.org/abs/2401.15884v3（arxiv, 2024-01-29）
[^12]: langchain-ai/langgraph — GitHub repository — https://github.com/langchain-ai/langgraph（github, 2026-10-25）
[^13]: run-llama/llama_index — GitHub repository — https://github.com/run-llama/llama_index（github, 2026-10-23）
[^14]: deepset-ai/haystack — GitHub repository (README) — https://github.com/deepset-ai/haystack（github, 2026-10-24）
[^15]: langgenius/dify — GitHub repository — https://github.com/langgenius/dify（github, 2026-10-25）
[^16]: When Agentic Retrieval Fails Silently: Framework and Production Evidence from a Creative AI Assistant — https://arxiv.org/abs/2610.30423（arxiv, 2026-10-21）
[^17]: Self-RAG 论文正文（HTML 版，第 1–3.2 节） — https://arxiv.org/html/2310.11511v1（arxiv, 2023-10-17）
[^18]: Self-RAG 官方项目页 (selfrag.github.io) — https://selfrag.github.io/（web）
[^19]: Corrective Retrieval Augmented Generation — 正文（Section 4 CRAG / 4.1-4.3，arXiv HTML v3） — https://arxiv.org/html/2401.15884v3（arxiv, 2024-10-07）
[^20]: Corrective Retrieval Augmented Generation — 正文（ar5iv 镜像，Section 4.3 Action Trigger 阈值定义） — https://ar5iv.labs.arxiv.org/html/2401.15884（arxiv, 2024-01-29）
[^21]: HuskyInSalt/CRAG — 官方代码仓库 README — https://github.com/HuskyInSalt/CRAG（github, 2024-10-08）
[^22]: alphaXiv 论文页（含 AI 生成概览）— Corrective Retrieval Augmented Generation — https://www.alphaxiv.org/abs/2401.15884（web）
[^23]: Adaptive-RAG 正文（HTML 版，Method 章节） — https://arxiv.org/html/2403.14403v2（arxiv, 2024-03-28）
[^24]: ACL Anthology: Adaptive-RAG (NAACL 2024, 2024.naacl-long.389) — https://aclanthology.org/2024.naacl-long.389/（web, 2024-06）
[^25]: starsuzi/Adaptive-RAG 官方代码仓库 README — https://github.com/starsuzi/Adaptive-RAG（github, 2024-04-05）
[^26]: IRCoT 正文（HTML 全文）§1–§4：方法、超参与实验设置 — https://arxiv.org/html/2212.10509v2（arxiv, 2023-06-23）
[^28]: LangGraph 官方文档：Persistence（checkpointer vs store） — https://docs.langchain.com/oss/python/langgraph/persistence（web）
[^29]: LangGraph 官方文档：Graph API（State / Nodes / Edges / StateGraph / 编译） — https://docs.langchain.com/oss/python/langgraph/graph-api（web）
[^30]: LlamaIndex 官方文档：Multi-agent patterns（AgentWorkflow / orchestrator / custom planner） — https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/（web）
[^31]: Haystack 官方文档：Agent 组件（tool-using loop、exit_conditions、state_schema） — https://docs.haystack.deepset.ai/docs/agent（web）
[^32]: Haystack 官方文档：Pipelines（分支、循环、agentic pipelines） — https://docs.haystack.deepset.ai/docs/pipelines（web）
[^33]: Dify 官方文档：Knowledge Retrieval 节点 — https://docs.dify.ai/en/guides/workflow/node/knowledge-retrieval（web）
[^34]: Dify 官方文档：Agent 节点（classic Agent 与 new Agent） — https://docs.dify.ai/en/guides/workflow/node/agent（web）
[^35]: Agentic retrieval in Azure AI Search (Microsoft Learn) — https://learn.microsoft.com/en-us/azure/search/search-agentic-retrieval-concept（web）
[^36]: BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents — https://arxiv.org/abs/2504.12516（arxiv, 2025-04-16）
[^37]: BrowseComp (HTML full text) — 人类表现与评测设计 — https://arxiv.org/html/2504.12516v1（arxiv, 2025-04-16）
[^38]: Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning — https://arxiv.org/abs/2503.09516（arxiv, 2025-03-12）