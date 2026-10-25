## 摘要

评估 LLM Agent 系统的好坏，不能沿用单轮 LLM 评测的范式。权威综述将 agent 定义为「在环境中随时间感知并行动、追求自身议程的系统」[^1]，并强调构建 agent「远非 QA」，agent 需扮演特定角色、自主感知环境并演化[^1][^2]。由此，评测对象从「一次输入输出」扩展为「一段带工具调用、记忆与环境交互的行为轨迹」。本报告梳理出四条主线：(1) 结果导向的端到端任务成功率基准（WebArena、SWE-bench、GAIA、OSWorld、AgentBench、τ-bench），其共同特征是真实环境、长时程、可执行判定，且当前最强系统与人类差距巨大[^4][^5][^6][^8][^3][^7]；(2) 过程导向的轨迹与单步评测，即评最终回复、评工具调用路径、孤立评某一步骤三类范式[^22]，以及工具选择、参数格式、记忆与上下文保持等能力维度[^20][^38]；(3) 可靠性维度，以 pass^k 衡量多次试验下的一致性[^7][^34]，以 time horizon 衡量「以某成功率可完成的人类任务时长」[^12][^35]；(4) 评测的统计效度与 judge 效度，前者要求报告标准误、处理题目聚类与题内随机性、使用配对差异检验[^13]，后者需正视位置偏差与自偏好偏差[^26][^27][^28][^29]。工程侧已形成 online/offline 双轨、trace/run/thread/trajectory 数据模型、LLM-as-a-Judge 与代码型确定性评测器并用的成熟工具链[^16][^17][^20][^21]。需要显式声明：基准污染与防污染设计、成本-延迟-成功率联合权衡这两项在给定证据中缺乏权威一手来源，本报告不作方法论层面的结论。

---

## 一、为什么 agent 评测不同于 LLM 评测

**定义与评测对象的位移。** 综述引用的经典定义是：「An autonomous agent is a system situated within and a part of an environment that senses that environment and acts on it, over time, in pursuit of its own agenda and so as to effect what it senses in the future.」[^1] 该定义的关键词是「over time」与「in an environment」——评测必须覆盖时间维度上的感知-行动序列，而非单点输出。综述提出的统一架构框架由 profiling（角色设定）、memory（存储环境感知信息）、planning、action 四个模块组成[^1][^2]，这四模块同时构成「评什么」的解剖学基础：能力获取问题关注「如何激发并增强 agent 完成不同任务的能力」[^2]，与架构设计并列为构建 agent 的两大问题。

**评测复杂度的来源。** KDD '25 综述指出，agent 评测比单独评测 LLM 更复杂，因为 agent 在动态交互环境中推理、规划、执行工具、使用记忆并与人类或其他 agent 协作，标准 LLM 评测方法不足[^38]；它同时指出 agent 评测不同于传统软件测试——软件测试关注确定性与静态行为，而 LLM agent 本质上是概率性的、动态行为的，因此需要新的评估方法[^38]。

**两条一级分类轴。** 早期综述在正文中把评测策略组织为「主观与客观策略」二分[^2]；KDD '25 综述给出更完整的二维体系：Evaluation Objectives（评什么：agent behavior、capabilities、reliability、safety）× Evaluation Process（怎么评：interaction modes、datasets/benchmarks、metric computation methods、evaluation tooling、evaluation environments）[^38]。在该体系中，Agent Behavior 是 outcome-oriented 方面（任务完成、输出质量、延迟、成本），Agent Capabilities 是 process-oriented 能力（工具使用、规划与推理、记忆与上下文保持、多智能体协作）[^38]；过程轴上则区分 static evaluation（对固定输入作答）与 interactive assessment（与用户交互），以及从 controlled simulations 到 open-world settings 的环境谱系[^38]。ACL Findings 综述从五个视角展开：核心 LLM 能力、应用特定基准、通用 agent 评测、agent 基准核心维度（数据构建、环境类型、指标）、面向开发者的框架与工具[^39]。注意：本次抓取的 [^2] 正文被裁剪至第 2.1.2 节，未包含其第 6 章评测章节的具体分类与基准清单，故该综述的细分结论无法在本文中展开[^2]。

---

## 二、结果导向：端到端任务成功率基准

这类基准的共同设计是：给 agent 一个真实或高保真环境、一个目标、可调用的工具，用「任务是否被功能性完成」作为唯一或主要指标。

**WebArena** 构建了四个领域的完整功能网站（电商、社交论坛、协作软件开发、内容管理），并配备地图类工具与用户手册类外部知识库以鼓励类人解题；任务具有多样性、长时程特征，指标为端到端任务成功率，最佳 GPT-4 基线仅 14.41%，人类为 78.24%[^4]。

**SWE-bench** 含 2,294 个取自 12 个流行 Python 仓库的真实 GitHub issue 及对应 PR，任务形式是给定代码库与 issue 描述、编辑代码库以解决该 issue；解题常需跨多个函数、类甚至文件协调修改，要求与执行环境交互、处理超长上下文并复杂推理，超出传统代码生成范畴；当时最佳模型 Claude 2 仅解决 1.96%[^5]。

**GAIA** 设计 466 个需要推理、多模态处理、网页浏览与通用工具使用的真实世界问题，公开问题但保留 300 个答案用于维护 leaderboard；人类受访者正确率 92%，配备插件的 GPT-4 仅 15%[^6]。其设计理念值得单独指出：不追求对人类越来越难的任务，而是考察系统能否达到普通人类在该类问题上的稳健性[^6]。

**OSWorld** 面向多模态 agent，覆盖 Ubuntu、Windows、macOS 的真实计算机环境，支持任务搭建、基于执行的评测与交互式学习；369 个任务涉及开放域真实 web 与桌面应用、文件 I/O 及跨应用工作流，每个样本含详细初始状态配置与自定义执行评测脚本以保证可靠可复现；人类可完成 72.36% 以上，最佳模型仅 12.24%，瓶颈在 GUI grounding 与操作知识[^8]。

**AgentBench** 以 8 个不同环境评估 LLM-as-Agent 的推理与决策能力，发现头部商用 LLM 在复杂环境中表现较强，但与不超过 70B 的许多开源竞品存在显著差距；主要失败原因归为长期推理、决策与指令遵循能力不足；并报告「在代码上训练」对不同 agent 任务的影响是矛盾的[^3]。

**τ-bench** 引入用户侧交互：由语言模型扮演用户，与配备领域 API 与政策准则的 agent 动态对话，判定方式是比对对话结束时的数据库状态与标注目标状态——这是一种高效且忠实的判定方法[^7][^34]。结果：gpt-4o 这类 SOTA function calling agent 任务成功率 <50%，且表现很不一致，retail 域 pass^8 <25%[^7][^34]。

**横向观察。** 上述基准的人类-模型差距普遍在一个数量级以上（14.41% vs 78.24%[^4]；1.96%[^5]；15% vs 92%[^6]；12.24% vs 72.36%[^8]），说明当前 agent 在长时程、真实环境任务上的能力仍处早期。ACL Findings 综述亦指出领域趋势是转向更真实、更具挑战性的评测与持续更新的基准[^39]。环境谱系上，该综述明确对照 Mind2Web 的 offline environment（通过对比预测操作与 gold-standard actions 做 intermediate goal evaluation）与 WebArena 的 sandboxed online environment（动态、功能完整网站 + 长程人类任务 + 功能正确性测试）[^39]。

---

## 三、过程导向：轨迹、单步与能力维度

只看最终成功率会丢失「agent 为什么失败」。LangSmith 官方教程把 chatbot/agent 最有效的评测分为三类：Final response（评最终回复）、Trajectory（评 agent 是否走了预期的工具调用路径）、Single step（孤立评估任一步骤，如是否选对第一个工具），并声明这些技术与框架无关[^22]。LangSmith 的评测概念文档同样建议按系统关键组件拆解评测对象——LLM 调用、检索步骤、tool invocations、输出格式，并对 agent 明确要求评测「正确的工具选择与参数格式，或 agent 所走的轨迹」[^20]。

**轨迹的数据模型。** 在 LangSmith 中，agent 的每个工作单元（模型调用、工具调用、信息检索）记录为一个 run，单次操作的 runs 集合为 trace，多轮会话的 traces 可链接为 thread；run 对应 OpenTelemetry 的 span 概念；trajectory 是把 thread 内 traces 展平为有序消息列表的投影，去掉 run 嵌套，展示 agent 从开始到结束的路径[^21]。单条 trace 上限为 25,000 个 runs[^21]。Thread 级评测使 online evaluator 能评估整段对话的跨轮连贯性、话题保持与用户满意度[^20]。Langfuse 侧提供 Agent Graphs 做轨迹可视化，可从 observations 自动推断（trace 中存在 span/event/generation 之外的 observation 类型时视为 agentic）或经 LangGraph 集成直接显示，并提供 Aggregated（按步骤名聚合、循环画成环，用于把握整体结构）与 Expanded（每次调用一个节点、循环展开为 DAG，用于逐步调试单次运行）两种视图[^18]。

**能力维度。** KDD '25 综述把 process-oriented 能力列为工具使用、规划与推理、记忆与上下文保持、多智能体协作[^38]。ACL Findings 综述指出 agent 由 backbone LLM 与 agent harness 组成，核心能力既可被孤立评测，也可作为完整 agent workflow 的一部分被评测[^39]；并指出自反思（self-reflection）仍缺乏标准化基准或评测方法论，是关键空白[^39]。

---

## 四、可靠性：一致性与时间跨度

单次成功率无法区分「稳定地能做」与「偶尔能做」。

**pass^k。** τ-bench 提出该指标评估 agent 在多次试验中的行为可靠性/一致性[^7][^34]。其经验意义在于：gpt-4o 任务成功率 <50% 的同时 retail 域 pass^8 <25%[^7][^34]，即多次尝试下的一致性衰减远快于单次成功率所示。作者据此指出需要能提升 agent 一致性与规则遵循可靠性的方法[^34]。

**Time horizon。** METR 提出以「AI agent 能自主完成的任务时长（按人类专家完成所需时间计）」作为能力指标，而非单步技能或知识题分数[^12]。方法完整表述为：在一组多样化的多步软件与推理任务上记录人类专家完成时间，用 logistic 曲线拟合「模型成功率 ~ 人类任务时长」，再取某固定成功率阈值处曲线的交点换算为时长[^12][^35]。50%-time horizon 即 agent 被预测以 50% 概率成功完成的任务时长，80% 同理[^35]；报告同时给出两个阈值，说明单一阈值不足以刻画能力[^12]。任务来源为 RE-Bench、HCAST 及一组更短的新颖软件任务，主要覆盖软件工程、ML 与网络安全，任务被设计为自包含、规格明确、可自动判定成功；人类时长多数通过雇佣专家实际尝试并取成功完成时间的几何均值获得[^35]。

实证结论：当前模型在人类 4 分钟以内任务上成功率接近 100%，但在人类耗时超过约 4 小时的任务上成功率低于 10%[^12]；以 50% 可靠性可完成的任务时长在过去 6 年约每 7 个月翻倍，作者称对总体趋势较有信心，区间约为每年 1–4 次翻倍[^12]。95% 置信区间通过对 task families、tasks、task attempts 做分层 bootstrap 计算[^12]。

该方法对自身局限的声明相当明确，应作为使用前提：估计依赖任务集选择与被测人类群体等方法学选择[^12]；「超过 16 小时的测量在当前任务集下不可靠」[^35]；time horizon 不是 AI 自主运行时长，而是任务难度度量，且任务分布主要限于软件工程/ML/网络安全，不能外推到所有职业[^35]；其他领域也呈指数趋势但绝对时间跨度不同，AI 能力相对人类是「jagged」的[^35]。版本方面，TH1.1（2026-01-29 发布，更大任务集与新的评测基础设施）为当前口径，TH1.0 为 2025-03 原始计算[^35][^36]；官方页面已标注不再积极更新，TH1.1 不再随新模型更新[^35][^12]。后续工作「How Does Time Horizon Vary Across Domains?」（2025-07-14）分析 9 个基准，观察到与原始工作约 7 个月翻倍大致相似的改进速率[^36]；「Expenditure Horizon」（2026-07-21）提出以 expenditure horizon 衡量 agent 优化能力，是不同于 time horizon 的新指标[^36]。

---

## 五、评测的统计效度：分数差异是否真实

Anthropic 工程文章主张，eval 报告的核心对象不是观测到的平均分，而是从「问题宇宙」中抽样得到的理论均值，建议在每个分数旁报告基于中心极限定理的标准误（SEM），95% 置信区间为均值 ± 1.96×SEM[^13]。两类已知失效模式：

- **题目非独立**：同一段落/材料派生多题（如 DROP、QuAC、RACE、SQuAD）会使朴素 SEM 低估误差，应在随机化单元（如 passage）上做聚类标准误；Anthropic 称在流行 eval 上聚类标准误可达朴素标准误的 3 倍以上，忽略题目聚类会导致误判模型间存在能力差异[^13]。
- **题内随机性**：模型非确定性输出抬高整体标准误，建议对同一题重采样多次并取题级平均（Inspect 通过 epochs 参数实现），或在非 CoT 场景下直接用下一 token 概率作为题分以消除该随机成分[^13]。

比较两个模型时应使用配对差异检验而非两样本 t 检验，以消除题目难度方差；前沿模型在流行 eval 上的题级得分相关性约为 0.3–0.7，即它们倾向于答对/答错同一批题[^13]。该文对应 Anthropic 发表的一篇研究论文，作者称此前社区对「分数差异是否真实」研究不足[^13]。

---

## 六、LLM-as-a-Judge 的效度与其偏差

**定义与构成。** LLM-as-a-Judge 指用一个 LLM 按评分 rubric 评估另一个 LLM 应用的输出，返回分数与推理理由；judge prompt 的典型组成为评测标准 rubric、输入上下文、待评输出、可选参考答案[^17]。分数类型可为 numeric（如 0–1 helpfulness）、categorical（correct / partially_correct / incorrect）、boolean[^17]。评测目标可为 Observations（单次操作：LLM 调用、检索、tool call）或 Experiments（受控数据集），官方称 observation 级是生产实时数据的推荐目标，trace 级 evaluator 已弃用[^17]；observation 级 evaluator 可选取该 observation 的 input、output、metadata 或 tool calls 作为变量[^17]。

**位置偏差。** 一项覆盖 15 个 LLM judge、22 个任务、约 40 个解答生成模型、累计超过 150,000 个评测实例的系统研究，提出 repetition stability、position consistency、preference fairness 三个度量，结论包括：能力较强的 judge 的位置偏差并非随机波动（repetition stability 高，偏差可复现）；同一 judge 在不同任务上偏好方向（primacy vs recency）波动很大；位置一致性受候选答案间质量差距强烈影响，而与内容长度只有非常弱的关联；不同 judge 的整体位置一致性可能相近，但具体实例判断差异显著，多模型一致同意的实例通常更易判、分歧实例更易出现位置偏差[^26]。实验采用双盲设置并对 pairwise 交换位置、list-wise 做全排列[^26]。

**自偏好偏差。** 一项研究发现，未微调情况下 GPT-4、GPT-3.5、Llama-2-7b-chat 区分自身输出与其他来源的准确率均超过 50%，GPT-4 区分自身与另外两个 LLM 及人类文本的准确率为 73.5%；500 条样本微调后 GPT-3.5 与 Llama 2 的自我识别准确率均超过 90%；自我识别能力与自我偏好偏差强度之间存在线性相关，作者通过受控实验排除了反向因果与常见混淆因素[^27]。顺序偏差方面，选项顺序反转时 GPT-4、GPT-3.5、Llama 反转 pairwise 偏好的比例分别为 25%、58%、89%[^27]。所有模型对「人类撰写的摘要」表现出最强的自我偏好；较弱模型难以区分自身与更强模型，Llama 2 完全无法区分自身与 GPT-3.5/GPT-4 的输出[^27]。

另一项工作提出基于 Equal Opportunity 概念的量化指标 Bias = P(Y'=1|S=1,Y=1) − P(Y'=1|S=0,Y=1)，0 表示无偏差、接近 1 表示高偏差、负值表示反向偏差；在 8 个评估器上测量，GPT-4 表现出显著自偏好偏差且强于其他模型；成因分析显示 LLM 对困惑度更低的输出给出显著更高评分（无论是否自己生成），作者据此认为自偏好偏差的本质是困惑度/熟悉度偏好，并提示用 GPT-4 作 judge 可能使其特有风格过度影响被评系统[^28]。

**judge 基准本身的效度。** JudgeBench 指出现有 judge 基准主要衡量「与人类偏好的一致性」，但在人类众包偏好无法反映事实与逻辑正确性的困难任务上失效；其构建覆盖知识、推理、数学、编程的困难回答对，偏好标签反映客观正确性，评测了 prompted judges、fine-tuned judges、multi-agent judges 与 reward models，结论是许多强模型（如 GPT-4o）表现仅略好于随机猜测[^29]。

**缓解实践。** Langfuse 官方建议用标注样本校准 judge，并在 Score Analytics 中比对 judge 分数与人工标注分数[^17]；其评测体系提供 LLM judge 与人工标注一致性的校准能力[^16]。上述自偏好研究亦采用交换 A/B 位置并对两次评分取平均来缓解位置偏差[^28]。

---

## 七、安全与内容风险评测

MLCommons AILuminate 是跨 12 个危害类别的安全基准族，含 Safety（英/法/中文 T2T）与 Jailbreak（T2T、T+I2T，英文）两类[^32]。Safety v1.0 面向通用聊天系统，每种语言提供超过 24,000 条测试提示词（12,000 公开练习 + 12,000 私有 Official Test），评分采用经调优的安全评估模型集成[^32]。评测对象区分 AI Systems（可能含护栏、算法过滤器等审核技术，通常以 API 或软件组件形式提供）与 Bare Models（无外部护栏的独立模型权重）[^33]。评分量表为 5 级（Poor / Fair / Good / Very Good / Excellent），依据违反评估标准的响应百分比，且是相对化的：以两个参数量小于 150 亿的开源权重模型中表现最佳者复合构成参考模型，Good 表示相对当前技术水平具有竞争力[^32][^33]。危害分三组：物理危害（儿童性剥削、CBRNE、性相关犯罪、自杀与自伤、暴力犯罪）、非物理危害（诽谤、仇恨、知识产权、非暴力犯罪、隐私）、情境危害（性内容、专业建议）[^33]。

其局限声明值得作为方法论警示：测试只具有负向预测力——在基准上表现好不代表模型安全，只表示未发现关键安全弱点；且因提示词从无限空间抽样、主观标准使用自动评估带来噪声，结果相对真实行为存在显著方差[^33]。此外当前版本仅支持单轮对话，因而仅限于内容类危害，需要更长上下文或重复交互才能体现的危害（如 bias）被排除[^32]。Jailbreak v0.5（草案）保留其危害分类法与评分逻辑，引入受控对抗提示以量化韧性，即从基线安全水平到受攻击下表现的下降幅度[^32]。另需注意，部分系统（如 Grok-3-Preview、腾讯 Hunyuan-TurboS、NVIDIA Llama 3.3 49b Nemotron Super）虽满足纳入要求，但因提供方禁止基准测试或要求省略结果而未列入报告[^33]。

---

## 八、工程工具链：online/offline 双轨与可复现性

**双轨结构。** Langfuse 将评测分为 online（对生产实时 trace 打分）与 offline（上线前对数据集跑实验）[^16]；LangSmith 同样区分 offline（benchmarking、回归测试、单元测试、backtesting，目标为 dataset 中的 examples）与 online（实时监控、异常检测、生产反馈，目标为 tracing 产生的 runs 与 threads）[^20]。LangSmith 的 Dataset 由 examples 组成（inputs、可选 reference outputs、可选 metadata），Experiment 表示某应用版本在某 dataset 上的评测结果，捕获每个 example 的 outputs、evaluator scores 与 execution traces，支持多实验并排对比[^20]。

**评测器类型。** Langfuse 支持代码型确定性评测器（Code Evaluators）、LLM-as-a-Judge 与 decision-model evaluators[^16]；在线侧提供 trace / session / agent / prompt 级追踪、监控看板、LLM-as-judge、用户反馈，离线侧提供 datasets、features-as-tests、experiments（对比 prompt / 模型 / 代码变体）、judges 与自定义评测、annotation 人工标注队列[^16]。Ragas 主张同时使用 LLM-based 与 traditional metrics 两类客观指标，并提供智能测试生成、与 LangChain 及主流可观测性工具集成、用生产数据构建反馈闭环[^10]；其 CLI 当前可用模板为 rag_eval，agent_evals 标注为 Coming Soon[^10]。

**CI 回归门禁。** Langfuse 提供 langfuse/experiment-action GitHub Action，在 PR 工作流中当分数低于阈值时抛 RegressionError 使任务失败[^16]。

**成本与用量追踪。** Langfuse 记录每次 LLM 调用的 usage 与 cost，可按模型、用例、时间维度监控；每个 generation 记录 usage details（input/output、cached_tokens、audio_tokens 等）与 cost details（按 usage type 的 USD 成本）；数据可来自 ingested（由响应经 API/SDK/集成上报）或 inferred（依据 model 参数匹配内置价格推算），两者都有时 ingested 优先；内置 OpenAI、Anthropic、Google 主流模型价格定义，支持自定义模型定义与 pricing tiers、成本看板、花费阈值告警及 Metrics API 聚合检索[^19]。

**框架与仓库热度（工程落地信号）。** Inspect 由英国 AI 安全研究所创建，README 声明的内置组件覆盖 prompt engineering、tool usage、multi-turn dialog、model graded evaluations，支持通过其他 Python 包扩展 elicitation 与 scoring，附带 200 多个预置评测，MIT 许可，3.1k stars / 860 forks / 8,109 commits[^9]。openai/evals 提供评估 LLM 或基于 LLM 构建的系统的框架与开源 benchmark registry，支持自定义 eval 与用自有数据构建 private evals，19,647 stars / 3,120 forks，许可证字段为 NOASSERTION 且具体类型未在本次取证中确认[^23]。Langfuse 36,417 stars / 4,069 forks，主语言 TypeScript，topics 含 evaluation、llm-evaluation、llm-observability、monitoring、llmops、self-hosted，许可证字段为 NOASSERTION[^14]。AgentOps 自我定位为「Observability and DevTool platform for AI Agents」，MIT，5,914 stars，topics 含 cost-estimation、evals、evaluation-metrics 及多种 agent 框架集成标签[^24]。Ragas 16.1k stars / 1.8k forks，Apache-2.0，默认收集匿名化使用数据（可用 RAGAS_DO_NOT_TRACK=true 关闭）[^10]。RagaAI-Catalyst 自述为面向 Agent AI 的可观测性、监控与评估 Python SDK，声明能力包括 agent/llm/tools tracing、多智能体系统调试、自托管 dashboard、带 timeline 与 execution graph 视图的高级分析，16,180 stars / 3,574 forks，最近 push 2026-02-11（仅取得仓库描述，未读 README 正文）[^11]。LangSmith 的 Python SDK 仓库已归档（2026-10-16），SDK 迁至 langchain-ai/langsmith-python 与 langsmith-javascript，且该仓库仅为客户端 SDK、平台本体为闭源托管服务，stars 不代表平台热度[^15]。

**标准收敛的迹象。** METR 已不再维护 Task Standard，转而使用 Inspect 来定义任务；该仓库此前提供任务环境生成（Dockerfile 与 drivers/）、任务定义模板与 Python 包，并含对 GAIA、AgentBench 等既有评测套件的 adaptor 示例[^37]。Inspect 文档站还提供面向 coding agent 的结构化索引（llms.txt / llms-guide.txt / llms-full.txt）[^9]。

---

## 九、矛盾与不确定性

**证据内部的分歧点。**

1. **「代码训练是否有帮助」存在矛盾结论。** AgentBench 报告在代码上训练对不同 agent 任务的影响是矛盾的（ambivalent impacts），与既有假设不同[^3]。这提示「用代码数据提升 agent 能力」不是无条件成立的，评测需按任务类型分别验证。
2. **judge 一致性指标可能掩盖实例级分歧。** 不同 judge 的整体位置一致性指标可能相近，但在具体实例上的判断差异显著[^26]；同时 JudgeBench 显示以「与人类偏好一致」为目标的 judge 基准在客观正确性任务上失效，强模型仅略好于随机[^29]。二者共同说明：judge 的聚合一致性分数不足以作为效度证明。
3. **安全基准的「通过」不等于安全。** AILuminate 明确声明测试只具负向预测力，基准上表现好只表示未发现关键安全弱点[^33]；且单轮限制使其无法覆盖需长上下文或重复交互才显现的危害[^32]。这与「用基准分数代表系统安全水平」的常见用法直接冲突。
4. **时间跨度指标的版本与时效。** TH1.1 为当前官方口径且不再随新模型更新[^35]，而原始论文页面标注部分静态图表与结论（如翻倍时间）已过时[^12]；同时「超过 16 小时的测量不可靠」[^35]。因此引用翻倍时间结论时必须绑定版本与任务集范围。

**显式的覆盖缺口（现有公开资料不足以确认）。**

- **基准污染与防污染设计。** 给定证据中没有任何关于 contamination、动态/防污染基准（如 LiveBench、SWE-bench Verified/Live）或过拟合基准的一手来源。GAIA「保留 300 个答案」[^6]与 METR「TH1.1 不再随新模型更新」[^35]只是间接相关，不足以支撑「污染与过拟合是已知失效模式」这一方法论共识主张。该子议题现有公开资料不足以确认。
- **成本-延迟-成功率的联合权衡。** 唯一直接证据是一个 0 star、0 fork、标注 Academic Project 的课程项目仓库，其描述列出 accuracy、latency、computational cost、consistency 四项指标[^30]；另一条是 arXiv cs.AI 列表页首页 50 条标题中未见相关标题的否定性观察[^31]。二者均无法支撑产业界方法论共识。现有公开资料不足以确认。可确证的仅是：成本与延迟已被纳入部分评测体系的目标维度（KDD '25 将延迟、成本列入 outcome-oriented 的 Agent Behavior[^38]）与工具能力（Langfuse 的 usage/cost 追踪[^19]）。
- **Braintrust 与 AgentOps 的具体评测维度主张。** Braintrust 仅取得 Python SDK 仓库描述（22 stars，2026-02 创建）与「platform for evaluating and shipping AI products」一句，未读 README 正文、未取得平台热度，且该仓库非平台主仓库、stars 不代表平台整体热度[^25]；AgentOps 仅取得 banner 定位与 topics 标签（含 cost-estimation、evals、evaluation-metrics），未取得 README 声明的具体评测维度[^24]。因此「主流商业评测平台 README 主张的评测维度」这一问题的证据不完整。
- **早期综述的评测章节细节。** [^2] 的 HTML 正文被信息密度裁剪，仅覆盖至第 2.1.2 节，未包含第 6 章评测章节的具体分类细节与基准清单[^2]；因此该综述「主观 vs 客观」二分之下的一级细分无法在本文中给出。
- **ACL Findings 综述指出的领域空白本身即不确定性来源**：成本效率、安全性与鲁棒性的评估，以及细粒度、可扩展评测方法的发展，被明确列为关键空白[^39]；自反思缺乏标准化基准或评测方法论[^39]。这意味着上述维度目前没有公认的评测口径。

**方法学层面的固有不确定性。** 所有基于任务集的指标都对评测设计敏感：METR 明示其估计依赖任务集选择与被测人类群体[^12]；Anthropic 指出忽略题目聚类会误判模型间存在能力差异，且前沿模型题级得分相关性 0.3–0.7 意味着配对检验是必要而非可选[^13]；AILuminate 指出提示词抽样与自动评估噪声使结果相对真实行为存在显著方差[^33]。因此，任何单一分数都不应脱离其抽样结构、置信区间与版本信息被引用。

## 参考来源

[^1]: A Survey on Large Language Model based Autonomous Agents (arXiv:2308.11432, v7) — https://arxiv.org/abs/2308.11432（arxiv, 2023-08-22 (v1); v7 2025-03-02）
[^2]: A Survey on Large Language Model based Autonomous Agents — HTML 全文 (v7), 第 1–2.1.2 节 — https://arxiv.org/html/2308.11432v7（arxiv, 2025-03-02）
[^3]: AgentBench: Evaluating LLMs as Agents — https://arxiv.org/abs/2308.03688v3（arxiv, 2023-08-07）
[^4]: WebArena: A Realistic Web Environment for Building Autonomous Agents — https://arxiv.org/abs/2307.13854v4（arxiv, 2023-07-25）
[^5]: SWE-bench: Can Language Models Resolve Real-World GitHub Issues? — https://arxiv.org/abs/2310.06770v3（arxiv, 2023-10-10）
[^6]: GAIA: a benchmark for General AI Assistants — https://arxiv.org/abs/2311.12983v1（arxiv, 2023-11-21）
[^7]: τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains — https://arxiv.org/abs/2406.12045v1（arxiv, 2024-06-17）
[^8]: OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments — https://arxiv.org/abs/2404.07972v2（arxiv, 2024-04-11）
[^9]: UKGovernmentBEIS/inspect_ai — Inspect: A framework for large language model evaluations — https://github.com/UKGovernmentBEIS/inspect_ai（github）
[^10]: explodinggradients/ragas — Supercharge Your LLM Application Evaluations — https://github.com/explodinggradients/ragas（github）
[^11]: raga-ai-hub/RagaAI-Catalyst — Python SDK for Agent AI Observability, Monitoring and Evaluation — https://github.com/raga-ai-hub/RagaAI-Catalyst（github）
[^12]: Measuring AI Ability to Complete Long Software Tasks (METR, 2025-03-19; 含 Time Horizon 1.0/1.1 更新) — https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/（web, 2025-03-19）
[^13]: Use the Central Limit Theorem — Anthropic Engineering, "A statistical approach to model evals" (Chapter 01) — https://www.anthropic.com/engineering/statistical-approach-to-model-evals（web）
[^14]: langfuse/langfuse — GitHub 仓库元数据与 README — https://github.com/langfuse/langfuse（github, 2023-05-18）
[^15]: langchain-ai/langsmith-sdk — GitHub 仓库元数据（已归档，SDK 已迁移） — https://github.com/langchain-ai/langsmith-sdk（github, 2023-05-30）
[^16]: Langfuse Docs — Evaluation Overview — https://langfuse.com/docs/evaluation/overview（web）
[^17]: Langfuse Docs — LLM-as-a-Judge — https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge（web）
[^18]: Langfuse Docs — Agent Graphs（轨迹可视化） — https://langfuse.com/docs/observability/features/agent-graphs（web）
[^19]: Langfuse Docs — Model Usage & Cost Tracking — https://langfuse.com/docs/observability/features/token-and-cost-tracking（web）
[^20]: LangSmith Docs — Evaluation concepts — https://docs.langchain.com/langsmith/evaluation-concepts（web）
[^21]: LangSmith Docs — Observability concepts（run / trace / thread / trajectory 结构） — https://docs.langchain.com/langsmith/observability-concepts（web）
[^22]: LangSmith Docs — Evaluate a complex agent（三类 agent 评测范式） — https://docs.langchain.com/langsmith/evaluate-complex-agent（web）
[^23]: openai/evals — Evals is a framework for evaluating LLMs and LLM systems, and an open-source registry of benchmarks — https://github.com/openai/evals（github, 2026-04-14）
[^24]: AgentOps-AI/agentops — Observability and DevTool platform for AI Agents — https://github.com/AgentOps-AI/agentops（github, 2026-06-25）
[^25]: braintrustdata/braintrust-sdk-python — Python Tracing & Evals library for Braintrust — https://github.com/braintrustdata/braintrust-sdk-python（github, 2026-10-23）
[^26]: Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge — https://arxiv.org/abs/2406.07791（arxiv, 2024-06-12 (v1), v9 2025-11-11; AACL-IJCNLP 2025）
[^27]: LLM Evaluators Recognize and Favor Their Own Generations — https://arxiv.org/abs/2404.13076（arxiv, 2024-04-15）
[^28]: Self-Preference Bias in LLM-as-a-Judge — https://arxiv.org/abs/2410.21819（arxiv, 2024-10-29 (v1), v2 2025-06-21; NeurIPS 2024 Safe Generative AI Workshop）
[^29]: JudgeBench: A Benchmark for Evaluating LLM-based Judges — https://arxiv.org/abs/2410.12784（arxiv, 2024-10-16 (v1), v2 2025-04-05; ICLR 2025）
[^30]: AI Agents Benchmarking for Data Analysis — 标准化 agent 框架基准测试框架（CrewAI vs LangGraph） — https://github.com/vaibhav-tiwari-7/AI-Agents-Benchmarking-for-Data-Analysis（github, 2026-08-03）
[^31]: arXiv cs.AI recent submissions 列表页（2026-10-23） — https://arxiv.org/list/cs.AI/recent（arxiv, 2026-10-23）
[^32]: MLCommons AILuminate Benchmark（官方基准主页） — https://mlcommons.org/benchmarks/ai-safety/ailuminate/（web）
[^33]: MLCommons AI Safety Benchmark 结果页（AILuminate v1.0 评测范围、危害清单与分级解释） — https://mlcommons.org/benchmarks/ai-safety/（web）
[^34]: τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains (arXiv:2406.12045) — https://arxiv.org/abs/2406.12045（arxiv, 2024-06-17）
[^35]: Task-Completion Time Horizons of Frontier AI Models（METR 官方时间跨度页面） — https://metr.org/time-horizons/（web, 2026-09-08）
[^36]: METR Research 索引页（含 Time Horizon 1.1 发布条目） — https://metr.org/research/（web）
[^37]: METR/task-standard（METR 任务标准仓库） — https://github.com/METR/task-standard（github, 2026-10-07）
[^38]: Evaluation and Benchmarking of LLM Agents: A Survey (KDD '25) — https://arxiv.org/html/2507.21504v1（arxiv, 2025-07-29）
[^39]: Survey on Evaluation of LLM-based Agents (ACL Findings) — https://arxiv.org/html/2503.16416v2（arxiv, 2025-03-20）