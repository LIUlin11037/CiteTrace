# 如何评测一个 RAG 系统：检索侧与生成侧的指标分解

## 摘要

RAG 评测的公开方法论已形成较稳定的骨架：把系统拆成检索模块与生成模块分别度量，再用端到端指标兜底[^1][^16]。检索侧主流做法是排序指标（Recall@k、Precision@k、NDCG@k、MRR、Hit Rate）[^13][^31] 加「上下文相关性」类 LLM 判分[^5][^17]；生成侧主流做法是忠实性/扎根性（faithfulness、groundedness）、答案相关性、答案正确性三类[^3][^6][^17][^31]。三篇关键论文提供了不同取向的证据：RAGChecker 主张 claim 级蕴含校验以支持检索错误与生成错误的分离诊断[^7]；ARES 用合成数据微调轻量 judge 加 prediction-powered inference 给出带置信区间的排序[^20][^21]；RAGBench 则报告在 100k 样本基准上微调过的 RoBERTa/DeBERTa 类模型优于 few-shot LLM judge[^29][^30]。三者在「LLM judge 是否够用」上结论方向不一致，且均为各自方法的自评，不能直接横向比较（见「矛盾与不确定性」）。

需要提前声明的两个证据缺口：其一，nDCG 与 MRR 在本次全部证据中没有任何定义或公式，经典 IR 教科书与 nDCG 原始论文均缺席，因此本报告只能说明「哪些工具用它们、怎么调用」，无法给出权威定义；其二，没有任何来源给出「低分组合 → 故障类型」的可操作归因规则，Ragas 官方 RAG 评测指南被明确记录为未提供此类决策树[^45]。

---

## 一、评测的整体分层：先确定你要回答哪个问题

Ragas 文档把 AI 应用指标分为三类：端到端指标（把系统当黑盒，只看最终输出）、组件级指标（独立评估各部件，如 retrieval accuracy）、业务指标（部署后数天到数月才可计算的滞后指标）[^16]。文档明确指出组件级指标「可立即行动、便于定向改进，但不一定与最终用户满意度直接相关」，并强调「单独改进一个组件并不保证更好的端到端结果」[^16]。RAGChecker 从使用者角度做了类似切分：只关心整体排名的用户需要单一数值指标，而需要定位错误来源（检索错误 vs 生成错误）的开发者需要诊断型指标[^7]。

按底层机制，Ragas 把指标分为 LLM-based（一次或多次 LLM 调用，可能非确定性，同一输入不总返回相同结果，但文档称其更准确、更接近人类评估）与 Non-LLM-based（确定性，依赖字符串相似度、BLEU 等，文档称其与人类评估相关性较低）[^16]。这一区分贯穿下文两侧指标。

---

## 二、检索侧指标

### 2.1 经典排序指标：工具怎么用，证据到哪一步

BEIR 是异构 IR 基准，提供在基准内评测检索模型的通用框架，覆盖 lexical、dense、sparse 与 reranking-based 检索架构[^13]。其 README 列出的检索侧指标为 NDCG@k、MAP@K、Recall@K、Precision@K，k 取 [1,3,5,10,100,1000]，另有 evaluate_custom 支持 MRR[^13]。使用路径为：GenericDataLoader 加载 corpus/queries/qrels，DenseRetrievalExactSearch + SentenceBERT 检索，EvaluateRetrieval 计算指标，并可保存 TREC runfile 供重排序复用[^13]。

LlamaIndex 的检索侧评测同样使用 MRR（mean reciprocal rank）、hit-rate、precision 等排序指标，流程是先在无结构语料上合成生成 (question, context) 对，再用排序指标评测 retriever[^31]。

**必须声明的缺口**：以上两条证据只给出指标名称、k 取值与调用方式，没有给出任何计算定义。nDCG 与 MRR 在全部证据中均无定义或公式；经典 IR 出处（Manning/Raghavan/Schütze 的 *Introduction to Information Retrieval*、Järvelin & Kekäläinen 的 nDCG 原始论文、TREC 评测方法综述）完全缺席。因此本报告无法满足「标准定义与权威出处」这一要求，现有公开资料不足以确认这些指标的标准公式表述。

### 2.2 Precision@k 与位置敏感性：Ragas Context Precision

Ragas 的 Context Precision 是本次证据中唯一给出完整公式的检索侧指标。它评测检索器在给定 query 下把相关 chunk 排在无关 chunk 之前的能力，即相关 chunk 在检索结果排序中靠前的程度；定义为上下文中每个 chunk 的 precision@k 的均值[^5]。

- Precision@k 定义：排名前 k 中相关 chunk 数 / 排名前 k 的 chunk 总数，即 true positives@k / (true positives@k + false positives@k)[^5]。
- Context Precision@K 公式：sum_{k=1..K}(Precision@k × v_k) / 前 K 个结果中相关条目总数，其中 K 为 retrieved_contexts 的 chunk 总数，v_k ∈ {0,1} 为第 k 位的相关性指示[^5]。
- 该指标对位置敏感：文档示例中无关 chunk 位于第二位时 context precision 不变，但若移到第一位则分数由约 1.0 降至约 0.5[^5]。
- 无参考版本 LLMContextPrecisionWithoutReference 用 LLM 将每个检索 chunk 与生成的 response 比对来判断相关性[^5]。

### 2.3 Context Recall：需要参考基准的召回度量

Context Recall 衡量有多少相关文档（或信息片段）被成功检索到，关注「不漏掉重要结果」；计算 context recall 始终需要参考（reference）作为对比基准[^4]。Ragas 给出三种实现：

- **LLM 版**：用 reference 作为 reference_contexts 的代理（因为标注参考上下文成本很高），把 reference 拆成 claim，逐条判断能否归因于检索上下文；公式为 Context Recall = 被检索上下文支持的 reference claim 数 / reference claim 总数[^4]。
- **非 LLM 版（NonLLMContextRecall）**：用字符串比较度量，公式为检索到的相关上下文数 / 参考上下文总数，取值 0 到 1[^4]。
- **ID 版（IDBasedContextRecall）**：直接比较 retrieved_context_ids 与 reference_context_ids，公式为在检索 ID 中找到的参考上下文 ID 数 / 参考上下文 ID 总数，支持字符串与整数 ID[^4]。

### 2.4 上下文相关性：RAG Triad 的检索边

TruLens 提出的 RAG triad 沿 RAG 架构的三条边分别评估幻觉，其中 Context Relevance 用于验证检索质量：确保每个 context chunk 与输入 query 相关；文档指出上下文中任何不相关信息都可能被编织进幻觉[^17]。ARES 的 context relevance 定义是「检索到的信息是否与测试问题相关」，与上述口径一致[^20]。

RAGBench 的 TRACe 框架给出了这类指标的 token 级形式化：对检索到的上下文文档 d_i，R_i 为其中对回答查询有用的 relevant tokens 集合，document relevance = Len(R_i)/Len(d_i)；example relevance 为所有上下文文档的 relevant token 长度之和除以文档总长度[^30]。计算 ground-truth 指标时采用 sentence-length 作为长度度量，因其最契合标注方案[^30]。

### 2.5 检索侧的其他可选项

Ragas 指标清单中检索相关项还包括 Context Entities Recall 与 Noise Sensitivity[^2]。DeepEval 的 RAG 指标集合包含 Contextual Recall、Contextual Precision、Contextual Relevancy[^10]。Phoenix 明确其评测能力覆盖「response and retrieval evals」，即生成侧与检索侧都有对应 eval[^12]。MTEB 定位为评测 embeddings 与检索系统的多模态工具箱，提供 mteb.get_model + mteb.get_tasks + mteb.evaluate 的 API 与 CLI，并维护交互式 leaderboard[^14]——它评的是嵌入/检索模型本身，而非端到端 RAG。

---

## 三、生成侧指标

### 3.1 Faithfulness / Groundedness：回答是否忠于检索上下文

Ragas 的 Faithfulness 定义完整可复现：衡量回答与检索到的上下文之间的事实一致性，取值 0 到 1；若回答中的所有 claim 都能被检索上下文支持，则该回答被视为 faithful[^3]。计算三步：(1) 识别回答中的所有 claim；(2) 逐条检查该 claim 能否由检索上下文推断；(3) 用公式计算[^3]。公式为 Faithfulness = 被检索上下文支持的回答 claim 数 / 回答 claim 总数[^3]。文档示例：回答拆为两条 statement，其中一条无法由上下文推断，故 Faithfulness = 1/2 = 0.5[^3]。文档另提供基于 Vectara HHEM-2.1-Open（T5 分类器，用于检测幻觉）的变体 FaithfulnesswithHHEM，用于第二步的 claim 与上下文交叉核验[^3]。

TruLens 的 Groundedness 口径相同：把回答拆成独立 claim，逐条在检索上下文中寻找支持证据；文档称 LLM 常偏离给定事实、夸大或扩展成「听起来正确」的答案[^17]。RAGChecker 的底层机制是 claim-level entailment checking——从模型回答与标准答案中抽取 claim，再与其他文本做蕴含校验，从而实现细粒度而非回答级的评估[^7]。

### 3.2 Answer / Response Relevancy：回答是否切题（不评事实准确性）

Ragas 的 Answer Relevancy 衡量回答与用户输入的相关程度，取值 0 到 1；若回答直接且恰当地回应了原始问题则视为相关；该指标关注回答是否契合问题意图，**而不评估事实准确性**，并会惩罚不完整的回答以及包含不必要细节的回答[^6]。计算方式：先用 LLM 基于 response 反向生成一组人工问题（默认 3 个），再计算用户输入嵌入 E_o 与每个生成问题嵌入 E_gi 的余弦相似度并取平均；公式为 Answer Relevancy = (1/N) Σ cosine_similarity(E_gi, E_o)，N 默认 3、可通过 strictness 参数配置[^6]。文档注明：由于余弦相似度数学范围为 -1 到 1，该分数通常落在 0 到 1 之间但并不保证[^6]。其背后思路是：若回答正确回应了问题，则仅凭回答就很可能重建出原始问题[^6]。

LlamaIndex 的 AnswerRelevancyEvaluator 与 ContextRelevancyEvaluator 采用分步打分法：judge LLM 需回答两个子问题（response 是否匹配 query 的主题；response 是否尝试回应用户 query 所关注的焦点/视角），每题 1 分，满分 2/2，返回 0 到 1 的分数与解释性 feedback[^34]。

### 3.3 答案正确性：需要 ground truth 的一类

LlamaIndex 的 CorrectnessEvaluator 对照参考答案（reference）评测生成答案的相关性与正确性，即需要 ground-truth 标签；输出数值 score（示例中为 2.5，即分档打分而非 0/1）与 feedback[^35]。Ragas 的 RAG 评测工作流示例组合使用 LLMContextRecall、Faithfulness、FactualCorrectness 三项，同时覆盖检索侧与生成侧，输出形如 {'context_recall': 1.0000, 'faithfulness': 0.8571, 'factual_correctness': 0.7280}[^45]。Ragas 文档把 Factual Correctness、Semantic Similarity、BLEU、CHRF、ROUGE、Exact Match 归入「Natural Language Comparison」组，与 RAG 专用指标区分[^2]。

LlamaIndex 的 SemanticSimilarityEvaluator 走非 LLM 路线：计算生成答案与参考答案的 embedding 相似度，默认阈值 0.8，超过则 passing 为 True[^36]。文档明确警告高相似度不代表答案正确——示例中 "The sky is yellow." 与 "The sky is blue." 得到 0.9179 的高分[^36]。

### 3.4 完整性与上下文利用：RAGBench 的 TRACe 增量

RAGBench 采用已有的 context relevance、answer faithfulness 定义，并新引入两个指标：context utilization 与 answer completeness，主张这套指标能更好描述 RAG 系统整体表现并提供细粒度、可行动的洞察[^30]。token 级定义为：U_i 为生成模型实际使用（utilized）的 tokens 集合，document utilization = Len(U_i)/Len(d_i)；example utilization 为所有文档 utilized token 长度之和除以文档总长度[^30]。

### 3.5 生成侧的其他维度

Ragas 指标清单还包含 Noise Sensitivity、Multimodal Faithfulness、Multimodal Relevance[^2]；Nvidia Metrics 组包含 Answer Accuracy、Context Relevance、Response Groundedness[^2]。DeepEval 的通用指标含 Hallucination、Summarization、Bias、Toxicity、JSON Correctness、Prompt Alignment，并支持合成数据集生成与 CI/CD 集成[^10]。LlamaIndex 的 response 评测指标共 6 个：Correctness、Semantic Similarity、Faithfulness、Context Relevancy、Answer Relevancy、Guideline Adherence[^31]。

---

## 四、端到端与「生成器能力」基准：换一个评测对象

上述指标评的是「某次检索 + 某次生成」。另一类工作评的是「模型在受控检索条件下的能力」，其指标设计不能直接搬到生产评测。

**RGB** 提出中英双语基准，按 RAG 所需的 4 项基本能力划分 4 个 testbed：noise robustness、negative rejection、information integration、counterfactual robustness[^23]。其构造细节值得注意：noisy documents 指与问题相关但不含任何答案信息的文档（问 2022 年诺贝尔文学奖得主，噪声文档是 2021 年该奖的报道）[^24]；反事实鲁棒性评估模型在被提示「检索信息可能存在风险」时能否识别检索文档中的已知事实错误[^24]。官方脚本的指标命名：all_rate 为准确率（noise_rate<1 时）或拒答率（noise_rate=1 时）；fact_check_rate 为错误检测率 (ED)；反事实评测输出 reject_rate 即错误检测率 (ED*)、correct_rate 即错误纠正率 (CR)[^25]。论文关键发现：模型有一定噪声鲁棒性但容易混淆相似信息；当所有外部文档都不含相关信息时 LLM 常常无法拒绝作答；LLM 缺乏跨多文档总结能力[^24]。

**MultiHop-RAG** 把 multi-hop 查询定义为需要检索并推理多份 supporting evidence 才能给出答案的查询[^27]。它明确从两个维度评估：retrieval evaluation（将检索集与每个查询的 ground-truth evidence 比较，null 查询除外）与 response evaluation（将 LLM 响应与 ground-truth 答案比较，以评估推理能力）[^27]。四类查询为 Inference、Comparison、Temporal、Null，其中 Null 查询被专门纳入以评估生成质量尤其是幻觉——即使提供了检索集，LLM 也应输出空响应而非编造答案[^27]。数据集含 2556 条查询，每条证据分布在 2 到 4 篇文档中[^28]。

**RAGChecker** 将 RGB（噪声鲁棒性、负样本拒答、信息整合、反事实鲁棒性四项能力）、RECALL、NoMIRACL 归类为「仅评测生成器能力」的工作，并批评这类人工构造数据集无法揭示生成结果与不同检索行为之间的纠缠关系[^7]。这一批评对上述基准的适用范围给出了明确边界。

---

## 五、judge 的可靠性：评测本身怎么被评测

### 5.1 LLM-as-a-judge 的已知偏差

MT-Bench 论文系统列出 LLM-as-a-judge 的局限：position bias、verbosity bias、self-enhancement bias，以及 limited reasoning ability[^44]。位置偏差幅度很大：default prompt 下交换两个相似答案的顺序后，判断一致率 Claude-v1 仅 23.8%、GPT-3.5 46.2%、GPT-4 65.0%[^44]。多数 judge 偏向第一个位置（default prompt 下偏向第一位比例 Claude-v1 75.0%、GPT-3.5 50.0%、GPT-4 30.0%）[^44]。还存在 name bias：Claude-v1 在 rename prompt 下偏向名为 "Assistant A" 的一方[^44]。judge 能力与偏差程度相关，更强的 judge 位置偏差更小[^44]。论文报告 GPT-4 judge 与人类偏好的一致率超过 80%，与人类之间的一致率处于同一水平[^44]。论文还指出单答案打分的绝对分数在 judge 模型变化时比相对成对比较波动更大，并提出 reference-guided grading 作为缓解手段之一[^44]。

### 5.2 用统计方法给 judge 纠偏：ARES 的 PPI

ARES 沿三个维度评测 RAG：context relevance、answer faithfulness、answer relevance[^20]。其三阶段流程为：(1) 用 LLM 生成合成查询/答案并过滤低质查询；(2) 用合成数据以对比学习方式微调 LLM judge，对 query–passage–answer 三元组按三个准则分类；(3) 用 judge 给 RAG 系统打分，并用 prediction-powered inference (PPI) + 人工偏好验证集给出排序的置信区间[^21]。合成数据主要由 FLAN-T5 XXL 生成，低质查询的过滤方式是检验该查询能否用其检索器把原始段落检索为 top-1 结果[^21]。judge 本体是微调的 DeBERTa-v3-Large，三个独立 judge 分别对应三个维度[^21]。人工偏好验证集约 150 条或更多标注数据，需同时包含三个维度的正例与负例；PPI 的 rectifier 函数用于估计 judge 的误差并给出置信界，实验采用 95% 置信水平、取各置信区间中点排序[^21]。论文明确对比了两种极端做法：直接报告 judge 在无标注数据上的平均分（可能不完全准确），或只用小规模人工验证集评测（需为每个 RAG 系统分别标注大量生成输出，成本高）[^21]。

### 5.3 小模型 vs LLM judge

RAGBench 的核心发现是：在大规模基准测试中，基于 LLM 的 RAG 评测方法在 RAG 评测任务上难以与一个微调过的 RoBERTa 模型竞争[^29]；论文报告一个 400M 参数的 DeBERTa-large 模型在 RAGBench 上微调后，在多个领域与任务类型上优于 few-shot LLM judges[^30]。该基准含 100k 条样本，覆盖五个行业特定领域，数据划分 train 78k / dev 12k / test 11k[^29][^30]。

### 5.4 元评测：与人类判断的相关性

RAGChecker 人工标注了一个人类判断数据集做元评测，声称其与人类判断的相关性显著优于其他评测指标（RAGAS、TruLens、ARES、CRUD-RAG 被纳入对比）[^7]。ARES 报告在 KILT 与 SuperGLUE 的六个知识密集型数据集上，其 context relevance 与 answer relevance 的评测准确率平均分别比 RAGAS 高 59.3 与 14.4 个百分点；在 AIS attribution 数据集上对答案幻觉发生率的预测与真实平均值相差在 2.5 个百分点以内；相比基于标注的评测方法所需标注量少 78%[^22]。ARES 对 RAGAS 的批评是：基于少量启发式手写提示，对新 RAG 评测场景适应性差[^22]。

---

## 六、工程落地：工具链与工作流

**Ragas** 定位为「指标 + 测试集生成 + 反馈闭环」三件套，Key Features 为 Objective Metrics（LLM-based 与传统指标并用）、Test Data Generation、Seamless Integrations（LangChain 与主流可观测性工具）、Build feedback loops[^9]。仓库当前 stars 16.1k、Apache-2.0[^9]；需注意项目已迁移：api.github.com/repos/explodinggradients/ragas 返回 HTTP 301 重定向到同一数字 ID 的新路径，表明是改名/转移而非新建[^49]，explodinggradients 组织页面显示已无公开仓库[^50]，PyPI 与 README 的安装说明均指向 vibrantlabsai/ragas[^51][^9]，Releases 版本线从 v0.3.5 连续覆盖至 v0.4.3[^52]。

**DeepEval** 定位为类似 Pytest 但专为 LLM 应用单元测试设计的框架，可评测黑盒端到端应用、完整 agent 轨迹、以及单步（LLM 调用、工具使用、检索、子 agent 交接）[^10]。其 judge 主张为「powered by ANY LLM of your choice, statistical methods, or NLP models that run locally」[^10]。RAG 指标为 Answer Relevancy、Faithfulness、Contextual Recall、Contextual Precision、Contextual Relevancy，以及名为 RAGAS 的指标（上述四项平均）[^10]。

**TruLens** 为 OpenTelemetry-native，用 decorator 埋点、用「会自我解释的 LLM judges」逐步打分，每个函数调用/LLM 生成/检索/工具调用记为结构化 OTEL span，可导出到 Jaeger、Grafana Tempo、Datadog、New Relic 或任意 OTLP 后端[^11]。支持 inline（tru_recorder 随应用运行打分）与 batch（RunConfig + add_run + compute_metrics）两种模式[^11]。其 README 自报的评测成绩属营销性主张，需交叉验证[^11]。

**Phoenix** 为 Arize 的开源 AI 可观测性平台，功能含 Tracing（OpenTelemetry 埋点）、Evaluation（用 LLM 对 response 与 retrieval 做基准评测）、Datasets（版本化）、Experiments（追踪 prompt/LLM/检索变更）[^12]。

**LangSmith** 按运行时机区分 Offline Evaluation（上线前在策划好的数据集上跑，用于比较版本、基准测试、捕获回归）与 Online Evaluation（生产环境实时评估真实用户交互）[^18]。离线四步为：创建数据集 → 定义 evaluator（human review、code rules、LLM-as-judge、pairwise comparison）→ 运行 experiment → 分析结果[^18]。在线四步为：部署应用 → 配置 online evaluator（含 reference-free LLM-as-judge，并通过过滤器与采样率控制成本）→ 实时监控 → 建立反馈闭环[^18]。反馈闭环的具体做法是把失败的生产 trace 加入数据集，文档将其价值表述为「把只见过一次的失败变成每次都会跑的测试」[^18]。需注意 LangSmith 平台本身非开源：自托管是 Enterprise 套餐附加项、需 license key[^39]，GitHub 搜索中不存在平台服务端的官方源码仓库[^43]，已归档的 langsmith-sdk 仅为客户端 SDK[^37]。

**LlamaIndex** 的评测模块把评测分为 Response Evaluation 与 Retrieval Evaluation 两类，所有评测模块实现 BaseEvaluator，返回含 passing（二元）、score（数值）、feedback（字符串）的 EvaluationResult，judge 通过构造函数 llm= 注入[^31][^32]。文档明确指出许多评测模块不需要 ground-truth 标签[^31]。官方集成第三方评测工具 UpTrain、Tonic Validate、DeepEval、Ragas、RAGChecker、Cleanlab[^31]。

---

## 七、矛盾与不确定性

**（1）「LLM judge 够不够用」的三组结论方向不一致，且不可直接比较。** AIMultiple 在 1,010 个对抗性问题上评测五个工具，结论是 W&B Weave Top-1 准确率最高（94.5%），TruLens 在排序类指标上领先（NDCG@5 0.932、Spearman ρ 0.750、MRR 0.594），并称领先 RAGAS、DeepEval、UpTrain[^47]；但同一份数据里 NDCG@5 为 TruLens 0.932、DeepEval 0.923、W&B Weave 0.910，三者差距很小，且 W&B Weave 的 NDCG@5 最低却 Top-1 最高，说明指标间存在张力[^47]。ARES 论文称其 context relevance 准确率比 RAGAS 高 59.3 个百分点[^22]。RAGBench 称微调 RoBERTa/DeBERTa 优于 few-shot LLM judges[^29][^30]。三者口径不同（排序相关性 vs 分类准确率 vs 标注一致性），AIMultiple 为单次 GPT-4o 运行、无参考答案、作者自陈局限[^47]，ARES 与 RAGBench 均为自评。证据中没有任何来源对这些冲突做交叉验证或解释口径差异，因此不能据此给出工具优劣排序。

**（2）相关性打分与事实准确性必须分开。** AIMultiple 报告的共性缺陷是：五个工具全部给 hard negatives 的平均分高于 partial contexts，说明相关性打分需另配事实准确性检查[^47]。这与 TruLens 对 RAG triad 的结论边界一致：三项达标只能说明「应用在其知识库范围内被验证为无幻觉」——即若向量库只含准确信息，则 RAG 给出的答案也准确[^17]。CRUD-RAG 也批评 RAGAS、ARES 这类 reference-free 框架只评估生成文本与检索上下文的一致性，若检索到的外部信息质量低则评测可能不可靠[^8]。

**（3）归因定位的方法论缺失（本课题的核心缺口）。** 证据只到「有诊断指标」这一层：RAGChecker 提供 Diagnostic Retriever Metrics 与 Diagnostic Generator Metrics[^7]，Ragas 区分组件级与端到端指标[^16]，但没有任何来源给出可操作的归因规则。Ragas 官方 RAG 评测指南被明确记录为「未给出低分组合到失败类型的归因规则或决策树」[^45]。因此「context recall 低 + faithfulness 高」与「context recall 高 + faithfulness 低」分别指向什么故障，现有公开资料不足以确认。

**（4）经典 IR 指标定义缺席。** 如第二节所述，nDCG 与 MRR 在全部证据中无定义或公式，经典 IR 教科书与 nDCG 原始论文均未纳入，本报告无法提供权威定义。

**（5）检索改进与噪声引入的权衡、以及「盲目信任上下文」。** RAGChecker 报告的关键发现包括：检索改进与噪声引入之间存在 trade-off；忠实的开源模型倾向于盲目信任上下文（blind trust on context）[^7]。这意味着提高 context recall 未必单调改善端到端质量，与 Ragas 关于「单独改进一个组件并不保证更好的端到端结果」的立场一致[^16]。

**（6）基准本身的适用边界。** RAGChecker 批评 RGB/RECALL/NoMIRACL 这类人工构造数据集无法揭示生成结果与不同检索行为之间的纠缠关系[^7]；CRUD-RAG 批评现有基准几乎只测问答任务、且通常只评测流水线中的 LLM 部分或知识密集型场景下的检索器[^8]；MultiHop-RAG 指出既有基准主要评测答案可由单条证据检索并解决的简单情形[^26]；RAGBench 批评既有研究各自设计新数据集、常用 LLM 作生成器与标注器导致不可复现，且跨域泛化仍是开放问题[^30]。

**（7）工具元数据的时间与命名不一致。** Ragas 仓库存在 explodinggradients 与 vibrantlabsai 两个路径，301 重定向与组织页空仓库共同指向迁移[^49][^50]，但搜索工具返回的 last push 日期（2026-02-24）与 README 内容需按迁移后仓库理解[^9][^48]。AIMultiple 页面标注 updated on Mar 17, 2026，与部分引用所称 2026-03-23 不一致[^47]。TruLens README 自报的 95% agent 错误捕获率引用 arXiv:2510.08847[^11]，而该论文作者 8 人全部隶属 Snowflake，属厂商自研自评、非独立第三方[^46]，且论文中 95% 与「76%–86%」不是同一口径[^46]。

## 参考来源

[^1]: Ragas: Automated Evaluation of Retrieval Augmented Generation (arXiv:2309.15217v2) — https://arxiv.org/abs/2309.15217v2（arxiv, 2023-09-26）
[^2]: Ragas 文档：List of available metrics — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/（web）
[^3]: Ragas 文档：Faithfulness — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/（web）
[^4]: Ragas 文档：Context Recall — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/（web）
[^5]: Ragas 文档：Context Precision — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/（web）
[^6]: Ragas 文档：Response Relevancy (Answer Relevancy) — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/（web）
[^7]: RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation — https://arxiv.org/html/2408.08067v2（arxiv, 2024-08-15）
[^8]: CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models — https://arxiv.org/html/2401.17043v3（arxiv, 2024-01-30）
[^9]: explodinggradients/ragas — Supercharge Your LLM Application Evaluations (README) — https://github.com/explodinggradients/ragas（github）
[^10]: confident-ai/deepeval — The LLM Evaluation Framework (README) — https://github.com/confident-ai/deepeval（github）
[^11]: truera/trulens — Evaluation and Tracking for LLM Experiments and AI Agents (README) — https://github.com/truera/trulens（github）
[^12]: Arize-ai/phoenix — Arize's open-source AI observability platform (README) — https://github.com/Arize-ai/phoenix（github）
[^13]: beir-cellar/beir — A Heterogeneous Benchmark for Information Retrieval (README) — https://github.com/beir-cellar/beir（github）
[^14]: embeddings-benchmark/mteb — Multimodal toolbox for evaluating embeddings and retrieval systems (README) — https://github.com/embeddings-benchmark/mteb（github）
[^16]: Ragas Docs — Overview of Metrics（指标类型与 LLM-based vs Non-LLM-based） — https://docs.ragas.io/en/stable/concepts/metrics/overview/（web）
[^17]: TruLens Docs — The RAG Triad（context relevance / groundedness / answer relevance） — https://www.trulens.org/getting_started/core_concepts/rag_triad/（web）
[^18]: LangSmith Docs — Evaluation 概念：离线评估与在线评估双轨流程 — https://docs.smith.langchain.com/evaluation/concepts（web）
[^20]: ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems (arXiv:2311.09476v2, NAACL 2024) — https://arxiv.org/abs/2311.09476（arxiv, 2023-11-16 (v1); v2 2024-03-31）
[^21]: ARES 全文（arXiv HTML v2）§1–§4.2：三阶段流程、judge 训练、PPI 打分与实验设置 — https://arxiv.org/html/2311.09476v2（arxiv, 2024-03-31）
[^22]: ARES 全文（ar5iv 镜像）§1 引言：与 RAGAS 的对比及报告的一致性/效率数字 — https://ar5iv.labs.arxiv.org/html/2311.09476（arxiv, 2024-03-31）
[^23]: Benchmarking Large Language Models in Retrieval-Augmented Generation (arXiv:2309.01431v2) — https://arxiv.org/abs/2309.01431v2（arxiv, 2023-09-04）
[^24]: Benchmarking Large Language Models in Retrieval-Augmented Generation — 正文（Benchmark 构造章节，HTML 版） — https://arxiv.org/html/2309.01431v2（arxiv, 2023-12-20）
[^25]: chen700564/RGB — 官方代码与数据仓库 README — https://github.com/chen700564/RGB（github, 2024-05-17）
[^26]: MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries (arXiv:2401.15391, abs page) — https://arxiv.org/abs/2401.15391（arxiv, 2024-01-27）
[^27]: MultiHop-RAG 正文（HTML v1）：multi-hop 查询定义、四类查询与评测维度 — https://arxiv.org/html/2401.15391v1（arxiv, 2024-01-27）
[^28]: yixuantt/MultiHop-RAG 官方仓库 README — https://github.com/yixuantt/MultiHop-RAG（github, 2024-01-23）
[^29]: RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems (arXiv:2407.11005, abs page) — https://arxiv.org/abs/2407.11005（arxiv, 2024-06-25）
[^30]: RAGBench 正文（HTML v2）：TRACe 指标定义与数据构造 — https://arxiv.org/html/2407.11005v2（arxiv, 2025-01-16）
[^31]: Evaluating — LlamaIndex 官方文档（evaluation 模块总览） — https://docs.llamaindex.ai/en/stable/module_guides/evaluating/（web）
[^32]: Usage Pattern (Response Evaluation) — LlamaIndex 官方文档 — https://docs.llamaindex.ai/en/stable/module_guides/evaluating/usage_pattern/（web）
[^34]: Answer Relevancy and Context Relevancy Evaluations — LlamaIndex 官方示例 notebook — https://docs.llamaindex.ai/en/stable/examples/evaluation/answer_and_context_relevancy/（web）
[^35]: Correctness Evaluator — LlamaIndex 官方示例 notebook — https://docs.llamaindex.ai/en/stable/examples/evaluation/correctness_eval/（web）
[^36]: Embedding Similarity Evaluator (SemanticSimilarityEvaluator) — LlamaIndex 官方示例 notebook — https://docs.llamaindex.ai/en/stable/examples/evaluation/semantic_similarity_eval/（web）
[^37]: langchain-ai/langsmith-sdk（已归档的 LangSmith 客户端 SDK 仓库） — https://github.com/langchain-ai/langsmith-sdk（github, 2023-05-30）
[^39]: Self-hosted LangSmith — LangChain 官方文档 — https://docs.langchain.com/langsmith/self-hosted（web）
[^43]: GitHub 仓库搜索结果（关键词 langsmith，按 stars 排序） — https://github.com/search?q=langsmith&s=stars（github）
[^44]: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (arXiv:2306.05685v4, NeurIPS 2023 D&B) — https://arxiv.org/html/2306.05685v4（arxiv, 2023-06-09 (v1); v4 2023-12-24）
[^45]: Ragas Docs — Evaluate a simple RAG system（RAG 评测工作流指南） — https://docs.ragas.io/en/stable/getstarted/rag_eval/（web, 2025-12-09）
[^46]: What Is Your Agent's GPA? A Framework for Evaluating Agent Goal-Plan-Action Alignment (arXiv:2510.08847v2) — https://arxiv.org/abs/2510.08847（arxiv, 2025-10-09 (v1); v2 revised 2026-03-27）
[^47]: RAG Evaluation Tools: W&B Weave vs Ragas vs TruLens (AIMultiple, Cem Dilmegani with Ekrem Sarı) — https://research.aimultiple.com/rag-evaluation-tools/（web, 页面标注 updated on Mar 17, 2026（与子问题所称 2026-03-23 不一致））
[^48]: vibrantlabsai/ragas — repository metadata (stars, forks, created, last push) — https://github.com/vibrantlabsai/ragas（github, 2026-02-24）
[^49]: explodinggradients/ragas — GitHub API 301 redirect to repository 637924634 — https://api.github.com/repos/explodinggradients/ragas（github）
[^50]: github.com/explodinggradients — organization page — https://github.com/explodinggradients（github）
[^51]: PyPI — ragas 0.4.3 project page — https://pypi.org/project/ragas/（web）
[^52]: vibrantlabsai/ragas — Releases 页面（最新版本与发布者） — https://github.com/vibrantlabsai/ragas/releases（github）