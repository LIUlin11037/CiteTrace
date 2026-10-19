# 如何评测一个 RAG 系统：检索侧与生成侧的指标拆解

## 摘要

RAG 系统由检索模块与基于 LLM 的生成模块组成，评测必须同时覆盖「检索器能否找到相关且聚焦的上下文」与「LLM 能否忠实利用这些上下文」两个维度[^8]。当前主流做法可归为三条路线：一是以 RAGAS 为代表的指标库路线，把评测拆成 Context Precision / Context Recall / Faithfulness / Answer Relevancy 等可计算指标，并强调无参考（reference-free）评估[^1][^8]；二是以 BEIR、KILT、RGB、MultiHop-RAG、RAGBench、CRUD-RAG 为代表的基准数据集路线，用固定语料与任务集做横向比较[^9][^11][^27][^28][^29]；三是以 RAGChecker、Azure RAG evaluators、Amazon Bedrock evaluations 为代表的「检索侧 / 生成侧解耦诊断」路线，把指标显式归因到 pipeline 的某个组件[^16][^23][^24]。检索侧的核心指标是排序质量类（NDCG@k、MAP@K、Recall@K、Precision@K、MRR）与上下文相关性类（Context Precision、Context Recall、Context Entities Recall、Retrieval 二元判定）[^3][^4][^6][^9][^23][^40][^41][^42][^43][^44]；生成侧的核心指标是忠实性 / 接地性（Faithfulness、Groundedness）、答案相关性（Answer Relevancy、Relevance）、完整性（Response Completeness）与噪声敏感度（Noise Sensitivity）[^2][^5][^7][^22][^23]。需要明确的是：本报告所依据的 44 条证据中，**没有任何一条提供可核验的公开评测数值量级**，也**没有任何一条来自 LLM-as-judge 偏差研究**，这两点是本报告的主要覆盖缺口。

---

## 一、评测对象的结构：为什么必须分侧

RAG 的原始定义来自 Lewis 等人 2020 年的论文：将预训练参数化记忆（seq2seq 模型）与非参数化记忆（Wikipedia 的 dense vector index，经预训练神经检索器访问）结合用于语言生成[^13]。该论文比较了两种 RAG 形式——一种在整个生成序列上条件于相同检索段落，另一种可对每个 token 使用不同段落[^13]。

RAGAS 论文进一步把评测需求拆成三个维度：检索系统识别相关且聚焦的上下文段落的能力、LLM 忠实利用这些段落的能力、以及生成本身的质量[^8]。RAGChecker 论文则指出 RAG 系统全面评测困难的原因包括 RAG 的模块化特性、长文本回答的评测、以及测量的可靠性[^30]，并据此为检索模块和生成模块**分别**提供一套诊断指标[^30]。

Azure 的官方文档给出了最清晰的分侧命名：把 RAG 评测器分为「过程评测（process evaluation）」与「系统评测（system evaluation）」两类，Document Retrieval 与 Retrieval 属过程评测，Groundedness / Groundedness Pro / Relevance / Response Completeness 属系统评测[^23]。Amazon Bedrock 则从作业类型上分侧：提供 Retrieve only（仅基于检索数据）与 Retrieve and generate（基于检索数据 + 生成模型摘要）两类 RAG 评测作业[^24]。

---

## 二、检索侧指标

### 2.1 经典 IR 排序指标（有 ground truth 标签时）

BEIR 官方评估使用 NDCG@k、MAP@K、Recall@K、Precision@K，k 取 [1,3,5,10,100,1000]，并支持通过 `evaluate_custom` 的 `metric='mrr'` 计算 MRR[^9]。

这些指标的标准定义以 TREC 官方评测工具 trec_eval 的实现为准：

- **Precision@k**：度量名为 `P`，定义为「在多个文档级 cutoff 处的 Precision@k」；若 cutoff 大于检索到的文档数，则假定其余位置由非相关文档填充。官方举例：若某方法检索 15 篇文档其中 4 篇相关，则 P20 = 0.2（4/20）。官方注释指出 Precision 是良好的面向用户的度量、适合单 topic 比较，但跨 topic 平均效果不佳[^44]。
- **Recall@k**：度量名为 `recall`，定义为「在多个文档级 cutoff 处的 Recall@k」；官方定义公式为 Recall = 检索到的相关文档数 / 相关文档总数（relevant retrieved / relevant），在排序的各个 cutoff 处测量；若 cutoff 大于实际检索到的文档数，则假定其余位置由非相关文档填充。官方注释指出 Recall 是良好的单 topic 度量，但跨 topic 平均效果不佳[^43]。
- **nDCG@k**：度量名为 `ndcg_cut`，计算在指定 cutoff 处的传统 nDCG；官方注释指明 nDCG 依据 Jarvelin and Kekalainen (ACM ToIS v.20, pp.422-446, 2002) 定义；增益（gain）取 qrels 文件中的 relevance 值；实现中 DCG 累加公式为 `sum += gain / log2(i+2)`（文档 i 的排名为 i+1），并以理想 DCG（IDCG）归一化得到 nDCG。默认 cutoff 参数为 5,10,15,20,30,100,200,500,1000[^42]。
- **MRR（Reciprocal Rank）**：度量名为 `recip_rank`，定义为「第一个相关检索文档的 Reciprocal Rank（倒数排名）」。官方注释说明该度量最适用于只有一个相关文档、或用户只想要一个相关文档的任务。实现逻辑为：遍历排序结果，找到第一个 relevance >= relevance_level 的文档位置 i，取值为 1.0/(i+1)。MRR 即对多个 query 的 recip_rank 取平均[^41]。

trec_eval 的 README 明确：该工具是 TREC 社区用于评测 ad hoc 检索 run 的标准工具，输入为结果文件与标准判定结果（qrels）；官方 TREC 用法示例为 `trec_eval -q -c -M1000 official_qrels submitted_results`[^40]。**注意**：trec_eval 的 README 未列出 Hit Rate 度量，其度量集合中不含 hit rate[^40]——因此若要在评测方案中使用 hit-rate，需明确它来自其他实现（如 LlamaIndex 的 Retrieval Evaluation 模块[^20]），而非 TREC 标准工具。

LlamaIndex 的 Retrieval Evaluation 模块使用 MRR、hit-rate、precision 等排序指标，需问题集与 ground-truth 排序，并支持从非结构化语料合成生成 (question, context) 对用于检索评测[^20]。

### 2.2 上下文相关性指标（无 ground truth 或部分无标签时）

**Context Precision（RAGAS）**：评估检索器将相关 chunk 排在无关 chunk 之前的能力，即相关 chunk 被置于排名顶部的程度。计算方式为上下文中每个 chunk 的 precision@k 的均值；Precision@k 是 rank k 处相关 chunk 数与该 rank 处 chunk 总数之比。公式为：

```
Context Precision@K = Σ_{k=1}^{K}(Precision@k × v_k) / (top K 结果中相关项总数)
```

其中 v_k ∈ {0,1} 为 rank k 的相关性指示；Precision@k = true positives@k / (true positives@k + false positives@k)[^3]。文档区分两种用法：有 reference answer 时用 ContextPrecision（对比每个 context 与参考答案）；无 reference 但有生成响应时用 ContextUtilization（对比每个 context 与生成响应）[^3]。

**Context Recall（RAGAS）**：衡量有多少相关文档（或信息片段）被成功检索到，关注不遗漏重要结果；由于关注不遗漏，计算 context recall 始终需要 reference 作为对比基准。LLM-based Context Recall 用 reference 作为 reference_contexts 的代理，将 reference 拆解为 claim，逐一判断能否归因于检索上下文。公式为：

```
Context Recall = 被检索上下文支持的 reference 中 claim 数 / reference 中 claim 总数
```

另有 NonLLMContextRecall，使用 retrieved_contexts 与 reference_contexts 做非 LLM 字符串比较，取值 0 到 1[^4]。

**Context Entities Recall（RAGAS）**：基于 reference 与 retrieved_contexts 中共同出现的实体数相对于 reference 中实体总数的比例，衡量检索上下文的实体召回。公式为：

```
Context Entity Recall = RCE 与 RE 的公共实体数 / RE 中实体总数
```

其中 RE 为 reference 中实体集合，RCE 为检索上下文中实体集合。该指标适用于事实型用例（如旅游客服、历史 QA），用于评估检索机制对实体的覆盖[^6]。

**Retrieval 评测器（Azure）**：在没有 ground truth 时，用 LLM judge 衡量检索到的上下文块与查询的相关性，输出基于 1–5 分阈值的二元 Pass/Fail[^23]。

**Document Retrieval 评测器（Azure）**：在有 ground truth 标签时，通过对比检索文档与标签计算搜索质量指标：Fidelity、NDCG、XDCG、Max Relevance、Holes（含 Pass/Fail）[^23]。

### 2.3 检索侧基准：BEIR 与 KILT

**BEIR** 是面向信息检索的鲁棒、异构评估基准，用于评估模型的分布外（OOD）泛化能力；论文选取 18 个公开数据集，覆盖多样文本检索任务与领域，评估 10 个 SOTA 检索系统（含 lexical、sparse、dense、late-interaction 与 re-ranking 架构）。论文结论：BM25 是鲁棒基线；re-ranking 与 late-interaction 模型平均取得最佳零样本性能但计算成本高；dense 与 sparse 检索模型计算更高效但常表现不佳。论文发表于 NeurIPS 2021 Dataset and Benchmark Track[^10]。仓库提供 17 个已预处理基准数据集，覆盖 lexical、dense、sparse 与 reranking-based 检索架构[^9]。

**KILT** 是面向知识密集型语言任务的基准，所有任务都基于同一份 Wikipedia 快照，以减少工程开销并加速任务无关记忆架构研究；覆盖开放域问答、事实核查、槽填充、实体链接等任务；除评估下游性能外，还评估模型提供 provenance（出处/证据）的能力。论文发现共享的 dense vector index 结合 seq2seq 模型是强基线，在事实核查、开放域问答与对话上优于更定制的方案；论文发表于 NAACL 2021[^11]。仓库方面：KILT 仓库已归档（2023-10-31 起只读）；知识源基于 2019/08/01 的 Wikipedia dump，`kilt_knowledgesource.json` 约 34.76GiB，含 5,903,530 个页面；数据格式中每条记录含 id、input、output，output 每个元素可含 answer（文本形式）与 provenance（来自 KILT 知识源的证据集，含 wikipedia_id、title、section、start/end paragraph 与 character 等字段）；KILT 仅发布测试集输入而不含答案，测试答案用于 EvalAI 上的 KILT challenge 排行榜，设有严格提交限制以防过拟合[^12]。

---

## 三、生成侧指标

### 3.1 忠实性 / 接地性（Faithfulness / Groundedness）

**Faithfulness（RAGAS）**：衡量响应与检索上下文在事实层面的一致性，取值 0 到 1，越高表示一致性越好。定义：若响应中所有 claim 都能被检索上下文支持，则该响应被视为 faithful。计算方式为：

```
Faithfulness = 被检索上下文支持的 claim 数 / 响应中 claim 总数
```

计算步骤：(1) 识别响应中所有 claim；(2) 检查每个 claim 是否可由检索上下文推断；(3) 按公式计算。示例：响应含 2 个 statement，其中 1 个可被上下文推断，则 Faithfulness = 1/2 = 0.5[^2]。

**Groundedness（Azure）**：衡量生成响应与给定上下文的一致性、是否编造内容（precision 方面），返回 1–5 分，基于模型判断（model-based judgment）[^22][^23]。Azure 另提供 Groundedness Pro（预览），使用 Azure AI Content Safety 服务，返回二元 pass/fail，无需部署模型[^22]。Azure 明确将 groundedness 定位为 precision 维度、response completeness 定位为 recall 维度，二者互补[^23]。

**Response Groundedness** 亦出现在 RAGAS 文档的 Nvidia 系列指标中[^1]。LlamaIndex 的 Response 指标清单中同样包含 Faithfulness[^20]。

### 3.2 答案相关性（Answer Relevancy / Relevance）

**Answer Relevancy（RAGAS）**：衡量响应与用户输入的相关程度，取值 0 到 1，越高表示与用户输入对齐越好。定义：若答案直接且恰当地回应原始问题，则视为相关；该指标关注答案是否匹配问题意图，**不评估事实准确性**，并惩罚不完整或含多余细节的答案。计算方式：基于响应生成一组人工问题（默认 3 个），计算用户输入 embedding 与每个生成问题 embedding 的余弦相似度，取平均。公式为：

```
Answer Relevancy = (1/N) Σ cosine_similarity(E_{g_i}, E_o)
```

N 默认为 3，可通过 strictness 参数配置。文档注明：由于余弦相似度数学范围为 -1 到 1，分数通常落在 0 到 1 之间但不保证[^5]。

**Relevance（Azure）** 是 Azure RAG 专用评测器之一[^22]，属系统评测类[^23]。**Context Relevance** 出现在 RAGAS 的 Nvidia 系列指标中[^1]，也出现在 LlamaIndex 的 Response 指标清单中[^20]。**Response Relevancy** 是 RAGAS 官方 RAG 相关指标之一[^1]。

### 3.3 完整性（Response Completeness）

**Response Completeness（Azure，预览）**：衡量响应是否遗漏 ground truth 中的关键信息（recall 方面）[^23]。Azure 明确将其与 Groundedness 配对使用：groundedness 管 precision、response completeness 管 recall[^23]。

### 3.4 噪声敏感度（Noise Sensitivity）

**NoiseSensitivity（RAGAS）**：衡量系统在使用相关或无关检索文档时产生错误响应的频率，取值 0 到 1，**越低越好**。计算基于 user_input、reference、response 与 retrieved_contexts；检查生成响应中每个 claim 是否正确（基于 ground truth）以及能否归因于相关/无关检索上下文。公式为：

```
noise sensitivity (relevant) = 响应中错误 claim 总数 / 响应中 claim 总数
```

可通过 mode 参数设为 `'irrelevant'` 来计算针对无关上下文的噪声敏感度[^7]。

### 3.5 事实正确性与文本相似度类

RAGAS 文档列出的自然语言比较类指标包括 Factual Correctness、Semantic Similarity、BLEU、CHRF、ROUGE、Exact Match 等[^1]。Azure 的文本相似度类评测器包括 Similarity、F1、BLEU、GLEU、ROUGE、METEOR[^22]。LlamaIndex 的 Response 指标中 Correctness 与 Semantic Similarity **需要 labels**[^20]。Vertex AI 的 computation-based metrics（如 ROUGE、BLEU）**需 ground truth**[^34]。

### 3.6 生成侧基准：RGB、MultiHop-RAG、RAGBench、CRUD-RAG

**RGB** 论文系统研究 RAG 对 LLM 的影响，提出 RAG 所需的 4 项基础能力：noise robustness（噪声鲁棒性）、negative rejection（负样本拒答）、information integration（信息整合）、counterfactual robustness（反事实鲁棒性）。RGB 是一个同时支持英文和中文的 RAG 评测语料库，依据上述 4 项基础能力将基准内的实例划分为 4 个独立的测试床。论文在 RGB 上评测了 6 个代表性 LLM，结论是：LLM 表现出一定程度的噪声鲁棒性，但在 negative rejection、information integration 以及处理虚假信息方面仍显著吃力。论文被 AAAI 2024 接收[^29]。

**MultiHop-RAG** 是面向多跳查询的 RAG 评测数据集，作者称此前没有 RAG 基准数据集专注于多跳查询。数据集包含四部分：一个知识库、大量多跳查询、其 ground-truth 答案、以及关联的支持证据；知识库基于一个英文新闻文章数据集构建。多跳查询被定义为需要检索并推理多块支持证据的查询。论文设计了两类评测实验：实验一比较不同 embedding 模型在多跳查询证据检索上的表现；实验二考察多种 SOTA LLM（含 GPT-4、PaLM、Llama2-70B）在给定证据下推理并回答多跳查询的能力。两类实验均显示现有 RAG 方法在多跳查询的检索与回答上表现不理想[^28]。

**RAGBench** 被作者定位为第一个大规模、综合性的 RAG 基准数据集，包含 100k 条样本，覆盖五个独特的行业特定领域以及多种 RAG 任务类型，样本来源于行业语料（如用户手册）。论文形式化了 TRACe 评测框架，作者称其为一组可解释、可行动的 RAG 评测指标，适用于所有 RAG 领域；RAGBench 提供可解释标签，用于对 RAG 系统做整体评测并给出可行动的改进反馈。论文报告：在 RAG 评测任务上，基于 LLM 的评测方法难以与微调后的 RoBERTa 模型竞争[^27]。

**CRUD-RAG** 是面向大语言模型检索增强生成的中文综合基准。论文指出现有 RAG 基准范围与多样性有限：多数基准主要评估问答应用，忽略了 RAG 可发挥优势的更广泛场景；且现有基准在实验中只评估 RAG 流水线中 LLM 组件的性能，而忽略了检索组件与外部知识库的影响。CRUD-RAG 将 RAG 应用范围划分为四种类型——Create（生成原创、多样内容）、Read（知识密集型情境下回答复杂问题）、Update（修订与纠正既有文本中的错误或不一致）、Delete（将长文本摘要为更简洁形式），并为每个类别构建了综合数据集。论文分析了 RAG 系统各组件的效果，包括检索器、上下文长度、知识库构建与 LLM。该论文提交于 2024-01-30，最新版本 v3 修订于 2024-07-15，共 40 页[^14]。

---

## 四、解耦归因：把指标绑定到组件

**RAGChecker** 定位为细粒度 RAG 诊断式自动评测框架，指标分为三类：Overall Metrics（整条 pipeline）、Diagnostic Retriever Metrics（检索组件）、Diagnostic Generator Metrics（生成组件）；采用 claim-level entailment（断言级蕴含）操作实现细粒度评测[^16]。论文指出 RAG 系统全面评测困难的原因包括 RAG 的模块化特性、长文本回答的评测、以及测量的可靠性；元评测验证 RAGChecker 与人类判断的相关性显著优于其他评测指标；作者用 RAGChecker 评测了 8 个 RAG 系统并做深入分析，揭示 RAG 架构设计选择中的模式与权衡[^30]。输入数据唯一必需标注为 ground truth answer（gt_answer），另需 query、response、retrieved_context；默认/示例 judge 配置为 AWS Bedrock 版 Llama3 70B 作为 claim extractor 和 checker[^16]。

**Azure** 的分侧命名（过程评测 vs 系统评测）已在第一节引用[^23]。**Amazon Bedrock** 从作业类型分侧：Retrieve only 可评测 Bedrock Knowledge Base 或自带外部 RAG 推理数据；Retrieve and generate 基于检索数据 + 生成模型摘要。Bedrock RAG 评测使用「评测模型（evaluator models）」计算指标，内置指标支持的评测模型包括 Amazon Nova 系列、Anthropic Claude 3.5/3.7/4.x 系列、OpenAI GPT-5.x、Meta Llama 3.1 70B、Mistral Large 等；支持自定义指标，其评测模型清单与内置指标不同。评测结果用于比较不同 Knowledge Base 或 RAG 系统，以选择最适合应用的方案[^24]。

**RAGAS** 官方将指标按应用场景分类：RAG 相关指标包括 Context Precision、Context Recall、Context Entities Recall、Noise Sensitivity、Response Relevancy、Faithfulness、Multimodal Faithfulness、Multimodal Relevance；Nvidia 系列指标包括 Answer Accuracy、Context Relevance、Response Groundedness；自然语言比较类指标包括 Factual Correctness、Semantic Similarity、BLEU、CHRF、ROUGE、Exact Match 等。RAGAS 说明其指标是评估应用某一特定方面的范式，LLM-based 指标可能使用一次或多次 LLM 调用来得出分数[^1]。

---

## 五、评测工具与平台现状

**RAGAS**：论文定位为面向 RAG 流水线的无参考（reference-free）评估框架，即不依赖人工标注的 ground truth；论文指出 RAG 系统由检索模块与基于 LLM 的生成模块组成，评估需考虑多个维度；论文定位该框架可加速 RAG 架构的评估循环[^8]。README 标题为 'Supercharge Your LLM Application Evaluations'，自我定位为评估与优化 LLM 应用的 toolkit；主张提供 'Objective Metrics'：同时使用 LLM-based 与传统（traditional）指标；提供 'Test Data Generation'，称支持 'production-aligned test set generation'；声称与 LangChain 等 LLM 框架及主流 observability 工具集成；README 示例中 judge LLM 通过 `llm_factory("gpt-4o", client=AsyncOpenAI())` 配置，需设置 `OPENAI_API_KEY` 环境变量；README 展示的指标示例为 DiscreteMetric（Aspect Critique），允许自定义 allowed_values 与 prompt；仓库页面显示 1,147 commits；安装方式为 `pip install ragas`[^15]。

**LlamaIndex**：提供 Response Evaluation 与 Retrieval Evaluation 两类评测模块；Response Evaluation 使用 'gold' LLM（如 GPT-4）判断预测答案是否正确，属 LLM-based 评测；文档明确说明许多评测模块不需要 ground-truth labels，可仅用 query、context、response 组合加 LLM 调用完成；Response 指标包括 Correctness（需 labels）、Semantic Similarity（需 labels）、Faithfulness、Context Relevancy、Answer Relevancy、Guideline Adherence；Retrieval Evaluation 使用 MRR、hit-rate、precision 等排序指标，需问题集与 ground-truth 排序；支持从非结构化语料合成生成 (question, context) 对用于检索评测；文档列出与社区评测工具集成：UpTrain、Tonic Validate、DeepEval、Ragas、RAGChecker、Cleanlab[^20]。

**Phoenix（Arize）**：开源 AI observability 平台，用于 experimentation、evaluation、troubleshooting；托管生产工作流由商业产品 Arize AX 提供。核心能力包括 Tracing（基于 OpenTelemetry 的 instrumentation）、Evaluation（用 LLM 对 response 与 retrieval 做 benchmark）、Datasets、Experiments、Playground、Prompt Management；README 明确 Evaluation 覆盖 'response and retrieval evals' 两类；声称 vendor 与 language agnostic，开箱支持 OpenAI Agents SDK、Claude Agent SDK、LangGraph、Vercel AI SDK、Mastra、CrewAI、LlamaIndex、DSPy 等框架；支持 LLM provider 包括 OpenAI、Anthropic、Google GenAI、Google ADK、AWS Bedrock、OpenRouter、LiteLLM 等；仓库页面显示 10,288 commits[^17]。

**TruLens**：定位为开源、OpenTelemetry-native 的 LLM/agent 评测与追踪工具，用 decorator 插桩、LLM judges 打分；核心概念包括 Metrics、RAG Triad、Honest/Harmless/Helpful Evals；judge 模型通过 provider 包配置：trulens-providers-openai / litellm / google / bedrock / cortex / huggingface 等，即支持多 provider 而非单一默认 LLM；仓库页面显示 1,957 commits；安装为 `pip install trulens`[^18]。README 自报评测成绩：Agent GPA 在 TRAIL/GAIA 上捕获 95% agent 错误（281 条人工标注错误中 267 条），基线 trace judge 为 55%（引用 arXiv:2510.08847）；自报 Groundedness F1 0.81（LLM-AggreFact，11,000 例 holdout），Context relevance NDCG@5 0.93。**但上述对比性主张（领先 WandB Weave、RAGAS、DeepEval、UpTrain）出处标注为 AIMultiple, 23 March 2026，属第三方评测来源**[^18]，不能作为产业界公开评测数字的证据（详见「矛盾与不确定性」）。

**DeepEval**：定位为开源 LLM 评测框架，类比 Pytest，用于对 LLM 应用做单元测试；指标由 'ANY LLM of your choice'、统计方法或本地运行的 NLP 模型驱动，即 judge 模型可配置、非固定默认。通用指标：JevEval、G-Eval（LLM-as-a-judge）、DAG（图结构确定性 LLM-as-a-judge 指标构建器）；Agentic 指标：Task Completion、Tool Correctness、Goal Accuracy、Step Efficiency、Plan Adherence、Plan Quality、Tool Use、Argument Correctness；RAG 指标：Answer Relevancy、Faithfulness、Contextual Recall（README 截断处）等；支持端到端黑盒评测、完整 agent trajectory 评测、单步（LLM 调用/工具使用/检索/sub-agent 交接）评测；仓库页面显示 10,366 commits；搜索工具返回 stars 18,853、forks 2,070、last push 2026-10-12[^19]。

**LangSmith**：仓库为 LangSmith 平台的 Python 与 JavaScript SDK，平台本身为闭源托管服务；定位为帮助团队 debug、evaluate、monitor 语言模型与智能 agent，兼容任意 LLM 应用，并与 LangChain Python/JS 原生集成；由 LangChain 公司开发维护；README 未列出具体评测指标清单，评测能力文档指向 docs.smith.langchain.com；README 页脚显示 Stars 1.1k、Forks 313、Watchers 14；仓库 3,175 commits[^21]。

**云厂商内置评测器**：Azure 将内置评测器分为四类：通用目的（Coherence、Fluency）、文本相似度（Similarity、F1、BLEU、GLEU、ROUGE、METEOR）、RAG 评测器、风险与安全评测器；RAG 专用评测器包括 Retrieval、Document Retrieval、Groundedness、Groundedness Pro（预览）、Relevance、Response Completeness（预览）；风险与安全评测器包含 Indirect Attack (XPIA)，用于衡量响应是否被通过检索上下文注入的间接越狱攻击所欺骗[^22]。Amazon Bedrock 提供 Retrieve only 与 Retrieve and generate 两类 RAG 评测作业[^24]。Vertex AI Gen AI evaluation service 采用 test-driven 框架，推荐默认使用 GENERAL_QUALITY 自适应 rubric 指标，提供 INSTRUCTION_FOLLOWING、TEXT_QUALITY、多轮对话指标、Agent 评测指标（final_response_reference_free、final_response_quality）；核心特性是 adaptive rubrics（为每个 prompt 生成定制的通过/失败测试）；支持四类评测方法：Adaptive rubrics（推荐）、Static rubrics、Computation-based metrics（如 ROUGE、BLEU，需 ground truth）、Custom functions（Python 自定义逻辑）；评测数据集可通过上传文件、从生产日志采样、或合成数据生成三种方式创建；支持 Google Cloud console 与 Python SDK 两种接口[^33][^34]。**该 Vertex AI 页面未列出 RAG 专属的 groundedness / context recall 指标定义**[^33][^34]。

**OpenAI**：官方将评测（evals）定义为测试模型输出是否满足指定风格与内容标准，并强调在升级或尝试新模型时评测是构建可靠应用的必要环节；给出构建并运行 LLM 应用评测的三步流程：把任务描述为一个 eval、用测试输入运行 eval、分析结果并迭代改进 prompt；**官方声明正在弃用 Evals 平台**：现有内容在过渡期保留，2026 年 10 月 31 日起对现有用户变为只读，平台计划于 2026 年 11 月 30 日关停；建议新用户或需要迭代实验环境的用户改用 Datasets[^25]。OpenAI 官方 Graders 指南将 graders 定义为对照参考答案评估模型表现的方式，返回 0 到 1 的分数，并支持部分得分而非仅二元 0/1；列出的 grader 类型包括：String check、Text similarity、Score model grader、Python code execution，并说明可在强化微调中用 multigrader 嵌套组合[^37]。

**NVIDIA NeMo Evaluator**：是一个 LLM 评测框架，提供基准环境、可插拔 solver 与多格式报告；内置 17 个基准，包括 MMLU、MMLU-Pro、MATH-500、GPQA、GSM8K、DROP、MGSM、TriviaQA、HumanEval、SimpleQA、HealthBench、XSTest 等——**这些属通用 LLM 能力基准，非 RAG 专用指标**；框架提供统计回归能力：用 McNemar 精确检验、配对翻转分析与置信区间比较两次运行，并支持按基准阈值做发布门禁（GO/NO-GO/INCONCLUSIVE）[^26]。

**Anthropic**：建议先定义成功标准（specific、measurable、achievable、relevant），再设计评测；量化指标示例包括任务特定（F1、BLEU、perplexity）、通用（accuracy、precision、recall）、运维（响应时间 ms、uptime %）；量化方法包括 A/B 测试、用户反馈、边缘案例分析；定性方法包括 Likert 量表与专家 rubric；列出的常见成功标准包含 'Context utilization'：模型多有效地利用所提供的上下文、引用并基于历史信息构建回答[^35][^38]。**该页为通用 LLM 评测方法论，未给出 RAG 专属的 faithfulness / context recall 指标定义**[^35][^38]。

---

## 六、评测方法论层面的通用建议

Anthropic 官方评估文档强调先明确成功标准再设计评估，成功标准应具备 Specific、Measurable、Achievable、Relevant 等特征[^38]。OpenAI 官方给出构建并运行 LLM 应用评测的三步流程：把任务描述为一个 eval、用测试输入运行 eval、分析结果并迭代改进 prompt[^25]。Vertex AI 采用 test-driven 框架，将评测从主观评分转为客观、可执行的结果[^33]。NeMo Evaluator 提供统计回归能力（McNemar 精确检验、配对翻转分析、置信区间）与发布门禁机制[^26]。这些方法论层面的建议与具体指标选择正交，但决定了指标数值能否被信任。

---

## 矛盾与不确定性

**1. LLM-as-judge 评测的可靠性与偏差研究：零覆盖。**
本报告所依据的 44 条证据中，**没有任何一条来自 LLM-as-judge 偏差或可靠性研究**。现有证据全部是端到端基准（RAGBench、CRUD-RAG、RGB、MultiHop-RAG）与解耦归因框架（RAGChecker）。仅有的间接线索是 RAGBench 论文的一句结论：在 RAG 评测任务上，基于 LLM 的评测方法难以与微调后的 RoBERTa 模型竞争[^27]。这不足以支撑关于 LLM-as-judge 偏差类型、校准方法或可复现性的任何结论。此外，OpenAI Graders 指南[^37]与 Anthropic 两份评测文档[^35][^38]的正文均**未出现**针对 LLM-as-judge 偏差、不可复现或需人工校准的明确警示或使用限制说明——这是官方文档层面的空白，而非「官方认为不存在偏差」的证据。**现有公开资料不足以确认 LLM-as-judge 在 RAG 评测中的偏差量级与缓解手段。**

**2. 公开评测数字：无官方来源可引用。**
Azure、AWS、OpenAI、Anthropic、Google、NVIDIA 六份官方文档均只描述指标定义与流程，**未给出可引用的评测数值**。唯一数字来自 TruLens README 自报（Groundedness F1 0.81、Context relevance NDCG@5 0.93）[^18]，但该主张的出处被标注为「AIMultiple, 23 March 2026」——第三方媒体转述、且日期晚于当前时点 2026-10-14，属不可核验来源，不能作为产业界公开评测数字的证据。**因此本报告无法给出「某基准上 faithfulness/recall 的量级」这类结论。**

**3. RAGAS 主仓库归属矛盾未解决。**
证据 [15] 显示 explodinggradients/ragas 的 README 安装说明指向 vibrantlabsai/ragas，仓库页面显示 1,147 commits[^15]；证据 [39] 则称 vibrantlabsai/ragas 是「当前主仓库」，stars 为 16,023、forks 为 1,761、last push 显示为 2026-02-24，但该证据自身标注「日期异常，不可作为可靠 commit 日期」[^39]。两条证据对同一项目的仓库归属、stars 与活跃度给出互斥描述。**关于 RAGAS 的 stars 量级与最近提交活跃度，现有公开资料不足以给出可信答案。**

**4. 指标命名的跨平台不一致。**
同一概念在不同平台命名不同：RAGAS 用 Faithfulness[^2]，Azure 用 Groundedness[^22][^23]，RAGAS 的 Nvidia 系列用 Response Groundedness[^1]；RAGAS 用 Answer Relevancy[^5]，Azure 用 Relevance[^22]，LlamaIndex 同时列 Context Relevancy 与 Answer Relevancy[^20]。这些指标的定义边界（例如 Azure Groundedness 返回 1–5 分[^22] vs RAGAS Faithfulness 返回 0–1[^2]）不可直接互换比较。

**5. 检索侧指标的可比性限制。**
trec_eval 官方注释明确指出：Precision 与 Recall 是良好的单 topic 度量，但**跨 topic 平均效果不佳**（P20 的期望特性随 topic 相关文档总数不同而差异很大）[^43][^44]。这意味着把 Precision@k / Recall@k 直接跨数据集平均，在方法论上是有问题的。BEIR 官方评估同时报告 NDCG@k、MAP@K、Recall@K、Precision@K 四类[^9]，但未说明如何处理跨数据集聚合问题。

**6. trec_eval 不含 hit rate。**
trec_eval 的 README 未列出 Hit Rate 度量，其度量集合中不含 hit rate[^40]。LlamaIndex 的 Retrieval Evaluation 使用 hit-rate[^20]，但其实现来源与定义未在证据中给出。**若评测方案要求 hit-rate，需明确其实现来源，不能假定 TREC 标准工具支持。**

**7. OpenAI Evals 平台关停的时间线。**
OpenAI 官方声明 Evals 平台 2026 年 10 月 31 日起对现有用户变为只读，计划于 2026 年 11 月 30 日关停[^25]。但抓取到的 OpenAI Deprecations 页面正文片段仅显示模型弃用条目（如 2026-10-14 的 GPT-4 系列弃用、2026-10-01 的 GPT-5.3-Codex 等），**未包含 Evals 平台关停条目**[^31]。两处官方文档对同一事项的记录不一致
**8. 评测数据集与生产数据的分布差异未被任何证据量化。**
RAGAS README 声称支持 'production-aligned test set generation'[^15]，Vertex AI 支持从生产日志采样创建评测数据集[^34]，LlamaIndex 支持从非结构化语料合成生成 (question, context) 对[^20]。但没有任何一条证据给出「合成 / 采样数据集上的指标」与「真实生产流量上的指标」之间的偏差量级。**现有公开资料不足以确认合成评测集能否代表生产分布。**

**9. 检索配置对指标的影响未被评测框架纳入。**
Amazon Bedrock 的检索配置文档显示：知识库查询默认返回最多 5 条结果，每条对应一个 source chunk；`numberOfResults` 参数设置返回结果的最大数量，若启用分层分块，该参数映射到子 chunk 数量；搜索类型包括 Default（由 Bedrock 决定）、Hybrid（向量嵌入语义搜索 + 原始文本搜索）、Semantic（仅向量嵌入）；Hybrid 搜索仅支持 Amazon RDS、Amazon OpenSearch Serverless 和 MongoDB 向量存储（需含可过滤文本字段）[^36]。**该页为检索配置文档，未包含 RAG 评测指标清单或评测数字**[^36]。这意味着：同一套检索指标（如 Context Recall）在不同 `numberOfResults` 与搜索类型下会给出不同数值，但证据中没有任何框架说明如何在报告指标时固定或披露这些配置。

**10. CRUD-RAG 指出的评测盲区尚未被后续证据填补。**
CRUD-RAG 论文明确指出：现有基准在实验中只评估 RAG 流水线中 LLM 组件的性能，而忽略了检索组件与外部知识库的影响[^14]。RAGChecker 与 Azure 的过程/系统评测二分[^23][^30]是对这一批评的回应，但证据中没有任何一条给出「解耦评测相比端到端评测能多发现多少问题」的量化对比。

**11. 多跳与拒答场景的评测缺口。**
MultiHop-RAG 论文称此前没有 RAG 基准数据集专注于多跳查询，且两类实验均显示现有 RAG 方法在多跳查询的检索与回答上表现不理想[^28]。RGB 论文结论指出 LLM 在 negative rejection、information integration 以及处理虚假信息方面仍显著吃力[^29]。这两项结论指向同一类缺口：**主流检索指标（NDCG@k、Recall@k）与生成指标（Faithfulness、Answer Relevancy）都不直接度量「该拒答时是否拒答」与「跨多块证据的整合能力」**。证据中未出现专门度量这两项能力的标准化指标定义。

**12. 风险与安全维度仅 Azure 单点覆盖。**
Azure 的风险与安全评测器包含 Indirect Attack (XPIA)，用于衡量响应是否被通过检索上下文注入的间接越狱攻击所欺骗[^22]。在全部 44 条证据中，**这是唯一一条涉及 RAG 检索上下文注入攻击的评测指标**。其他平台（AWS、OpenAI、Anthropic、Google、NVIDIA）的评测文档中均未出现同类指标。**现有公开资料不足以确认 RAG 安全评测是否已形成跨平台共识。**

**13. 指标定义的「LLM 调用次数」成本未被系统比较。**
RAGAS 说明其 LLM-based 指标可能使用一次或多次 LLM 调用来得出分数[^1]，但未给出各指标的调用次数清单。RAGChecker 默认使用 AWS Bedrock 版 Llama3 70B 作为 claim extractor 和 checker[^16]，DeepEval 的指标可由 'ANY LLM of your choice' 驱动[^19]，TruLens 支持多 provider[^18]。**证据中没有任何一条给出「跑完一套完整 RAG 评测需要多少次 judge LLM 调用、成本量级如何」的数据**，因此无法在指标选择与评测成本之间做定量权衡。

**14. 基准的时效性与维护状态。**
KILT 仓库已归档（2023-10-31 起只读）[^12]，其知识源基于 2019/08/01 的 Wikipedia dump[^12]。BEIR 论文发表于 NeurIPS 2021[^10]，另有 SIGIR 2024 Resource Track 的参考模型与官方排行榜论文[^9]。CRUD-RAG 最新版本 v3 修订于 2024-07-15[^14]。RAGBench v2 修订于 2025-01-16[^27]。**证据中没有任何一条说明这些基准的知识源是否会随 LLM 训练数据截止时间推移而失效**——若被测 LLM 的训练数据已覆盖基准语料，基准分数将失去区分度。这是一个证据层面的空白，而非「基准已失效」的结论。

**15. 中文与多语言评测覆盖不均。**
CRUD-RAG 是中文综合基准[^14]，RGB 同时支持英文和中文[^29]，MultiHop-RAG 基于英文新闻文章数据集[^28]，RAGBench 覆盖五个行业特定领域但未在证据中说明语言分布[^27]，BEIR 覆盖多样文本检索任务与领域但未在证据中说明语言分布[^10]。**除 CRUD-RAG 与 RGB 外，证据中其余基准的语言覆盖情况不足以确认。**

---

## 小结：一个可操作的分侧指标清单

综合以上证据，一个 RAG 评测方案在**检索侧**至少应覆盖：

- 有标签时：NDCG@k、MAP@K、Recall@K、Precision@K、MRR[^9][^40][^41][^42][^43][^44]，或 Azure Document Retrieval 的 Fidelity、NDCG、XDCG、Max Relevance、Holes[^23]；
- 无标签时：Context Precision[^3]、Context Recall（需 reference）[^4]、Context Entities Recall[^6]、Azure Retrieval 的二元 Pass/Fail[^23]；
- 组件归因：RAGChecker 的 Diagnostic Retriever Metrics[^16][^30]。

在**生成侧**至少应覆盖：

- 忠实性 / 接地性：Faithfulness[^2]、Groundedness / Groundedness Pro[^22][^23]、Response Groundedness[^1]；
- 相关性：Answer Relevancy[^5]、Relevance[^22]、Context Relevance[^1][^20]；
- 完整性：Response Completeness[^23]；
- 噪声鲁棒性：NoiseSensitivity[^7]、RGB 的 noise robustness / negative rejection / information integration / counterfactual robustness 四项能力[^29]；
- 事实正确性与相似度：Factual Correctness、Semantic Similarity、BLEU、CHRF、ROUGE、Exact Match[^1]，或 Azure 的 Similarity、F1、BLEU、GLEU、ROUGE、METEOR[^22]；
- 组件归因：RAGChecker 的 Diagnostic Generator Metrics[^16][^30]。

需要再次强调：**上述清单中的每一项指标，其数值在证据中都没有可引用的公开基准量级**；**指标之间的相关性、以及 LLM-as-judge 作为打分器的可靠性，在证据中均无研究支撑**。任何声称「某 RAG 系统 faithfulness 达到 X」的比较，若其 X 来自厂商自报或第三方媒体转述，均不满足本报告的证据标准。

## 参考来源

[^1]: Ragas Documentation — List of available metrics — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/（web, 2025-12-09）
[^2]: Ragas Documentation — Faithfulness — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/（web, 2025-12-09）
[^3]: Ragas Documentation — Context Precision — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/（web, 2025-12-09）
[^4]: Ragas Documentation — Context Recall — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/（web, 2025-12-09）
[^5]: Ragas Documentation — Answer Relevancy — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/（web, 2025-12-09）
[^6]: Ragas Documentation — Context Entities Recall — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_entities_recall/（web, 2025-12-09）
[^7]: Ragas Documentation — Noise Sensitivity — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/noise_sensitivity/（web, 2025-12-09）
[^8]: Ragas: Automated Evaluation of Retrieval Augmented Generation — https://arxiv.org/abs/2309.15217（arxiv, 2023-09-26）
[^9]: BEIR — A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models (repo) — https://github.com/beir-cellar/beir（github）
[^10]: BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models — https://arxiv.org/abs/2104.08663（arxiv, 2021-04-17）
[^11]: KILT: a Benchmark for Knowledge Intensive Language Tasks — https://arxiv.org/abs/2009.02252（arxiv, 2020-09-04）
[^12]: KILT — A Benchmark for Knowledge Intensive Language Tasks (repo) — https://github.com/facebookresearch/KILT（github）
[^13]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — https://arxiv.org/abs/2005.11401（arxiv, 2020-05-22）
[^14]: CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models — https://arxiv.org/abs/2401.17043（arxiv, 2024-01-30）
[^15]: explodinggradients/ragas (README) — https://github.com/explodinggradients/ragas（github）
[^16]: amazon-science/RAGChecker (README) — https://github.com/amazon-science/RAGChecker（github）
[^17]: Arize-ai/phoenix (README) — https://github.com/Arize-ai/phoenix（github）
[^18]: truera/trulens (README) — https://github.com/truera/trulens（github）
[^19]: confident-ai/deepeval (README) — https://github.com/confident-ai/deepeval（github）
[^20]: LlamaIndex Evaluating (官方文档) — https://docs.llamaindex.ai/en/stable/module_guides/evaluating/（web）
[^21]: langchain-ai/langsmith-sdk (README) — https://github.com/langchain-ai/langsmith-sdk（github）
[^22]: Built-in evaluators reference — Microsoft Foundry (Azure AI) — https://learn.microsoft.com/en-us/azure/ai-studio/concepts/evaluation-metrics-built-in（web）
[^23]: Retrieval-Augmented Generation (RAG) evaluators — Microsoft Foundry (Azure AI) — https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/rag-evaluators（web）
[^24]: Evaluate the performance of RAG sources using Amazon Bedrock evaluations — AWS Documentation — https://docs.aws.amazon.com/bedrock/latest/userguide/evaluation-kb.html（web）
[^25]: Evaluations (Evals) guide — OpenAI Platform Documentation — https://platform.openai.com/docs/guides/evals（web）
[^26]: NeMo Evaluator — NVIDIA Documentation — https://docs.nvidia.com/nemo/evaluator/latest/index.html（web）
[^27]: RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems — https://arxiv.org/abs/2407.11005（arxiv, 2024-06-25 (v1); v2 2025-01-16）
[^28]: MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries — https://arxiv.org/abs/2401.15391（arxiv, 2024-01-27）
[^29]: Benchmarking Large Language Models in Retrieval-Augmented Generation (RGB) — https://arxiv.org/abs/2309.01431（arxiv, 2023-09-04 (v1); v2 2023-12-20; AAAI 2024）
[^30]: RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation — https://arxiv.org/abs/2408.08067（arxiv, 2024-08-15）
[^31]: OpenAI Platform Docs — Deprecations — https://platform.openai.com/docs/deprecations（web）
[^33]: Define your evaluation metrics - Vertex AI Gen AI Evaluation Service — https://cloud.google.com/vertex-ai/generative-ai/docs/models/determine-eval（web）
[^34]: Gen AI evaluation service overview - Vertex AI — https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview（web）
[^35]: Create strong empirical evaluations - Anthropic Claude Docs — https://docs.anthropic.com/en/docs/build-with-claude/develop-tests（web）
[^36]: Configure and customize queries and response generation - Amazon Bedrock — https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-config.html（web）
[^37]: OpenAI Platform Docs — Graders guide — https://platform.openai.com/docs/guides/graders（web）
[^38]: Anthropic Docs — Define success criteria and build evaluations (eval tool) — https://docs.anthropic.com/en/docs/test-and-evaluate/eval-tool（web）
[^39]: vibrantlabsai/ragas (RAGAS 主仓库，原 explodinggradients/ragas) — https://github.com/vibrantlabsai/ragas（github）
[^40]: usnistgov/trec_eval — README (standard TREC evaluation tool) — https://raw.githubusercontent.com/usnistgov/trec_eval/main/README（github）
[^41]: trec_eval m_recip_rank.c — Reciprocal Rank (MRR) implementation — https://raw.githubusercontent.com/usnistgov/trec_eval/main/m_recip_rank.c（github）
[^42]: trec_eval m_ndcg_cut.c — nDCG@k implementation — https://raw.githubusercontent.com/usnistgov/trec_eval/main/m_ndcg_cut.c（github）
[^43]: trec_eval m_recall.c — Recall@k implementation — https://raw.githubusercontent.com/usnistgov/trec_eval/main/m_recall.c（github）
[^44]: trec_eval m_P.c — Precision@k implementation — https://raw.githubusercontent.com/usnistgov/trec_eval/main/m_P.c（github）