# 混合检索（向量+关键词）为什么优于单一检索？主流融合方法有哪些？

## 摘要

混合检索（hybrid search）指在同一次检索中并行执行词法（lexical）检索与语义（向量）检索，再把两路结果融合为一个排序列表。其优于单一检索的核心理由是两类表示在相关性建模上互补：稀疏/词法检索（TF-IDF、BM25）通过倒排索引做精确关键词匹配，但存在词汇鸿沟（lexical gap），只能召回包含查询关键词的文档，且把查询与文档当作 bag-of-words、不考虑词序[^12]；稠密检索把查询与文档编码为潜在语义向量，能把「bad guy」与「villain」这类无 token 重叠的表述匹配起来[^1]。主流融合方法可归为两大类：基于排名的融合（以 RRF 为代表，只读排名位置、丢弃分数幅度）与基于分数的融合（凸组合/加权归一化，如 Milvus WeightedRanker、Weaviate relativeScoreFusion、Qdrant DBSF）。需要明确声明的是：**现有证据无法给出「混合相对单一检索」在 BEIR / MS MARCO / TREC DL 等基准上的量化提升幅度**，也无法给出混合检索失效或反而更差的场景；且 RRF 与凸组合的相对优劣在证据内部存在方向性矛盾，尚无统一实验设置下的裁决。

---

## 1. 两类检索各自的机制与失效模式

### 1.1 稀疏/词法检索：BM25 的定义与局限

BM25 是一种 bag-of-words 检索函数，根据查询词是否出现在文档中对文档排序，不考虑词在文档内的邻近关系（proximity）[^10]。其评分公式为对查询词求和：

score(D,Q) = Σ IDF(qi) · [ f(qi,D)·(k1+1) ] / [ f(qi,D) + k1·(1 − b + b·|D|/avgdl) ]

其中 f(qi,D) 为词 qi 在文档 D 中的出现次数，|D| 为文档长度，avgdl 为集合平均文档长度；k1 与 b 为自由参数，未做高级优化时通常取 k1 ∈ [1.2, 2.0]、b = 0.75；IDF(qi) 通常计算为 ln( (N − n(qi) + 0.5) / (n(qi) + 0.5) + 1 )，N 为集合中文档总数，n(qi) 为包含 qi 的文档数[^10]。BM25 基于 1970–1980 年代由 Stephen E. Robertson、Karen Spärck Jones 等人发展的概率检索框架，其 IDF 分量在原始推导中源自二元独立模型（Binary Independence Model）；名称中的 Okapi 来自首个使用该函数的系统，即 1980–1990 年代在伦敦 City University 实现的 Okapi 信息检索系统[^10]。BM25 及其变体（如可考虑文档结构与锚文本的 BM25F）属于用于文档检索的 TF-IDF 类检索函数[^10]。

DPR 论文把这类方法定义为：使用 TF-IDF 或 BM25，通过倒排索引高效匹配关键词，可视为把问题和上下文表示为高维稀疏向量（带权重）[^1]；该论文摘要把 TF-IDF/BM25 称为开放域 QA 中「de facto method」（事实标准）的稀疏向量空间模型[^2]。

其结构性缺陷即 lexical gap：只能检索包含查询关键词的文档，且将查询与文档视为 bag-of-words、不考虑词序[^12]。DPR 给出了一个具体失效例子：问题「Who is the bad guy in lord of the rings?」可由上下文「Sala Baker is best known for portraying the villain Sauron in the Lord of the Rings trilogy.」回答；基于词项的系统难以检索到该上下文，而稠密检索能把「bad guy」与「villain」匹配上[^1]。

### 1.2 稠密检索：定义、优势与代价

DPR 把稠密检索定义为 dense, latent semantic encoding，与稀疏表示在设计上互补；同义或改写（完全不同的 token）仍可被映射到相近的向量[^1]。DPR 使用简单的 dual-encoder（双编码器）框架，从少量问题-段落对学习嵌入，实现纯稠密表示的检索[^2]；稠密编码可通过调整嵌入函数学习，提供任务特定表示的额外灵活性，检索可用最大内积搜索（MIPS）等内存数据结构与索引方案高效完成[^1]。

稠密检索的代价同样被 DPR 明确指出：通常认为学习好的稠密向量表示需要大量标注的问答对；在 ORQA 之前，稠密检索从未在开放域 QA 上被证明优于 TF-IDF/BM25[^1]。DPR 的实验结论是：在广泛的开放域 QA 数据集上，DPR 稠密检索在 top-20 段落检索准确率上比强 Lucene-BM25 系统绝对高出 9%–19%[^1]（该论文发表于 EMNLP 2020，arXiv v3 最后修订于 2020-09-30[^2]）。

### 1.3 零样本泛化：BM25 是稳健基线，稠密并非全面占优

BEIR 由 18 个公开数据集构成，覆盖多样文本检索任务与领域，用于评估检索模型的分布外（OOD）泛化能力，评估了 10 个 SOTA 检索系统，涵盖 lexical、sparse、dense、late-interaction 与 re-ranking 架构[^11][^15]。其结论是：BM25 是一个稳健的基线（robust baseline）；re-ranking 与 late-interaction 模型平均取得最佳 zero-shot 性能，但计算成本高；dense 与 sparse 检索模型计算上更高效，但常常表现不如其他方法，其泛化能力仍有相当大的改进空间[^11][^15]。论文动机是既有神经 IR 模型多在单一、同质的窄设定下研究，限制了对 OOD 泛化能力的认识；多数先前工作在训练集同分布数据上评估，展示了相对 BM25 的显著提升，但 zero-shot 场景下表现不明[^11]。BEIR 引言进一步指出：创建大规模训练语料耗时昂贵，因此许多检索系统在 zero-shot 设定下应用（无可用训练数据）[^12]。该论文发表于 NeurIPS 2021 Dataset and Benchmark Track，v4 版本 2021-10-21[^11][^15]。

这一组结论构成混合检索动机的另一半：既然词法与稠密各有短板、且在不同域上强弱互换，融合两路信号在原理上比押注单一路径更稳。

---

## 2. 混合检索为什么优于单一检索：证据支持的理由

综合上述证据，可归纳出四条有直接依据的理由：

**理由一：互补的匹配能力。** 多个数据库厂商文档明确以互补性作为混合检索的动机。Qdrant 文档写明动机是结合 dense 向量的语义理解与 sparse 向量的精确词匹配，以「get the best of both worlds」[^3]。Milvus 文档明确支持 sparse-dense 向量混合检索，指出 dense 向量擅长捕捉语义关系、sparse 向量擅长精确关键词匹配，二者结合可同时提供概念理解与精确词相关性[^5]。Weaviate 的混合检索将向量检索与关键词（BM25F）检索的结果集进行融合[^4]。Elasticsearch 文档说明 RRF 会给在任一查询中得分高的文档更多权重，最终列表在精确关键词匹配与基于相似度的匹配之间取得平衡[^8]。这一互补性有 DPR 的「bad guy / villain」实例作为机制层面的支撑[^1]，也有 BEIR 关于 lexical gap 的论述作为反面支撑[^12]。

**理由二：单一路径的失效模式可被另一路覆盖。** 词法路径的失效是词汇鸿沟[^12]；稠密路径的失效则体现在 OOD 泛化不足、常常不如其他方法[^11][^15]。两路融合后，任一路的强项可在另一路失效时提供兜底召回。Elasticsearch 文档把混合检索描述为在单个请求中同时运行全文检索与向量检索，返回一个融合了关键词匹配与相似度检索的排序列表[^8]。

**理由三：工程生态已把混合检索作为一等能力。** Qdrant 自 v1.10.0 起提供混合查询，基于「每个 point 多个命名向量（named vectors）」与 Query API 的 prefetch 参数实现多阶段查询[^3]。OpenSearch 自 2.11 版本引入混合检索[^7]。Milvus 通过允许在多个向量字段上同时执行多个 ANN 搜索来实现 multi-vector hybrid search[^5]。Elasticsearch 官方推荐使用 reciprocal rank fusion (RRF) 实现混合检索，用于合并全文查询与向量查询的排名[^8]，并推荐通过 semantic_text workflow 使用混合检索[^8]。Weaviate 的混合检索使用单一查询字符串（single query string）同时驱动向量与关键词两路检索[^4]。GitHub 仓库元数据也显示多个开源数据库以混合检索为卖点：infiniflow/infinity（★4741）自述为面向 LLM 应用的 AI-native 数据库，提供 dense vector、sparse vector、tensor（multi-vector）与 full-text 的快速混合检索；manticoresoftware/manticoresearch（★12056）自述为支持全文、向量与混合检索的开源搜索数据库；apache/doris（★16086）自述为面向 AI agent 的实时分析与混合检索数据库；tensorchord/pgvecto.rs（★2190）自述为 Postgres 中可扩展、低延迟且支持混合的向量搜索；langchain-ai/langchain-milvus（★59）自述为 Milvus 的 LangChain 包装器，支持 vector search、full-text search、hybrid retrieval 与 RAG[^9]。

**理由四：稀疏侧的表示本身在进化，进一步缩小与稠密的差距。** SPLADE 是一种基于显式稀疏正则化与词项权重 log-saturation 效应的 first-stage ranker，产生高度稀疏的表示，继承 bag-of-words 模型的优点：词项精确匹配与倒排索引的效率，且简单、单阶段端到端训练[^18]。SPLADE v2 修改 pooling 机制、benchmark 仅基于 document expansion 的模型、引入用蒸馏训练的模型，报告了 BEIR benchmark 上的结果，相比原 SPLADE 在 TREC DL 2019 上 NDCG@10 提升超过 9%，在 BEIR benchmark 上达到 state-of-the-art 结果[^19]。Milvus 文档点名可用于全文/稀疏检索的算法与模型包括 BM25，以及 BGE-M3、SPLADE 等稀疏嵌入模型[^5]。Qdrant 调优指南建议先用 BM25（查询时无需模型），再实测 learned sparse 模型（如 miniCOIL、SPLADE）是否值得采用[^25]。

**成本侧的现实约束。** 混合检索相对单一检索会增加存储、索引与查询开销（storage, indexing, and query work），Qdrant 官方建议实测收益是否值得该成本[^25]。BM25 稀疏检索在查询时不需要模型推理（no model inference）[^25]。现有证据未给出混合检索相对纯向量检索的延迟增量：Qdrant 公开的单节点基准指标包括 RPS、Latency、p95 latency、Index time（数据更新于 2024 年 1 月/6 月），但该基准对比的是各向量引擎的纯向量检索性能，未单独给出混合检索相对纯向量检索的延迟增量[^27]。

---

## 3. 主流融合方法

### 3.1 基于排名的融合：RRF

**定义（完整写出）。** Reciprocal Rank Fusion（RRF）的评分公式为 score = Σ 1/(k + rank(result(q), d))，其中 k 是 ranking constant，rank 从 1 开始计数[^16]。即：输入是若干路检索各自返回的、按相关性排序的文档列表；步骤是对每个文档在所有路中取其排名，按 1/(k+rank) 求和；产出是一个融合后的分数，据此重新排序[^16]。OpenSearch 概念页给出同一公式 score(d) = Σ 1/(k + rank_q(d))，rank 从 1 开始，k 由 rank_constant 参数配置，该 RRF 能力自 OpenSearch 2.19 引入[^22]。

**参数默认值与语义。** Elasticsearch 的 RRF retriever 中 rank_constant 参数默认值为 60，且必须 ≥ 1；rank_constant 值越大，表示排名靠后的文档影响力越大[^16]。OpenSearch 的 score-ranker-processor 使用 RRF，combination.rank_constant 默认值为 60，有效范围为 [1, 10000]；较大的 rank constant 使分数更均匀、降低头部结果影响，较小的值加大排名间分差、更重视头部结果[^21]。Milvus 的 RRFRanker 公式中 k 为平滑参数，文档称「typically set at 60」，示例中明确「Assume that the smoothing parameter k is set at 60」，并用 1/(60+rank) 计算最终分数；RRFRanker 用于混合搜索，基于排名位置而非原始相似度分数来平衡多路检索结果[^23]。Weaviate 的 rankedFusion（即 RRF 式排名融合）分数按 1/(RANK + 60) 计算，隐含 k=60；Weaviate 文档未将 k 暴露为可配置参数，仅在公式中硬编码 60[^24]。

**Qdrant 的 RRF 变体。** Qdrant 官方给出的融合方式有两种：rrf（Reciprocal Rank Fusion）与 dbsf（Distribution-Based Score Fusion）[^3]。Qdrant 的 RRF 公式中 k 默认取 2，权重 w_r 默认取 1；采用零基排名（top 结果 r_d = 0），因此默认权重下公式可简化为 score = Σ 1/(k + r_d)[^3]。注意这与 Elasticsearch/OpenSearch/Milvus/Weaviate 的 k=60、一基排名不同。

**RRF 被主张的优点。** Elasticsearch 文档将 RRF 描述为无需调参（requires no tuning），且不同相关性指标之间无需相互关联即可获得高质量结果；RRF retriever 至少需要两个 child retriever，每个 child retriever 在 RRF 公式中权重相等[^16]。Qdrant 调优指南解释 RRF 的机制：稠密相似度分数有界，而 BM25 分数幅度取决于命中词数与词稀有度，因此固定权重无法在所有查询上保持平衡；RRF 通过丢弃分数幅度（只读排名位置）来规避尺度不匹配[^25]。

**RRF 的配置形态。** OpenSearch 提供两种 search phase results processor：normalization processor（基于分数，将各查询子句分数转换到统一尺度后合并，保留文档间分差）与 score ranker processor（基于排名，使用 RRF 按各查询子句结果中的位置合并文档，忽略分数本身）[^7]。OpenSearch 建议：当文档间分数差本身包含需在最终排序中体现的信息、或需要细粒度控制（归一化技术、组合技术、分数边界）时选 normalization processor；当希望无需先测量各子句打分即可工作的配置时选 score ranker processor（RRF）[^7]。OpenSearch 还提供自动化 workflow 模板（use_case=hybrid_search）一键创建 ingest pipeline、index 与 search pipeline，默认模型维度为 1024（可通过 output_dimension 参数覆盖）[^7]。

### 3.2 基于分数的融合：凸组合 / 加权归一化

**定义（完整写出）。** 凸组合（convex combination, CC）指把 lexical 与 semantic 两路检索的分数做加权线性组合，权重之和约束为凸组合形式[^14]。Milvus 的 WeightedRanker 是这一类的工程实现：先对每路检索分数归一化到 [0,1]（Milvus 使用 arctan 函数处理不同度量类型，如 IP 范围 [-∞,+∞]、L2 范围 [0,+∞]），再按用户指定的权重（每路权重范围 [0,1]）做加权平均；权重个数必须等于 AnnSearchRequest 实例个数，示例中 WeightedRanker(0.8, 0.8, 0.7) 对应三路检索[^6]。Milvus 通过 hybrid_search() API 提供混合检索，并对多个 AnnSearchRequest 的结果应用重排策略，当前提供两种重排策略：WeightedRanker（加权评分）与 RRFRanker（基于排名的倒数排名融合）[^6]。

**Weaviate 的两种策略。** Weaviate 支持两种融合策略：relativeScoreFusion 与 rankedFusion[^24]。rankedFusion 在 1.24 之前是默认算法；relativeScoreFusion 于 1.20 引入、1.24 起成为默认，官方认为对多数场景更优[^26]。Weaviate 的混合检索在底层执行两次检索：一次向量检索、一次关键词（BM25F）检索，然后融合结果；混合查询通过 alpha 参数（示例 alpha=0.5）控制向量与关键词的相对权重[^26]。对含命名向量（named vectors）的集合做混合检索时必须指定目标向量（target_vector），Weaviate 用查询向量在该目标向量空间中检索[^4]。Weaviate 还提供 Query Agent 将自然语言问题自动翻译为优化后的 Weaviate 查询（Cloud only）[^4]。

**Qdrant 的 DBSF。** DBSF（Distribution-Based Score Fusion）按查询对每个分数分布做归一化，与 RRF 丢弃分数幅度的思路相对[^25]。

### 3.3 多阶段 / 多向量检索管线

Qdrant 的混合查询基于「每个 point 多个命名向量」与 Query API 的 prefetch 参数实现多阶段查询；官方示例展示了对 sparse 与 dense 两个命名向量各取 limit=20 的 prefetch，再用 {"query": {"rrf": {}}} 融合并返回 limit=10 的结果；prefetch 可嵌套（prefetches can have prefetches themselves）；offset 参数只作用于主查询，prefetch 的 limit 需至少为 limit + offset[^3]。

OpenSearch 通过 search pipeline（搜索管道）在搜索时拦截中间结果并归一化、合并文档分数[^7]。Milvus 的混合检索流程包含 reranking（重排）策略以合并多路结果[^5]，并支持多模态混合检索（文本、图像、音频），示例中用 CLIP 生成图像嵌入、用 BERT/Transformers 或 OpenAI 生成文本嵌入[^5]。

### 3.4 与 late-interaction / 重排的关系（相邻但不同的技术路线）

需要区分：late-interaction 与 re-ranking 不是混合检索的融合方法，而是与混合检索并列的、提升检索质量的另一条路线。ColBERT 提出 late interaction 架构：用 BERT 分别独立编码 query 和 document，再用轻量交互步骤建模细粒度相似度；通过延迟但保留细粒度交互，ColBERT 可离线预计算文档表示，从而显著加速查询处理；其剪枝友好交互机制支持借助向量相似度索引直接从大规模文档集合做端到端检索，而不仅是重排传统模型召回的文档；在 passage search 数据集上，ColBERT 效果与已有 BERT-based 模型相当（并优于所有非 BERT 基线），同时查询执行快两个数量级、每查询所需 FLOPs 少四个数量级[^17]。BEIR 的结论中，re-ranking 与 late-interaction 模型平均取得最佳 zero-shot 性能，但计算成本高[^11][^15]。monoT5 将预训练 sequence-to-sequence 模型适配到文档排序任务，与基于 encoder-only 架构（如 BERT）的分类式排序范式根本不同：训练 seq2seq 模型生成相关性标签作为 target words，并将这些 target words 的底层 logits 解释为用于排序的相关性概率[^20]；在 MS MARCO passage ranking 上至少与之前的分类式模型相当，且用更大、更新的模型可以超越它们；在 TREC 2004 Robust Track 测试集上，零样本迁移方法优于此前需要 in-dataset 交叉验证的 SOTA 模型；在数据稀缺（few training examples）场景下显著优于 encoder-only 模型[^20]。Sentence Transformers（a.k.a. SBERT）是用于使用与训练 embedding 与 reranker 模型的 Python 模块，由 UKP Lab 创建、现由 Hugging Face 维护；通过 Sentence Transformer 模型从文本计算 embeddings，并用余弦相似度计算相似度分数[^13]。其当前 v6.0 文档介绍 MultiVectorEncoder（ColBERT 风格 late-interaction）等新特性，非 1908.10084 原文内容[^13]。

---

## 4. 矛盾与不确定性

### 4.1 量化提升幅度：证据缺口（blocking）

本报告无法回答「混合检索相对纯稀疏/纯稠密检索在召回率与 nDCG 上的提升幅度」这一核心量化问题。现有证据中：

- BEIR 论文只给出 BM25 与 dense 的零样本基线对比，未做混合融合实验；其摘要未直接给出 hybrid（BM25+dense 融合）相对纯 sparse/纯 dense 的 nDCG@10 提升百分点[^15]。
- SPLADE v2 报告的是自身相对原 SPLADE 的提升（TREC DL 2019 NDCG@10 +9%），不是混合 vs 单一[^19]。
- OpenSearch 的 3.86% 是 RRF 与 score-based 两种混合管线之间的差异，不是混合 vs 单一检索[^22]。
- DPR 的 9%–19% 是稠密 vs BM25 的对比，不是混合 vs 单一[^1]。

因此，**在 BEIR、MS MARCO、TREC DL、MTEB 等基准上，混合检索相对单一检索的量化提升幅度，现有公开资料不足以确认。**

### 4.2 混合检索失效或反而更差的场景：证据缺口

计划中要求覆盖「混合反而无增益或更差」的场景（如短查询、强词汇匹配域、dense 已饱和的域），但全部证据中没有任何一条给出此类反例或边界条件。现有证据只提供了成本侧的提示（混合检索增加存储、索引与查询开销，需实测收益是否值得该成本[^25]），以及 Qdrant 基准强调 ANN 检索是精度与速度的权衡、只有在精度相近时比较结果才有意义[^27]。**混合检索在哪些场景下无增益或反而更差，现有公开资料不足以确认。**

### 4.3 RRF 与凸组合的相对优劣：三方结论方向不一致（blocking）

存在未解决的直接矛盾：

- arXiv 2210.11934《An Analysis of Fusion Functions for Hybrid Retrieval》声称：与既有研究相反，RRF 对其参数敏感（sensitive to its parameters）；CC 融合的学习通常对分数归一化的选择不敏感（agnostic）；CC 在 in-domain 与 out-of-domain 设置下均优于 RRF；CC 具有样本效率，仅需少量训练样本即可将其唯一参数调至目标域[^14]。该论文为 v2 版本，2023-05-04 修订，关联 DOI 10.1145/3596512；摘要层面未给出 BEIR / MS MARCO / TREC DL 上的 nDCG@10 或 Recall 具体百分点数值[^14]。
- Elasticsearch 官方文档称 RRF「requires no tuning」，且不同相关性指标之间无需相互关联即可获得高质量结果[^16]。
- OpenSearch 基准给出相反方向的量化：在六个 BEIR 数据集基准上，RRF 的 NDCG@10 平均比 score-based 管线低 3.86%[^22]。

三方结论方向不一致，且现有证据中无第三方独立复现或统一实验设置下的对照来裁决。**RRF 与 CC/分数归一化融合的相对优劣，现有公开资料不足以确认。**

### 4.4 其他需注意的口径差异

- **RRF 的 k 默认值与排名基准不统一**：Elasticsearch/OpenSearch/Milvus/Weaviate 为 k=60、rank 从 1 开始[^16][^21][^23][^24]；Qdrant 为 k 默认 2、零基排名[^3]。跨系统比较 RRF 分数时不可直接对照。
- **Weaviate 默认融合策略在版本间发生过变更**：rankedFusion 是 v1.23 及更早版本的默认，relativeScoreFusion 自 v1.24 起为默认[^24][^26]。
- **Qdrant 基准不含混合检索延迟数据**：其单节点基准只覆盖纯向量检索性能[^27]。

---

## 参考来源

[^1]: Dense Passage Retrieval for Open-Domain Question Answering (DPR) — https://ar5iv.labs.arxiv.org/html/2004.04906 （arxiv, 2020-04-10）
[^2]: Dense Passage Retrieval for Open-Domain Question Answering (DPR) — arXiv metadata page — https://arxiv.org/abs/2004.04906v3 （arxiv, 2020-04-10）
[^3]: Qdrant Documentation — Hybrid Queries (Hybrid and Multi-Stage Queries) — https://qdrant.tech/documentation/concepts/hybrid-queries/ （web）
[^4]: Weaviate Documentation — Hybrid search — https://weaviate.io/developers/weaviate/search/hybrid （web）
[^5]: Milvus Documentation — Multi-Vector Hybrid Search — https://milvus.io/docs/multi-vector-search.md （web）
[^6]: Milvus Documentation — Reranking — https://milvus.io/docs/reranking.md （web）
[^7]: OpenSearch Documentation — Hybrid search — https://opensearch.org/docs/latest/vector-search/ai-search/hybrid-search/index/ （web）
[^8]: Elasticsearch Documentation — Hybrid search — https://www.elastic.co/docs/solutions/search/hybrid-search （web）
[^9]: GitHub search results — hybrid search vector database (repo metadata) — https://github.com/search?q=hybrid+search+vector+database&type=repositories （github）
[^10]: Okapi BM25 — Wikipedia — https://en.wikipedia.org/wiki/Okapi_BM25 （web）
[^11]: BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models (arXiv:2104.08663v4) — https://arxiv.org/abs/2104.08663v4 （arxiv, 2021-04-17）
[^12]: BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models (arXiv:2104.08663v4) — Introduction — https://arxiv.org/html/2104.08663v4 （arxiv, 2021-04-17）
[^13]: SentenceTransformers Documentation (sbert.net) — https://www.sbert.net/ （web）
[^14]: An Analysis of Fusion Functions for Hybrid Retrieval — https://arxiv.org/abs/2210.11934 （arxiv, 2022-10-21）
[^15]: BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models — https://arxiv.org/abs/2104.08663 （arxiv, 2021-04-17）
[^16]: Elasticsearch Docs — Reciprocal rank fusion — https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion （web）
[^17]: ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT — https://arxiv.org/abs/2004.12832v2 （arxiv, 2020-04-27）
[^18]: SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking — https://arxiv.org/abs/2107.05720 （arxiv, 2021-07-12）
[^19]: SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval — https://arxiv.org/abs/2109.10086 （arxiv, 2021-09-21）
[^20]: Document Ranking with a Pretrained Sequence-to-Sequence Model (monoT5) — https://arxiv.org/abs/2003.06713 （arxiv, 2020-03-14）
[^21]: OpenSearch — Score ranker processor (RRF) reference — https://docs.opensearch.org/latest/search-plugins/search-pipelines/score-ranker-processor/ （web）
[^22]: OpenSearch — Reciprocal rank fusion (concept page) — https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/rrf/ （web）
[^23]: Milvus — RRF Ranker documentation — https://milvus.io/docs/rrf-ranker.md （web）
[^24]: Weaviate — Hybrid search concepts (fusion strategies) — https://weaviate.io/developers/weaviate/concepts/search/hybrid-search （web）
[^25]: Hybrid Search in Qdrant — Tuning Guide (cost, fusion, learned sparse) — https://qdrant.tech/articles/hybrid-search/ （web）
[^26]: Hybrid Search Fusion Algorithms — Weaviate Blog (rankedFusion vs relativeScoreFusion) — https://weaviate.io/blog/hybrid-search-fusion-algorithms （web）
[^27]: Vector Search Benchmarks — Qdrant (single-node RPS/latency/p95/index time) — https://qdrant.tech/benchmarks/ （web, 2024-06）