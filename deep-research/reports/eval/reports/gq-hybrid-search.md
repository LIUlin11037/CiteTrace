# 混合检索（向量 + 关键词）为什么优于单一检索？主流融合方法有哪些？

## 摘要

混合检索的立论基础不是"两种方法各有所长"这种笼统说法，而是两条可验证的失效模式互补：词法检索（TF-IDF/BM25）只能召回包含查询关键词的文档，且把查询与文档当词袋处理、忽略词序，这一失效被形式化为 lexical gap[^3]；稠密检索虽能捕捉语义匹配、试图克服潜在词汇不匹配[^3]，但无法显式建模词项匹配（inability to explicitly model term matching），因此仍需与词袋模型结合[^2]。BEIR 在 18 个数据集、9 类任务、10 个系统上的核心结论是：没有任何单一方法在所有数据集上持续优于其他方法[^3]。这构成了混合检索的"必要性"论证，但**不是"混合一定更好"的充分论证**——2026 年的 Fusion Tax 研究给出了反面的量化证据：即使最好的融合方法（稠密加权分数插值），也在 13–48% 的查询上把相关文档排在其两个组件中较好者的后面，在 8–21% 的查询上排在其可部署的纯稠密检索器后面[^4]。融合方法层面，主流可归为三类：分数归一化后加权求和（convex combination，单参数 α）[^7][^10][^15]、基于排名的 RRF（k 常数，业界默认 60 或 2）[^9][^11][^18]、以及分布/相对分数归一化类变体（dbsf、relativeScoreFusion）[^9][^10]。学术侧对 CC 与 RRF 的优劣有明确但存在争议的结论：CC 在域内与域外均优于 RRF 且样本高效，RRF 对参数敏感并丢弃分数分布信息[^7][^8]；而厂商文档侧仍普遍把 RRF 作为"无需调参"的默认推荐[^11]。**本报告最重要的不确定性是：现有证据中没有任何一条提供同实验设置下"混合 vs 纯 BM25 vs 纯稠密"的三方 nDCG@10 / Recall 对比**，因此混合检索的增益幅度在本报告证据范围内无法量化。

---

## 一、为什么需要混合：两类检索器的互补失效模式

### 1.1 词法检索的失效模式：lexical gap 与词袋假设

词法方法（TF-IDF/BM25）的失效模式被形式化为 lexical gap：只能检索包含查询中关键词的文档；且把查询与文档当作词袋处理，不考虑词序[^3]。词袋模型依赖倒排索引，仍是强基线，但存在长期未解的词汇不匹配（vocabulary mismatch）问题：相关文档可能不包含查询中出现的词[^2]。DPR 论文给出了一个具体例证：对于「Sala Baker is best known for portraying the villain Sauron...」这类上下文，词项系统难以检索到，而稠密检索能更好地把「bad guy」与「villain」匹配起来[^1]。

### 1.2 稠密检索的失效模式：无法显式建模词项匹配

稠密检索的机制是：把查询与文档映射到共享的稠密向量空间，使文档表示可预计算并建索引；能捕捉语义匹配、试图克服（潜在的）词汇不匹配[^3]。DPR 把检索实现为双编码器框架：两个独立 BERT-base（uncased）分别编码问题与段落，取 [CLS] 表示，维度 d=768；相似度函数必须可分解以便段落表示离线预计算，DPR 选择内积并用 FAISS 对全部段落向量离线建索引，在线做最大内积搜索取 top-k[^1]。但 SPLADE 明确指出：基于近似最近邻（ANN）的稠密检索效果显著，但仍需与词袋模型结合，原因是它无法显式建模词项匹配[^2]。

### 1.3 互补性的边界条件：稠密检索的优势依赖标注数据与域内评测

DPR 论文自己承认边界条件：一般认为学习好的稠密向量表示需要大量标注的问题-上下文对；在 ORQA 之前，稠密检索从未在开放域 QA 上被证明优于 TF-IDF/BM25[^1]。Vespa 官方博客进一步指出：在域内训练并评测时，稠密向量、late interaction、cross-encoder 三类神经排序方法都优于 BM25 词法基线，但该优势不一定能迁移到零样本场景——把在 MS MARCO 等域内数据上训练的排序模型直接应用到新领域而不做适配或微调时，域内嵌入排序的有效性并不延续到不同领域[^24]。BEIR 的结论与此一致：稠密与稀疏检索模型计算更高效，但常常不如其他方法，说明其泛化能力仍有相当大的改进空间[^3]。

### 1.4 为什么"混合"是自然答案——以及它的代价

BEIR 把混合式（hybrid lexical-dense）方法描述为旨在结合两类方法优势的后续方向[^3]。但"结合优势"不等于"没有代价"。Fusion Tax 研究把「不融合」(do not fuse) 当作一等结果，考察融合在单条查询层面的代价，而非只看聚合平均增益[^4]。其结论是：融合在平均意义上可以有益，同时损害个别查询——论文明确承认「That fusion can help on average while hurting individual queries is not a new observation」，并称 risk-sensitive evaluation of rank fusion、fusion-oriented query performance prediction、per-query routing between retrievers 三条既有研究线均以此为出发点[^14]。

---

## 二、融合方法的形式化定义与主流实现

### 2.1 问题形式化

Hybrid search 的形式化定义为：给定词法检索与语义检索分别产出的两个有序列表 R_Lex 与 R_Sem，融合任务是构造最终列表 R_Fusion 以提升检索质量[^7]。

### 2.2 方法一：加权分数线性融合（Convex Combination, CC）

**定义（完整写出）**：加权分数线性融合的算法为 f_Convex = α·f_Sem + (1−α)·f_Lex，其中 0 ≤ α ≤ 1，f_Lex 为词法检索分数（如 BM25），f_Sem 为语义/向量检索分数[^7]。其输入是两个检索器对同一文档给出的原始分数与一个权重 α；步骤是先对两路分数做归一化（因 BM25 等词法分数与点积等语义分数可能无界，通常用 min-max scaling），再按 α 加权求和；产出是单一融合分数，据此排序[^7]。

**超参数**：CC 只有一个可调参数 α（词法/语义的相对权重），论文称其为 "its only parameter"，并强调该参数高度可解释，无训练样本时可依据领域知识手工设定[^7]。

**归一化的作用**：归一化的作用是构造有界函数、使融合在不同领域间保持一致（consistency）[^7]。论文的核心论点之一是：归一化的具体选择只是很小的细节——对 min-max、standard score（标准化）或任意其他线性变换归一化后的分数，总存在彼此 rank-equivalent 的 CC 组合；以 per-query 学习问题表述时，用一种归一化方案在某数据集上求得的解，与用另一种归一化方案求得的解等价；摘要明确结论为 "the learning of a CC fusion is generally agnostic to the choice of score normalization"[^7][^8]。

**工程实现中的 α**：Weaviate 的 alpha 参数控制向量检索在最终结果中的权重，取值 0–1：alpha=0 为纯关键词检索，alpha=0.5 为等权，alpha=1 为纯向量检索；服务器默认值为 0.75（偏向向量检索），但该默认仅在请求未携带 alpha 时生效（GraphQL 场景，以及 v1.36.7 起 gRPC 允许客户端不设 alpha）[^10]。官方明确警告：各客户端库并非都会不设 alpha，因此实际生效权重可能偏离 0.75，某些情况下甚至退化为纯关键词检索；建议在权重重要时显式设置 alpha[^10]。OpenSearch 的 normalization processor 示例默认配置为 normalization.technique = min_max、combination.technique = arithmetic_mean，weights 数组按小数百分比为各 query clause 赋权[^15]。M3-Embedding 的混合检索采用加权分数融合 s_rank ← w1·s_dense + w2·s_lex + w3·s_mul，权重依下游场景确定；其流程为先由各方法分别召回候选，再基于整合后的相关性分数重排，multi-vector 因开销大被豁免于候选召回阶段[^13]。

### 2.3 方法二：Reciprocal Rank Fusion（RRF）

**定义（完整写出）**：RRF 的打分公式为 score(d) = Σ_{q∈Q} 1/(k + rank_q(d))，其中 rank_q(d) 从 1 开始计数，未出现的 query clause 不贡献分数；k 为 rank constant，通过 rank_constant 参数配置[^16]。其输入是多路检索各自返回的有序列表；步骤是取每个文档在各列表中的排名（而非原始分数），按 1/(k+rank) 求和；产出是融合分数[^16]。Elasticsearch 的实现要求至少两个子检索器（child retrievers），且每个子检索器在 RRF 公式中权重相等；rank_constant 默认值为 60，取值必须 ≥ 1，值越大表示排名靠后的文档影响越大[^11]。Milvus 的 RRFRanker 基于各检索路径中的排名位置而非原始相似度分数进行融合，k 默认值为 60，可调范围 (0, 16384)，须为浮点数，官方推荐取值区间 [10, 100]；官方给出 topK=5、k=60 的完整算例：文档 101 在 sparse 排第 1、dense 排第 2，最终分数为 1/(60+1)+1/(60+2)=0.03252247[^18]。

**k 值的实现分歧**：Qdrant 的 RRF 公式中 k 常量默认取 2，各 ranking 的权重 w_r 默认取 1；当权重均为 1 时公式退化为原始 RRF：score = Σ 1/(k + r_d)；Qdrant 使用 zero-based 排名，即首位结果 r_d = 0[^9]。这与 Elasticsearch/Milvus 的 k=60 形成明显差异，是跨系统迁移时必须注意的实现细节。

**RRF 的宣称优势**：Elasticsearch 官方说明 RRF 无需调参（requires no tuning），且不同相关性指标之间无需相关即可获得高质量结果；官方称 RRF 相比线性组合免去了确定合适权重的麻烦，并称 RRF 已被证明比任一单独查询带来更好的相关性[^11]。OpenSearch 官方建议：当各 query clause 分数不可比、数据有离群值/高方差、相关性信号稀疏、数据持续变化，或希望奖励在多个结果列表中排名都靠前的文档时，优先用 RRF[^16]。

**RRF 的学术批评**：Bruch 等人的论文反驳了 RRF 是"非参数、可零样本使用"的既有说法：作者采用参数化视角看 RRF，认为待融合的每个检索函数各有一个参数，参数数量总比 CC 多一个；并发现 RRF 对其参数敏感；论文还指出 RRF 因是 rank 的函数，会忽略分数的分布信息（"it disregards the distribution of scores and, as such, discards useful information"），原始分数之间的距离对混合分数没有任何作用[^7]。论文报告 CC 在 in-domain 与 out-of-domain 设置下均优于 RRF，且 CC 样本高效，只需少量训练样本即可把唯一参数调到目标领域；相反，调过参的 RRF 在 out-of-domain 数据集上泛化很差[^7][^8]。该论文引述 Chen et al. (2022) 的观点作为被检验对象：该文认为 convex combination 对参数 α 与分数归一化选择敏感，因而主张 RRF 更合适[^7]。

### 2.4 方法三：分布/相对分数归一化类变体

Qdrant 原生支持的融合方式有两种：rrf 与 dbsf（Distribution-Based Score Fusion）[^9]。Weaviate 支持两种融合策略：relativeScoreFusion 与 rankedFusion；relativeScoreFusion 自 v1.24 起为默认，rankedFusion 是 v1.23 及更早版本的默认[^10]。rankedFusion 的分数按 1/(RANK + 60) 计算，仅保留排名信息，与原始分数大小无关；relativeScoreFusion 将最高分归一化为 1、最低分归一化为 0，其余按相对距离线性缩放，再对归一化向量距离与归一化 BM25 分数做加权求和[^10]。官方认为 relativeScoreFusion 通常是更好的选择，理由是它比只保留排名的 rankedFusion 保留了更多原始检索信息[^10]。

### 2.5 融合之外的另一条路线：多索引 + hybrid query

Blended RAG 的混合检索由 BM25 索引、dense vector (KNN) 索引与 Sparse Encoder 索引分别搭配 hybrid query（Multi-Fields / Best Fields）构成，**并非** BM25 与 dense 的 RRF 或加权分数融合[^12]。这提示"混合检索"在文献中至少有两种不同含义：分数/排名融合，与多字段查询下的布尔式组合，二者不可混为一谈。

### 2.6 各系统实现对照

| 系统 | 融合方式 | 关键参数与默认值 |
|---|---|---|
| Qdrant | rrf、dbsf | RRF k 默认 2，权重默认 1，zero-based 排名[^9] |
| Weaviate | relativeScoreFusion（v1.24+ 默认）、rankedFusion（≤v1.23 默认） | alpha 0–1，服务器默认 0.75[^10] |
| Elasticsearch | rrf retriever | rank_constant 默认 60，≥1；rank_window_size 默认等于 size[^11] |
| OpenSearch | normalization processor（min_max + arithmetic_mean）、score ranker processor（RRF） | normalization 自 2.10 可用，score ranker 自 2.19 可用[^15][^16] |
| Milvus | RRFRanker 等 reranking 策略 | k 默认 60，范围 (0, 16384)，推荐 [10, 100][^18] |

工程落地热度方面，OpenSearch 仓库 stars 13904、forks 2982，最近 push 2026-10-24[^19]；pgvector stars 23472、forks 1365，最近 push 2026-10-24[^20]；LangChain stars 148407、forks 24907，最近 push 2026-10-25[^21]。这些元数据只反映生态活跃度，不构成对混合检索效果的证据。

---

## 三、混合检索的收益：证据强度分层

### 3.1 强证据：单点场景下的显著提升

StrucChunk 用 BM25 与 BGE-M3 经 RRF 组合成混合稠密-稀疏检索，用于层级化法律文档，在 LegalBench-RAG（6,858 个 query-passage 对）上达到 Char-Recall@8 = 48.18%，最佳基线为 23.59%（+104%）；在印度法律 CrPC 的 501 条查询上 Recall@5 从 0.557 提升到 0.804（+44.4%）；CPC 上从 0.158 提升到 0.259（+64.6%）[^5]。**但作者明确归因：分块（chunking）贡献了上述增益的 88%（CrPC）与 73%（CPC），即增益主要来自结构感知分块而非混合检索本身**[^5]。这一自述大幅削弱了该文作为"混合检索优于单一检索"证据的效力。

### 3.2 中等证据：聚合指标上的稳定优势，但伴随单查询损失

Fusion Tax 研究在七个 BEIR 数据集、三种融合函数上评测，实验设置为标准 Anserini BM25 + 五个稠密编码器（22M–596M 参数，跨三个模型世代）+ 一个学习型稀疏替代方案 SPLADE + 一个 cross-encoder 二阶段重排[^4]。结论是双面的：即使最好的方法（dense-weighted score interpolation），在 13–48% 的查询上把相关文档排在其两个组件中较好者的后面，在 8–21% 的查询上排在其可部署的纯稠密检索器后面；在 leave-one-corpus-out 验证下选定权重时，上述两个比例分别为 14–48% 与 11–25%[^4]。尽管存在上述单查询损失，在使用主编码器时，该方法在每一个语料上聚合指标都优于纯稠密检索；在七个语料中的六个上，按损失厌恶的风险敏感评估口径，这一下行损失是统计显著的[^4]。作者将这一损失命名为 fusion tax（融合税）；它主要是排序效应，聚合层面召回率得以保持[^4]。融合税最可能出现在「只有一个检索器能覆盖」的查询上，并且在编码器规模阶梯与 SPLADE 替换下都持续存在[^4]。RRF 在稠密检索器明显更强时承担更高的融合税，只有在两个组件势均力敌时才优于纯稠密检索[^4]。

### 3.3 弱证据或不可用证据

- **Blended RAG**：报告 NQ 上 NDCG@10 为 0.67，相对其引用的 monoT5-3B benchmark 0.633 提升 5.8%；TREC-COVID 上 NDCG@10 为 0.87，相对 COCO-DR Large benchmark 0.804 提升 8.2%[^12]。但其对照基线是文献中既有的 benchmark 分数，而非同一实验设置下重跑的纯 BM25 与纯稠密检索；论文未在 BEIR 或 MS MARCO 上报告结果，也未报告 Recall@100[^12]。
- **M3-Embedding**：抓取到的正文片段未包含结果表格，因此未能提取 BEIR / MS MARCO 上的 nDCG@10 或 Recall@100 绝对数值[^13]。
- **NevIR 重排研究**：在固定候选池（来自稠密、学习型稀疏与 BM25 三种检索）上评估，共 2,736 个 Test 查询方向；一个 pairwise 训练的排序器把严格逐查询偏好率从固定融合的 52.30% 提升到 56.62%，但把偏好段落的 MRR 从 0.5764 降到 0.3055——说明「偏好正确」与「排得靠前」是两件事[^6]。该文评估的是 reranker 而非混合 vs 单一，不能直接用作混合检索增益的证据。

### 3.4 厂商文档侧的反向数据点

OpenSearch 官方对比给出：在六个 BEIR 数据集基准上，RRF 的 NDCG@10 平均比基于分数的混合流水线低 3.86%，搜索延迟与协调节点 CPU 利用率相当[^16]。这是本报告证据中少见的、同设置下的融合方法横向对比，且结论与学术侧"CC 优于 RRF"一致[^7][^8]。

---

## 四、融合方法的选型与调参实践

### 4.1 α 的领域敏感性

Weaviate 社区 issue #11627 指出：Weaviate 的混合检索要求用户手动设置 alpha 参数来平衡关键词与语义检索，官方无自动调优机制；提交者称通用语料上 alpha≈0.5 尚可，但在金融文档（10-K、财报电话会、资产负债表）上最优 alpha 随查询类型剧烈变化：精确匹配查询 alpha≈0.2、语义查询 alpha≈0.8、表格单元格查询 alpha≈0.1[^22]。提交者称在自建的 finrag-eval 实验中，盲目使用 alpha=0.5 在金融 QA 上 precision@5 约 58%，而按查询扫描出的最优 alpha 约 78%，差距约 20 个百分点[^22]。**该 issue 由社区用户提交，标签为 community，截至抓取时状态为 Open，非厂商官方工程报告**[^22]，其数值应视为待复现的线索而非结论。

评论者 cschanhniem 指出 alpha 接近 0.0（BM25 主导）时结果集词汇精确但稀疏——高 precision、低 recall；alpha 接近 1.0（向量主导）时 recall 提升但 precision 常下降，因为语义相似但不相关的片段会浮现[^22]。评论者提出用 'regret'（静态 alpha 相对 oracle 逐查询 alpha 损失的精度）来判断查询类型感知选择是否值得实现，并指出若 oracle 仅比静态 alpha=0.5 高 3-4% nDCG@5，则工程复杂度可能不划算[^22]。提交者承认查询分类器方案存在误差累积风险：把表格单元格查询误判为语义查询并使用 alpha=0.8 而非 0.1，会在 LLM 看到上下文之前就造成检索失败[^22]。

### 4.2 融合决策应绑定实测的稀疏-稠密平衡

Fusion Tax 给出的实践建议是：把是否融合的决策绑定到目标语料实测的稀疏-稠密平衡上[^4]。其依据是：融合失败在单一语料内只有弱可预测性，跨语料迁移不可靠，因为预测它的特征会随稀疏-稠密平衡而改变符号[^4]。cross-encoder 只能在其自身可靠的范围内挽回部分融合税：在三个语料上，可部署税的普遍程度在一个语料上下降三分之一、另两个完全没有下降，71-90% 原本受损的查询被单独修复但同时有新的受损查询补位[^14]。rerank 后 interpolated 与 dense 候选集的差距至多 0.011 nDCG@10（配对区间 ±0.02 内），但 sparse 与 dense 均衡的情形除外[^14]。

### 4.3 阈值与参数的可比性问题

Weaviate 官方指出：混合检索支持 max vector distance 阈值，但该阈值只作用于向量检索分量，不过滤关键词分数；BM25 分量与最终融合分数没有等价阈值，因为 BM25 分数未归一化、无界[^10]。混合检索支持 BM25 的全部参数，包括分词方法、停用词、BM25 的 k1 与 b、搜索操作符（and / or / and_cross）、指定检索属性与属性加权[^10]。Qdrant 方面，使用 offset 参数只影响主查询，因此 prefetch 的 limit 必须至少为「主查询 limit + offset」，否则可能返回空结果[^9]。Elasticsearch 方面，若 knn 检索的 k 大于 rank_window_size，结果会被截断到 rank_window_size；若 k 小于 rank_window_size，则结果数为 k[^11]。

### 4.4 工程成本

混合检索的延迟是持续处理的工程问题：Weaviate 官方 issue 库中检索 'hybrid search latency slow' 返回 9 条结果，其中多条为性能优化类 PR/issue；issue #12351 标题为 'Build the filtered hybrid allow list once per shard, not once per leg'，标题本身表明带过滤的混合检索会按 'leg'（各路检索）重复构建 allow list，存在重复计算开销[^23]。**以上仅为 issue 标题与元数据，未读取正文，因此不能据此断言混合检索延迟收益不及预期的具体量化结论**[^23]。

---

## 五、矛盾与不确定性

**矛盾 1：CC 与 RRF 的优劣，学术侧与厂商文档侧结论相反。** Bruch 等人报告 CC 在 in-domain 与 out-of-domain 设置下均优于 RRF、样本高效，且 RRF 对参数敏感、丢弃分数分布信息[^7][^8]；OpenSearch 官方在六个 BEIR 数据集上测得 RRF 的 NDCG@10 平均比基于分数的混合流水线低 3.86%[^16]，与学术结论方向一致。但 Elasticsearch 官方仍称 RRF "无需调参"、已被证明比任一单独查询带来更好的相关性[^11]，Weaviate 也把只保留排名的 rankedFusion 作为旧版默认[^10]。这一分歧部分可由"比较对象不同"解释：Elasticsearch 的对比对象是"任一单独查询"，而非 CC；但"RRF 无需调参"这一表述与 Bruch 等人的参数化分析直接冲突[^7]。现有证据不足以判定哪一方在何种条件下成立。

**矛盾 2：混合检索的净收益方向。** BEIR 把混合式描述为结合两类方法优势的后续方向[^3]，StrucChunk 报告了 +44% 至 +104% 的召回提升[^5]，Fusion Tax 报告聚合指标在每一个语料上优于纯稠密检索[^4]；但 Fusion Tax 同时报告 13–48% 的查询上融合反而劣于其较好组件、8–21% 的查询上劣于纯稠密检索[^4]，且 StrucChunk 自述增益的 73–88% 来自分块而非混合本身[^5]。两者并不逻辑矛盾（聚合增益与单查询损失可以并存），但"混合优于单一"作为无条件命题不成立。

**不确定性 1（最关键）：混合检索相对纯 BM25 与纯稠密检索在 BEIR/MS MARCO 上的 nDCG@10 / Recall 提升幅度，现有证据无一提供同设置下的三方对比。** StrucChunk 自述增益 73–88% 来自分块[^5]；NevIR 文评估的是 reranker 而非混合 vs 单一[^6]；Blended RAG 的基线是文献引用值（monoT5-3B、COCO-DR）而非重跑的 BM25/dense，且未在 BEIR/MS MARCO 报告[^12]；M3-Embedding 抓取正文缺结果表[^13]。该子问题的核心量化结论目前无一手支撑。

**不确定性 2：混合检索增益在稀有实体、数字/ID、领域外、长尾专有名词等查询类型上是否最大，无任何证据按查询类型做分层量化。** Fusion Tax 只给出「融合税最可能出现在只有一个检索器能覆盖的查询上」的定性描述，未按实体/数字/长尾分类给出增益幅度[^4]。Weaviate issue #11627 给出了金融文档下按查询类型（精确匹配/语义/表格单元格）的最优 alpha 差异[^22]，但那是社区自建实验、非厂商报告、状态 Open，且讨论的是 α 调优而非混合 vs 单一的增益分层。

**不确定性 3：混合检索在真实生产负载下收益不及预期的工程报告缺失。** 仅有 Weaviate 社区 issue #11627（自建 finrag-eval、非厂商报告、状态 Open）[^22] 与一组未读正文的 issue 标题（#12351 等）[^23]，后者明确标注「不能据此断言量化结论」。缺厂商工程博客或可复现的负载测试数据。Vespa 官方博客虽讨论零样本排序，但抓取内容被截断于数据集统计部分，未读到混合检索的延迟/成本数据或收益不及预期的结论章节[^24]。

**不确定性 4：Fusion Tax 一文的发表状态与作者背景。** 作者版全文以 IEEE Access 期刊排版呈现，标注 "Date of publication October 15, 2026"、DOI 10.1109/ACCESS.2026.3743502，并声明为正式发表版本的作者自备版[^14]；作者为两位独立研究者，无大学或企业机构隶属，论文声明无外部资助、无利益冲突[^14]。该文自述其增量在于给出实践者决策所需的量：现代 sparse-dense 融合把相关文档排到其两个组件中较好者之下、以及排到本可单独部署的单一组件之下的频率与幅度[^14]。其研究范围明确限定为第一阶段检索质量加一个 reranking 阶段的恢复效果，不评估答案生成[^14]。引用其结论时应注意这是单一研究团队的诊断性分析，尚未见独立复现。

## 参考来源

[^1]: Dense Passage Retrieval for Open-Domain Question Answering (DPR) — https://arxiv.org/abs/2004.04906v3（arxiv, 2020-04-10）
[^2]: SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking — https://arxiv.org/abs/2107.05720v1（arxiv, 2021-07-12）
[^3]: BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models — https://ar5iv.labs.arxiv.org/html/2104.08663（arxiv, 2021-04-17）
[^4]: The Fusion Tax: A Diagnostic Analysis of When Hybrid Retrieval Hurts — https://arxiv.org/abs/2610.30612（arxiv, 2026-10-21）
[^5]: StrucChunk: Structure-Aware Retrieval for Legal Document RAG — https://arxiv.org/abs/2610.26313（arxiv, 2026-10-19）
[^6]: Preference Is Not Position: Evaluating Reranking on NevIR in Fixed Hybrid Candidate Pools — https://arxiv.org/abs/2610.26260（arxiv, 2026-10-19）
[^7]: An Analysis of Fusion Functions for Hybrid Retrieval (Bruch, Gai, Ingber; arXiv:2210.11934v2; ACM TOIS, DOI 10.1145/3596512) — https://arxiv.org/html/2210.11934v2（arxiv, 2022-10-21 (v1); 2023-05-04 (v2)）
[^8]: An Analysis of Fusion Functions for Hybrid Retrieval — arXiv 摘要页（元数据与摘要） — https://arxiv.org/abs/2210.11934v2（arxiv, 2022-10-21 (v1); 2023-05-04 (v2)）
[^9]: Hybrid and Multi-Stage Queries — Qdrant Documentation — https://qdrant.tech/documentation/concepts/hybrid-queries/（web）
[^10]: Hybrid search concepts — Weaviate Documentation — https://docs.weaviate.io/weaviate/concepts/search/hybrid-search（web）
[^11]: Reciprocal rank fusion — Elasticsearch Reference — https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion（web）
[^12]: Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search and Hybrid Query-Based Retrievers — https://arxiv.org/html/2404.07220v2（arxiv, 2024-03-22）
[^13]: M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation — https://arxiv.org/html/2402.03216v5（arxiv, 2024-02-05）
[^14]: The Fusion Tax — 作者版全文（arXiv HTML, v1） — https://arxiv.org/html/2610.30612v1（arxiv, 2026-10-22）
[^15]: OpenSearch Documentation — Hybrid search (search pipelines, normalization vs score ranker) — https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/（web）
[^16]: OpenSearch Documentation — Reciprocal rank fusion (score ranker processor, rank_constant) — https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/rrf/（web）
[^18]: Milvus Documentation — RRF Ranker (default k = 60) — https://milvus.io/docs/rrf-ranker.md（web）
[^19]: GitHub — opensearch-project/OpenSearch (repository metadata) — https://github.com/opensearch-project/OpenSearch（github, 2026-10-24）
[^20]: GitHub — pgvector/pgvector (repository metadata) — https://github.com/pgvector/pgvector（github, 2026-10-24）
[^21]: GitHub — langchain-ai/langchain (repository metadata) — https://github.com/langchain-ai/langchain（github, 2026-10-25）
[^22]: [Feature / Research] Hybrid search alpha parameter auto-tuning for domain-specific corpora — financial document retrieval benchmarks (weaviate/weaviate #11627) — https://github.com/weaviate/weaviate/issues/11627（github, 2026-06-03）
[^23]: Weaviate issue tracker search results: 'hybrid search latency slow' (9 results, incl. #12351 'Build the filtered hybrid allow list once per shard, not once per leg') — https://github.com/weaviate/weaviate/issues?q=hybrid+search+latency+slow（github）
[^24]: Improving Zero-Shot Ranking with Vespa Hybrid Search (Vespa Blog, Jo Kristian Bergum) — https://blog.vespa.ai/improving-zero-shot-ranking-with-vespa/（web, 2023-01-05）