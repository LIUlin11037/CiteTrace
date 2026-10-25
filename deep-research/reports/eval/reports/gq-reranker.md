## 摘要

在 RAG 与开放域检索中，reranker 是两阶段「retrieve & re-rank」流水线的第二阶段：第一阶段检索器（词法检索或 bi-encoder 稠密检索）从百万级文档集合中召回约 100 个候选，第二阶段用 cross-encoder 对每个 (query, document) 对联合打分并重排 [^1][^5]。其存在理由有两个层面：一是第一阶段检索器「可能返回与查询并不那么相关的文档」，召回集本身含噪 [^5]；二是 cross-encoder 因跨 query 与 document 做 attention 而精度更高，但对全库逐一打分「会相当慢」，只能施加于小候选集 [^1][^5]。bi-encoder 与 cross-encoder 的根本差异在于是否联合编码：bi-encoder 对 query 与段落「独立」生成 embedding，再用 cosine 相似度比较，因此文档向量可离线预计算并索引 [^1][^5]；cross-encoder 把 query 与 document 同时送入同一个 transformer，输出 0–1 的单一相关性分数 [^1][^5]。这一差异直接决定了二者的成本结构与适用位置：bi-encoder 可扩展到百万级集合但精度受限，cross-encoder 精度更高但只能用于重排小候选集 [^1][^5]。需要说明的是，加入 reranker 的量化收益与绝对延迟代价，现有公开证据均不足以给出可采信的数字（详见「矛盾与不确定性」）。

## 一、reranker 在 RAG 中的位置与作用

**位置：检索之后、下游使用之前。** 官方文档把 RAG/QA 检索描述为两阶段 pipeline：第一阶段 retrieval 取回约 100 个候选（可用 Elasticsearch 等词法检索，或用 SentenceTransformer 即 bi-encoder 做稠密检索），第二阶段用基于 CrossEncoder 的 re-ranker 对候选打分重排 [^5]。reranker 的输入是检索器已召回的候选集，因此它只能重排、不能召回第一阶段未取回的文档——recall 上限由第一阶段候选集决定 [^5]。在工程框架中这一位置被一致地复现：LlamaIndex 的 node postprocessor 运行于 retrieval 与 response synthesis 之间，接收 retriever 返回的节点并在 LLM 看到之前做变换、过滤或重排，最常见的一类就是 reranker [^34]；LangChain 的示例把 Cohere rerank 接在 FAISS 向量检索器之后，并刻意把基础检索器的 `k` 设为 20 以放大候选集 [^39]；Elasticsearch 的 `text_similarity_reranker` 是一个以子 retriever 为输入的 retriever，用 NLP 模型按与查询的语义相似度重排 top-k 文档 [^35]；Vespa 则把重排表达为 rank-profile 中的 `second-phase`/`global-phase`，默认对每个 content node 上 first-phase 的前 100 条做第二阶段计算 [^38]。

**作用一：修正第一阶段的召回噪声。** 文档明确指出第一阶段检索器「可能返回与查询并不那么相关的文档」，这正是引入 reranker 的直接动因 [^5]。两类第一阶段检索器各有不同的失效模式：词法检索只做查询词的字面匹配，无法识别同义词、缩写或拼写变体；稠密检索把 query 编码进向量空间取近邻，可覆盖同义词与缩写 [^5]。Cohere 的官方示例给出了 reranker 压制「关键词命中但语义无关」候选的具体形态：对「美国首都是哪里」的查询，包含 Washington, D.C. 的文档被排到第 1 位（relevance_score 0.943264），而字面含 capital 但语义无关的「Capital punishment in the US」被排到最后（0.4664567）[^6]。Cohere Rerank 的接口形态即「给定一个 query 和一个 documents 列表，返回按语义相关性从高到低排序的文档索引及 relevance_score」，是对已有候选列表重排，不负责从全库召回 [^6]。

**作用二：用更高的相关性判别精度换取小候选集上的可控成本。** RankRAG 全文指出，cross-encode 问题并对候选上下文独立排序的模型，在从 top-N 候选中取 top-k（N≫k）时可比稠密 embedding 检索器效果更好——这是 reranker 在 RAG 中被需要的直接理由 [^32]。该文同时指出现有 RAG 流程的局限：LLM 不擅长读过多 chunk（top-100），更短的 top-k（如 5、10）通常带来更高生成准确率，而小 k 下需要机制保证召回 [^32]。BEIR 的零样本评测结论与此一致：re-ranking 与 late-interaction 类模型平均取得最佳零样本性能，但计算成本高；dense 与 sparse 检索模型计算效率更高，却常常不如上述方法 [^10]。

**作用三：在混合检索中充当分数/排名融合层（同名但不同义的「rerank」）。** 需注意术语歧义：Milvus 文档中的 reranking 指 hybrid search 中整合多个向量字段结果的阶段，原生策略只有 WeightedRanker（用 arctan 把不同度量的分数归一化到 [0,1] 后按权重加权平均）与 RRFRanker（基于各路径排名的倒数排名融合，平滑参数 k 默认 60），该阶段属于分数/排名融合层，不涉及神经 cross-encoder 推理 [^9]。Elasticsearch 也把 `rrf`（reciprocal rank fusion）与 `text_similarity_reranker` 列为两种不同的 retriever [^36]。因此讨论「reranker 的作用」时须区分：融合式 rerank 解决的是多路召回结果的合并，模型式 rerank 解决的是单路候选的语义精排。

**reranker 的适用边界。** 当段落集合很小时（如单文档内搜索）可跳过 retrieval 阶段，直接用 CrossEncoder 对全部段落打分 [^1][^5]。官方示例的完整流程是：把 Wikipedia 文章切分为段落并用 bi-encoder 编码，查询时取余弦相似度最高的段落，再由 CrossEncoder 重排，最终把 CrossEncoder 分数最高的 5 个段落呈现给用户 [^5]。

## 二、bi-encoder 与 cross-encoder 的差异

**bi-encoder 的定义与计算范式。** bi-encoder 指检索阶段使用的 dense retrieval 模型，文档中明确称 SentenceTransformer 实现的稠密检索为 "a.k.a. bi-encoder" [^1]。其计算范式是：为段落和查询「独立」生成 embedding（"produces embeddings independently for your paragraphs and for your search queries"），再用向量相似度（cosine-similarity）比较 [^1]。因为编码相互独立，文档向量可离线预计算并索引，这是其能支撑百万级集合的原因 [^5]。DPR 证明检索可仅用稠密表示实现，embedding 由「简单的 dual-encoder 框架」从少量问题与段落中学习得到，在广泛开放域 QA 数据集上 top-20 段落检索准确率较强 Lucene-BM25 系统大幅领先 9%–19% 绝对值 [^3]。SBERT 则用 siamese 与 triplet 网络结构产出可用 cosine-similarity 比较的语义句子 embedding [^2]。

**cross-encoder 的定义与计算范式。** cross-encoder 把 query 与候选 document「同时」送入 transformer 网络，网络输出 0 到 1 之间的单一分数，表示该文档对给定查询的相关性 [^1][^5]。其优势来源被明确归因于「跨 query 与 document 做 attention」（"they perform attention across the query and the document"），因此性能更高 [^1][^5]。Vespa 的表述一致：cross-encoder 类文本排序模型通常比 text embedding 模型更有效，因为它们把 query 与 document 同时作为输入、在所有 query 与 document token 之间做 full cross-attention；代价是计算复杂度 [^37]。Voyage 官方也把 reranker 定义为 cross-encoder：与分别编码 query 和 document 的 embedding 模型不同，reranker 联合处理 query-document 对，从而给出更准确的相关性预测 [^40]。

**差异一：联合编码 vs 独立编码，决定了精度上限。** cross-encoder 的 attention 跨越 query 与 document 两侧，模型可以直接建模词级交互；bi-encoder 把两侧各自压缩为单一向量后才比较，交互只发生在向量相似度层面 [^1][^5][^37]。SBERT 论文从反面印证了全交叉编码的精度价值与其代价：BERT/RoBERTa 用于句子对回归任务（如 STS）时要求「两个句子同时输入网络」，由此带来巨大的计算开销 [^2]。

**差异二：成本结构，决定了可施加的候选集规模。** cross-encoder 对成千上万乃至上百万个 (query, document) 对逐一打分「会相当慢」（"Scoring thousands or millions of (query, document)-pairs would be rather slow"），故只用于对检索器产出的约 100 个候选做重排 [^1][^5]。其调用形态是成对打分：`scores = model.predict([(query, passage) for passage in passages])`，即每个候选都需一次完整的前向计算 [^11]。SBERT 论文给出了这一复杂度的量级：在 10,000 个句子中找最相似的一对需要约 5000 万次推理计算（约 65 小时）；改用 siamese 结构的 SBERT 后，同一任务从 65 小时降至约 5 秒，同时保持 BERT 的准确率 [^2]。论文据此明确结论：BERT 的构造方式使其不适合语义相似度搜索以及聚类等无监督任务 [^2]。反过来，retriever 必须对含数百万条目的文档集合保持高效，但可能返回不相关候选，这正是需要 re-ranker 的原因 [^1]。

**差异三：输出形态与分数语义。** cross-encoder 输出 0–1 的单一相关性分数 [^1][^5]；但 FlagEmbedding 的 reranker 文档指出，reranker 以 question 与 document 同时作为输入、直接输出相似度而非向量，相关性分数由交叉熵损失优化，因此不限定在特定区间内 [^42]。工程实现上分数还会被平台改写：Elasticsearch 在返回前对 rerank 分数做归一化 `score = max(score, 0) + min(exp(score), 1)`，并提供反变换以恢复原始分数 [^35]。这意味着跨模型、跨平台的 reranker 分数不可直接横向比较。

**差异四：工程可部署性。** bi-encoder 的离线索引能力使其成为第一阶段默认选择 [^5]；cross-encoder 的推理优化集中在单次前向的成本上：Sentence Transformers 提供 PyTorch、ONNX、OpenVINO 三种后端，GPU 上可用 fp16 半精度加速（官方描述为「以极小的精度损失换取 GPU 推理加速」），并明确不推荐对 Cross Encoder 启用 Flash Attention 2——因为让 Flash Attention 发挥优势的自动 input unpadding 只适用于 feature-extraction 任务，classification head 与 flattened inputs 不兼容；官方基准测得在没有 unpadding 的情况下，单纯替换 kernel 在长文本上与 fp16 速度持平、在短文本上慢 20% 到 45%，该结论覆盖全部四个被基准测试的 reranker [^11]。torch.compile 的收益强烈依赖模型与硬件，收益随模型规模增大而增大，非常小的模型在快 GPU 上可能收益很小甚至略微变慢 [^11]。

**中间形态：late interaction。** ColBERT 的 late interaction 被定位为「近似 cross-encoder 的 query-document 联合注意力」，同时因 bi-encoder 架构与近期索引/搜索优化而保持接近传统稠密检索的推理效率 [^22]。PLAID 通过 centroid interaction 与 centroid pruning 加速 late interaction 检索，声称在不影响检索质量的前提下相对 vanilla ColBERTv2 将搜索延迟最多降低 7×（GPU）与 45×（CPU），评测规模最大到 1.4 亿 passages [^21]。在 RAG 工程侧，RAGatouille（约 3972 stars）定位为在任意 RAG pipeline 中便捷使用与训练 late-interaction 检索 [^24]，PyLate 则构建在 Sentence Transformers 之上用于 ColBERT 类模型的微调、推理与检索 [^23]。LlamaIndex 把 `ColbertRerank` 列为 token-level late interaction 的 reranker 选项 [^34]。

## 三、reranker 的模型族与选型现状

**cross-encoder 是 reranker 的主流形态。** BGE reranker 系列为 cross-encoder 模型，2023-09-12 发布 base 与 large 两版，README 建议用其对 embedding 模型返回的 top-k 文档做重排 [^8]；其后 2024-03-18 发布基于 M3 与 LLM（GEMMA、MiniCPM）骨干的新 reranker，支持多语言与更长输入 [^8]；2024-07-26 发布基于 gemma-2-9b 的 bge-reranker-v2.5-gemma2-lightweight，支持 token 压缩与逐层轻量化 [^8]。BGE reranker 由 xlm-roberta-base 初始化，在多语言数据混合上训练（中文 788,491 对、英文 933,090 对、其他语言 97,458 对），主要支持中英文，在其他低资源语言上可能出现性能下降 [^42]。其 C-MTEB 类评测平均分为 bge-reranker-large 66.10、bge-reranker-base 65.42，高于同表所列的 bge-large-zh-v1.5（63.97）与 multilingual-e5-large（57.4）等 embedding 模型 [^42]。

**reranker 已形成多层模型族。** AnswerDotAI/rerankers 的 README 列出支持的族：transformers 系 cross-encoder、T5 seq2seq、ColBERT（含 PyLate ColBERT/late-interaction 重排）、FlashRank（ONNX 优化，README 称在 CPU 上很快）、RankGPT、RankLLM（RankZephyr/RankVicuna 等本地模型）、LLM-layerwise、多模态 MonoVLM；API 型提供方包括 Cohere、Jina、Pinecone、Isaacus、MixedBread [^7]。LlamaIndex 的分类更贴近选型决策：本地 cross-encoder（`SentenceTransformerRerank`）、托管 API（CohereRerank、JinaRerank、VoyageAIRerank、MixedbreadAIRerank、NVIDIARerank）、LLM 型（LLMRerank、RankGPTRerank、RankLLMRerank）、token-level late interaction（ColbertRerank）、多模态（ColPaliRerank）[^34]。其官方延迟/质量指引为：无 API key 的本地默认推荐 `SentenceTransformerRerank`，速度用 `cross-encoder/ms-marco-MiniLM-L6-v2`、更强多语言质量用 `Qwen/Qwen3-Reranker-0.6B`；质量最高、可容忍延迟则用 LLM 型 reranker [^34]。该文档还特别提示构造器默认模型 `cross-encoder/stsb-distilroberta-base` 是为语义文本相似度而非段落重排训练的，检索场景「强烈建议」传入 MS MARCO 训练的模型 [^34]。

**LLM reranker 的核心矛盾是效率-精度权衡。** GroupRank 把当前 LLM reranking 的困境表述为 efficiency-accuracy trade-off：pointwise 方法高效但忽略文档间比较、精度次优；listwise 方法能捕捉全局上下文，但受上下文窗口限制且推理延迟过高 [^15]。ERank 的表述一致：基于 SFT 的 pointwise 方法把相关性当作二分类任务，缺乏必要的打分区分度；而为复杂推理设计的方法多采用强大但低效的 listwise 形式，使其在低延迟应用中不可行 [^16]。两条主流应对路线是：GroupRank 提出 groupwise 折中范式，融合局部 pointwise 信号与全局 listwise 排序 [^15]；ERank 不放弃 pointwise 低延迟架构，先用 SFT 生成细粒度整数分数提升区分度，再用源自 listwise 的 reward 做 RL 注入全局排序意识，其 ERank-4B 在 BRIGHT 上 nDCG@10 为 38.7、32B 变体为 40.2（作者称 SOTA，属自述主张）[^16]。长文档场景下，EviRerank 指出 decoder-only LLM reranker 推理代价高且相关性信号会被无关上下文稀释，其做法是先用轻量 selector（BM25、bi-encoder 或 cross-encoder）给文档块打分，再在硬性 token 上限下动态分配证据预算并由 LLM 重排，在 TREC DL'19 上 nDCG@10 最高达 0.744 [^17]。这一「轻量预筛 + LLM 重排」的级联结构说明 cross-encoder/bi-encoder 与 LLM reranker 并非替代关系 [^17]。

**另一条路线是把排序内化进生成器。** RankRAG 用单个 LLM 同时承担 context ranking 与 answer generation，关键发现是训练混合中加入一小部分 ranking 数据即可让指令微调后的 LLM 在排序上超过现有专家排序模型，包括用 10 倍以上 ranking 数据专门微调的同一 LLM [^31]。该文指出现有 RAG 流程中专家排序模型的零样本泛化能力弱于通用 LLM，且此前将排序用于 RAG 的方法仍依赖额外的中等规模模型（如 BERT、T5）做排序，难以充分刻画 query 与 context 的相关性 [^32]。Self-RAG 则用特殊 token（reflection tokens）对检索到的段落及自身生成进行反思，把「是否相关/是否有支持」的判定内化进生成器，而非外挂独立 reranker [^33]。需注意 Self-RAG 的 v1 提交于 2023-10-17，属该方向的奠基性前作 [^33]。

**托管 API 的约束条件。** Voyage 当前主力 reranker 为 rerank-3 与 rerank-3-lite，上下文长度均为 32,000 tokens；单次请求文档数不得超过 1,000 篇；rerank-2 要求 query token 数与任一单文档 token 数之和不超过 16,000，总 token 预算（query tokens × 文档数 + 所有文档 token 之和）不超过 600K [^40]。Cohere 的模型表列出 rerank-v3.5 上下文长度 4k tokens，rerank-v4.0-fast 与 rerank-v5.0-pro 均为 32k tokens；官方建议新项目使用 rerank-v5.0-pro，rerank-v4.0-fast 定位低延迟高吞吐场景 [^41]。Cohere Rerank 的计费单位中包含 search_units，说明 rerank 调用按搜索单元计费 [^6]。Elasticsearch 的 `rank_window_size` 控制参与重排的 top 文档数，默认 10，官方示例设为 100；其 `chunk_rescorer` 只把得分最好的块送入重排模型，用于缓解长文档被模型 token 上限截断的问题，并可通过控制推理 token 量来控制成本 [^35]。

## 四、reranker 的训练与数据构造

reranker 通常用 hard negative mining 训练，目标是选出对每个 query 排名高但实际不相关的页面 [^26]。DocReRank 认为传统 hard negative mining 是被动的、受限于检索器在现有语料中能找到的内容，存在多样性有限、负例常常不够 hard、可控性低、以及频繁的 false negative 损害训练四个固有局限，并提出反向做法：不按 query 检索负例页面，而是按页面用 LLM-VLM 流水线生成 hard negative query [^26]。ReasonIR 面向推理任务构建合成数据流水线：对每个文档生成一个具有挑战性且相关的 query，同时生成一个看似相关但最终无用的 hard negative [^27]。PETRA 在石油工程领域构造了 1.36M 精选 chunks、约 859k 条 embedding 训练行与约 400k 条 teacher 打分的 reranker 候选行，其负面发现值得注意：在合成标签上取得高 train-holdout 准确率并不能预测检索增益；检索挖掘的数据只有在被重新打包为从推理时候选分布中采样的 teacher 打分候选列表后才有帮助 [^28]。另一条降低标注成本的路径是用 LLM 做数据生成与监督而非推理：用 LLM 从领域语料生成合成 query、用基于 LLM 的分类器标注 positive 与 hard-negative 对，再以 Localized Contrastive Estimation loss 微调较小的 transformer reranker [^29]。

## 矛盾与不确定性

**1. reranker 的量化收益缺乏可采信证据。** sbert.net 与 Cohere 文档对 reranker 的价值均为定性描述，Cohere 页面明确未提供 nDCG/recall 提升幅度或延迟基准数据 [^6]。仅有的数字来自 0–7 stars 的个人仓库（如「cross-encoder reranker 带来 8 倍延迟增加而无成比例收益」），这些仓库未经同行评审或第三方复现，按取证纪律不可作为结论依据 [^12][^13][^14]。PETRA 报告的 nDCG 0.703→0.763（score fusion）与 reranker 适配带来的 +44%（Earth Science 基准）、+23%（六任务 reasoning-intensive 面板）属石油/地球科学领域数据集结论，不能外推为通用 RAG 增益 [^28]。因此「加入 reranker 能提升多少 recall@k / nDCG / 端到端答案准确率」现有公开资料不足以确认。

**2. reranker 的绝对延迟与成本量级同样缺失。** Vespa 只给出 second-phase 默认重排每节点 100 条与 `total-rerank-count` 上界，并强调 second-phase 的价值在于「为成本提供可预测的上界」，未给实测延迟 [^38]；sbert.net 只给后端优化与 kernel 替换的相对速度结论（短文本慢 20%–45%），无绝对延迟数字 [^11]；GroupRank 的 6.4 倍推理加速与 EviRerank 的「显著降低延迟」都是相对基线，且未说明硬件与 batch 配置 [^15][^17]。每 query 增加多少毫秒、多少 token/费用，现有公开资料不足以确认。

**3. 「当前最强开源 reranker」存在未裁决的冲突。** AnswerDotAI/rerankers README 称 MXBai V2 为「current open-source state-of-the-art」，但这是仓库自述的营销性主张，需外部验证 [^7]；LlamaIndex 文档推荐 `Qwen/Qwen3-Reranker-0.6B` 作为更强多语言质量选项 [^34]；FlagEmbedding README 的 reranker 更新条目停在 2024-07-26，未见 2025-03 之后的 reranker 更新 [^8]。三者对「当前最强开源 reranker」给出不同指向，且均无模型卡、技术报告或公开榜单等一手依据。此外，BGE reranker 的 C-MTEB 分数表 [^42] 与 GroupRank/ERank 在 BRIGHT 上的 nDCG@10 [^15][^16] 分属不同基准与不同任务设定，不可直接比较，无法据此排出统一名次。

**4. 分数语义不可跨模型比较。** cross-encoder 输出 0–1 分数 [^1][^5]，但 FlagEmbedding 指出其 reranker 分数由交叉熵优化、不限定区间 [^42]，Elasticsearch 又在返回前做非线性归一化 [^35]。同一候选在不同 reranker 下的分数不具可比性，任何跨模型阈值设定都需重新校准。

**5. 术语歧义风险。** 「rerank」在 Milvus 中指多路向量检索结果的分数/排名融合（WeightedRanker、RRFRanker），不涉及神经 cross-encoder 推理 [^9]；Elasticsearch 也把 `rrf` 与 `text_similarity_reranker` 并列为两种不同 retriever [^36]。将融合式 rerank 的收益与模型式 rerank 的收益混为一谈，是引用文献时的常见错误来源。

## 参考来源

[^1]: Retrieve & Re-Rank — Sentence Transformers documentation (sbert.net) — https://www.sbert.net/examples/applications/retrieve_rerank/README.html（web）
[^2]: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (EMNLP 2019) — https://arxiv.org/abs/1908.10084v1（arxiv, 2019-08-27）
[^3]: Dense Passage Retrieval for Open-Domain Question Answering (DPR) — https://arxiv.org/abs/2004.04906v3（arxiv, 2020-04-10）
[^5]: Retrieve & Re-Rank — Sentence Transformers documentation — https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html（web）
[^6]: How Rerank Works — Cohere Rerank API documentation — https://docs.cohere.com/docs/rerank-overview（web）
[^7]: AnswerDotAI/rerankers — A lightweight unified API for various reranking models — https://github.com/AnswerDotAI/rerankers（github, 2025-12-20）
[^8]: FlagOpen/FlagEmbedding — BGE: One-Stop Retrieval Toolkit For Search and RAG (README) — https://github.com/FlagOpen/FlagEmbedding（github）
[^9]: Milvus 官方文档 — Reranking（hybrid search 中的重排阶段） — https://milvus.io/docs/reranking.md（web）
[^10]: BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models (arXiv:2104.08663v4) — https://arxiv.org/abs/2104.08663（arxiv, 2021-04-17 (v1); v4 2021-10-21）
[^11]: Speeding up Inference — Sentence Transformers documentation (Cross Encoder) — https://www.sbert.net/docs/cross_encoder/usage/efficiency.html（web）
[^12]: oladri-renuka/rag-sec-project — RAG pipeline benchmarking 4 retrieval strategies on SEC 10-K filings — https://github.com/oladri-renuka/rag-sec-project（github）
[^13]: tkarim45/embedding-reranker-bench — benchmark 5 embedding models + a cross-encoder reranker — https://github.com/tkarim45/embedding-reranker-bench（github）
[^14]: umstek/zero-shot-ie-bench — 60 zero-shot IE & classification systems benchmarked — https://github.com/umstek/zero-shot-ie-bench（github）
[^15]: GroupRank: A Groupwise Paradigm for Effective and Efficient Passage Reranking with LLMs — https://arxiv.org/abs/2511.11653（arxiv, 2025-11-10 (v1); v2 2026-04-30; accepted ACL-Findings 2026）
[^16]: ERank: Fusing Supervised Fine-Tuning and Reinforcement Learning for Effective and Efficient Text Reranking — https://arxiv.org/abs/2509.00520（arxiv, 2025-08-30）
[^17]: EviRerank: Adaptive Evidence Construction for Long-Document LLM Reranking — https://arxiv.org/abs/2411.06254（arxiv, 2024-11-09 (v1); v2 2026-10-14; Findings of EMNLP 2026）
[^21]: PLAID: An Efficient Engine for Late Interaction Retrieval — https://arxiv.org/abs/2205.09707（arxiv, 2022-05-19）
[^22]: Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever — https://arxiv.org/abs/2408.16672（arxiv, 2024-09-14）
[^23]: lightonai/pylate — Flexible Training and Retrieval for Late Interaction Models — https://github.com/lightonai/pylate（github）
[^24]: AnswerDotAI/RAGatouille — late-interaction (ColBERT) retrieval in RAG pipelines — https://github.com/AnswerDotAI/RAGatouille（github）
[^26]: DocReRank: Single-Page Hard Negative Query Generation for Training Multi-Modal RAG Rerankers — https://arxiv.org/abs/2505.22584（arxiv, 2025-05-28）
[^27]: ReasonIR: Training Retrievers for Reasoning Tasks — https://arxiv.org/abs/2504.20595（arxiv, 2025-04-29）
[^28]: PETRA: Transforming Web Text for Petroleum-Engineering Domain Adaptation — https://arxiv.org/abs/2606.24346（arxiv, 2026-06-23）
[^29]: Enhancing Transformer-Based Rerankers with Synthetic Data and LLM-Based Supervision — https://arxiv.org/abs/2510.01229（arxiv, 2025-09-23）
[^31]: RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs (arXiv:2407.02485) — https://arxiv.org/abs/2407.02485（arxiv, 2024-07-02）
[^32]: RankRAG 全文（HTML 版）§1–§4：动机、两阶段指令微调与训练数据构成 — https://arxiv.org/html/2407.02485v1（arxiv, 2024-07-02）
[^33]: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (arXiv:2310.11511) — https://arxiv.org/abs/2310.11511（arxiv, 2023-10-17）
[^34]: LlamaIndex Docs — Node Postprocessor Modules (rerankers) — https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/（web）
[^35]: Elasticsearch Docs — Text similarity re-ranker retriever — https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/text-similarity-reranker-retriever（web）
[^36]: Elasticsearch Docs — Retrievers (rrf, linear, text_similarity_reranker overview) — https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers（web）
[^37]: Vespa Docs — Ranking With Transformer Cross-Encoder Models — https://docs.vespa.ai/en/ranking/cross-encoders.html（web）
[^38]: Vespa Docs — Phased Ranking — https://docs.vespa.ai/en/phased-ranking.html（web）
[^39]: LangChain Docs — Cohere Reranker (retriever integration) — https://docs.langchain.com/oss/python/integrations/retrievers/cohere-reranker（web）
[^40]: Voyage AI Docs — Reranker（模型选择与参数限制） — https://docs.voyageai.com/docs/reranker（web）
[^41]: Cohere Docs — Models（Rerank 模型表与平台可用性） — https://docs.cohere.com/docs/models（web）
[^42]: FlagEmbedding research/reranker README — Reranker 用法、训练数据与评测表 — https://github.com/FlagOpen/FlagEmbedding/tree/master/research/reranker（github）