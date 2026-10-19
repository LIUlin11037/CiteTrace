# RAG 中的 Reranker：作用与 Cross-Encoder / Bi-Encoder 差异

## 摘要

在 RAG（检索增强生成）系统中，reranker 是运行在 retrieval 与 response synthesis 之间的组件：它接收 retriever 返回的候选 nodes，在 LLM 看到它们之前对其进行 transform、filter 或 re-order[^3]。其核心价值在于提升进入生成阶段的上下文的精度（precision）[^4]。从架构上看，reranker 的主流实现是 cross-encoder：query 与 document 被同时送入同一个 transformer 网络，直接输出相关性分数，而不产生 embedding[^9][^17]。与之相对的是 bi-encoder（dual-encoder）：query 与 document 分别独立编码为固定大小向量，再用余弦相似度等度量比较[^2][^6][^7]。这一差异决定了二者的计算复杂度量级：cross-encoder 无法预计算文档表示，成对推理代价随候选集大小线性增长；bi-encoder 可离线预计算全部文档向量，在线仅需一次 query 编码加向量检索[^2][^6]。本报告梳理 reranker 的作用定位、两类编码器的定义与差异、主流 reranker 的实现谱系，并显式标注证据缺口与来源冲突。

---

## 1. Reranker 在 RAG 中的作用

### 1.1 阶段定位：retrieval 与 generation 之间的中间层

LlamaIndex 将 node postprocessor 定义为运行在 retrieval 与 response synthesis 之间的组件：它接收 retriever 返回的 nodes，在 LLM 看到它们之前对其进行 transform、filter 或 re-order[^3]。文档明确指出，最常见的一类 node postprocessor 就是 reranker[^3]。

这一定位意味着 reranker 不负责从全库中找候选，而是对已召回的候选集合重新排序或过滤。cross-encoder/ms-marco-MiniLM-L6-v2 的模型卡把这一流程描述为典型的 "Retrieve & Re-rank"：给定 query，先用 ElasticSearch 等检索出候选 passage，再用 cross-encoder 对候选打分排序[^24]。

### 1.2 目标：提升精度、减少幻觉

LlamaIndex 在生产级 RAG 优化指南中，将 "Reranking retrieved nodes for precision" 列为一项通用技术[^4]。该指南把标准 RAG 栈描述为 top-k retrieval + basic text splitting，并指出其在文档规模扩大时表现不佳[^4]。指南将优化目标表述为提升 retrieval 与 generation 性能，以在更复杂数据集上更准确作答并减少幻觉[^4]。

### 1.3 输入与产出的标准定义

以 Cohere Rerank API 为例，其 endpoint 的定义是：给定一个 query 和一个 documents 列表，Rerank 将文档按与 query 的语义相关性从最相关到最不相关排序（indexes the documents from most to least semantically relevant to the query）[^5]。API 返回带 relevance_score 的排序结果，文档示例中 top_n=5，最高分文档 relevance_score 为 0.943264[^5]。Cohere 建议：若文档含结构化数据，为获得最佳性能应格式化为 YAML 字符串[^5]。

### 1.4 历史起点：monoBERT

reranker 的代表性早期工作是 Nogueira 与 Cho 的 *Passage Re-ranking with BERT*：论文提出对 BERT 的简单重实现，用于基于 query 的 passage re-ranking（query-based passage re-ranking）[^1]。该系统在 TREC-CAR 数据集上达到 state of the art，并在 MS MARCO passage retrieval 任务排行榜上位列第一，在 MRR@10 指标上相对此前 state of the art 提升 27%[^1][^8]。该工作提供了可复现代码仓库 https://github.com/nyu-dl/dl4marco-bert[^1][^8]。

---

## 2. Cross-Encoder 与 Bi-Encoder 的定义与差异

### 2.1 Cross-Encoder 的定义

Sentence-BERT 论文给出了 cross-encoder 的完整定义：两个句子被一起送入 transformer 网络，然后预测目标值（'BERT uses a cross-encoder: Two sentences are passed to the transformer network and the target value is predicted'）[^6]。

据此，cross-encoder 的输入是 query 与 document 组成的句对，处理步骤是二者拼接后共同经过同一个 transformer 编码器并输出预测目标，产出是一个标量相关性分数——而不是向量表示。BGE reranker 模型卡对这一产出有明确说明：reranker 以 query 和 document 同时作为输入，直接输出相似度而非 embedding；分数可经 sigmoid 映射到 [0,1][^9][^17]。

### 2.2 Bi-Encoder / Dual-Encoder 的定义

检索可以用纯稠密表示实现，嵌入由一个简单的 dual-encoder（双编码器）框架从少量问题和段落中学习得到[^7]。该稠密检索器在 top-20 段落检索准确率上比强 Lucene-BM25 系统绝对提升 9%-19%，并帮助端到端 QA 系统在多个开放域 QA 基准上达到新的 SOTA[^7]。论文将稠密检索定位为对 TF-IDF / BM25 等稀疏向量空间模型的替代方案，用于开放域 QA 中高效选取候选上下文[^7]。

SBERT 一侧给出了 bi-encoder 式结构的定义：SBERT 使用 siamese / triplet 网络结构，为输入句子导出固定大小的向量（fixed-sized vectors），可用 cosine-similarity 或 Manhattan/Euclidean 距离比较[^6]。即：输入是单个句子，处理步骤是独立经过共享权重的编码器，产出是固定大小的向量。

### 2.3 差异的核心：计算复杂度

差异的根源在于「是否必须成对计算」。

cross-encoder 设置不适合成对回归任务，因为可能的组合数过多：在 n=10,000 的句子集合中找最相似的一对需要 n·(n-1)/2 = 49,995,000 次推理计算，在 V100 GPU 上约需 65 小时[^6]。对 Quora 的 4000 万问题做单次查询的最相似匹配，用 BERT 成对比较需超过 50 小时[^6]。

SBERT 将寻找最相似句对的开销从 BERT/RoBERTa 的约 65 小时降至约 5 秒，同时保持 BERT 的准确率[^2][^6]。这一数量级差异正是 bi-encoder 可预计算文档向量的直接结果。

### 2.4 为什么不能简单地对 BERT 输出取平均

一个常见误解是：把单句独立输入 BERT，再取平均输出层或 [CLS] token 作为句向量即可。SBERT 论文明确指出，这种常见做法效果很差，常常比平均 GloVe 词向量还差——这是 SBERT 需要专门训练 siamese 结构的原因[^6]。SBERT 使 BERT 可用于大规模语义相似度比较、聚类与语义检索等此前不适用 BERT 的任务[^6]。

### 2.5 分工关系：bi-encoder 召回 + cross-encoder 重排

两类编码器在 RAG 中并非替代关系，而是两阶段流水线的两端：bi-encoder / dual-encoder 负责从大规模语料中高效选取候选上下文[^7]，cross-encoder reranker 负责对已检索出的 passage 重排（'sort the passages in a decreasing order'）[^24]。LlamaIndex 的 node postprocessor 定义正是把 reranker 放在 retriever 之后[^3]。

---

## 3. Reranker 的实现谱系

### 3.1 LlamaIndex 的分类

LlamaIndex 文档给出 reranker 分类[^3]：

- 本地 cross-encoder（如 cross-encoder/ms-marco-MiniLM-L6-v2、Qwen/Qwen3-Reranker-0.6B）
- 托管 API（CohereRerank、JinaRerank、VoyageAIRerank、MixedbreadAIRerank、NVIDIARerank）
- LLM 判定（LLMRerank、RankGPTRerank、RankLLMRerank）
- token 级 late interaction（ColbertRerank）
- 多模态（ColPaliRerank）

LlamaIndex 将 SentenceTransformerRerank 描述为 cross-encoder（基于 sentence-transformers），并列为无 API key、本地、推荐的默认选项[^3]。

### 3.2 RankLLM 的三类谱系

castorini/rank_llm 提供三类 reranker 的完整谱系：pointwise（如 MonoT5）、pairwise（如 DuoT5）、以及以开源 LLM 为主的 listwise 模型，后者兼容 vLLM 推理[^13]。仓库同时支持 RankGPT 与 RankGemini 变体，README 明确将其归类为「proprietary listwise rerankers」（专有/闭源 listwise 重排器）[^13]。仓库支持「仅使用 first-token logits 进行重排」以提升推理效率[^13]。

### 3.3 LLM listwise reranking 的定位

RankZephyr 论文将 RankGPT、LRL 等 LLM listwise reranking 方法归类为 "prompt-decoders"，并指出其共同局限是依赖专有模型、难以复现[^29]。论文称 RankVicuna（开源）效果优于更大的专有模型 RankGPT-3.5，但仍落后于 SOTA 的 RankGPT-4[^29]。论文提出的研究问题之一为：仅 7B 参数的开源 LLM（RankZephyr）能否在零样本设定下超越大得多的专有模型 RankGPT-4 的 listwise reranking 效果[^29]。

RankZephyr 摘要称其弥合了与 GPT-4 的效果差距，并在某些情况下超越该专有模型[^12]。评测覆盖 TREC Deep Learning Tracks，以及 BEIR 中的 NEWS 和 COVID 子集[^12]。作者称 RankZephyr 对初始文档排序变化和被重排文档数量具有鲁棒性[^12]，并在 NovelEval 测试集（查询与 passage 均在其训练期之后）上优于 GPT-4，以回应数据污染担忧[^12]。

RankGPT 一侧：论文研究生成式 LLM（ChatGPT、GPT-4）用于 IR 相关性排序，发现经恰当指令的 LLM 可给出与 SOTA 监督方法相当甚至更优的结果[^30]。为应对数据污染，论文构建了 NovelEval 测试集[^30]。论文提出 permutation distillation 方案，将 ChatGPT 的排序能力蒸馏到小型专用模型；蒸馏得到的 440M 模型在 BEIR 基准上超过 3B 的监督模型[^30]。该论文发表于 EMNLP 2023[^30]。

### 3.4 具体模型与规模

BGE reranker 家族[^9][^17]：

- bge-reranker-base（xlm-roberta-base）
- bge-reranker-large（xlm-roberta-large）
- bge-reranker-v2-m3（bge-m3）
- bge-reranker-v2-gemma（gemma-2b）
- bge-reranker-v2-minicpm-layerwise（MiniCPM-2B-dpo-bf16，支持 8-40 层 layerwise 输出）

参数量：bge-reranker-v2-m3 的 safetensors 参数总量为 567,755,777（约 5.68 亿参数，F32），单文件非分片，总文件大小约 2.27 GB[^9]。bge-reranker-v2-gemma 的参数总量为 2,506,172,416（约 25.06 亿参数，F32），分片存储，总文件大小约 10.02 GB[^10]。

Jina 一侧：jina-reranker-v2 是 cross-encoder 模型，输入 query-document 对并输出相关性分数[^22]；模型参数量为 278,437,633（约 0.28B，BF16 safetensors）[^22]。jina-reranker-v3 是 0.6B 参数的多语言文档 reranker，采用 'last but not late interaction' 架构，基于 Qwen3-0.6B，含 28 层 transformer 与轻量 MLP projector（1024→512→256）[^23]。

Cohere 一侧：rerank-v3.5 的 context length 为 4096 tokens[^19]；rerank-english-v3.0 与 rerank-multilingual-v3.0 的 context length 同样为 4096 tokens[^19]。文档说明：每个文档的 query tokens 与 document tokens 合并计入单文档 context 上限；超出时文档会被自动分块并多次推理[^19]。文档当前最新模型为 rerank-v4.0-pro 与 rerank-v4.0-fast（多语言，支持 JSON 半结构化数据）[^19]。

### 3.5 训练与损失

BGE reranker（base 版）由 xlm-roberta-base 初始化，在多语言数据集混合上训练[^11]。训练数据规模与来源[^11]：

- 中文：788,491 个文本对，来自 T2ranking、MMmarco、dulreader、Cmedqa-v2、nli-zh
- 英文：933,090 个文本对，来自 msmarco、nq、hotpotqa、NLI(SimCSE)
- 其他语言：97,458 个文本对，来自 Mr.TyDi

为增强跨语言检索能力，基于 MMarco 构造了两个跨语言检索数据集：采样 100,000 条英文 query 检索中文 passage，以及 100,000 条中文 query 检索英文 passage[^11]。reranker 以 cross-entropy loss 优化，因此相关性分数不限定在特定范围内[^11]。官方使用示例中 tokenizer 采用 max_length=512 进行截断（代码示例，非规格声明）[^11]。该模型主要支持中英文，其他低资源语言可能出现性能下降[^11]。

---

## 4. 工程生态与可复现性

### 4.1 统一调用层

AnswerDotAI/rerankers：★1641，forks 106，定位为「轻量、低依赖、统一 API 以调用所有常见 reranking 与 cross-encoder 模型」，最近 push 2025-12-20[^15]。

### 4.2 检索微调框架

NovaSearch-Team/RAG-Retrieval：★1136，forks 89，定位为统一 RAG 检索（Embedding、ColBERT、ReRanker）的高效微调框架，最近 push 2026-08-28[^15]。netease-youdao/BCEmbedding：★1882，forks 129，定位为网易有道面向 RAG 产品的开源 embedding 与 reranker 模型，最近 push 2026-09-03[^15]。

### 4.3 sentence-transformers

搜索 'sentence-transformers' 返回的最高星仓库为 huggingface/sentence-transformers，stars 19171、forks 2954，描述为 "State-of-the-Art Embeddings, Retrieval, and Reranking"，主语言 Python，last push 2026-10-14[^25]。UKPLab/sentence-transformers 仓库页面可正常访问，默认分支为 main，累计 commit 数为 3,241；README 正文说明该框架用于计算 embedding，并可用 Cross-Encoder（即 reranker）模型计算相似度分数[^26]。README 称 Hugging Face 上有超过 15,000 个预训练 Sentence Transformers 模型可直接使用[^26]。

### 4.4 RankLLM 的 CLI 化

RankLLM 的 News 记录显示：2025.05.25 RankLLM 资源论文被 SIGIR 2025 接收；2025.07.23 支持 YAML 自定义 prompt 模板（v0.25.0）；2025.08.25 增加 OpenRouter API 支持（v0.25.7）；2026.03.26 新增 rank-llm 命令行界面[^13]。

关于 2026-03-26 这一条，需要区分「Release」与「commit」：该仓库的 GitHub Releases 页面显示没有任何 release（"There aren't any releases here"）[^31]，Tags 页面同样显示没有任何 release/tag[^32]。2026 年 3 月 26 日存在一条 commit（SHA 8754de9），标题为 "docs: announce rank-llm CLI"，作者 ronakice；该 commit 仅修改 README.md，新增 1 行、删除 0 行，新增内容为 README 的 News 条目："[2026.03.26] RankLLM now supports the new `rank-llm` command-line interface (CLI)."[^33]。README News 中相邻的历史条目包括 [2025.08.25] 支持 OpenRouter API - Release v0.25.7，以及 [2025.07.23] 支持 YAML 自定义 prompt 模板 - Release v0.25.0[^33]。因此「2026.03.26 新增 CLI」应理解为 README 公告与代码提交，而非 GitHub Release 形式的版本发布。

2026-03-26 当天有多条 commit，包括 "docs: announce rank-llm CLI"（8754de9）、"docs: remove outdated macOS warning"（baa5d0d）、"Split RankLLM API and MCP extras (#369)"（22412e9）[^34]。2026-03-26 前后集中出现一系列 CLI 相关 commit（#356–#367），2026-03-27 有 #370、#371，2026-03-31 有 #373，2026-04-02 至 2026-04-04 有 #378、#379、#380 等后续 commit[^34]。src/rank_llm/cli 目录最早的相关 commit 出现在 2026-03-26，后续在 2026-03-27、2026-03-31、2026-05-18、2026-05-30、2026-06-02、2026-08-29、2026-09-03、2026-09-29 持续有 commit[^35]。2026-06-02 的 commit "feat: add LiteLLM as listwise reranker for 100+ LLM providers (#391)" 也修改了该 CLI 目录[^35]。main 分支最新可见 commit 为 2026-10-12 "docs: record onboarding reproduction results (#462)"（4c586c2）[^36]。

仓库提供轻量 HTTP API 依赖组（FastAPI、Flask、Uvicorn），以及 MCP 相关支持，指向服务化部署场景[^13]。仓库提供 Model Zoo、Training、End-to-end Run 和 2CR 等章节，并包含 trec_eval_aggregated_results.jsonl 聚合评测结果文件[^13]。

### 4.5 BGE 工具包动态

FlagEmbedding 仓库（BGE 工具包）在工具返回的仓库列表中未出现，因此其 stars 与最近提交时间未取得；仅确认仓库存在 master 分支、1,476 次提交历史[^14]。2024/7/26 发布轻量 reranker bge-reranker-v2.5-gemma2-lightweight，基于 gemma-2-9b，支持 token compression 与 layerwise lightweight 操作，项目方自述「在节省大量资源的同时仍能保证良好性能」[^14][^18]。2024/7/26 同时发布 embedding 模型 bge-en-icl（引入 in-context learning 能力）与 bge-multilingual-gemma2（基于 gemma-2-9b 的多语言 embedding 模型）[^14]。2025/3/6 发布 BGE-VL 多模态 embedding 模型与 MegaPairs 合成数据集，采用 MIT 许可[^14]。2024/12/5 建立集中式 BGE 文档；2024/9/2 开始维护 tutorials 并声明将持续更新[^14]。2024/5/21 与 Jina AI、Zilliz、HuggingFace 等合作发布 AIR-Bench，定位为面向 Neural IR & RAG 的公平分布外（out-of-distribution）评测基准，并设有 leaderboard[^14]。

---

## 5. 矛盾与不确定性

### 5.1 基准分数存在系统性冲突（blocking）

jina-reranker-v3 模型卡对比表给出：bge-reranker-v2-m3 BEIR 56.51、MIRACL 69.32；mxbai-rerank-base-v2 BEIR 58.40；mxbai-rerank-large-v2 BEIR 61.44；Qwen3-Reranker-0.6B BEIR 56.28；Qwen3-Reranker-4B BEIR 61.16；jina-reranker-v3 BEIR 61.94（模型卡称其为 SOTA）[^23]。该对比表来自 Jina 自家模型卡，属竞品自评，与 BGE 官方模型卡/论文（arXiv:2312.15503、2402.03216）自述的 SOTA 主张方向相反。BGE 官方评测表在两次抓取中均被截断——bge-reranker-v2-m3 模型卡正文在 Usage 章节处被截断，未包含 Evaluation 表格内容[^17]；FlagEmbedding main README 抓取内容在 News 章节处被截断，未进入 Model List 与 Evaluation 表格[^18]。因此同一模型在不同来源的 BEIR 分数存在系统性冲突且未解决，属 blocking，本报告不采信任一方的绝对数值作为结论。

### 5.2 基准分数覆盖度严重不足

目前仅 cross-encoder/ms-marco-MiniLM-L6-v2 有明确基准分数（NDCG@10 (TREC DL 19) 74.30、MRR@10 (MS Marco Dev) 39.01）与吞吐（1800 docs/sec on V100）[^24]。该模型卡还提供其他 cross-encoder 的吞吐量：ms-marco-MiniLM-L12-v2 为 960 docs/sec，ms-marco-MiniLM-L4-v2 为 2500 docs/sec，ms-marco-MiniLM-L2-v2 为 4100 docs/sec，TinyBERT-L2-v2 为 9000 docs/sec；模型卡明确注明 'Runtime was computed on a V100 GPU.'[^24]。

Cohere Rerank 3/3.5 的官方博客正文为 JS 渲染，两次抓取均未取到任何基准分数或语言数量：Rerank 3.5 页面仅获得标题、日期、作者与标签[^20]；Rerank 3 页面同样仅获得标题、日期、作者与标签[^21]。RankZephyr 仅有定性主张（"弥合与 GPT-4 的差距"）而无具体 nDCG 数值[^12]。M3-Embedding 论文摘要声称在多语言、跨语言、长文档检索基准上取得新的 state-of-the-art 结果，属作者自述主张，未在该来源中给出具体数值[^16]。

### 5.3 模型结构细节缺失

参数量事实本身可信（来自 HF safetensors 元数据）[^9][^10][^22][^23]，但层数、隐藏维度、输入长度上限、训练数据规模与来源，在 2024–2026 新模型（bge-reranker-v2 系列、Cohere Rerank 3.x/4.0、Jina Reranker v2/v3、RankZephyr）上基本缺失。仅 BGE base 版有训练数据规模（788,491/933,090/97,458 对）[^11]。已知的输入长度约束：jina-reranker-v2 context length 上限为 1024 tokens，超过 1024 tokens 的长文本采用 sliding window 分块后分别 rerank[^22]；Cohere rerank-v3.5 为 4096 tokens[^19]；jina-reranker-v3 可在 131K token context 内同时处理最多 64 个文档[^23]。

### 5.4 生产部署证据缺失

top-k 召回量、两阶段延迟预算、蒸馏/量化/ONNX 加速这一子问题实际无有效证据。两条被检索到的来源与主题无关：Hybrid LLM 论文主题为 LLM 查询路由（在大小模型间按查询难度与目标质量分配），与 cross-encoder reranker 的 top-k 召回或加速无关[^27]；另一条属数学（Classical Analysis and ODEs, math.CA），研究双线性 Calderón-Zygmund 算子的紧性刻画，与 RAG、reranker、cross-encoder 或推理加速完全无关[^28]。二者属编号误猜，应剔除。蒸馏方面仅有 RankGPT 的 permutation distillation（440M 蒸馏模型超 3B 监督模型）一条[^30]，量化/ONNX 加速零证据。

### 5.5 仓库活跃度数据不完整

两个最核心的开源 reranker 仓库在两次抓取中均未取得 stars 与最近提交时间：FlagEmbedding 明确记录「未出现，stars 与最近提交时间未取得」[^14]；UKPLab/sentence-transformers 页面抽取文本中未包含 stars 数、forks 数或最近 commit 日期[^26]。工具取到了 huggingface/sentence-transformers ★19171、last push 2026-10-14[^25]，但该结果的组织归属为 huggingface，而非 UKPLab；工具未说明两者是否为同一仓库的迁移关系[^25]。因此仓库活跃度对比不完整。

### 5.6 其他

Marker-Inc-Korea/RAGchain：★282，forks 28，最近 push 2024-01-09，显示该仓库已长期不活跃[^15]。anush008/fastembed-rs：★1037，forks 145，Rust 库，用于本地生成向量 embedding 与 reranking，最近 push 2026-10-12[^15]。castorini/rank_llm：★660，forks 105，最近 push 2026-10-12[^15]。上述 stars 与 push 时间均来自单次检索结果，未做交叉验证。

---

## 参考来源

[^1]: Passage Re-ranking with BERT (Nogueira & Cho, 2019) — https://arxiv.org/abs/1901.04085 （arxiv，2019-01-13）
[^2]: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (Reimers & Gurevych, 2019) — https://arxiv.org/abs/1908.10084 （arxiv，2019-08-27）
[^3]: LlamaIndex Docs — Node Postprocessor Modules (reranker 阶段定义) — https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/ （web）
[^4]: LlamaIndex Docs — Building Performant RAG Applications for Production — https://docs.llamaindex.ai/en/stable/optimizing/production_rag/ （web）
[^5]: Cohere Docs — How Rerank Works (Rerank API Overview) — https://docs.cohere.com/docs/rerank-overview （web）
[^6]: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks — https://ar5iv.labs.arxiv.org/html/1908.10084 （arxiv，2019-08-27）
[^7]: Dense Passage Retrieval for Open-Domain Question Answering — https://arxiv.org/abs/2004.04906v3 （arxiv，2020-04-10）
[^8]: Passage Re-ranking with BERT — https://arxiv.org/abs/1901.04085v5 （arxiv，2019-01-13）
[^9]: BAAI/bge-reranker-v2-m3 (Hugging Face model card) — https://huggingface.co/BAAI/bge-reranker-v2-m3 （web，2024-03-15）
[^10]: BAAI/bge-reranker-v2-gemma (Hugging Face model card) — https://huggingface.co/BAAI/bge-reranker-v2-gemma （web，2024-03-16）
[^11]: FlagEmbedding research/reranker README (BGE reranker training data & usage) — https://raw.githubusercontent.com/FlagOpen/FlagEmbedding/master/research/reranker/README.md （github）
[^12]: RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze! (arXiv:2312.02724) — https://arxiv.org/abs/2312.02724 （arxiv，2023-12-05）
[^13]: castorini/rank_llm — RankLLM: A Python toolkit for reproducible information retrieval research using rerankers, with a focus on listwise reranking — https://github.com/castorini/rank_llm （github）
[^14]: FlagOpen/FlagEmbedding — BGE: One-Stop Retrieval Toolkit For Search and RAG — https://github.com/FlagOpen/FlagEmbedding （github）
[^15]: GitHub 仓库检索结果（reranker 关键词，按 stars 排序） — https://github.com/search?q=reranker&s=stars （github）
[^16]: M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation (arXiv:2402.03216v5) — https://arxiv.org/abs/2402.03216v5 （arxiv，2024-02-05）
[^17]: BAAI/bge-reranker-v2-m3 model card (Hugging Face) — https://huggingface.co/BAAI/bge-reranker-v2-m3/raw/main/README.md （web，2024-03-15）
[^18]: FlagEmbedding main README (BGE: One-Stop Retrieval Toolkit For Search and RAG) — https://raw.githubusercontent.com/FlagOpen/FlagEmbedding/master/README.md （github）
[^19]: Rerank models — Cohere Documentation (v2 API) — https://docs.cohere.com/docs/rerank （web）
[^20]: Introducing Rerank 3.5: Precise AI Search (Cohere Blog) — https://cohere.com/blog/rerank-3pt5 （web，2024-12-02）
[^21]: Introducing Rerank 3: A New Foundation Model for Efficient Enterprise Search & Retrieval (Cohere Blog) — https://cohere.com/blog/rerank-3 （web，2024-04-11）
[^22]: jinaai/jina-reranker-v2-base-multilingual (Hugging Face Model Card) — https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual （web，2024-06-19）
[^23]: jinaai/jina-reranker-v3 (Hugging Face Model Card / README) — https://huggingface.co/jinaai/jina-reranker-v3 （web）
[^24]: cross-encoder/ms-marco-MiniLM-L6-v2 (HuggingFace model card) — https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 （web）
[^25]: huggingface/sentence-transformers (search result for 'sentence-transformers') — https://github.com/huggingface/sentence-transformers （github）
[^26]: UKPLab/sentence-transformers README (repository page) — https://github.com/UKPLab/sentence-transformers （github）
[^27]: Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing (arXiv:2404.14618) — https://arxiv.org/abs/2404.14618 （arxiv，2024-04-22）
[^28]: A characterization of compactness via bilinear T1 theorem (arXiv:2404.14013) — https://arxiv.org/abs/2404.14013 （arxiv，2024-04-22）
[^29]: RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze! (full text, Introduction) — https://arxiv.org/html/2312.02724v1 （arxiv，2023-12-05）
[^30]: Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents (RankGPT) — https://arxiv.org/abs/2304.09542 （arxiv，2023-04-19）
[^31]: castorini/rank_llm — Releases 页面 — https://github.com/castorini/rank_llm/releases （github）
[^32]: castorini/rank_llm — Tags 页面 — https://github.com/castorini/rank_llm/tags （github）
[^33]: castorini/rank_llm — commit 8754de9 "docs: announce rank-llm CLI" — https://github.com/castorini/rank_llm/commit/8754de9 （github，2026-03-26）
[^34]: castorini/rank_llm — main 分支 commit 历史（2026-03-01 至 2026-04-15 区间） — https://github.com/castorini/rank_llm/commits/main/?since=2026-03-01&until=2026-04-15 （github）
[^35]: castorini/rank_llm — src/rank_llm/cli 目录 commit 历史 — https://github.com/castorini/rank_llm/commits/main/src/rank_llm/cli （github）
[^36]: castorini/rank_llm — main 分支 commit 历史（全时段，最新在前） — https://github.com/castorini/rank_llm/commits/main （github）