# RAG 中文档切分（Chunking）的主流策略与粒度对检索质量的影响

## 摘要

文档切分（chunking）被 Pinecone 定义为「把大段文本拆成更小片段（chunks）的过程」，是一种预处理技术，用于优化最终存入向量数据库的内容的相关性[^1]。其必要性来自两点：让 embedding 模型能把数据装进上下文窗口（超出部分会被截断丢弃），以及保证 chunk 本身含有可供检索的信息[^1]。Anthropic 将传统 RAG 预处理描述为三步：把语料拆成通常不超过几百 token 的小 chunk、用 embedding 模型把 chunk 转为向量、存入向量数据库以支持语义相似度检索[^2]。

主流策略可归为四类：固定长度切分、递归/分隔符切分、语义切分、结构感知切分；在此之上还有一类不改变切分边界、但改变「chunk 如何被编码」的方法（late chunking、contextual retrieval），以及一类解耦「检索用 chunk」与「合成用 chunk」的父子/窗口方案。粒度对检索质量的影响在证据上呈现明显的不对称：**chunk size 与 overlap 的量化关系只有极少数公开数字**，而语义切分相对固定长度切分的收益在唯一一项系统评测中被判定为「不足以抵消计算成本」[^11]。中文场景的切分证据几乎完全缺失。

---

## 一、切分策略的分类体系

### 1.1 厂商/文档给出的分类

Pinecone 采用两级分类：Fixed-size chunking（最常用、推荐作为起点）与 "Content-aware" chunking，后者下含 Simple Sentence and Paragraph splitting、Recursive Character Level Chunking、Document structure-based chunking、Semantic Chunking[^1]。

学术侧的分类略有不同。arXiv:2402.05131v3 把既有切分策略归纳为四类：fixed size（固定长度，忽略文本结构）、recursive（用标点等分隔符迭代细分）、contextual（借助句子切分等 NLP 技术保留上下文语义）、hybrid（组合多种方式）[^4]。

工程库的实现清单可作为「主流策略」的操作性定义。Chonkie 官方文档列出 12 种 chunking 策略：CodeChunker（基于 AST 按代码结构切分）、FastChunker（SIMD 加速的字节级切分）、LateChunker（Late Chunking 算法）、NeuralChunker（微调 BERT 模型按语义漂移切分）、RecursiveChunker（递归切分，适合结构清晰的长文档）、SemanticChunker（按语义相似度分组）、SentenceChunker（按句子边界切分）、SlumberChunker（通过 Genie 接口用生成式 LLM 做 agentic chunking）、TableChunker（按行切分大型 markdown 表格并保留表头）、TeraflopAIChunker（调用 TeraflopAI Segmentation API，面向法律等垂直领域）、TokenChunker（固定 token 大小切分）[^22]。README 另列有 11 种 chunker 的对应描述，其中 FastChunker 宣称 100+ GB/s、SlumberChunker 又称 AgenticChunker[^6]。

LlamaIndex 的 node parser 分类是「基于文件的解析器」与「文本切分器」两类：前者包括 SimpleFileNodeParser、HTMLNodeParser（基于 beautifulsoup，默认标签集为 p/h1-h6/li/b/i/u/section）、JSONNodeParser、MarkdownNodeParser；后者包括 CodeSplitter（参数 chunk_lines / chunk_lines_overlap / max_chars）、SentenceSplitter（参数 chunk_size / chunk_overlap）、TokenTextSplitter、SemanticSplitterNodeParser（按句子间 embedding 相似度自适应选择断点，参数 buffer_size / breakpoint_percentile_threshold）[^23]。LlamaIndex 还将 node parser 定义为「接收 document 列表、切分为 Node 对象的抽象」，每个 node 是父文档的一个 chunk，父文档的 metadata、text、metadata templates 等属性会被子节点继承[^24]。

### 1.2 各策略的定义与关键参数

**固定长度切分（fixed-size / token chunking）**：按预设的 token 数顺序切分文本。Pinecone 指出 chunk 大小通常取 embedding 模型的最大上下文窗口（文中举例 llama-text-embed-v2 为 1024、text-embedding-3-small 为 8196），并提醒不同模型分词方式不同，需准确估算 token 数[^1]。arXiv:2410.13070 给出的固定长度基线定义更具体：按预定义或用户指定的「每 chunk 句子数」顺序切分文档，并在相邻 chunk 之间使用重叠句子以维持上下文连续性[^11]。Azure AI Search 的 Text Split skill 提供 textSplitMode 参数（pages 或 sentences）；maximumPageLength 仅在 pages 模式下生效，以字符为单位时最小 300、最大 50000、默认 5000，算法会尽量在句子边界断开，因此实际 chunk 可能略小于该值；pageOverlapLength 控制相邻页重叠，每一页以上一页末尾的该数量字符或 token 开头，设为 0 则无重叠；unit 参数（preview）可指定按 characters（默认）或 azureOpenAITokens 切分[^21]。

**递归切分（recursive character level）**：Pinecone 将其描述为「在按特定字符切分与使用更语义化切分器之间的良好折中」，同时尽可能保证固定 chunk 大小[^1]。Chonkie 的实现是「按可定制规则层级切分」[^6]。

**语义切分（semantic chunking）**：按语义相似度分组[^22]，Chonkie 版灵感来自 Greg Kamradt[^6]。LlamaIndex 的 SemanticSplitterNodeParser 按句子间 embedding 相似度自适应选择断点[^23]。semchunk 则自称采用一种「新颖的层级切分算法（hierarchical chunking algorithm）」，目标是在切分文本时尽可能保留局部语义上下文[^8]。

**结构感知切分（document structure-based）**：针对 PDF、DOCX、HTML、代码、Markdown、LaTeX 等格式，利用标题、表格、标签、Markdown 语法层级、LaTeX 章节与公式等结构信息生成更语义连贯的 chunk[^1]。Chonkie 的 CodeChunker 基于 AST 按代码结构切分、TableChunker 按行切分大型 markdown 表格并保留表头[^22]。arXiv:2402.05131v3 主张以文档结构元素（structural element components）为主要依据切分，而非仅按段落级切分，并称这种切分方式「无需调参即可得到最佳 chunk size」[^4]；该研究范围限定为美国 SEC 财务报告（10-K、10-Q、8-K），作者称据其所知这是首个针对文档理解、特别是财务报告处理的 chunking 系统性研究[^4]。

**神经/LLM 切分**：NeuralChunker 用微调 BERT 模型按语义漂移切分；SlumberChunker 通过 Genie 接口用生成式 LLM 做 agentic chunking[^22]。

### 1.3 不改变边界、改变编码的三类方法

**Late chunking**：先用长上下文嵌入模型对长文本的全部 token 做嵌入，切分发生在 transformer 之后、mean pooling 之前，因此得名 "late"[^10]。动机是：稠密向量检索系统通常在较短文本段上表现更好，实践者因此常把文档切成小块分别编码，但这样产生的 chunk 嵌入会丢失来自周围 chunk 的上下文信息，导致表示次优[^10]。该方法无需额外训练即可应用于多种长上下文嵌入模型[^10]。作者为 Jina AI 团队（Michael Günther, Isabelle Mohr, Daniel James Williams, Bo Wang, Han Xiao）[^10]。

**Contextual Retrieval**：在 embedding 之前为每个 chunk 前置一段针对该 chunk 的解释性上下文，形成 Contextual Embeddings，并对同一上下文构建 BM25 索引形成 Contextual BM25[^2]。Anthropic 明确将自身方法与若干既有「用上下文改进检索」的方案区分开，并列出这些方案：给 chunk 添加通用文档摘要（其实验显示收益非常有限）、HyDE 假设文档嵌入、summary-based indexing（评估后性能较低）[^2]。

**父子/小到大（small-to-big）**：LlamaIndex 把「解耦用于检索的 chunk 与用于合成的 chunk」列为构建生产级 RAG 的首要技术，理由是检索最优的 chunk 表示与合成最优的表示可能不同[^3]。官方给出两种实现：embed 文档摘要并链接到该文档下的 chunk（先文档级召回再 chunk 级召回）；或 embed 句子并链接到该句子周围的窗口[^3]。后者既能做更细粒度的相关上下文检索（因为 embed 巨大 chunk 会导致 lost in the middle 问题），又能保证 LLM 合成时有足够上下文[^3]。LlamaIndex 的 SentenceWindowNodeParser 将文档切成单句节点，并在 metadata 中保存前后各 window_size 句的「窗口」，配合 MetadataReplacementNodePostProcessor 在送入 LLM 前替换为上下文；HierarchicalNodeParser 将输入切成多个层级大小的节点，每个节点持有父节点引用，配合 AutoMergingRetriever 使用[^23]。

---

## 二、切分粒度如何影响检索质量

### 2.1 机制层面的三条路径

**（a）上下文破坏**。Anthropic 指出传统 RAG 的核心缺陷是「破坏上下文」：单个 chunk 往往缺乏足够上下文（如「公司营收环比增长 3%」未说明是哪家公司、哪个时期），导致检索失败或信息无法被有效使用[^2]。Late chunking 论文把 naive chunking 的失败模式命名为 lost context problem——长距离语义依赖跨 chunk 时，指代无法解析[^19]。

**（b）嵌入质量与上下文窗口约束**。chunk 的 token 数受嵌入模型 token 上限约束；不同 chunk size 可能导致不理想的检索结果；被检索 chunk 的 token 数还可能影响 LLM 生成答案的质量[^4]。Pinecone 给出的核心权衡是：chunk 要大到含有意义信息，又要小到能支撑低延迟的 RAG 与 agentic 工作负载；其经验法则是——若一段文本脱离上下文对人仍有意义，对语言模型也有意义[^1]。

**（c）下游 LLM 的 lost-in-the-middle**。Pinecone 指出长上下文模型仍存在 lost-in-the-middle 问题，即使文档能整体放入上下文，埋在中部的相关信息也可能被漏掉，因此仍需控制送入下游 LLM 的信息量[^1]。LlamaIndex 同样以 lost in the middle 解释为何不应 embed 巨大 chunk[^3]。

### 2.2 有具体数字的证据

**Late chunking 的 Berlin 示例（余弦相似度，jina-embeddings-v2-small）**：含 "Berlin" 的首句 naive 0.8486 vs late 0.8495（几乎无差异）；仅含指代 "Its" 的第二句 naive 0.7084 vs late 0.8249（提升约 0.117）；仅含指代 "the city" 的第三句 naive 0.7535 vs late 0.8498（提升约 0.096）[^19]。论文解释：late chunking 在切分前已让模型看到上下文，因此把 "Berlin" 的语义编码进了指代句的嵌入中[^19]。

**Late chunking 的 BEIR nDCG@10（固定 256 token 字符串块，jina-embeddings-v2-small-en）**：SciFact 传统 64.20% / late 66.10% / 不切分 63.89%；TRECCOVID 63.36% / 64.70% / 65.18%（不切分最高）；FiQA2018 33.25% / 33.84% / 33.43%；NFCorpus 23.46% / 29.98% / 30.40%（不切分最高）；Quora 三者均为 87.19%（无差异）[^20]。README 结论：所有数据集上 late chunking 都不低于传统切分；部分数据集上 late chunking 优于整文档单嵌入，另一些数据集则不切分最优；文档平均长度越长，late chunking 带来的 nDCG 提升越大[^20]。该 README 另指出 jina-embeddings-v2-base-en 支持 8192 token 输入（约十页标准文本）[^20]。

**语义切分 vs 固定长度切分（arXiv:2410.13070，F1@5 %）**：文档检索实验中，Miracl* 固定 69.45 / Breakpoint 81.89 / Clustering 67.35；NQ* 43.79 / 63.93 / 41.01；Scidocs* 16.82 / 17.60 / 19.87；Scifact* 35.27 / 36.27 / 35.70；BioASQ* 61.86 / 61.87 / 62.49；NFCorpus* 21.36 / 21.07 / 22.12；HotpotQA 90.59 / 87.37 / 84.79；MSMARCO 93.58 / 92.23 / 93.18；ConditionalQA 68.11 / 64.44 / 65.94；Qasper 90.99 / 89.27 / 90.77[^11]。作者观察到：固定长度切分在非拼接数据集上占优，语义切分在拼接数据集上更好；随着文档长度增加，两者差距缩小[^11]。证据检索实验中，固定长度在 5 个数据集中的 3 个上表现最好，但固定长度与两种语义切分的差异极小[^11]。作者总体结论是：语义切分带来的计算成本并未被一致的性能增益所证明合理[^11]。

该研究的实验设置：文档检索用 10 个 BEIR 派生数据集（其中六个短文档数据集被拼接成平均约 100 句的长合成文档），每数据集采样 100 条 query，k ∈ [1,3,5,10]；证据检索用 RAGBench 中含长文档与 ground-truth 证据句的数据集，指标是检索到的 top-k chunk 中出现的 ground-truth 证据句数量；答案生成用 gpt-4o-mini、top-5 检索 chunk 为输入，以语义相似度对比生成答案与标准答案[^11]。作者说明 Recall@k 与 NDCG@k 不适用于其设置（相关文档或证据句数量不固定），故报告 F1@5[^11]。所有报告结果使用的 embedder 为 dunzhang/stella_en_1.5B_v5，是在测试过的三个（依 MTEB Leaderboard 排名初选）中挑出的最佳[^11]。

### 2.3 只有定性主张、无公开数字的部分

**结构感知切分**：arXiv:2402.05131v3 声称基于元素类型的切分在财务报告场景「大幅提升」RAG 结果，并称其框架可评估不同元素类型对检索上下文与准确性的贡献[^4]。但摘要层面仅给出定性结论，未报告任何具体 chunk size 数值、overlap 设置或 Recall@k / nDCG / faithfulness 指标分数[^5]。作者机构为 Unstructured Technologies，即该研究与其商业文档解析产品同源，属需交叉验证的利益相关方主张[^4]。

**Contextual Retrieval**：Anthropic 报告该方法可将检索失败率降低 49%，与 reranking 结合后降低 67%[^2]。此为厂商自报数据，需注意其营销属性[^2]。

**semchunk**：README 声称「相比最接近的竞品，RAG 性能提升 15%」，并指向仓库内 Benchmarks 章节（该章节正文本次未能抓取，属项目方自述主张）[^8]。

**速度与包体积（Chonkie 自测）**：Token chunking（100K 篇文章）Chonkie 58 秒 / 4.82 MB/s，LangChain 1 分 10 秒 / 4 MB/s（慢 1.21x），LlamaIndex 50 分钟 / 0.09 MB/s（慢 51.7x）；Sentence chunking Chonkie 59 秒，LlamaIndex 3 分 59 秒（慢 4.05x），表中标注 LangChain 无对应 chunker；Recursive chunking Chonkie 1 分 19 秒，LangChain 2 分 45 秒（慢 2.09x），表中标注 LlamaIndex 无对应 chunker；Semantic chunking（sentence-transformers/all-minilm-l6-v2）Chonkie 默认设置 13 分 59 秒，LangChain 1 小时 13 秒（慢 4.35x），LlamaIndex 1 小时 24 分 15 秒（慢 6.07x）[^7]。包体积（默认安装）Chonkie 15 MiB、LangChain 80 MiB（约 5.3x）、LlamaIndex 171 MiB（约 11.4x）；含语义功能时 62 / 625 / 678 MiB[^7]。该基准由 Chonkie 项目方自行发布，属厂商自测，非第三方独立评测[^7]。

### 2.4 参数推荐值与默认值

Azure AI Search 文档明确给出面向 embedding 模型的推荐值：对文本 embedding 模型，page length 的通用建议是 512 tokens[^21]。LlamaIndex 文档示例默认参数为 SentenceSplitter(chunk_size=1024, chunk_overlap=20)，即默认以 token 计 1024 大小、20 重叠[^24]。Pinecone 建议固定长度切分取 embedding 模型最大上下文窗口[^1]。semchunk 支持 chunk overlap（按比例 <1 或绝对 token 数 >=1）与 chunk offsets 返回[^8]。

### 2.5 适用边界

Anthropic 补充说明：若知识库小于约 200,000 token（约 500 页材料），可直接把整个知识库放进 prompt，无需 RAG 或类似方法[^2]。这构成 chunking 适用边界的一个官方表述。

### 2.6 策略选择的影响变量

Pinecone 列出选择切分策略时需考虑的四个变量：被切分数据的类型、所用 embedding 模型、用户查询的长度与复杂度预期、检索结果在下游应用中的用途[^1]。LlamaIndex 补充两点：标准 RAG 栈（top-k 检索 + 基础文本切分）在文档数量规模化时表现不佳，需要借助结构化信息（元数据过滤、文档层级摘要 + 递归检索）提升检索精度[^3]；「动态按任务类型检索」是独立优化维度，top-k 相似度检索针对事实型问答优化，而摘要型、对比型查询需要不同的检索技术[^3]。LlamaIndex 另将 reranking 作为独立环节：稠密/混合检索器面向召回调优，需用更强的 cross-encoder 对 top-k 候选重排以提升精度[^3]。

---

## 三、工具生态与评估现状

Chonkie 自我定位为「轻量级 ingestion 库」，覆盖 fetch → chunk → refine → embed → 写入向量库的端到端流程[^6]。支持 Pipeline 组合：可链式 chunk_with / refine_with（如 recursive→semantic，再 overlap 与 embeddings 精修），并提供异步 arun 接口[^6]。宣称 32+（正文另一处写 45+）集成，开箱支持 56 种语言[^6]。提供自托管 REST API server（chonkie serve / docker compose up），端点覆盖所有 chunker、refinery 与 pipeline，pipeline 配置存于本地 SQLite[^6]。仓库提交历史显示 2,219 commits[^6]。所有 chunker 共享统一接口 chunk(text)、chunk_batch(texts)、直接调用 chunker(text)，以及异步变体 achunk / achunk_batch / achunk_document；文档称所有 chunker 均为线程安全，异步方法通过 asyncio.to_thread 在线程池中执行同步方法[^22]。文档给出「Availability」矩阵，说明不同 chunker 随安装方式（默认 / chonkie[all] / Chonkie JS / API）可用性不同[^22]。LlamaIndex 提供 Chunker 这一多用途 node parser 封装 chonkie 的 chunker，可用别名初始化（如 Chunker("recursive", chunk_size=2048)）或直接传入 chonkie chunker 实例[^23]。LangchainNodeParser 可把任意 LangChain text splitter 包装为 LlamaIndex node parser（示例使用 RecursiveCharacterTextSplitter）[^23]。

semchunk 与 tokenizer 无关：chunkerify() 可接受 Tiktoken 编码、Transformers/tokenizers 分词器、模型名（如 gpt-4、cl100k_base、isaacus/kanon-2-tokenizer）或自定义 token 计数函数；支持多进程（processes>1）与进度条；提供可选的 AI-powered chunking（传入 Isaacus enrichment 模型如 kanon-2-enricher 作为 chunking_model，需 Isaacus SDK 与 API key）[^8]。README 声称被 Docling、Microsoft Intelligence Toolkit 与 Isaacus API 采用，且「每月下载数百万次」[^8]。仓库有 196 commits，stars 约 667、最近 push 2026-06-13，另有社区维护的 Rust 移植 semchunk-rs[^8][^9]。

评估工具方面，GitHub 上存在多个 chunking 评估仓库：RAGChunkEval（12 stars，最后推送 2026-08-14，自述为评估和优化 RAG 系统中文档切片策略的基准工具）[^12]、RAG-Chunking-Benchmark（1 star，2026-06-25，自述使用语义检索、多指标评估与加权排序为不同文档类型推荐最优 chunking 方法）[^13]、chunktuner（2 stars，2026-06-21，面向用户自有语料的参数调优工具，提供 CLI、Python 库与 MCP server 三种形态）[^14]、rag-retrieval-evaluation-framework（1 star，2026-07-22，使用 ChromaDB 与 Sentence Transformers）[^15]、RAG-Scope-RU（9 stars，2026-10-17，俄语语料的 RAG 组件消融）[^16]。rag-chunking-eval 自述按 chunk size × overlap × splitter 网格评估 hit-rate@k 与 MRR[^9]。上述 chunk 评估类仓库 stars 均在个位数到十余量级，说明该细分方向尚无高热度事实标准工具[^9]。另有 chunky（189 stars，2026-10-17，带 chunk 可视化与策略对比）与 chonkiejs（367 stars，2026-10-03，Chonkie 的 TypeScript 移植）[^9]。

arXiv 列表页可正常抓取，2026-10-23 当日 cs.IR 共 27 条新提交；该 recent 列表页前 50 条标题中未出现 chunking 相关论文，说明 chunking 策略对比类工作并非 cs.IR 近期高频主题[^17]。

---

## 矛盾与不确定性

**1. 语义切分的价值存在直接冲突。** arXiv:2410.13070 的结论是语义切分的计算成本未被一致的性能增益证明合理，固定长度切分在多个数据集上不劣甚至更优[^11]；而 Pinecone 将 Semantic Chunking 列入其 content-aware 分类体系并作为推荐选项[^1]，Chonkie 以 SemanticChunker 为主打策略之一[^6][^22]，semchunk 以「保留局部语义上下文」为核心卖点并声称 RAG 性能提升 15%[^8]。这些立场未在同一基准上被调和。需注意 arXiv:2410.13070 的结论建立在特定设置上（单一 embedder dunzhang/stella_en_1.5B_v5、F1@5、部分数据集为拼接合成的长文档），其外推性有限[^11]。

**2. chunk size / overlap 与检索指标的量化关系证据严重不足。** 现有证据中仅 arXiv:2410.13070 提供具体数字（F1@5）[^11]。overlap 对指标的影响完全没有量化证据——Azure 文档只给出 pageOverlapLength 的参数定义与语义（每页以上一页末尾的该数量字符或 token 开头，设为 0 则无重叠）[^21]，LlamaIndex 只给出默认值 chunk_overlap=20[^24]，均无效果数字。Azure 的 512 tokens 推荐值[^21]、Pinecone 的「取模型最大上下文窗口」建议[^1]、LlamaIndex 的 1024/20 默认值[^24] 三者口径不一致，且都未附支撑性评测。**单一来源不足以支撑「主流策略效果对比」这一核心结论。**

**3. Contextual Retrieval 的 49% / 67% 数字未经独立验证。** 该数字为 Anthropic 自报[^2]。GitHub 上存在多个声称复现或扩展的项目，例如 tunahanaliozturk/contextual-retrieval-ablation 描述称在 BEIR 上对比 naive、sentence-window、contextual 与 late chunking 四种方案并做配对 bootstrap 显著性检验；alexPantoja90210/contextual-retrieval-reproduction 描述称以 737 chunks、248 queries 复现并称基线被精确复现（80.92 / 87.15 / 90.06）；YUALAB/contextual-retrieval-japanese 描述称检验该方法在日文场景是否有效；syedahmad0786/contextual-retrieval 描述称在人为构造歧义的语料上给出 recall@5 基准[^18]。但这些仓库 stars 均极低（0-33），属社区个人项目，且本次仅取得元数据、未读 README 正文，**其结论不可采信为独立验证**[^18]。

**4. contextual retrieval 与 late chunking 无法横向比较。** 两者分别有 Anthropic 的失败率下降数字[^2] 与 Jina 的 nDCG@10 数字[^20]，但基准、指标、语料均不同，并列呈现易造成误导。目前缺少同一基准下的对照证据。

**5. 结构感知切分的收益缺乏可核验数字，且来源有利益相关性。** arXiv:2402.05131v3 只给出「大幅提升」的定性表述，摘要中无任何 chunk size、overlap 或 Recall@k / nDCG / faithfulness 分数[^4][^5]；作者机构 Unstructured Technologies 与商业文档解析产品同源[^4]。该论文提交于 2024-02-05、v3 修订于 2024-03-16，学科分类 cs.CL[^5]。

**6. 中文 RAG 场景的切分证据几乎完全缺失。** 现有证据几乎全部基于英文语料（BEIR、Wikipedia、SEC 财报、Paul Graham Essays）[^7][^11][^20]。仅 JellyZhang0101/Rag-Eval-Lab 的简介提及中文（自述为中文 RAG 评估 dashboard，比较 chunk size、top-k 检索、答案质量与延迟），但 stars=1、未读正文[^9]。LlamaIndex 官方文档自陈 SemanticSplitterNodeParser 的局限：正则主要适用于英文句子，且 breakpoint percentile 阈值可能需要调参[^23]——这提示语义切分在中文上的可用性存疑，但**中文分词/标点边界对 chunking 的影响无任何实质证据**。RAG-Scope-RU 的语料限定为俄语，同样不能外推[^16]。

**7. 厂商自测数据需谨慎对待。** Chonkie 的 BENCHMARKS.md 由项目方自行发布，非第三方独立评测[^7]；其 FastChunker「100+ GB/s」与 SlumberChunker「S-tier chunk quality」均为厂商自述，未见第三方基准佐证[^22]。semchunk 的「每月下载数百万次」与「RAG 性能提升 15%」同为项目方自述，且 Benchmarks 章节正文未抓取[^8]。

**8. 长文档场景下「不切分」有时最优。** Late chunking 的 BEIR 结果显示，TRECCOVID 与 NFCorpus 上不切分的 nDCG@10 高于 late chunking 与传统切分（65.18% vs 64.70% / 63.36%；30.40% vs 29.98% / 23.46%）[^20]。这与 Anthropic「知识库小于约 200,000 token 可直接整库入 prompt」的表述[^2] 方向一致，但两者的阈值与适用条件（模型上下文长度、语料规模）不同，无法直接换算。

## 参考来源

[^1]: Chunking Strategies for LLM Applications (Pinecone Learn) — https://www.pinecone.io/learn/chunking-strategies/（web, 2025-06-28）
[^2]: Introducing Contextual Retrieval (Anthropic) — https://www.anthropic.com/news/contextual-retrieval（web）
[^3]: Building Performant RAG Applications for Production (LlamaIndex Docs) — https://docs.llamaindex.ai/en/stable/optimizing/production_rag/（web）
[^4]: Financial Report Chunking for Effective Retrieval Augmented Generation (arXiv:2402.05131v3) — https://arxiv.org/html/2402.05131v3（arxiv, 2024-02-05）
[^5]: Financial Report Chunking for Effective Retrieval Augmented Generation — arXiv abstract/metadata page (arXiv:2402.05131v3) — https://arxiv.org/abs/2402.05131（arxiv, 2024-02-05）
[^6]: chonkie-inc/chonkie — The lightweight ingestion library for fast, efficient and robust RAG pipelines (README) — https://github.com/chonkie-inc/chonkie（github）
[^7]: chonkie-inc/chonkie — BENCHMARKS.md（Chonkie vs LangChain vs LlamaIndex 速度与包体积） — https://raw.githubusercontent.com/chonkie-inc/chonkie/main/BENCHMARKS.md（github）
[^8]: isaacus-dev/semchunk — A fast, lightweight Python library for splitting text into semantically meaningful chunks (README) — https://github.com/isaacus-dev/semchunk（github）
[^9]: GitHub 仓库检索结果：chunking 生态候选库（stars / 最近 push / 简介） — https://github.com/search?q=chunking+evaluation+RAG+chunk+quality&type=repositories（github）
[^10]: Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models (arXiv:2409.04701v3) — https://arxiv.org/abs/2409.04701v3（arxiv, 2024-09-07）
[^11]: Is Semantic Chunking Worth the Computational Cost? — https://arxiv.org/abs/2410.13070（arxiv, 2024-10-16）
[^12]: sitong721/RAGChunkEval — benchmark toolkit for evaluating and optimizing document chunking strategies in RAG — https://github.com/sitong721/RAGChunkEval（github）
[^13]: harishkumard24/RAG-Chunking-Benchmark — modular benchmarking framework for RAG chunking strategies — https://github.com/harishkumard24/RAG-Chunking-Benchmark（github）
[^14]: shantanu-deshmukh/chunktuner — benchmark chunking strategies for your RAG corpus — https://github.com/shantanu-deshmukh/chunktuner（github）
[^15]: boddetijayanth22/rag-retrieval-evaluation-framework — benchmarking chunking strategies with ChromaDB and Sentence Transformers — https://github.com/boddetijayanth22/rag-retrieval-evaluation-framework（github）
[^16]: TheNarhawl/RAG-Scope-RU — ablation studies of RAG components including chunking on Russian corpora — https://github.com/TheNarhawl/RAG-Scope-RU（github）
[^17]: arXiv cs.IR recent submissions listing (Fri, 23 Oct 2026) — 检索通道可用性证据 — https://arxiv.org/list/cs.IR/recent（arxiv, 2026-10-23）
[^18]: GitHub 仓库搜索结果：Anthropic Contextual Retrieval 的复现/消融类项目（仅元数据） — https://github.com/tunahanaliozturk/contextual-retrieval-ablation（github）
[^19]: Late Chunking 论文正文 Table 1：Berlin 示例的 naive vs late chunking 余弦相似度（ar5iv HTML 全文） — https://ar5iv.labs.arxiv.org/html/2409.04701（arxiv, 2024-09-07）
[^20]: jina-ai/late-chunking README：BEIR 检索任务 nDCG@10 对比表 — https://raw.githubusercontent.com/jina-ai/late-chunking/main/README.md（github）
[^21]: Text Split skill - Azure AI Search (Microsoft Learn 官方文档) — https://learn.microsoft.com/en-us/azure/search/cognitive-search-skill-textsplit（web）
[^22]: Chonkie Docs — Chunkers Overview（官方文档） — https://docs.chonkie.ai/oss/chunkers/overview（web）
[^23]: LlamaIndex Docs — Node Parser Modules（官方文档） — https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/（web）
[^24]: LlamaIndex Docs — Node Parser Usage Pattern（官方文档） — https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/（web）