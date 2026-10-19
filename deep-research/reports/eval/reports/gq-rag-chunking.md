# RAG 中文档切分（chunking）主流策略与粒度对检索质量的影响

## 摘要

本报告基于 25 条证据，梳理 RAG 流水线中文档切分的主流策略与其对检索质量的影响机制。核心发现如下：切分是 RAG 预处理的关键步骤，其动机在于让嵌入模型能容纳数据并保证 chunk 本身含有可检索信息[^15]；主流策略可归为固定长度切分、基于分隔符/结构的递归切分、语义切分，以及「先编码后切分」的 late chunking 四类[^3][^8][^1]。粒度对检索质量的影响机制在定性层面有较一致的描述——chunk 过大导致主题混杂与 embedding 被「平均化」，过小则丢失上下文[^16][^1][^2]——但在**量化层面，现有证据不足以给出 chunk size / overlap 与 recall、precision、answer faithfulness 之间的具体数值关系**，且语义切分相对固定长度切分的收益被一项系统评估判定为高度依赖任务、不足以证明其额外计算成本[^5][^6]。此外，本报告在核验中发现一处未解决的参数矛盾（LlamaIndex `SentenceSplitter` 默认 overlap 的取值），已在「矛盾与不确定性」一节显式声明。

---

## 一、切分在 RAG 中的位置与动机

Jina AI 官方博客将 RAG 典型流水线定义为 chunking-embedding-retrieving-generating（切分-嵌入-检索-生成）[^2]。Anthropic 官方工程博客对预处理步骤的描述是：将知识库（语料）拆分为较小的文本块（chunk），通常不超过几百个 token；再用嵌入模型将 chunk 转为向量嵌入；存入向量数据库以支持语义相似度检索[^4]。Jina AI 博客进一步给出切分的具体动机：RAG 需要把文档切分为较小文本块（例如 512 token 以内），存入向量数据库，由嵌入模型生成向量表示[^2]。

Pinecone 官方博客明确说明 chunking 的两大必要性：一是让 embedding 模型能把数据装进其 context window（超出会被截断丢弃，导致重要上下文无法被检索到），二是保证 chunk 本身含有可供检索的信息[^15]。该文同时指出 chunking 的目标是让 chunk 大到足以包含有意义信息、小到能保证低延迟检索[^15]。Weaviate 官方博客则强调切分是 RAG 管线中最重要的决策之一，并指出 RAG 系统表现不佳时问题往往不在 retriever 而在 chunks（'the issue is often not the retriever—it's the chunks'）[^16]。

---

## 二、主流切分策略

### 2.1 策略分类框架

Jina AI 官方博客将 chunking 拆解为两个正交的核心问题：一是确定断点（breakpoints），即如何切分文档；二是每个 chunk 内部上下文的丢失[^3]。其中断点确定方式被列举为：固定 token 长度、固定句子数、或更高级的技术如 regex 或语义分割模型（semantic segmentation models）[^3]。这一「断点 + 上下文丢失」的二分框架是理解各策略差异的基础。

### 2.2 固定长度切分（fixed-size / fixed-length chunking）

论文将传统做法称为 fixed-size chunking，即把文档切成固定长度（如 200 tokens）的连续片段[^6]。这是最基础的策略，其断点由长度阈值决定，不依赖文本内容结构。

### 2.3 基于分隔符与结构的递归切分

LangChain 的实现代表了这一路线。`CharacterTextSplitter` 默认 `separator="\n\n"`，`is_separator_regex` 默认 False，按字符切分[^8]。`RecursiveCharacterTextSplitter` 默认 separators 为 `["\n\n", "\n", " ", ""]`，`keep_separator` 默认 True，递归尝试不同分隔符直至可用；其 `_split_text` 递归逻辑为：先选合适分隔符切分，再对过长片段用后续分隔符递归切分并合并[^8]。LangChain 的 `TextSplitter` 基类默认 `chunk_size=4000`、`chunk_overlap=200`，`length_function` 默认为 `len`（即按字符计长）；构造时校验 chunk_size 必须 >0，chunk_overlap 必须 >=0 且不得大于 chunk_size，否则抛 ValueError[^7]。

LlamaIndex 的 `SentenceSplitter` 走的是段落—句子—子句的层级路线：默认 `separator=" "`、`paragraph_separator="\n\n\n"`，`secondary_chunking_regex` 默认 `"[^,.;。？！]+[,.;。？！]?|[,.;。？！]"`（支持中英文标点）[^9]。其切分函数链为：先按段落分隔符切分，再用句子 tokenizer 切分；子句级回退使用 `secondary_chunking_regex`、`separator`、单字符切分[^9]。构造时校验 chunk_overlap 不得大于 chunk_size，否则抛 ValueError[^9]。

Haystack 的 `DocumentSplitter` 提供更细的切分单位枚举：默认 `split_by="word"`、`split_length=200`、`split_overlap=0`、`split_threshold=0`[^11]。`split_by` 支持取值：function、page、passage、period、word、line、sentence、token（token 模式用 tiktoken，默认 `tokenizer_encoding="o200k_base"`）[^11]。字符映射为：page→`"\f"`、passage→`"\n\n"`、period→`"."`、word→`" "`、line→`"\n"`；sentence 模式使用 NLTK 句子分词器[^11]。其他默认包括 `respect_sentence_boundary=False`、`language="en"`、`use_split_rules=True`、`extend_abbreviations=True`、`skip_empty_documents=True`[^11]。

RAGFlow 采用「Template-based chunking」（基于模板的切分），提供多种模板选项，并提供文本切分的可视化（Visualization of text chunking）以支持人工干预[^12]。其支持 Word/PPT/Excel/TXT/图片/PDF/扫描件/结构化数据/网页等异构数据源[^12]。

### 2.4 语义切分（semantic chunking）

Jina AI 博客给出 'semantic chunking' 的定义：将嵌入空间中余弦相似度高的句子分组，以尽量减少对语义单元的破坏[^3]。论文测试的第二种 chunker 是 breakpoint-based semantic chunker，即基于相邻句子间语义距离阈值检测断点[^6]。

### 2.5 Late chunking（先编码后切分）

Late chunking 是与上述断点策略正交的另一条路线。论文提出的方法为：先对长文本的所有 token 做嵌入，切分（chunking）在 transformer 模型之后、mean pooling 之前进行，故称 'late'[^1]。该方法利用长上下文嵌入模型，使每个 chunk 的嵌入保留完整上下文信息，在多种检索任务上取得更优结果[^1]。方法通用，可应用于多种长上下文嵌入模型，且无需额外训练；论文还提出针对嵌入模型的专用微调方法以进一步增强 late chunking 效果[^1]。

Jina AI 博客明确 late chunking 与边界/语义切分是正交的两个问题，可分别处理；并认为上下文丢失问题（第二问题）更为关键[^3]。late chunking 仍需边界线索（boundary cues），但关键区别在于使用时机：先编码整个文档（jina-embeddings-v3 为 8192 token 上下文窗口），再依据边界线索对每个 chunk 应用 mean pooling[^3]。博客称实验表明 late chunking 消除了对完美语义边界的需求，部分解决了断点问题[^3]。late chunking 已集成进 jina-embeddings-v3 API[^2]。

### 2.6 Contextual Retrieval（上下文增强路线）

Anthropic 提出的 'Contextual Retrieval' 方法包含两个子技术：Contextual Embeddings（上下文嵌入）与 Contextual BM25[^4]。该博客指出传统 RAG 方案在编码信息时会移除上下文（remove context），常导致系统无法从知识库检索到相关信息[^4]。博客声称该方法可将检索失败率降低 49%，结合 reranking 后降低 67%[^4]——此为厂商自报数据，需交叉验证[^4]。

### 2.7 检索用 chunk 与合成用 chunk 解耦

LlamaIndex 官方文档提出「解耦检索用 chunk 与合成用 chunk」的技术：用于检索的最优 chunk 表示可能与用于合成答案的最优 chunk 不同[^17]。该文档指出原始文本 chunk 可能含填充词/信息从而偏置 embedding 表示，或缺乏全局上下文而在相关查询到来时根本检索不到[^17]。一种具体做法是：embed 单个句子，再链接到该句子周围的窗口（window），以兼顾细粒度检索与足够的合成上下文[^17]。该文档明确将「embedding 巨型 chunk」与 'lost in the middle' 问题关联，作为采用句子级检索+窗口扩展的理由[^17]。

### 2.8 向量数据库是否内置切分器

证据显示，主流向量数据库通常不内置 chunker。Chroma 自述为「the open-source data infrastructure for AI」，核心 API 为 create_collection / add / query 等向量库操作[^13]。其 `collection.add` 直接接收 documents 列表，由 Chroma 负责 tokenization、embedding 与索引，未提及内置文档切分器（chunker）[^13]。Chroma 文档说明 `add` 接口要求每条记录有唯一字符串 id，并接收 documents / embeddings / metadatas 三类字段，且 'You must provide either documents, embeddings, or both.'，即文档以整条形式写入，Chroma 不负责切分[^21]。Chroma 官方文档列出的能力清单（Document Storage、Embeddings、Vector Search、Full-Text & Regex Search、Metadata Filtering、Multi-Modal Retrieval）中不含任何文本切分/chunking 功能；仅在 'Example Projects' 的 Code Search 条目中出现 'AST-aware chunking' 一词，作为示例项目描述，而非 Chroma 内置能力[^22]。其 Guides 页面内容为 LLM 应用入门教程，未提供任何关于文档切分策略或内置 chunker 的指导[^24]。LangChain 仓库 README 将自身定位为 'The agent engineering platform'，正文未提及 text splitter / chunking 相关组件（该功能位于 libs 子包中，README 未展开）[^23]。

---

## 三、切分粒度如何影响检索质量

### 3.1 定性机制

**chunk 过大。** Weaviate 官方博客给出定性结论：过大的 chunk 会把多个主题混在一起，产生被「平均化」的噪声 embedding，难以清晰代表任一主题，从而妨碍向量检索找到全部相关上下文[^16]。

**chunk 过小。** 论文指出，稠密向量检索系统通常在较短文本片段上表现更好，因为语义不易在嵌入中被过度压缩（over-compressed），因此实践者常将文档切分为较小 chunk 并分别编码[^1]。但论文同时指出：这样分别编码产生的 chunk 嵌入会丢失来自周围 chunk 的上下文信息，导致次优表示[^1]。

**小而聚焦的 chunk。** Weaviate 官方博客给出定性结论：小而聚焦的 chunk 能捕获单一清晰概念，产生更精确的 embedding，更利于系统找到正确信息[^16]。该文提出 chunk 需同时满足两个目标：易于被向量搜索命中，同时给 LLM 足够上下文生成有用答案[^16]。

**上下文丢失问题（The Lost Context Problem）。** Jina AI 博客描述：简单 RAG 流水线会破坏长距离上下文依赖；当相关信息分散在多个 chunk 时，把文本片段脱离上下文会导致其失效[^2]。博客举例：将维基百科文章按句子切分后，'its'、'the city' 等指代 'Berlin' 的短语因 Berlin 只出现在首句而难以被嵌入模型正确关联，产生低质量向量表示[^2]。

**经验法则。** Pinecone 官方博客给出定性经验法则：如果一段文本脱离上下文仍能被人理解，那么它对语言模型也同样可理解（'if the chunk of text makes sense without the surrounding context to a human, it will make sense to the language model as well'）[^15]。该文指出 chunk 过小或过大都会导致检索结果不精确或漏检相关内容，但本次抓取未能读到该文后续的 'Figuring out the best chunking strategy' 章节，故未取得任何具体数值建议[^15]。Weaviate 博客的具体数值建议章节同样在本次抓取中被截断，未读到[^16]。

### 3.2 语义切分相对固定长度切分的收益评估

论文由 Vectara, Inc. 与 University of Wisconsin–Madison 的作者（Renyi Qu, Ruixuan Tu, Forrest Bao）于 2024-10-16 提交至 arXiv（cs.CL; cs.IR）[^5]。研究系统评估 semantic chunking 相对 fixed-size chunking 的有效性，使用三类检索相关任务：document retrieval、evidence retrieval、retrieval-based answer generation[^5]。论文结论：semantic chunking 带来的计算成本并未被一致的性能提升所证明（'the computational costs associated with semantic chunking are not justified by consistent performance gains'）[^5]。论文指出 semantic chunking 的优势高度依赖任务（'highly task-dependent'），且往往不足以证明其额外计算开销的合理性[^5]。

该论文的评估设计为三类代理任务（proxy tasks）：(1) document retrieval 衡量识别相关文档的能力；(2) evidence retrieval 衡量定位 ground-truth evidence 的能力；(3) answer generation 测试生成模型使用检索到的 chunk 所产出答案的质量[^6]。论文指出缺乏可直接用于比较 chunking 策略的数据，因此采用间接评估设计[^6]。

### 3.3 长上下文模型对切分必要性的影响

论文分析语言模型在需要从输入上下文中定位相关信息的两类任务上的表现：多文档问答（multi-document QA）与键值检索（key-value retrieval）[^18]。研究发现：当改变相关信息在上下文中的位置时，模型性能会显著下降，说明当前语言模型并不能稳健地利用长输入上下文中的信息[^18]。性能通常在相关信息位于输入上下文的开头或结尾时最高，而当模型必须访问长上下文中间位置的相关信息时性能显著下降——即使对显式支持长上下文的模型也是如此[^18]。该结论直接反驳了『长上下文模型已能有效利用全部上下文、因而无需切分/检索』的观点[^18]。

另一篇论文对 RAG 与长上下文（LC）LLM 进行了全面对比，使用三个最新 LLM 在多个公开数据集上进行基准测试[^19]。结果显示：在资源充足（resourced sufficiently）时，长上下文（LC）在平均性能上持续优于 RAG[^19]。但 RAG 显著更低的成本（significantly lower cost）仍是一项明显优势[^19]。论文提出 Self-Route 方法：基于模型自我反思（self-reflection）将查询路由到 RAG 或 LC，在保持与 LC 相当性能的同时显著降低计算成本[^19]。该论文部分支持『长上下文削弱 RAG 必要性』的观点（性能上 LC 更优），但同时指出成本因素使 RAG 仍具价值，因此并非完全否定 RAG[^19]。

### 3.4 嵌入模型上下文窗口的约束

Voyage AI 文档显示，当前文本 embedding 模型（voyage-4-large / voyage-4 / voyage-4-lite / voyage-code-4 / voyage-finance-2 / voyage-law-2）的 context length 为 32,000 tokens（voyage-law-2 为 16,000）[^14]。voyage-4 系列支持 1024（默认）、256、512、2048 多种 embedding 维度，且同系列 embedding 互相兼容[^14]。voyage-4-large 被描述为通用与多语言检索质量最佳；voyage-4-lite 针对延迟与成本优化[^14]。文档列出旧版模型（voyage-3 系列等）context length 亦多为 32,000 tokens，并建议迁移到新模型[^14]。这一窗口容量是切分粒度决策的上界约束之一。

---

## 四、矛盾与不确定性

### 4.1 未解决的参数矛盾：LlamaIndex `SentenceSplitter` 默认 overlap

证据 [9] 显示 LlamaIndex 的 `SentenceSplitter` 默认 `chunk_size=DEFAULT_CHUNK_SIZE`、`chunk_overlap=SENTENCE_CHUNK_OVERLAP=200`（token 计）[^9]。但证据 [10] 显示 LlamaIndex 全局常量 `DEFAULT_CHUNK_OVERLAP = 20`（tokens）[^10]，且证据 [25] 明确指出：`TokenTextSplitter` 从 constants 导入 `DEFAULT_CHUNK_OVERLAP` 与 `DEFAULT_CHUNK_SIZE`，其 `chunk_overlap` 字段默认值为 `DEFAULT_CHUNK_OVERLAP`（20），`__init__` 的 `chunk_overlap` 形参默认值同样为 `DEFAULT_CHUNK_OVERLAP`（20），因此全局常量 `DEFAULT_CHUNK_OVERLAP=20` 实际服务于 `TokenTextSplitter`，而非 `SentenceSplitter`[^25]。

两条证据对同一框架同一参数给出 200 与 20 两个互斥数值，且都声称来自官方源码。证据 [25] 的表述实际上为证据 [9] 的 200 提供了「20 不属于 SentenceSplitter」的旁证，但并未直接确认 `SENTENCE_CHUNK_OVERLAP` 的具体取值。**本报告无法在现有证据下判定 `SentenceSplitter` 默认 overlap 究竟为 200 还是 20，故不将其作为确定事实写入正文。** 该矛盾未解决即无法给出该框架的确定性默认参数。

### 4.2 量化关系的证据缺口

本课题的核心问题「切分粒度如何影响检索质量」，在现有证据中**没有任何量化证据支撑**。具体而言：

- 现有关于语义切分评估的证据 [5] 与 [6] 实为同一篇论文（arXiv:2410.13070）的两个来源（摘要页与 HTML 全文），**不构成两个独立来源**，且只给出「semantic chunking 计算成本未被性能提升证明」的定性结论，未报告任何 chunk size / overlap 的具体数值与 recall、precision、answer faithfulness 等指标的变化[^5][^6]。
- Pinecone[^15] 与 Weaviate[^16] 两篇博客的具体数值建议章节在本次抓取中均被截断，未取得任何具体数值建议。
- Anthropic 声称的「检索失败率降低 49%、结合 reranking 后降低 67%」为厂商自报数据，需交叉验证，且该数据针对的是 Contextual Retrieval 方法整体，而非单纯的切分粒度变量[^4]。

因此，**「chunk size / overlap 与 recall、precision、answer faithfulness 之间的量化关系」现有公开资料不足以确认**。任何关于「最优 chunk size 为 N token」的断言均超出本报告证据范围。

### 4.3 其他不确定性

- 证据 [6] 提到论文测试了三种 chunker，但第三种「正文被截断未读到」，故本报告只能确认前两种（fixed-size chunker 与 breakpoint-based semantic chunker）[^6]。
- 证据 [12] 中 RAGFlow README 的 Latest Updates 列出 2026 年的日期（2026-09-29、2026-08-19、2026-07-02），与当前时间线存在明显异常，该时间戳的可靠性存疑，本报告仅如实转述其内容而不对其时间真实性背书[^12]。
- 证据 [14] 未标注文档日期，其模型清单的时效性无法确认[^14]。

---

## 参考来源

[^1]: Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models — https://arxiv.org/abs/2409.04701v3 (arxiv, 2024-09-07)
[^2]: Late Chunking in Long-Context Embedding Models (Jina AI 官方博客 Part I) — https://jina.ai/news/late-chunking-in-long-context-embedding-models/ (web, 2024-08-22)
[^3]: What Late Chunking Really Is & What It's Not: Part II (Jina AI 官方博客) — https://jina.ai/news/what-late-chunking-really-is-and-what-its-not-part-ii/ (web, 2024-10-03)
[^4]: Introducing Contextual Retrieval (Anthropic 官方工程博客) — https://www.anthropic.com/news/contextual-retrieval (web, 2024-09-19)
[^5]: Is Semantic Chunking Worth the Computational Cost? — https://arxiv.org/abs/2410.13070 (arxiv, 2024-10-16)
[^6]: Is Semantic Chunking Worth the Computational Cost? (HTML full text, Introduction) — https://arxiv.org/html/2410.13070v1 (arxiv, 2024-10-16)
[^7]: langchain_text_splitters/base.py — TextSplitter 基类默认参数 — https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/text-splitters/langchain_text_splitters/base.py (github)
[^8]: langchain_text_splitters/character.py — CharacterTextSplitter 与 RecursiveCharacterTextSplitter — https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/text-splitters/langchain_text_splitters/character.py (github)
[^9]: llama_index/core/node_parser/text/sentence.py — SentenceSplitter — https://raw.githubusercontent.com/run-llama/llama_index/main/llama-index-core/llama_index/core/node_parser/text/sentence.py (github)
[^10]: llama_index/core/constants.py — 默认 chunk 常量 — https://raw.githubusercontent.com/run-llama/llama_index/main/llama-index-core/llama_index/core/constants.py (github)
[^11]: haystack/components/preprocessors/document_splitter.py — DocumentSplitter — https://raw.githubusercontent.com/deepset-ai/haystack/main/haystack/components/preprocessors/document_splitter.py (github)
[^12]: infiniflow/ragflow README — 模板化切分与近期更新 — https://raw.githubusercontent.com/infiniflow/ragflow/main/README.md (github)
[^13]: chroma-core/chroma README — 定位为向量数据库 — https://raw.githubusercontent.com/chroma-core/chroma/main/README.md (github)
[^14]: Voyage AI Embeddings Documentation (Model Choices) — https://docs.voyageai.com/docs/embeddings (web)
[^15]: Chunking Strategies for LLM Applications (Pinecone Learn) — https://www.pinecone.io/learn/chunking-strategies/ (web, 2025-06-28)
[^16]: Chunking Strategies for RAG (Weaviate Blog) — https://weaviate.io/blog/chunking-strategies-for-rag (web)
[^17]: Building Performant RAG Applications for Production (LlamaIndex Docs) — https://docs.llamaindex.ai/en/stable/optimizing/production_rag/ (web)
[^18]: Lost in the Middle: How Language Models Use Long Contexts — https://arxiv.org/abs/2307.03172v3 (arxiv, 2023-07-06)
[^19]: Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach — https://arxiv.org/abs/2407.16833 (arxiv, 2024-07-23)
[^20]: chroma-core/chroma — GitHub repository README — https://github.com/chroma-core/chroma (github)
[^21]: Chroma Docs — Adding Data (collections/add-data) — https://docs.trychroma.com/docs/collections/add-data (web)
[^22]: Chroma Docs — Introduction (overview/introduction) — https://docs.trychroma.com/docs/overview/introduction (web)
[^23]: langchain-ai/langchain — GitHub repository README — https://github.com/langchain-ai/langchain (github)
[^24]: Chroma Docs — Guides (docs.trychroma.com/guides) — https://docs.trychroma.com/guides (web)
[^25]: llama_index/core/node_parser/text/token.py (TokenTextSplitter source) — https://raw.githubusercontent.com/run-llama/llama_index/main/llama-index-core/llama_index/core/node_parser/text/token.py (github)