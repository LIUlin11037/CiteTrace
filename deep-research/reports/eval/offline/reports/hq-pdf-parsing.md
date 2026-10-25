# PDF/表格类文档构建 RAG：解析环节的难点与主流流程

## 摘要

本报告基于本地知识库的 13 条证据撰写。核心结论是一个**证据边界的结论**：知识库能够较完整地刻画 RAG 流水线中 **chunking 及其之后**的阶段（切分在流水线中的位置、主流切分器的默认参数、GraphRAG 的六阶段索引数据流、检索与生成侧的评测指标），但**对 chunking 之前的「文档解析」环节——PDF 为何是视觉格式而非语义格式、版面分析、阅读顺序重建、表格结构识别（TSR）、解析器的失败模式与量化瓶颈、主流解析工具的能力对比——库内均无覆盖**。因此，本课题「解析环节的难点是什么」在现有证据下**无法给出有引用支撑的实质回答**；「主流处理流程」只能给出**从切分开始的后半段**，前半段（解析→版面/表格结构化）在库内缺失，无法拼出完整链路。下文逐项说明库内有什么、缺什么，并显式标注每一处缺口。

---

## 一、解析环节在 RAG 流水线中的位置：库内能确认的部分

库内证据一致地把 RAG 流水线描述为「切分—嵌入—检索—生成」四阶段，并把切分定位为解析之后的一环：

- Jina AI 官方博客将 RAG 典型流水线定义为 chunking-embedding-retrieving-generating（切分-嵌入-检索-生成）[^2][^5]。
- Anthropic 官方工程博客把预处理描述为：将知识库（语料）拆分为较小的文本块（chunk），通常不超过几百个 token；再用嵌入模型将 chunk 转为向量嵌入；存入向量数据库以支持语义相似度检索[^2][^5]。
- Jina AI 博客给出的切分动机是：RAG 需要把文档切分为较小文本块（例如 512 token 以内），存入向量数据库，由嵌入模型生成向量表示[^5]。

需要明确的是：上述证据**只把 chunking 定位为 RAG 流水线的一环，位于文档解析之后，但未描述解析环节本身的定义或难点**[^2]。也就是说，「解析在前、切分在后」这个顺序关系在库内可确认，但「解析这一步做什么、怎么做、难在哪」在库内没有对应段落。

GraphRAG 的官方索引流程提供了另一种阶段划分，共 6 个 phase：Phase 1 Compose TextUnits、Phase 2 Document Processing、Phase 3 Graph Extraction、Phase 4 Graph Augmentation、Phase 5 Community Summarization、Phase 6 Text Embedding[^6]。其中 Phase 1 将文本切分为 TextUnit，chunk 大小以 token 计、用户可配置，默认 1200 tokens[^6]。官方文档将 GraphRAG 索引包定义为一个数据管道与转换套件，旨在用 LLM 从非结构化文本中提取有意义的结构化数据[^6]。这里的 Phase 2「Document Processing」在名称上最接近解析环节，但库内证据**只给出阶段名称，未展开该阶段的具体做法**（是否包含 PDF 转文本、版面还原、表格转 Markdown/HTML 等），现有公开资料不足以确认。

---

## 二、解析环节的难点：库内无有效证据

这是本课题的核心问题，也是证据缺口最严重的地方。逐条说明：

**2.1 PDF 的格式特性与解析难点的定义性内容：库内无覆盖。**
现有两条与「解析」沾边的证据均自述不能支撑该问题：RAGFlow 相关片段仅枚举输入数据源类型（Word/PPT/Excel/TXT/图片/PDF/扫描件/结构化数据/网页等异构数据源），并明确声明「未涉及 PDF 格式特性、版面分析、阅读顺序重建或表格结构识别的任何定义」[^1]；Jina/Anthropic 的流水线描述同样自述「未描述解析环节本身的定义或难点」[^2]。因此，「PDF 为何是视觉格式而非语义格式」「版面分析」「阅读顺序重建」「表格结构识别（TSR）」这几个核心概念的标准定义与经典出处，**当前知识库未收录**。

**2.2 解析器的实证失败模式与量化瓶颈：库内无覆盖。**
多栏阅读顺序错乱、跨页表格、合并单元格、扫描件 OCR 误差、公式图表丢失、解析误差向检索/生成传播放大——这些具体失败模式在库内没有任何 arxiv 论文级证据，也没有任何量化瓶颈数字。唯一沾边的证据是 Qwen-VL-Plus / Qwen-VL-Max 的基准分数（DocVQA 93.1%、ChartQA 79.8%、AI2D 79.3%、TextVQA 79.5%、MMMU 51.4%、MathVista 51.0%、MM-Bench-CN 75.1%）[^3]，但该证据自身标注为**弱证据**：DocVQA/ChartQA 在此仅作为模型基准分数出现，未讨论文档解析器的失败模式或瓶颈，不能作为本子问题的结论依据[^3]。此外该报告声明，所引性能数字绝大多数来自论文或厂商自述，缺少独立第三方基准交叉验证[^3]。

**2.3 主流解析工具的能力对比：库内无覆盖。**
PyMuPDF、pdfplumber、Unstructured、Marker、MinerU、Docling、Table Transformer、PaddleOCR/PP-StructureV3 等文档解析工具的 stars、近期提交活跃度、输入输出格式、表格/公式支持、许可证与部署成本，库内均无数据。库内唯一的 GitHub 生态热度证据来自 Agentic RAG 报告，其仓库清单为 langgenius/dify（158443 stars、25058 forks）、langchain-ai/langchain（147842 stars、24795 forks）、infiniflow/ragflow（92201 stars、10985 forks）、langchain-ai/langgraph（43254 stars、7354 forks）、bytedance/deer-flow（83852 stars、11680 forks）、mem0ai/mem0（67164 stars、7903 forks）、ruvnet/ruflo（74633 stars、8874 forks）、headroomlabs-ai/headroom（75321 stars、5846 forks），抓取时点为 2026-10-14[^4]。该清单**明确不含**上述任何文档解析工具[^4]，对解析工具选型零支撑。

---

## 三、主流处理流程：库内可还原的后半段

虽然解析环节缺失，库内对「解析之后」的流程有较扎实的参数级证据，可还原为以下阶段。

**3.1 切分（chunking）阶段。** 切分的目标是把文档拆成不超过几百 token 的文本块[^2][^5]。库内记录了四类主流切分器的默认配置：

- **LangChain**：`TextSplitter` 基类默认 `chunk_size=4000`、`chunk_overlap=200`，`length_function` 默认为 `len`（即按字符计长）[^8]；`RecursiveCharacterTextSplitter` 默认 separators 为 `["\n\n", "\n", " ", ""]`，`keep_separator` 默认 True，递归尝试不同分隔符直至可用[^8]。
- **LlamaIndex**：`SentenceSplitter` 走段落—句子—子句层级路线，默认 `separator=" "`、`paragraph_separator="\n\n\n"`，`secondary_chunking_regex` 默认 `"[^,.;。？！]+[,.;。？！]?|[,.;。？！]"`（支持中英文标点）[^8]。
- **Haystack**：`DocumentSplitter` 默认 `split_by="word"`、`split_length=200`、`split_overlap=0`、`split_threshold=0`；`split_by` 支持 function、page、passage、period、word、line、sentence、token，token 模式用 tiktoken，默认 `tokenizer_encoding="o200k_base"`[^8]。
- **GraphRAG**：依据仓库 main 分支 defaults.py，分块类型 ChunkerType.Tokens、分块大小 1200、分块重叠 100、编码模型 o200k_base[^7]；社区报告 max_length 默认 2000、max_input_length 默认 8000，默认 embedding 模型 text-embedding-3-large，默认 completion 模型 gpt-4.1[^7]。官方文档明确声明：许多配置值有默认值但不在文档中复述，需直接参考代码中的 constants[^7]。

**3.2 嵌入与索引阶段。** chunk 由嵌入模型转为向量嵌入并存入向量数据库以支持语义相似度检索[^2][^5]。GraphRAG 的 Phase 6 即 Text Embedding[^6]。

**3.3 检索阶段。** Amazon Bedrock 知识库查询默认返回最多 5 条结果，每条对应一个 source chunk；`numberOfResults` 参数设置返回结果的最大数量，若启用分层分块，该参数映射到子 chunk 数量；搜索类型包括 Default（由 Bedrock 决定）、Hybrid（向量嵌入语义搜索 + 原始文本搜索）、Semantic（仅向量嵌入），Hybrid 搜索仅支持 Amazon RDS、Amazon OpenSearch Serverless 和 MongoDB 向量存储[^13]。

**3.4 评测阶段。** 检索侧核心指标为排序质量类（NDCG@k、MAP@K、Recall@K、Precision@K、MRR）与上下文相关性类（Context Precision、Context Recall、Context Entities Recall）；生成侧核心指标为忠实性/接地性（Faithfulness、Groundedness）、答案相关性、完整性（Response Completeness）与噪声敏感度（Noise Sensitivity）[^12]。RAGAS 的 Context Precision 评估检索器将相关 chunk 排在无关 chunk 之前的能力，计算方式为上下文中每个 chunk 的 precision@k 的均值，公式为 `Context Precision@K = Σ_{k=1}^{K}(Precision@k × v_k) / (top K 结果中相关项总数)`，其中 v_k ∈ {0,1} 为 rank k 的相关性指示[^11]。

**3.5 流程前半段的缺口。** 上述 3.1–3.4 覆盖的是「切分→嵌入→检索→评测」。**从 PDF 到可检索 chunk 的标准流水线中，「解析」与「版面/表格结构化」这两个前置阶段在库内完全缺失**，现有证据无法拼出完整链路。表格处理策略（摘要化/行级化/文本化）与解析质量评估指标（如表格还原准确率、TEDS、阅读顺序正确率）同样无任何证据：库内评测材料全部针对检索侧与生成侧，无针对解析/分块产物质量的指标[^11][^12]。

---

## 四、矛盾与不确定性

**4.1 核心缺口（q1：解析难点的定义性内容）。** PDF 为何是视觉格式而非语义格式、版面分析、阅读顺序重建、表格结构识别（TSR）的学术/官方标准定义与经典出处，**当前知识库未收录**。现有两条 q1 证据（RAGFlow 数据源枚举[^1]、Jina/Anthropic 对 chunking 流水线的描述[^2]）均自述未涉及这些定义，不能支撑 q1 任何结论。

**4.2 核心缺口（q2：实证失败模式与量化瓶颈）。** 2023–2026 arxiv 文献中 PDF/表格解析的实证失败模式与量化瓶颈（多栏阅读顺序错乱、跨页表格、合并单元格、扫描件 OCR 误差、公式图表丢失、解析误差向检索/生成传播放大），**当前知识库未收录**。唯一沾边的 Qwen-VL 基准分数[^3]已被其来源自标为不能作为结论依据，且缺少独立第三方交叉验证[^3]。无任何 arxiv 论文级证据，也无任何量化瓶颈数字。

**4.3 核心缺口（q3：解析工具对比）。** PyMuPDF/pdfplumber、Unstructured、Marker、MinerU、Docling、Table Transformer、PaddleOCR/PP-StructureV3 的 stars、近期提交活跃度、输入输出格式、表格/公式支持、许可证与部署成本，**当前知识库未收录**。库内生态热度清单明确不含这些工具[^4]，对 q3 零支撑。

**4.4 核心缺口（q4：完整流水线与表格策略）。** 从 PDF 到可检索 chunk 的标准流水线阶段划分中，解析与表格结构化两个前置阶段完全缺失，无法拼出完整流水线；表格处理策略与解析质量评估指标的具体做法**当前知识库未收录**。库内评测证据全部针对检索侧与生成侧[^11][^12]，无解析环节指标。

**4.5 库内已有材料中是否有 chunking 之前的解析/预处理描述。** 在 Jina AI、Anthropic、GraphRAG、RAGFlow、LangChain/LlamaIndex/Haystack 切分文档中，**未发现任何段落描述 PDF 转文本、版面还原、表格转 Markdown 或 HTML 的具体做法**；GraphRAG 的 Phase 2「Document Processing」仅有阶段名称[^6]，无展开内容。离线模式下无法补充取证。

**4.6 库内评测材料中是否有针对解析/分块产物质量的指标。** RAGAS、RAGChecker、Azure RAG evaluators、Amazon Bedrock evaluations、LlamaIndex Retrieval Evaluation 相关证据中，**未发现任何针对 chunk 完整性、表格还原正确率、上下文召回中的解析丢失归因的指标**；现有指标均为检索排序质量与生成质量[^11][^12]。离线模式下无法补充取证。

**4.7 一处与解析无直接关系但影响流程参数确定性的矛盾。** LlamaIndex `SentenceSplitter` 的默认 overlap 在库内证据间存在冲突：一处显示 `chunk_overlap=SENTENCE_CHUNK_OVERLAP=200`（token 计）[^10]，另一处显示全局常量 `DEFAULT_CHUNK_OVERLAP = 20`（tokens）[^10]；原报告判定无法在现有证据下确定其为 200 还是 20，故不作为确定事实[^10]。此外，库内没有任何量化证据支撑「chunk size / overlap 与 recall、precision、answer faithfulness 之间的量化关系」，Anthropic 声称的「检索失败率降低 49%、结合 reranking 后降低 67%」为厂商自报数据、需交叉验证，且针对 Contextual Retrieval 方法整体而非单纯切分粒度变量[^9]。

**4.8 总体判断。** 在离线知识库边界内，本课题只能确认「解析位于切分之前」这一顺序关系，以及切分之后各阶段的参数与指标；**解析环节本身的难点、失败模式、工具选型与表格处理策略，均超出当前知识库覆盖范围**，需补充 arxiv 论文与解析工具官方文档后方可作答。

## 参考来源

[^1]: RAG 中文档切分（chunking）主流策略与粒度对检索质量的影响 — RAGFlow 数据源与模板切分 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-rag-chunking.md · ### 2.3 基于分隔符与结构的递归切分 · L27-36（kb）
[^2]: RAG 中文档切分（chunking）主流策略与粒度对检索质量的影响 — 切分在 RAG 流水线中的位置 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-rag-chunking.md · ## 一、切分在 RAG 中的位置与动机 · L9-13（kb）
[^3]: 2025年多模态大模型的主流架构路线 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-multimodal.md · ## 六、矛盾与不确定性 · L80-84（kb）
[^4]: Agentic RAG：定义、架构关键差异与工程现状（生态热度参考节） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-agentic-rag.md · ### 4.5 生态热度参考 · L133-138（kb）
[^5]: RAG 中文档切分（chunking）主流策略与粒度对检索质量的影响 — 切分在 RAG 中的位置与动机 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-rag-chunking.md · ## 一、切分在 RAG 中的位置与动机 · L9-11（kb）
[^6]: GraphRAG 深度研究报告：问题定位与索引构建流程 — 六阶段数据流 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-graph-rag.md · ### 2.2 六阶段数据流 · L53-63（kb）
[^7]: GraphRAG 深度研究报告：问题定位与索引构建流程 — 关键默认参数 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-graph-rag.md · ### 2.6 关键默认参数 · L87-108（kb）
[^8]: RAG 中文档切分（chunking）主流策略与粒度对检索质量的影响 — 基于分隔符与结构的递归切分 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-rag-chunking.md · ### 2.3 基于分隔符与结构的递归切分 · L27-35（kb）
[^9]: RAG 中文档切分（chunking）主流策略与粒度对检索质量的影响 — 量化关系的证据缺口 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-rag-chunking.md · ### 4.2 量化关系的证据缺口 · L101-107（kb）
[^10]: RAG 中文档切分（chunking）主流策略与粒度对检索质量的影响 — 未解决的参数矛盾：LlamaIndex SentenceSplitter 默认 overlap — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-rag-chunking.md · ### 4.1 未解决的参数矛盾：LlamaIndex SentenceSplitter 默认 overlap · L95-99（kb）
[^11]: 如何评测一个 RAG 系统：检索侧与生成侧的指标拆解 — 上下文相关性指标 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-rag-eval.md · ### 2.2 上下文相关性指标（无 ground truth 或部分无标签时） · L36-44（kb）
[^12]: 如何评测一个 RAG 系统：检索侧与生成侧的指标拆解 — 摘要（三条评测路线与覆盖缺口） — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-rag-eval.md · ## 摘要 · L3-8（kb）
[^13]: 如何评测一个 RAG 系统：检索侧与生成侧的指标拆解 — Bedrock 检索配置与 chunk 返回数量 — C:/Users/workbien/Desktop/agent/deep-research/reports/eval/reports/gq-rag-eval.md · ## 矛盾与不确定性 · L172-201（kb）