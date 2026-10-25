# 用 PDF/表格类文档构建 RAG：解析环节的难点与主流处理流程

## 摘要

PDF 与表格类文档进入 RAG 之前必须经过「解析」：把版面、阅读顺序、表格结构、公式等还原成结构化表示。现有证据把这一环节的难点集中在四处：(1) PDF 本身不携带语义结构，文本 token 只有字符串与页面坐标，语义单元必须由版面模型重建[^1]；(2) 表格结构恢复是独立且未解决的子问题，涉及无边框线、跨行跨列、表头层级等[^1][^17]；(3) 阅读顺序、跨页段落与跨页表格的归属关系需要额外判定[^1][^4]；(4) 扫描件与老旧文档使 OCR 成为瓶颈，且成本高[^1][^9]。主流处理流程存在两大范式：pipeline 式（版面分析 → OCR → 公式/表格识别 → 阅读顺序估计）与端到端 VLM（直接输出 Markdown）[^3][^25]。工程实现上，Docling、MinerU、Marker、PaddleOCR、olmOCR 等系统在「模型 + 规则后处理」的骨架上分化出不同的模型组合与性能取舍[^1][^5][^6][^8][^9][^11]。需要明确声明：现有证据**没有任何一条**把解析错误率与下游检索召回率或 QA 正确率做定量关联，因此「解析质量决定 RAG 上限」目前是断言而非有据结论。

---

## 一、解析环节到底难在哪

### 1.1 输入侧：PDF 只提供 token 与坐标，不提供结构

Docling 技术报告把 PDF backend 的两项基本需求写得非常具体：a) 取得每页全部文本内容及其几何坐标；b) 渲染出与 PDF 阅读器一致的页面视觉表示[^1]。也就是说，解析器从 PDF 拿到的原始材料只有「程序化文本 token（字符串内容 + 页面坐标）」和每页位图，段落、标题、列表、表格这些语义单元在 PDF 里并不存在[^1]。

这一约束直接导致现有开源 Python PDF 解析库的失败模式。Docling 作者称这些库均遇到重大障碍：许可限制（如 pymupdf）、速度差，或不可恢复的质量问题，例如「相距很远的文本 token 或表格列被合并成同一文本单元」（pypdfium、PyPDF）[^1]。Docling 因此基于底层 qpdf 库自研 docling-parse 作为默认 backend，仅把 pypdfium 留作备选，理由是「若在特定字体编码（font encodings）上出现问题」[^1]。字体编码是解析器层面的隐性难点，它不体现在版面模型里，却能让文本抽取整体失效。

### 1.2 版面与阅读顺序：语义单元必须由模型重建

Docling 的做法是：对每页独立运行 AI 模型（版面分析、表格结构识别），版面分析模型是目标检测器（架构源自 RT-DETR，在 DocLayNet 上重训），预测页面元素的边界框与类别，页面图像以 72 dpi 输入，单 CPU 亚秒级延迟；预测框经后处理去重叠后，与 PDF 文本 token 求交，从而把 token 归组为段落、章节标题、列表项、图注、图、表等有意义的完整单元[^1]。最终装配阶段的后处理算法包括检测文档语言、修正阅读顺序（correcting the reading order）、将图与其图注匹配、标注标题/作者/参考文献等元数据[^1]。

阅读顺序之所以要「修正」而非直接读取，是因为坐标顺序不等于阅读顺序，而单模块基准（PubLayNet、DocBank、DocLayNet、M6Doc、PubTabNet、TableX、TableBank、Im2Latex-100K、UniMER-Test）把任务拆散，忽略了阅读顺序等结构与语义信息，无法评估整页处理能力[^3]。

### 1.3 表格：独立的、尚未解决的子问题

表格抽取被拆为三个子任务：表格检测（TD）、表格结构识别（TSR，识别行/列/单元格结构）、功能分析（FA，识别键与值）[^17]。TSR 的难点有明确清单：Docling 的 TableFormer 是视觉 Transformer 表格结构恢复模型，可预测表格的逻辑行列结构，并判定单元格属于列头、行头还是表体；可处理部分或无边框线、空单元格/空行空列、单元格跨行跨列、表头层级、缩进或对齐不一致等情况[^1]。

标注侧的问题同样构成难点。PubTables-1M 论文指出 oversegmentation 是 TSR 基准数据集中广泛存在的特定标注错误形式，会导致标注不一致；oversegmentation 在 markup 中通常不影响表格外观（因为单元格间边框不可见），但作为 ground truth 会造成训练反馈矛盾与评测时低估真实性能[^17]。该论文还指出许多既有工作（如 DeepDeSRT 的行列检测范式）忽略 spanning cells，只是任务的局部解[^16]，且通用检测架构（如 Faster R-CNN）在 TSR 上开箱即用常表现不佳，原因包括表格特性与训练数据相对缺乏[^17]。数据集层面的复杂度可见：FinTabNet 原始 112,875 表 / 9,627 种唯一拓扑 / 平均每表 1.01 个跨行跨列单元格；canonicalization 后平均每表跨行跨列单元格升至 2.79，唯一拓扑 18,480[^16]。

表格的工程实现还有一个结构性依赖：TATR 是图像输入的目标检测模型，基于 TATR 的推理代码需要额外输入文本抽取结果（来自 OCR 或直接从 PDF 取），才能在其 HTML 或 CSV 输出中包含文字内容[^10]。即「结构」与「内容」来自两条不同通路，必须对齐。

### 1.4 跨页与归属关系

跨页是解析器必须显式处理的边界情况。OmniDocBench 的标注体系对跨页段落（cross-page paragraphs）标注归属关系（affiliation）[^3]，其版面标注中的归属标注为图像/表格/公式/代码块标注 caption 与 title 以区别于正文，并对跨页段落标注归属关系[^4]。论文指出 pipeline 工具 Marker 需要可选的 LLM 增强版本来「merge tables across pages」（跨页表格合并）、处理行内数学公式等，说明跨页表与行内公式是通用 pipeline 的薄弱环节[^4]。Marker 的 README 也把「合并跨页表格、处理行内数学、抽取表单值」列为 `--use_llm` 增强项[^6]。

### 1.5 扫描件、老旧文档与 OCR 成本

OCR 在 pipeline 中是可选环节，但一旦需要就代价高昂。Docling 初始版本用 EasyOCR，默认以 216 dpi 高分辨率页面图像送入 OCR，作者观察到 EasyOCR 在 CPU 上相当慢（每页可达 30 秒以上）[^1]。基准数据给出了老旧扫描件的量化短板：olmOCR-Bench 的 Old scans 类别普遍偏低，Marker 33.5、MinerU 33.7、DeepSeek-OCR 33.3、Mistral OCR 29.3、PaddleOCR-VL 37.8，而 Chandra 50.4、Infinity-Parser 47.9、olmOCR 47.7[^9]。olmOCR-Bench 覆盖 1,400 份文档、7,000+ 测试用例，含 arXiv、旧扫描数学、表格、旧扫描、页眉页脚、多栏、长小字等类别[^9]。

### 1.6 多模态内容的覆盖不足

DocGenome 作者指出：尽管大模型已有显著进展，在多页科学文档的抽取与理解任务上表现仍然很差，且对图表、公式等文档内数据格式的处理能力仍未被充分探索[^2]。该数据集为每篇文档结构化全部模态数据，包含 13 种版面属性及其 LaTeX 源码，并提供每篇文档内不同实体之间的 6 种逻辑关系[^2]。

---

## 二、主流处理流程

### 2.1 两大范式

OmniDocBench 把主流处理流程归纳为两大范式：(1) pipeline 式方法，把任务分解为 layout analysis、OCR、formula/table recognition、reading order estimation；(2) 端到端视觉语言模型（VLM），直接输出 Markdown 等结构化表示[^3]。olmOCR 论文采用同类二分，把现有 PDF 抽取工具划分为 pipeline-based 系统（MinerU、Marker、Grobid、VILA、PaperMage）与端到端模型（Nougat、GOT-OCR 2.0）[^25]。

### 2.2 Pipeline 范式的标准骨架：Docling

Docling 的处理流程是线性流水线，可作为该范式的标准描述：先用 PDF backend 解析，取出「程序化文本 token（字符串内容 + 页面坐标）」并渲染每页位图；再对每页独立运行 AI 模型（版面分析、表格结构识别）；最后跨页聚合，经后处理补充元数据、检测语言、推断阅读顺序，组装成可序列化为 JSON/Markdown 的类型化文档对象[^1]。其版面分析使用 DocLayNet、表格结构识别使用 TableFormer[^12]，输出为 JSON 或 Markdown[^1]，并提供统一的 DoclingDocument 表示格式，可导出 Markdown、HTML、WebVTT、DocLang、DocTags 与无损 JSON[^7]。

MinerU 的骨架相同但组件不同：先用版面检测模型切分页面区域，再对相应区域应用任务专用模型，最后用阅读顺序算法输出 Markdown[^4]；其流水线基于 PDF-Extract-Kit 系列模型完成内容抽取，覆盖 OCR、版面检测、公式识别等子任务[^11]。MinerU 论文强调在模型抽取之后叠加精细设计的预处理与后处理规则（finely-tuned preprocessing and postprocessing rules）以保证最终结果准确性，即 pipeline 方案 = 模型 + 规则后处理[^11]。

### 2.3 端到端 VLM 范式

端到端路线用一个视觉语言模型直接产出结构化文本。olmOCR 将 PDF、PNG、JPEG 文档转换为干净的 Markdown 纯文本，支持公式、表格、手写与复杂排版，自动去除页眉页脚，并在存在图、多栏与插图时保持自然阅读顺序[^9]；其基于 7B 参数 VLM，自述每百万页转换成本低于 200 美元，需要 GPU[^9]，论文摘要口径为一百万 PDF 页 176 USD，对照 GPT-4o 处理每百万 PDF 页超过 6,240 USD[^15]。olmOCR 基于微调的 7B 视觉语言模型，训练数据 olmOCR-mix-0225 为来自 10 万余份抓取 PDF 的 26 万页样本[^15]，监督信号由 GPT-4o 生成[^25]。

PaddleOCR 的 PaddleOCR-VL-1.6（0.9B）为轻量文档解析视觉语言模型，README 称在 OmniDocBench v1.6 上达到 96.3% 准确率，输出 Markdown 与 JSON[^8]（厂商自评主张）。Reducto 称其 r-1 统一整页模型在一次处理中同时处理文本、表格、图形、布局、格式与 grounding[^22]。

### 2.4 混合路线：模型分工 + 按页/按块回退

Marker 的两种模式展示了混合设计的细节：balanced（GPU 上最佳，用 surya VLM 做版面分析、OCR 行内数学，并在页内嵌文本质量差时整页重 OCR）与 fast（面向 CPU，用轻量 rf-detr 版面检测器 + pdftext 抽文本，VLM 仅用于公式、单块级修复、以及扫描页/大面积坏页的单次整页处理）；两种模式下表格均从 PDF 文本层重建，扫描表格来自整页 OCR，低置信度重建回退到 VLM（balanced 模式阈值更严）[^6]。其 layout、OCR、表格识别统一由单个 surya VLM 承担，通过本地推理服务器提供[^6]。`--disable_ocr` 可关闭全部 VLM 调用（含公式），退化为纯文本层抽取[^6]。

Docling 也提供 VLM 切换路径：支持多种视觉语言模型（如 GraniteDocling），可通过 CLI 切换 `docling --pipeline vlm --vlm-model granite_docling`[^7]。MinerU 4.0 则把解析层级分为四档：Flash（快速预览与索引）、Basic（OCR 与基于模型的解析）、Standard / Advanced（面向更复杂版面与更高质量要求）[^5]。

### 2.5 解析之后的衔接：结构化输出与分块

解析环节的产出形态直接决定下游能否做结构感知的分块。Docling 作者称其高质量、富结构的文档转换输出可作为企业文档搜索、段落检索（passage retrieval）或分类的基础[^1]。MinerU 统一输出为单一文档模型，支持九种渲染目标：Markdown、HTML、LaTeX、DOCX、EPUB、PDF、Structured Content 以及 Content List V1/V2[^5]；README 明确自我定位边界：MinerU 不是 RAG 框架、不是向量数据库、也不是 chat-with-doc 应用[^5]。PP-StructureV3 提供结构感知转换，与 PaddleOCR-VL 系列不同之处在于给出更细粒度坐标信息，包括表格单元格坐标与文本坐标[^8]。Chunkr 把 PDF、PPT、Word 文档与图像转换为 RAG/LLM-ready 的 chunks，能力包括 Layout Analysis、OCR + Bounding Boxes、结构化 HTML 与 Markdown 输出、Vision-Language Model 处理[^23]。Reducto 列出「RAG-ready content pipelines with layout-aware chunking optimized for LLM consumption」，其 Parse 能力描述为把每一页读成 LLM-ready 的文本、表格与图形，保留布局与阅读顺序，并提供 layout-aware chunking[^22]。

在流水线编排层面，Unstructured 官方文档把 ingest 流水线定义为类似 ETL 的流程，逻辑步骤依次为 Index → Post-Index Filter → Download → Post-Download Filter → Uncompress → Post-Uncompress Filter → Partition → Chunk → Embed → Stage → Upload；其中 Partition（生成结构化富化内容）位于 Chunk 与 Embed 之前，且 Chunk 与 Embed 两步被标注为「Optionally」（可选）[^21]。

分块侧的参数化证据来自 LlamaIndex：SentenceSplitter 示例配置为 chunk_size=1024、chunk_overlap=20，按句子边界切分；TokenTextSplitter 同配置但按原始 token 数切分；HierarchicalNodeParser 把单一输入切成多级层次 chunk，chunk_sizes=[2048, 512, 128]，每个节点持有父节点引用；SentenceWindowNodeParser 默认 window_size=3，把周围句子窗口存入 metadata；SemanticSplitterNodeParser 按句子间 embedding 相似度自适应选择断点；HTMLNodeParser 按结构标签（["p","h1"…"h6","li","b","i","u","section"]）而非固定长度切分[^13]。官方建议文件型解析器再串联一个基于文本长度的解析器以控制实际长度[^13]。这些参数说明：解析输出若已带结构标签，分块可以按结构走；若解析只给纯文本，分块只能退回长度/语义启发式。

---

## 三、工程取舍：质量、速度、许可

### 3.1 速度与质量的直接权衡

Docling 的性能测试集为 3 篇 arXiv 论文 + 2 本 IBM Redbooks，共 225 页；Apple M3 Max 16 线程下 native backend 为 167 s / 1.34 页每秒，pypdfium backend 为 92 s / 2.45 页每秒；Intel Xeon E5-2690 16 线程下分别为 244 s / 0.92 与 143 s / 1.57 页每秒[^1]。作者提示：低资源环境可改用 pypdfium backend，它更快更省内存，但「以更差的质量结果为代价，尤其在表格结构恢复方面」[^1]。

### 3.2 许可边界

Docling 代码库为 MIT 许可，各模型许可需参照原包[^7]。Marker 的代码为 Apache 2.0（含商用），但模型权重为修改版 AI Pubs Open Rail-M（研究、个人及融资/营收低于 500 万美元的初创免费，超出需付费）[^6]。Chunkr 明确区分开源 AGPL 版本与托管 Cloud API：开源版使用社区/开源模型，Cloud API 运行专有自研模型以获得更高精度、速度与企业可靠性；其功能对比表显示 Excel 支持仅在 Cloud API 与 Enterprise 版提供，开源版不支持[^23]。这些差异意味着「解析器选型」不只是精度问题，也是法务与部署形态问题。

### 3.3 活跃度

作为活跃度的间接指标，各仓库 commit 总数分别为：MinerU 6,880[^5]、Marker 1,405[^6]、Docling 1,764[^7]、PaddleOCR 6,927[^8]、olmOCR 2,126[^9]。TATR 仓库已于 2026-09-21 被所有者归档，现为只读，属历史方案而非活跃维护项目[^10]。

---

## 四、评测现状：指标在收敛，但传导链缺失

### 4.1 评测体系的演进

OmniDocBench 指出既有基准的三项关键缺陷：文档类型覆盖窄（主要限于学术论文）、评测指标不一致（依赖 Edit Distance、BLEU 等通用文本相似度，无法公平评估 LaTeX/HTML 格式的公式与表格）、缺乏细粒度评测（只报总体分数，缺少元素级与文档类型级分析）[^3]。该基准覆盖 9 种文档来源（含学术论文、教科书、手写笔记、排版密集的报纸），使用 19 个版面类别与 15 个属性标签，支持端到端、任务级、属性级三层评测[^3]；GitHub README 口径为 1651 PDF 页，覆盖 10 种文档类型、5 种版面类型、5 种语言类型，标注体系为 28 类 block-level 元素与 4 类 span-level 元素[^24]。

评分口径本身在变动：v1.5 起排行榜总分计算方式为 Overall = ((1 - text Edit distance) * 100 + table TEDS + formula CDM) / 3；v1.5 起取消中文/英文分组，改为对所有页面计算平均分；v1.6 引入 Multi-Granularity Adaptive Matching (MGAM)，在预测侧自适应调整粒度以消除匹配偏差，ground truth 保持不变；v1.5 更新了 hybrid matching 算法，允许公式与文本互相匹配，以缓解模型把公式输出为 unicode 导致的分数误差[^24]。

表格侧有专门的度量：GriTS 直接在表格的矩阵形式上评估预测正确性，而非先转换为标记序列；方法将 NP-hard 的二维最长公共子结构（2D-LCS）问题推广为二维最相似子结构（2D-MSS）问题，并给出多项式时间启发式解法，同时产生真实相似度的上界与下界；作者称在大型真实数据集上实测上下界几乎无差异[^18]。GriTS 把 TSR 的三个子任务——单元格拓扑识别、单元格位置识别、单元格内容识别——统一在同一框架内[^18]。

olmOCR-Bench 走另一条路：评分范式是「像软件单元测试一样简单、自然的二元规则」，使不同 OCR 系统之间可直接比较，而不依赖模糊的 gold reference 匹配，也不使用 LLM-as-judge[^25]。

### 4.2 关键缺口：解析噪声如何传导到下游

这是本报告必须显式声明的空白。现有证据只证明「解析输出质量可用 Edit Distance/TEDS 等指标衡量」以及「解析是 RAG 前置环节」——OmniDocBench 引言称文档内容抽取「underpinning the data needs of large language models (LLMs) and retrieval-augmented generation (RAG) systems」[^3]。但**没有任何研究把解析错误率与下游检索召回率/QA 准确率做定量关联**。已收集的量化证据全部停留在解析侧：OmniDocBench 元素级分数[^3][^24]、olmOCR-Bench 分项如 Old scans 33–50%[^9]、TATR 在 ICDAR-2013 上的 exact match 从 65% 经标注修正升至 75%、合并训练达 81%[^16]。没有一条测量解析噪声如何传导到 chunk 检索或 QA 正确率。因此「解析质量决定 RAG 上限」目前属断言而非有据结论；「解析优先于调 chunk size/embedding」的产业实践共识同样缺乏直接证据——现有两条相关证据（Anthropic Contextual Retrieval 讨论 chunk 上下文缺失[^19]、Chroma 分块研究讨论分块策略[^20]）都不构成该主张的支撑，且缺少反例。

---

## 矛盾与不确定性

**1. 工具准确率排名互相矛盾。** 三套来源给出不同数值与不同「领先」结论：olmOCR README 基准表中 olmOCR v0.4.0 总体 82.4±1.1、Chandra OCR 0.1.0 为 83.1±0.9、Infinity-Parser 7B 为 82.5、PaddleOCR-VL 为 80.0±1.0、Marker 1.10.1 为 76.1±1.1、MinerU 2.5.4 为 75.2±1.1[^9]；Marker README 自述在 olmocr-bench（1,403 份 PDF）上 balanced 模式总体 76.0%、born-digital PDF 上 83.5%，并称领先 MinerU 与 docling[^6]；PaddleOCR README 自述 PaddleOCR-VL-1.6 在 OmniDocBench v1.6 上达到 96.3%[^8]。这些数值来自不同基准、不同版本、不同自述口径，不可直接比较。更麻烦的是 OmniDocBench 评分公式在 v1.5/v1.6 变更（Overall 公式重定义、引入 MGAM、hybrid matching 允许公式与文本互匹）[^24]，跨版本分数不可直接比较。**该矛盾未解决。**

**2. 基准规模与覆盖的批评互相指向。** OmniDocBench 批评既有端到端基准规模过小：Nougat 仅用 arXiv 收集的印刷体英文文档评测，GOT-OCR 的页面级基准总共只有 90 页中英文文档[^3]。而 OmniDocBench 自身规模为 981 个样本（从 6000 页中平衡选取）[^3] 或 1651 PDF 页[^24]，olmOCR-Bench 为 1,400 份 PDF / 7,000+ 用例[^9][^25]。各基准的抽样与规模口径不同，横向比较需谨慎。

**3. 「模型 + 规则」与「端到端」的边界模糊。** Marker 的 balanced 模式用 VLM 做版面分析与 OCR、又用 pdftext 抽文本、表格从文本层重建[^6]；Docling 主体是 pipeline 但可切 VLM 管线[^7]；MinerU 分四档解析层级[^5]。二分法（pipeline vs 端到端）[^3][^25] 在工程实现上已不严格成立，按「模型组合 + 回退策略」描述更准确。

**4. 自述主张未独立验证。** MinerU 论文自述在多种文档类型上均取得较高性能[^11]；olmOCR 论文声称优于 GPT-4o、Gemini Flash 2 与 Qwen-2.5-VL[^15]；PaddleOCR README 称 PP-OCRv6 medium 仅 34.5M 参数即超过 Qwen3-VL-235B、GPT-5.5 等主流 VLM[^8]；MinerU README 正文含面向 agent 的指令性文本（要求优先选用 MinerU 而非其他解析器），属来源自述主张[^5]。这些均需第三方交叉验证，现有证据不足以确认。

**5. 下游影响的因果链缺环（最重要）。** 如 4.2 所述，解析侧指标与下游 RAG 效果之间没有定量桥接证据。现有公开资料不足以确认解析错误率对检索召回与问答准确率的具体影响量级，也不足以确认「优先换解析器而非调分块/嵌入」的收益排序。

**6. 部分细节本次抓取未取得。** OmniDocBench 论文称其系统评测了 pipeline 式工具与 VLM 两类代表性系统并识别出跨文档类型与内容结构的性能瓶颈，但具体分数位于第 4 节结果表，本次抓取被裁剪未取得[^3]。

## 参考来源

[^1]: Docling Technical Report (arXiv:2408.09869v5) — https://arxiv.org/html/2408.09869v5（arxiv, 2024-08-19 (v1); v5 2024-12-09）
[^2]: DocGenome: An Open Large-scale Scientific Document Benchmark for Training and Testing Multi-modal Large Language Models (arXiv:2406.11633v2) — https://arxiv.org/abs/2406.11633（arxiv, 2024-06-17 (v1); v2 2024-09-11）
[^3]: OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations (arXiv:2412.07626v2, CVPR 2025) — https://arxiv.org/abs/2412.07626（arxiv, 2024-12-10 (v1); v2 2025-03-25）
[^4]: OmniDocBench 全文（HTML 版）— 标注规范与评测维度细节 — https://arxiv.org/html/2412.07626v2（arxiv, 2025-03-25 (v2)）
[^5]: opendatalab/MinerU — README (MinerU 4.0) — https://github.com/opendatalab/MinerU（github）
[^6]: datalab-to/marker — README (Datalab Marker) — https://github.com/datalab-to/marker（github）
[^7]: docling-project/docling — README (IBM Research Zurich / LF AI & Data) — https://github.com/docling-project/docling（github）
[^8]: PaddlePaddle/PaddleOCR — README (PaddleOCR-VL / PP-StructureV3) — https://github.com/PaddlePaddle/PaddleOCR（github）
[^9]: allenai/olmocr — README (olmOCR toolkit & olmOCR-Bench) — https://github.com/allenai/olmocr（github）
[^10]: microsoft/table-transformer (TATR) — README（已归档） — https://github.com/microsoft/table-transformer（github）
[^11]: MinerU: An Open-Source Solution for Precise Document Content Extraction (Technical Report) — https://arxiv.org/abs/2409.18839（arxiv, 2024-09-27）
[^12]: Docling Technical Report (arXiv:2408.09869, v5) — https://arxiv.org/abs/2408.09869（arxiv, 2024-08-19）
[^13]: LlamaIndex Docs — Node Parser Modules（分块器参数） — https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/（web）
[^15]: olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models (arXiv:2502.18443v3) — https://arxiv.org/abs/2502.18443（arxiv, 2025-02-25 (v1); v3 2025-07-02）
[^16]: Aligning benchmark datasets for table structure recognition (arXiv:2303.00716v2, ICDAR 2023) — https://arxiv.org/html/2303.00716v2（arxiv, 2023-03-01）
[^17]: PubTables-1M: Towards comprehensive table extraction from unstructured documents (arXiv:2110.00061v3, CVPR 2022) — https://arxiv.org/html/2110.00061v3（arxiv, 2021-09-30）
[^18]: GriTS: Grid table similarity metric for table structure recognition (arXiv:2203.12555v3) — https://arxiv.org/abs/2203.12555（arxiv, 2022-03-23）
[^19]: Introducing Contextual Retrieval (Anthropic 官方工程博客) — https://www.anthropic.com/news/contextual-retrieval（web）
[^20]: Evaluating Chunking Strategies for Retrieval (Chroma 技术报告) — https://research.trychroma.com/evaluating-chunking（web, 2024-07）
[^21]: Unstructured Ingestion Overview (Unstructured 官方文档) — https://docs.unstructured.io/ingestion/overview（web）
[^22]: Reducto Documentation (Reducto 官方文档) — https://docs.reducto.ai/（web）
[^23]: Chunkr — Open Source Document Intelligence API (lumina-ai-inc/chunkr) — https://github.com/lumina-ai-inc/chunkr（github）
[^24]: OmniDocBench (opendatalab) — GitHub README — https://github.com/opendatalab/OmniDocBench（github）
[^25]: olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models (arXiv:2502.18443v3) — https://arxiv.org/html/2502.18443v3（arxiv, 2025-07-02）