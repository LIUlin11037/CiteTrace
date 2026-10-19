# 向量数据库主流选型差异与 pgvector 的适用边界

## 摘要

Qdrant、Milvus、Weaviate、pgvector 四者的差异不在"是否支持 ANN 搜索"，而在**架构定位、索引/量化能力矩阵、过滤与混合检索的实现层次、以及运维复杂度**。Milvus 是四者中唯一采用数据平面/控制平面分离与存储计算分离的分布式系统，构建于 Faiss、HNSW、DiskANN、SCANN 之上[^10]；Qdrant 以 Rust 实现、向量索引与载荷索引分离、量化选项最丰富（四种方法，最高 64x）[^7][^8][^13]；Weaviate 采用"向量优先"存储、提供 HNSW/Flat/Dynamic/HFresh 四种索引与四种量化，但许可证为混合模式（wl 目录为企业许可）[^11][^12][^15]；pgvector 不是独立数据库，而是 Postgres 扩展，把向量检索放进既有事务、备份、JOIN 体系内[^6][^18]。

pgvector 的合理选择场景，从现有证据可确认的是：**向量规模与查询量尚未使索引构建成本无法摊销、且应用已经在 Postgres 上、需要 ACID/JOIN/时间点恢复与向量检索在同一事务边界内完成时**。但本报告必须明确指出：**没有任何可用证据给出"多少向量、多高 QPS、多长延迟"应迁移到专用向量库的量化阈值**，这一子问题的核心结论在现有资料下无法确认。

---

## 一、四个系统的定位与架构差异

### 1.1 Milvus：分布式优先

Milvus 是开源、云原生向量数据库，构建于 Faiss、HNSW、DiskANN、SCANN 之上[^10]。其架构采用数据平面与控制平面分离（data plane and control plane disaggregation），存储与计算层完全分离[^10]。四层结构为：Access Layer（无状态 Proxy）、Coordinator、Worker Nodes、Storage[^10]。Coordinator 在全集群同一时刻仅有一个 active，负责维护集群拓扑、调度任务、保证集群级一致性[^10]。Worker Nodes 分三类：Streaming Node、Query Node、Data Node[^10]。系统采用 MPP（massively parallel processing）架构，由 Proxy 聚合与后处理中间结果[^10]。共享存储 + 存储计算分离架构支持计算节点水平扩展[^10]。WAL 层使用 Woodpecker 作为零磁盘 WAL，以提升弹性、降低运维开销[^10]。

工程侧：仓库 46,383 stars、4,297 forks、主语言 Go、累计 25,758 commits、最近 push 2026-10-14[^14]。项目归属 LF AI & Data Foundation，Apache 2.0 许可，Zilliz 为主要贡献者[^14]。提供 Milvus Lite（pip 安装、本地文件持久化）与 Standalone 单机模式，以及 Zilliz Cloud 全托管（Serverless/Dedicated/BYOC）[^14]。

### 1.2 Qdrant：Rust 实现、索引分层清晰

Qdrant 的核心设计特征是**向量索引与载荷（payload）索引分离**：向量索引加速向量搜索，载荷索引加速过滤[^7]。载荷索引支持 keyword、integer、float、bool（v1.4.0+）、geo、datetime（v1.8.0+）、text（全文检索）、uuid（v1.11.0+）[^7]。索引按 segment 独立存在，但索引参数按整个 collection 配置；是否建索引由 optimizer 设置决定，通常取决于存储的点数[^7]。keyword 索引可选启用前缀匹配，text 类型索引支持全文检索过滤条件[^7]。官方提示载荷索引占用额外内存与磁盘，建议仅对用于过滤条件的字段建索引[^7]。

工程侧：35,073 stars、2,745 forks、主语言 Rust、累计 7,117 commits、最近 push 2026-10-14[^13]。官方客户端覆盖 Go、Rust、JavaScript/TypeScript、Python、.NET/C#、Java，社区客户端有 Kotlin、PHP[^13]。提供 Qdrant Edge 轻量版（可嵌入应用进程运行，支持与 Qdrant server 同步）、Qdrant Cloud 托管服务（含免费层），以及面向 AI 编码助手的 Agent Skills 集合[^13]。

### 1.3 Weaviate：向量优先存储 + 混合许可

Weaviate 采用**向量优先（vector-first）存储系统**[^11]。支持四种向量索引类型：HNSW、Flat、Dynamic、HFresh[^11]。其中 HNSW 索引构建较慢，但查询为对数时间复杂度，可良好扩展至大数据集；Flat 索引简单轻量，面向小数据集；Dynamic 索引可从 flat 自动切换到 HNSW（按对象数阈值）；HFresh 为基于聚类的索引，用 HNSW 做质心索引，将大部分数据留在磁盘以提升内存效率[^11]。可通过水平扩展或足够的分片在不降低性能的前提下存储超大数据量[^11]。

工程侧：16,881 stars、1,421 forks、主语言 Go、累计 30,926 commits、最近 push 2026-10-14[^15]。许可证为**混合模式**：wl 目录外源码采用 BSD-3-Clause；wl 目录及子目录为 Weaviate BV 保留所有权利的企业许可，需 license key 启用[^15]。支持导入时自动向量化（集成 OpenAI、Cohere、HuggingFace 等模型）或导入预计算向量；提供 Docker、Kubernetes、Weaviate Cloud 部署方式[^15]。README 示例镜像版本为 1.36.0，Python 客户端为 weaviate-client[^15]。

### 1.4 pgvector：Postgres 扩展，而非独立数据库

pgvector 是 Postgres 的开源向量相似度搜索扩展，支持精确搜索与近似最近邻搜索[^6][^18]。支持单精度、半精度、二进制、稀疏向量[^6][^18]；支持 L2、内积、余弦、L1、Hamming、Jaccard 六种距离度量[^6][^18]。内积运算符 `<#>` 返回负内积，原因是 Postgres 仅支持对运算符做 ASC 顺序索引扫描[^6]。依托 Postgres 原生能力：ACID 合规、时间点恢复（point-in-time recovery）、JOINs[^6][^18]。支持 Postgres 13+，当前版本 v0.8.7，可经 Docker/Homebrew/PGXN/APT/Yum/pkg/APK/conda-forge 安装，并预装于 Postgres.app 及多家托管 Postgres 提供商[^18]。过滤即 SQL WHERE 条件，可与向量排序组合（示例 `WHERE id != 1 ORDER BY embedding <-> ...`）[^6]。

---

## 二、索引与量化能力矩阵

### 2.1 底层算法：HNSW 是共同基线

HNSW 是四者共同依赖或可选的核心 ANN 算法，其定义需要完整写出。HNSW 是一种基于可导航小世界图（navigable small world graphs）并带可控层次结构的近似 K 近邻搜索方法[^1]。其构建与搜索机制为：增量式构建多层结构，由嵌套元素子集的层次化近邻图层组成；元素存在的最大层以指数衰减的概率分布随机选取，从而产生类似 NSW 的图，同时使链接按特征距离尺度分离；搜索从上层开始并结合尺度分离，相比 NSW 提升性能，并允许对数复杂度（logarithmic complexity）扩展[^1]。该方案完全基于图，不需要其他近邻图技术通常用于粗搜索阶段的额外搜索结构[^1]。额外采用近邻图邻居选择的启发式方法，在高召回率和高聚类数据场景下显著提升性能[^1]。论文声称该通用度量空间搜索索引能大幅超越此前开源的最先进纯向量方法，且算法与跳表（skip list）结构相似，便于直接实现均衡的分布式部署[^1]。

Supabase 文档对 HNSW 的表述与之一致：HNSW 是一种近似最近邻（ANN）搜索算法，适用于高维向量查询；基于邻近图（proximity graph），由分层（Hierarchical）与可导航小世界（NSW）两部分构成[^21]。

### 2.2 各系统索引类型

**Milvus**（v3.0.x）浮点（dense）向量索引包括：FLAT、IVF_FLAT、IVF_PQ、IVF_SQ8、HNSW、HNSW_SQ、HNSW_PQ、HNSW_PRQ、SCANN；二进制向量索引：BIN_FLAT、BIN_IVF_FLAT；稀疏向量索引：SPARSE_INVERTED_INDEX，仅支持 IP 与 BM25（全文检索）度量[^9]。一个向量字段仅支持一种索引类型，切换索引类型时 Milvus 自动删除旧索引[^9]。量化通过索引类型体现：IVF_SQ8、HNSW_SQ 为标量量化类；IVF_PQ、HNSW_PQ、HNSW_PRQ 为乘积量化类[^9]。浮点向量距离度量使用欧氏距离（L2）与内积（IP）；二进制向量使用 JACCARD 与 HAMMING[^9]。SPARSE_WAND 自 Milvus 2.5.4 起被弃用，建议改用 `inverted_index_algo: DAAT_WAND`[^9]。

**Weaviate** 支持 HNSW、Flat、Dynamic、HFresh 四种索引[^11]（见 1.3）。

**pgvector** 目前支持两种索引类型：HNSW 与 IVFFlat[^20]。Supabase 官方推荐使用 HNSW，理由是性能更好且对数据变化更鲁棒（"we recommend using HNSW because of its performance and robustness against changing data"）[^20]。无索引时查询会退化为顺序扫描（sequential scan），在记录很多时是资源密集型操作[^20]。pgvector 0.7.0 及以上版本创建索引的最大维度限制：vector 类型最多 2,000 维，halfvec 最多 4,000 维，bit 最多 64,000 维[^20]。pgvector 提供 3 个距离操作符及其对应操作符类：`<->` 对应 vector_l2_ops，`<#>` 对应 vector_ip_ops，`<=>` 对应 vector_cosine_ops[^21]。对于超过 2,000 维的向量，可通过 halfvec 类型建立索引，文档给出 3,072 维的示例（`embedding::halfvec(3072) halfvec_cosine_ops`）[^21]。

**Qdrant** 的索引体系以载荷索引为特色（见 1.2），文档未在同一页面枚举向量索引类型清单[^7]。

### 2.3 量化：Qdrant 选项最细，Weaviate 有硬性约束

**Qdrant** 支持四种量化方法：TurboQuant、Scalar Quantization、Binary Quantization、Product Quantization[^8]。TurboQuant（v1.18.0+，源自 Google 的方法）支持最高 32x 压缩[^8]；Scalar Quantization 将每个向量分量从 32-bit float 压缩为 8-bit integer，实现 4x 压缩[^8]；Binary Quantization 将每个向量分量压缩为 1–2 bit，最高 32x 压缩，适合高维、中心化分布的向量[^8]；Product Quantization 在内存最小化为首要目标时最高可达 64x 压缩[^8]。量化以引入近似误差为代价，可能轻微降低搜索质量[^8]。官方建议：4x 压缩用 Scalar Quantization；8x 用 4-bit TurboQuant；16x 用 2-bit TurboQuant 或 2-bit binary；32x 用 1-bit TurboQuant 或 1-bit binary；最高 64x 用 Product Quantization[^8]。TurboQuant 的引入有独立发布记录佐证：Qdrant v1.18.0 由 timvisee 于 5 月 11 日发布，Features 中列有 "milestone!48 - Add TurboQuant quantization variant, 8x vector compression without the recall tax"[^30]。

**Weaviate** 提供四种量化技术：Binary quantization (BQ)、Product quantization (PQ)、Scalar quantization (SQ)、Rotational quantization (RQ)[^12]。自 v1.33 起可通过 `DEFAULT_QUANTIZATION` 环境变量为新 collection 设置默认量化；默认未设置即不应用量化[^12]。**量化一旦在 collection 上设置便无法禁用**[^12]。PQ 仅适用于 HNSW 索引；PQ 与 SQ 不支持 flat 索引[^12]。PQ 分两步压缩：先将向量维度归约为更少的 segment，再将每个 segment 量化到更少位数；PQ 默认将每个 segment 聚类为 256 个质心，构成 codebook[^12]。PQ 压缩示例：768 维 4 字节向量（3072 字节）压缩后为 128 字节，约 24x 压缩[^12]。

**Milvus** 的量化通过索引类型体现（IVF_SQ8、HNSW_SQ、IVF_PQ、HNSW_PQ、HNSW_PRQ）[^9]。

**pgvector** 的量化路径是类型层面的：README 以 "Have a lot of vectors? Use quantization to scale" 指向 Scaling 章节，量化通过 halfvec/bit 类型实现[^6]。

### 2.4 容量与限制的表述方式差异

Milvus 用**硬性上限**描述规模：单个向量最大维度 32,768；单实例最多 65,536 个 Collection；每个 Collection 最多 1,024 个 Partition、16 个 Shard、64 个 Field、1 个 Index；Entity 无上限；搜索时 topk 上限 16,384，nq 上限 16,384；单次 RPC 的 Insert/Search/Query 输入输出上限均为 64 MB；待加载数据必须低于所有 query node 总内存资源的 90%；资源名长度上限 255 字符；VARCHAR 字符串长度上限 65,535[^27]。

Qdrant 用**容量规划公式**而非硬性上限：base = number_of_points * replication_factor；稠密向量存储量 dense_size = base * dimensions * bytes_per_dim，bytes_per_dim 取决于数据类型（float32=4 字节默认、float16=2、uint8=1、turbo4=0.5）[^28]。官方示例：100 万点、768 维、复制因子 2、float32、默认 cached 层，dense_size ≈ 5.72 GB[^28]。支持命名向量，每个命名向量有各自的维度、数据类型、内存层与 HNSW 索引，需分别计算容量[^28]。量化会额外存储压缩副本：quantized_size = base * dimensions * quant_bytes；例如 4-bit TurboQuant 为 8x 压缩、0.5 字节/维[^28]。向量可配置内存层：pinned（堆内存，永不驱逐）、cached（mmap，启动预热）、cold（mmap，按需加载）[^28]。

这一差异本身是选型信号：Milvus 的文档形态面向"集群资源规划"，Qdrant 的文档形态面向"按公式估算内存"。

---

## 三、生态集成与功能覆盖

LangChain 官方集成列表中，Qdrant 对应专用包 langchain-qdrant，Milvus 对应 langchain-milvus，Weaviate 对应 langchain-weaviate，三者均支持 JS/TS；LangChain 生态共 1000+ 集成，vector store 为其中一类组件[^16]。

LlamaIndex 支持 20+ 种向量存储选项[^17]。在 LlamaIndex 的特性支持矩阵中：Qdrant 支持 Metadata Filtering、Hybrid Search、Delete、Store Documents、Async 全部五项；Milvus / Zilliz 支持 Metadata Filtering、Hybrid Search、Delete、Store Documents 四项；Weaviate 支持 Metadata Filtering、Hybrid Search、Delete、Store Documents 四项；Postgres（即 pgvector 路径）支持 Metadata Filtering、Hybrid Search、Delete、Store Documents、Async 全部五项[^17]。另有独立的 pgvecto.rs 集成项[^17]。

按此矩阵，Qdrant 与 pgvector 在 LlamaIndex 中并列覆盖最全（五项），Milvus 与 Weaviate 各少 Async 一项[^17]。

---

## 四、pgvector 何时是合理选择

### 4.1 可确认的适用理由

**理由一：向量检索与业务数据在同一事务边界内。** pgvector 依托 Postgres 原生能力提供 ACID 合规、时间点恢复、JOINs[^6][^18]。过滤即 SQL WHERE 条件，可与向量排序直接组合（示例 `WHERE id != 1 ORDER BY embedding <-> ...`）[^6]。这意味着"元数据过滤 + 向量排序"不需要跨系统同步，也不需要在应用层做结果合并。

**理由二：部署与运维路径已存在。** pgvector 支持 Postgres 13+，可经 Docker/Homebrew/PGXN/APT/Yum/pkg/APK/conda-forge 安装，并预装于 Postgres.app 及多家托管 Postgres 提供商[^18]。对于已经在托管 Postgres 上的应用，引入向量检索不需要新增一套数据库运维体系。

**理由三：小规模搜索场景下索引本身可能是过度设计。** Faiss 官方选型指南给出的一般性原则：若只做少量搜索（约 1000–10000 次），索引构建时间无法被搜索时间摊销，直接计算（Flat 索引）最高效[^3]。该原则虽出自 Faiss 文档而非 pgvector 文档，但描述的是 ANN 索引的通用成本结构：构建成本需要足够的查询量来摊销。

**理由四：混合检索可在库内完成。** ParadeDB 博客主张用 ParadeDB 提供 BM25 词法相关性 + pgvector 提供向量语义相似度，二者组合成混合检索系统，并明确采用 Reciprocal Rank Fusion (RRF) 作为融合两路排序的方法，称可构建"production-ready search strategy that lives entirely within your database, with no external dependencies or synchronization headaches"[^23]。该文批评 Postgres 原生全文检索（tsvector/tsquery）的 ts_rank 只考虑单文档内部统计、不理解全局语料统计（如某词出现在 80% 还是 5% 文档中），因此无法做现代相关性排序[^23]。文中称 BM25 通过三个信号解决排序问题：词频（Term Frequency，带递减收益）、逆文档频率（IDF，稀有词权重更高）、文档长度归一化（偏好简短聚焦文档）[^23]。BM25 是 Elasticsearch 与 Solr 等现代搜索引擎背后的算法[^23]。ParadeDB 仓库自述定位为「One Postgres for your application data, full-text search, vector retrieval, and aggregations」，实现方式为 Postgres 扩展（pg_search），底层依赖 pgrx、Tantivy、Apache DataFusion[^22]。其设计目标是「keeps your application data and search engine in one database with standard SQL and ACID transactions」[^22]。许可证：ParadeDB Community 采用 AGPL v3.0；ParadeDB Enterprise 为商业许可[^22]。

**理由五：索引类型选择简单。** pgvector 只有 HNSW 与 IVFFlat 两种索引[^20]，Supabase 官方推荐 HNSW，理由是性能更好且对数据变化更鲁棒[^20]。相比 Milvus 的九种浮点索引类型[^9]或 Weaviate 的四种索引 × 四种量化的组合[^11][^12]，pgvector 的决策面显著更窄。

### 4.2 可确认的约束

**维度约束是硬性的。** pgvector 0.7.0 及以上版本创建索引的最大维度限制：vector 类型最多 2,000 维，halfvec 最多 4,000 维，bit 最多 64,000 维[^20]。超过 2,000 维需改用 halfvec（文档给出 3,072 维示例）[^21]。对比之下，Milvus 单向量最大维度为 32,768[^27]。若使用 3,072 维 embedding，pgvector 路径必须走 halfvec。

**无索引即顺序扫描。** 无索引时查询会退化为顺序扫描，在记录很多时是资源密集型操作[^20]。

**量化能力以类型形式提供。** pgvector 的量化通过 halfvec/bit 类型实现，README 以 "Have a lot of vectors? Use quantization to scale" 指向 Scaling 章节[^6]。这与 Qdrant 的四种量化方法（含最高 64x 的 PQ）[^8]、Weaviate 的四种量化技术（PQ 约 24x 示例）[^12]在粒度上不同。

**扩展生态提供额外路径。** pgvectorscale 构建在 pgvector 之上，定位为更高性能的 embedding 搜索与更省成本的存储；引入新索引类型 StreamingDiskANN（灵感来自 Microsoft 的 DiskANN 算法）、Statistical Binary Quantization（SBQ，由 Timescale 研究者开发，声称优于标准 Binary Quantization）、以及基于标签的过滤向量搜索（基于 Microsoft 的 Filtered DiskANN 研究）[^19]。pgvectorscale 用 Rust（PGRX 框架）开发，而 pgvector 用 C 编写[^19]。安装方式：`CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE`（CASCADE 会自动安装 pgvector）；macOS X86 (Intel) 上构建当前不受支持[^19]。

### 4.3 无法确认的部分（关键缺口）

**没有任何可用证据给出"何时应从 pgvector 迁移到专用向量库"的量化阈值。** 现有证据中不存在向量数量、维度、QPS 或延迟层面的迁移触发条件。Milvus 的硬性上限[^27]与 Qdrant 的容量公式[^28]描述的是各自系统的边界，不是 pgvector 的迁移阈值。Faiss 的"1000–10000 次搜索"原则[^3]描述的是索引构建摊销问题，且出自 Faiss 而非 pgvector 语境。

**pgvectorscale 的性能主张缺少独立佐证。** pgvectorscale README 声称：在 5000 万条 768 维 Cohere embedding 基准数据集上，PostgreSQL + pgvector + pgvectorscale 相比 Pinecone 的 storage-optimized (s1) 索引，在 99% recall 的 ANN 查询下实现 28x 更低 p95 延迟、16x 更高查询吞吐，自托管于 AWS EC2 时成本低 75%[^19][^24]。该主张的出处是 Timescale 官方的 pgvector vs Pinecone comparison blog post，即厂商自测（vendor benchmark），非独立第三方；README 明确将性能细节与基准方法论指向 Timescale 官方博客[^24]。VectorDBBench 是一个公开的向量数据库基准测试工具，支持插入、向量搜索、过滤搜索和全文搜索的可复现工作负载，数据集涵盖 SIFT、GIST、Cohere、OpenAI/C4、VIBE 等，明确支持 pgvector 与 pgvectorscale 作为被测后端（`pip install vectordb-bench[pgvector]`），同时也支持 qdrant、pinecone、weaviate、milvus 等[^25]。但 VectorDBBench 由 Zilliz（Milvus 背后的开源向量数据库公司）赞助，因此其基准结果存在与 Milvus 相关的利益关联，不能视为完全中立的第三方[^25]。而 Zilliz 托管的 VectorDBBench leaderboard 在 1M Cohere Medium 数据集（1M × 768 维）上给出各系统在 $1000/月成本下的 P99 延迟、QPS 与 recall：ZillizCloud-8cu-perf 为 P99 2ms / 13,316 QPS / recall 0.9383；Milvus-16c64g-sq4u-fp16 为 2.3ms / 9,575 QPS / 0.9189；QdrantCloud-16c64g 为 6.4ms / 1,242 QPS / 0.9474；Pinecone-p2.x8-1node 为 13.7ms / 1,146 QPS / 0.9262[^26]。**该 leaderboard 当前展示的结果条目中未出现 pgvector 或 pgvectorscale 的对比数据**，因此无法据此验证 Timescale 的 28x/16x/75% 主张[^26]。该 leaderboard 的对比对象为 ZillizCloud、Milvus、OpenSearch、ElasticCloud、QdrantCloud、Pinecone、TurboPuffer 等，属于云服务/托管产品层面的对比[^26]。

---

## 五、矛盾与不确定性

**不确定性一：DiskANN 与 IVF-PQ 缺少一手定义来源。** 本报告涉及 DiskANN 的全部内容均来自二手转述：Milvus 架构文档称其构建于 DiskANN 之上[^10]；pgvectorscale README 称 StreamingDiskANN 灵感来自 Microsoft 的 DiskANN 算法、过滤搜索基于 Microsoft 的 Filtered DiskANN 研究[^19]。**DiskANN 的原始论文未被成功定位**：本次尝试用 arXiv ID 1905.01255 定位 DiskANN 失败，该条目实为天体物理学论文（AGILE-GRID 对超新星遗迹 SNR G78.2+2.1 的伽马射线辐射研究），与 DiskANN 或向量检索无关[^29]。因此 DiskANN 的标准定义、复杂度与召回/内存权衡在本报告中无法给出。

**不确定性二：IVF-PQ 缺少算法级定义。** 现有证据仅有 Faiss wiki 的工程选型建议：IVF 类方案（如 IVF1024,PQ Nx4fs,RFlat）比 HNSW 更快，但需要重排序阶段，因此有两个参数需调整——重排序的 k_factor 和 IVF 的 nprobe[^3]。当内存相当重要时，使用 OPQ M_D,...,PQ Mx4fsr：先做 OPQ 变换降维到 D，再将向量 PQ 量化为 M 个 4-bit 码，总存储为每向量 M/2 字节；当内存非常重要时，使用 OPQ M_D,...,PQ M，PQ M 用乘积量化器输出 M 字节码，M 通常 <= 64，更大的码通常用 SQ 更精确且更快，OPQ 是使向量更易压缩的线性变换，D 必须是 M 的倍数且 D <= 输入维度 d[^3]。这些是参数与存储量的工程描述，**缺少 IVF-PQ 的算法定义、复杂度与召回率权衡的权威出处**（原始论文或教科书级综述）。此外，Faiss wiki 关于 HNSW 的若干工程细节——HNSW 内存占用为每向量 (d * 4 + M * 2 * 4) 字节；HNSW 仅支持顺序添加（不支持 add_with_ids），需要时前缀 IDMap；HNSW 无需训练，且不支持从索引中删除向量；M 为每向量链接数，4 <= M <= 64，越大越精确但占用更多 RAM；速度-精度权衡由 efSearch 参数设定；若 RAM 充足或数据集较小，HNSW 是最佳选择——均出自 Faiss 语境[^3]，**不能直接外推到 Qdrant/Milvus/Weaviate/pgvector 的实现**，各系统的参数命名与默认值可能不同。

**不确定性三：pgvector 的迁移阈值完全缺失。** 如 4.3 所述，现有证据不包含任何向量数量/维度/QPS/延迟层面的"何时用 pgvector、何时迁移到专用向量库"的经验阈值与时间范围。**该子问题的核心结论在现有公开资料下不足以确认。**

**不确定性四：pgvectorscale 性能主张的可信度未解决。** 28x 更低 p95 延迟、16x 更高吞吐、成本低 75% 的主张仅出自 Timescale 自家 README 与官方博客（vendor benchmark）[^19][^24]，且 Zilliz 托管的 VectorDBBench leaderboard 中根本没有 pgvector/pgvectorscale 条目，无法交叉验证[^26]。同时，VectorDBBench 本身由 Zilliz 赞助，存在与 Milvus 相关的利益关联[^25]。**两个方向的基准都存在利益关联，本报告不将其中的任何数字作为中立性能结论。**

**不确定性五：ParadeDB 混合检索主张为厂商营销性表述。** ParadeDB 博客称"the easiest and most performant approach is with the ParadeDB pg_search extension"，将 BM25 作为原生索引类型引入 Postgres[^23]。该主张出自 ParadeDB 自家博客，属厂商营销性表述，无第三方基准佐证[^23]。且该文正文在 BM25 介绍段落后即被截断，RRF 具体实现、性能基准与适用场景的后续章节未能读取[^23]。因此"在 Postgres 内做混合检索"这一路径的完整可行性，本报告只能确认其被主张，无法确认其性能表现。

**不确定性六：Qdrant 向量索引类型清单缺失。** Qdrant 文档的 Indexing 页面聚焦于载荷索引与向量索引的分离关系[^7]，未在同一页面枚举向量索引类型清单；本报告因此无法像 Milvus[^9]或 Weaviate[^11]那样列出 Qdrant 的完整索引类型矩阵。

**不确定性七：四系统之间的中立性能对比缺失。** 现有唯一的横向 leaderboard 数据来自 Zilliz 托管的 VectorDBBench，覆盖 ZillizCloud、Milvus、QdrantCloud、Pinecone 等云服务/托管产品，且由 Milvus 背后的公司赞助[^25][^26]。**没有任何中立第三方给出 Qdrant/Milvus/Weaviate/pgvector 四者在同一硬件、同一数据集、同一 recall 目标下的对比数据。**

## 参考来源

[^1]: Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs (HNSW) — https://arxiv.org/abs/1603.09320（arxiv, 2016-03-30 (v1), last revised 2018-08-14 (v4)）
[^3]: Faiss Wiki — Guidelines to choose an index — https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index（github, last edited 2026-10-01 (31 revisions)）
[^6]: pgvector — Open-source vector similarity search for Postgres (README, v0.8.7) — https://github.com/pgvector/pgvector（github）
[^7]: Qdrant Documentation — Indexing — https://qdrant.tech/documentation/concepts/indexing/（web）
[^8]: Qdrant Documentation — Quantization — https://qdrant.tech/documentation/guides/quantization/（web）
[^9]: Milvus Documentation — In-memory Index (v3.0.x) — https://milvus.io/docs/index.md（web）
[^10]: Milvus Documentation — Architecture Overview (v3.0.x) — https://milvus.io/docs/architecture_overview.md（web）
[^11]: Weaviate Documentation — Vector Index — https://docs.weaviate.io/weaviate/concepts/vector-index（web）
[^12]: Weaviate Documentation — Vector Quantization — https://docs.weaviate.io/weaviate/concepts/vector-quantization（web）
[^13]: qdrant/qdrant — https://github.com/qdrant/qdrant（github）
[^14]: milvus-io/milvus — https://github.com/milvus-io/milvus（github）
[^15]: weaviate/weaviate — https://github.com/weaviate/weaviate（github）
[^16]: LangChain Vector Stores Integration Docs — https://python.langchain.com/docs/integrations/vectorstores/（web）
[^17]: LlamaIndex Vector Store Options & Feature Support — https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/（web）
[^18]: pgvector/pgvector — Open-source vector similarity search for Postgres (README) — https://raw.githubusercontent.com/pgvector/pgvector/master/README.md（github）
[^19]: timescale/pgvectorscale — README (StreamingDiskANN, SBQ, filtered search) — https://raw.githubusercontent.com/timescale/pgvectorscale/main/README.md（github）
[^20]: Supabase Docs — Vector indexes (HNSW / IVFFlat) — https://supabase.com/docs/guides/ai/vector-indexes（web）
[^21]: Supabase Docs — HNSW indexes — https://supabase.com/docs/guides/ai/vector-indexes/hnsw-indexes（web）
[^22]: paradedb/paradedb — One Postgres for your application data, full-text search, vector retrieval, and aggregations (Home of the pg_search extension) — https://github.com/paradedb/paradedb（github）
[^23]: Hybrid Search in PostgreSQL: The Missing Manual (ParadeDB Blog, James Blackwood-Sewell, 2025-10-22) — https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual（web, 2025-10-22）
[^24]: timescale/pgvectorscale (official README) — https://github.com/timescale/pgvectorscale（github）
[^25]: zilliztech/VectorDBBench (VDBBench) — https://github.com/zilliztech/VectorDBBench（github）
[^26]: VectorDBBench Performance Leaderboard (Zilliz) — https://zilliz.com/benchmark（web）
[^27]: Milvus Limits (Official Documentation) — https://milvus.io/docs/limitations.md（web）
[^28]: Qdrant Capacity Planning (Official Documentation) — https://qdrant.tech/documentation/capacity-planning/（web）
[^29]: AGILE study of the gamma-ray emission from the SNR G78.2+2.1 (Gamma Cygni) — https://arxiv.org/abs/1905.01255（arxiv, 2019-05-03）
[^30]: Qdrant v1.18.0 Release Notes — https://github.com/qdrant/qdrant/releases/tag/v1.18.0（github, 2025-05-11）