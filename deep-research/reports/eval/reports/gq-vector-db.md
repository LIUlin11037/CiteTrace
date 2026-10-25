# 向量数据库主流选型差异与 pgvector 的适用边界

## 摘要

四个主流选型在架构定位上分属两类。Qdrant、Milvus、Weaviate 是独立的向量搜索引擎/数据库，各自围绕 HNSW 构建了不同的分布式、过滤与索引体系：Milvus 是云原生四层分离架构，控制面与数据面解耦 [^7]；Qdrant 以「向量索引 + payload 索引」双索引耦合为核心设计 [^5]；Weaviate 采用 vector-first 存储并提供 HNSW/Flat/Dynamic/HFresh 四种索引 [^8]。pgvector 则是 Postgres 扩展，向量与业务数据同库，直接继承 ACID、PITR、JOIN 等数据库能力 [^4]，其近似索引只有 HNSW 与 IVFFlat 两种 [^4]，并有明确的维度上限（vector 2,000 维、halfvec 4,000 维、bit 64,000 维、sparsevec 1,000 非零元素）[^4][^13]。

关于「pgvector 何时合理」，可从证据中确立的判据是：数据已在 Postgres 内、需要事务一致性与 JOIN、向量规模与维度在上限之内、可接受 HNSW/IVFFlat 两种索引与后过滤语义、混合检索愿意用全文检索 + RRF 自行拼装。但**「pgvector 与专用库的召回-延迟差距有多大」这一核心量化问题，现有证据无法给出任何数值结论**——详见「矛盾与不确定性」。

---

## 一、索引内核：四者共享 HNSW 血统，但暴露的参数与索引族不同

HNSW 是这四者共同的近似最近邻基础。其标准定义为：一种纯图结构的 ANN 索引，不需要 IVF 式粗聚类/倒排桶等额外搜索结构 [^1][^22]；索引由多层邻近图组成，每层对应存储元素的一个嵌套子集，构成层级结构 [^1]；每个元素出现的最高层按指数衰减的概率分布随机选取，从而把链接按特征距离尺度分离 [^1]；搜索从上层开始，结合尺度分离相比 NSW 提升性能并带来对数复杂度的扩展性 [^1][^22]；并采用邻近图邻居选择启发式，在高召回区间与高度聚簇数据上显著提升性能 [^1][^22]。论文作者称该通用度量空间索引显著优于此前的开源 SOTA 纯向量方法，这属作者自述主张 [^1][^22]。

**pgvector** 的近似索引只有两种：HNSW 与 IVFFlat [^4]。HNSW 参数 m 默认 16、ef_construction 默认 64，查询参数 hnsw.ef_search 默认 40；IVFFlat 参数 lists，查询参数 ivfflat.probes 默认 1 [^4][^13]。README 明确 HNSW 相比 IVFFlat 有更好的速度-召回权衡，但构建更慢、内存占用更高；无索引时默认走精确最近邻搜索，召回完美 [^13]。IVFFlat 调参建议：lists 在 1M 行以内取 rows/1000、超过 1M 取 sqrt(rows)，probes 起始值取 sqrt(lists) [^13]。并行构建受 max_parallel_maintenance_workers 控制（默认 2，README 建议调高），图能放进 maintenance_work_mem 时构建显著更快 [^13]。

**Qdrant** 的核心设计是向量索引与传统 payload 索引的组合，文档明确「仅有向量索引不足以让向量检索与过滤有效协同」[^5]。索引在 segment 级别存在且相互独立，但参数按 collection 全局配置 [^5]。量化方面提供四种方法：TurboQuant、标量量化（float32→uint8，4x 压缩，v1.1.0 起）、二值量化（1 bit，内存降 32 倍、最高 40x 加速，v1.5.0 起）、产品量化（最高 64x 压缩）[^28]。二值量化仅对高维且分量呈中心化分布的向量有效，官方给出 text-embedding-3-large 3072d 在 DBpedia 1M 上 0.9966 recall（3x oversampling + rescoring）等实测 [^28]。稠密向量维度范围 1–65,536，稀疏向量长度动态分配 [^29]。

**Milvus** 构建在 Faiss、HNSW、DiskANN、SCANN 等检索库之上 [^7]，浮点向量索引类型包括 FLAT、IVF_FLAT、IVF_SQ8、IVF_PQ、GPU_IVF_FLAT、GPU_IVF_PQ、HNSW、DISKANN，metric 支持 L2/IP/COSINE [^32]。另有二值向量索引 BIN_FLAT/BIN_IVF_FLAT（metric 为 JACCARD/HAMMING，属二值数据类型而非对浮点向量做二值量化）与稀疏向量索引 SPARSE_INVERTED_INDEX（支持 IP 与 BM25）[^32]。IVF_SQ8 由 IVF 聚类与 SQ8 量化组成，SQ8 用 8-bit 整数替代 32-bit 浮点，nlist 默认 128、nprobe 默认 8 [^31]。

**Weaviate** 支持四种向量索引：HNSW（默认，自研实现，完整 CRUD，查询对数复杂度）、Flat（单列表全量扫描，内存极省但检索时间线性增长）、Dynamic（从 Flat 起步，超阈值自动切 HNSW）、HFresh（基于聚类，用 HNSW 做质心索引，大部分数据留磁盘）[^8]。HNSW 内存占用与向量数及连接数成正比：节点约 4B×维度（1M 向量时 2–12GB，100M 时 200–1200GB），边约 10B×20 连接 [^8]。ef 可显式设置或设为 -1 启用动态 ef，由 dynamicEfMin/dynamicEfMax/dynamicEfFactor 依查询 limit 自动计算 [^8]。

---

## 二、过滤（pre-filter）语义：最容易被忽略的选型分水岭

过滤与向量检索如何协同，是四者差异最实质的地方，也直接决定「带条件查询」的召回是否可控。

**pgvector**：近似索引下过滤在索引扫描之后应用，可能返回不足结果；自 0.8.0 起支持 iterative index scans 自动扩大扫描，受 hnsw.max_scan_tuples（默认 20000）与 ivfflat.max_probes 限制 [^4]。0.8.0 同时改进了过滤场景下的代价估算以改善索引选择 [^12]。

**Qdrant**：使用 filterable HNSW——只有在 payload 索引创建之后生成的 HNSW 索引才会获得额外的 filter-aware 边；若先灌数据后建 payload 索引，需重建 HNSW 索引才能受益 [^5]。payload 索引还用于精确估算过滤基数，帮助查询规划器选择检索策略，即过滤与向量检索存在代价模型耦合 [^5]。官方建议 payload 索引在数据写入前创建，且只为实际用于过滤的字段建索引 [^5]。strict mode 可将 unindexed_filtering_retrieve 设为 false，使对未索引字段的过滤查询直接报错，Qdrant Cloud 默认对所有 collection 应用该设置 [^5]。

**Milvus**：用 bitset（0/1 位数组）承载属性过滤结果，满足条件的实体标记为 1；过滤位图与删除位图经布尔逻辑合并为 result_bitset，只有其中有效的实体才进入后续 search/query 计算阶段 [^36]。这说明过滤在检索计算之前以位图形式完成 [^36]。

**Weaviate**：基于 pre-filtering，过滤器在向量检索执行之前构建，且不要求退化为暴力检索 [^37]。每个 shard 内倒排索引与 HNSW 索引并存：先用倒排索引生成候选 allow-list（uint64 id 列表），再传入 HNSW，HNSW 正常沿边遍历但只把在 allow-list 上的 id 加入结果集 [^37]。文档明确列出 post-filtering 的两大缺点：无法预测结果集大小；若过滤条件极具限制性，原始向量检索结果中可能一个匹配都没有 [^37]。HNSW 支持三种过滤策略 sweeping、acorn、pathseer，自 v1.34 起 ACORN 为默认；PathSeer 于 v1.40 加入，不额外存储数据、无导入成本，已有 collection 可随时切换而无需重建索引 [^37]。

这一节的实际含义：pgvector 的过滤是「索引扫描后应用 + 迭代扩大扫描」，而 Qdrant/Weaviate/Milvus 都在索引结构或查询规划层面做了过滤感知。对于「高选择性过滤 + 向量检索」的负载，这一差异比原始 QPS 更关键——但现有证据没有给出跨引擎的可比数值。

---

## 三、分布式、扩展与多租户

**Milvus** 采用数据面/控制面分离，分四层：Access Layer（无状态 Proxy，MPP 架构下聚合中间结果）、Coordinator（全集群唯一 active，管理 DDL/DCL/TSO）、Worker Nodes（Streaming Node 作分片级 mini-brain、Query Node 加载历史数据、Data Node 做 compaction 与索引构建）、Storage（etcd 存元数据、MinIO/S3/Azure Blob 存索引文件、Kafka/Pulsar/Woodpecker 作 WAL）[^7]。写入路径为 Proxy → Streaming Node → WAL → 实时可查 → segment 满转 sealed → Data Node 建索引写对象存储 → Query Node 加载替换 growing data；查询为多级归并 [^7]。Worker 节点因存储计算分离而无状态，可在 Kubernetes 上横向扩缩容 [^7]。

**Qdrant** 分布式自 v0.8.0 起支持，cluster.enabled=true 开启，节点间 p2p 默认端口 6335，共识用 Raft（tick_period_ms 默认 100）[^6]。开启分布式不会自动复制数据，需创建 replication_factor ≥ 2 的 collection，且 shard 数应为节点数的整数倍 [^6]。官方建议每节点至少 2 个 shard，预期大幅增长建议 12 个 shard，可从 1 节点扩到 2/3/6/12 节点而无需 reshard [^6]。Resharding 仅在 Qdrant Cloud 提供（v1.13.0+），自托管需重建 collection，shard 再平衡需手动完成 [^6]。支持 user-defined sharding（v1.7.0+），官方明确其多租户用途：每个租户数据隔离在独立 shard [^6]。

**Weaviate** 元数据复制用 Raft（基于 Hashicorp raft 库），数据复制采用无主设计、最终一致，所有节点都可接受读写 [^39]。复制因子按 collection 配置，RF=3 常被采用；读写一致性可调（ALL/QUORUM/ONE，v1.18 起），默认等待 QUORUM [^39]。多租户默认关闭，需设 multiTenancyConfig.enabled = true；每个 tenant 存储在独立 shard 上，一个 tenant 的数据对另一个 tenant 不可见 [^38]。tenant 可设 ACTIVE/INACTIVE/OFFLOADED 状态（v1.21 起）[^38]。

**pgvector** 没有内置分布式或租户机制：多租户建议用 list partitioning 或独立表（PARTITION BY LIST(customer_id)）实现租户隔离 [^4]。它继承的是 Postgres 本身的运维体系（ACID、PITR、JOIN）[^4]。

多租户能力对比在 Milvus 文档中有最细的分级：Database 级（默认最多 64 租户，物理隔离，支持 RBAC）、Collection 级（默认最多 65,536 collection）、Partition 级（每 collection 最多 1,024 partition，共享 schema，不支持 RBAC）、Partition Key 级（共享单 collection 与 schema，按 key 自动路由到 16 个物理 partition，支持数百万租户但隔离最弱）[^35]。官方扩展性排序为 Partition Key > Partition > Collection > Database，隔离强度排序为 Database = Collection > Partition > Partition Key；跨租户检索仅 Partition 与 Partition Key 级支持 [^35]。

---

## 四、混合检索与全文能力

**Qdrant**：混合与多阶段查询自 v1.10.0 起可用，支持稠密向量、稀疏向量（如 SPLADE++）、以及对带全文索引的 text payload 做 BM25 全文检索三种方式 [^30]。核心构件是 prefetch（可嵌套），提供 rrf 与 dbsf 两种融合方式；RRF 中常数 k 默认 2、排名权重默认 1，使用零基排名 [^30]。注意 offset 只影响主查询，prefetch 的 limit 至少需为主查询 limit + offset，否则可能返回空结果 [^30]。

**Milvus**：全文检索用 BM25 打分，接受原始文本、自动转稀疏向量，schema 需主键 + 设 enable_analyzer=True 的文本字段 + SPARSE_FLOAT_VECTOR 字段，通过 Function(function_type=FunctionType.BM25) 定义映射 [^33]。混合检索通过多向量字段上的多路 ANN 实现，覆盖 Sparse-Dense 与多模态两类场景 [^34]。单 collection 向量字段数默认受限，可通过 proxy.maxVectorFieldNum 调至最多 10 个 [^34]。

**Weaviate**：仓库 topics 含 hybrid-search、generative-search [^11]，但给定证据未展开其混合检索的具体算子与融合公式，**现有公开资料（本次证据范围内）不足以确认其混合检索的实现细节**。

**pgvector**：混合检索需与 Postgres 全文检索（tsvector/ts_rank_cd）组合，并用 Reciprocal Rank Fusion 或 cross-encoder 融合结果，非内置混合查询算子 [^4]。

---

## 五、许可与商业托管

**许可**：pgvector 为 PostgreSQL/UC Berkeley 风格宽松许可，允许为任何目的免费使用、复制、修改和分发，保留版权声明与免责条款即可，无 copyleft 要求 [^16]。Qdrant 为 Apache-2.0 [^9]；Milvus 为 Apache-2.0 [^10]。Weaviate 是双许可：源码「variously licensed under the BSD-3-Clause License, or the Weaviate License」，"wl" 目录之外的源码为 BSD-3-Clause，"wl" 目录及其子目录版权归 Weaviate BV、all rights reserved，仅可通过单独企业许可获得，企业功能通过 license key 启用 [^27]。因此 Weaviate 并非纯 BSD-3-Clause，核心为 BSD-3-Clause 但企业功能含商用限制 [^27]。仓库元数据中 Weaviate 的 license 字段为 NOASSERTION，需以 LICENSE 正文为准 [^11][^27]。

**托管**：Qdrant Cloud 分 Free（单节点 0.5 vCPU/1GB RAM/4GB Disk）、Standard（按用量计费）、Premium（有最低消费，含 SSO、Private VPC Links、99.9% SLA）、Hybrid/Private Cloud；按 compute/memory/storage/备份/token 小时计费；SLA 分层 Free 99.5%、Standard 单 AZ 99.9%/多 AZ 99.95%、Premium 99.9% [^17]。Zilliz Cloud（Milvus 托管）Free 层 5GB 存储、每月 2.5M vCU、最多 5 collection；Dedicated 按类型给出容量档位（768 维评估）：Performance-optimized 每 CU 约 200 万向量、QPS 500–1500、延迟 10ms、从 $63/百万向量/月；Capacity-optimized 每 CU 约 800 万向量、QPS 100–300、从 $16/百万向量/月；Tiered-storage 每 CU 约 4000 万向量、QPS 10–50、从 $5/百万向量/月 [^18]。Weaviate Cloud 免费层：100,000 objects、1GB 内存、10GB 磁盘、1 个 collection（最多 3 tenant）[^19]。

**社区规模**（2026-10-25 前后快照）：milvus 46,471 stars / 4,311 forks（Go）、qdrant 35,239 / 2,774（Rust）、pgvector 23,472 / 1,365（C）、weaviate 16,910 / 1,429（Go），四者 last push 均在 2026-10-23 至 10-25 之间，均为活跃仓库 [^15]。同期还出现 alibaba/zvec（16,147 stars，in-process 向量库）与 apache/doris（16,139 stars，实时分析+混合检索）等新进入者 [^15]。

---

## 六、pgvector 何时是合理选择

把上述证据收敛成可操作的判据：

**1）数据已经在 Postgres 内，且向量检索需要与业务数据强一致地联合查询。** pgvector 是 Postgres 扩展，向量与业务数据同库存储，README 明确主张附带 ACID、point-in-time recovery、JOINs 等 Postgres 特性 [^4]。这是专用向量库在架构上无法直接复制的能力——它们要么需要外部数据库承担事务与关联，要么依赖各自的 payload/标量字段体系。

**2）向量维度与规模落在硬上限之内。** vector 最多 2,000 维、halfvec 最多 4,000 维、bit 最多 64,000 维、sparsevec 最多 1,000 个非零元素，HNSW 与 IVFFlat 均适用（IVFFlat 不支持 sparsevec）[^4][^13]。超过 2,000 维的稠密向量必须走 halfvec 或降维，这是硬约束而非调优项。

**3）可接受「只有 HNSW 与 IVFFlat 两种近似索引」这一索引族上限。** 若负载需要 DiskANN 式磁盘驻留索引、产品量化、二值量化等更丰富的索引/量化组合，pgvector 本体不提供；社区扩展 pgvectorscale 在 pgvector 之上引入受微软 DiskANN 启发的 StreamingDiskANN 索引、SBQ 量化与 label-based 过滤 [^14]，但该 README 顶部注明 Timescale 正在开发新一代 pg_vectorsearch 索引（用分区+压缩替代图索引），目前为 early alpha、尚未 production-ready [^14]。另需注意 pgvectorscale 仓库元数据可确认存在（Rust、PostgreSQL 许可、约 3,152 stars、最近 push 2026-10-14），但本次抓取未读到其 README 正文 [^24]。

**4）过滤负载不是「高选择性 + 高召回」的极端组合。** pgvector 的近似索引下过滤在索引扫描之后应用，可能返回不足结果，靠 iterative index scans 自动扩大扫描兜底，并受 hnsw.max_scan_tuples（默认 20000）与 ivfflat.max_probes 限制 [^4]。相比之下 Qdrant 有 filterable HNSW 与过滤基数估算 [^5]，Weaviate 有倒排索引生成 allow-list 的 pre-filter 与 ACORN/PathSeer 策略 [^37]，Milvus 有 bitset 前置过滤 [^36]。若业务是「按租户/类别强过滤后再做向量检索」，需要重点验证 pgvector 的召回稳定性。

**5）混合检索可以接受自行拼装。** pgvector 需与 Postgres 全文检索（tsvector/ts_rank_cd）组合，并用 RRF 或 cross-encoder 融合，非内置混合查询算子 [^4]；而 Qdrant 提供 prefetch + rrf/dbsf 的原生 Query API [^30]，Milvus 提供 BM25 Function + 多向量混合检索 [^33][^34]。

**6）多租户可以接受「用分区/独立表自己搭」。** pgvector 无内置租户机制，官方建议 list partitioning 或 PARTITION BY LIST(customer_id) [^4]；Milvus 提供 Database/Collection/Partition/Partition Key 四级策略 [^35]，Weaviate 提供 per-tenant shard 隔离与 tenant 状态管理 [^38]，Qdrant 提供 user-defined sharding 按租户分 shard [^6]。

**7）运维上愿意接受 Postgres 的构建代价。** HNSW 构建更慢、内存占用更高 [^13]；并行构建需调高 max_parallel_maintenance_workers，图放不进 maintenance_work_mem 时会提示 "hnsw graph no longer fits into maintenance_work_mem" [^13]。版本演进上，0.6.0 引入 HNSW 并行构建并降低内存与 WAL 生成 [^12]，0.7.0 一次性引入 halfvec/sparsevec/bit 类型与 binary_quantize 等函数 [^12]，0.8.0 引入 iterative index scans 与过滤代价估算改进 [^12]，0.9.0（unreleased）加入 HNSW 异步 I/O 并放弃 Postgres 13/14 支持 [^12]。当前最新发布版为 0.8.7（2026-10-01）[^12]。

**反向判据（何时不该选 pgvector）**：需要云原生存算分离与 K8s 横向扩缩容 [^7]、需要内置多级多租户与 RBAC [^35]、需要原生混合检索算子 [^30][^33][^34]、需要丰富量化组合以压内存 [^28][^31]、需要 DiskANN 式磁盘索引 [^14]——这些场景专用库有明确的结构性优势。

---

## 矛盾与不确定性

**（1）pgvector 与专用向量库的召回-延迟差距：无任何可用数值。** 这是本课题最核心的量化问题，但现有证据完全无法支撑数值结论。ANN-Benchmarks 官方结果站的 sift-128-euclidean 与 glove-100-angular（k=10）页面仅列出图表标题与坐标轴含义，未内嵌可读的 recall-QPS 或 index size 数值，数据以交互式图表呈现 [^20][^21]；首页虽列出 pgvector、qdrant、Milvus(Knowhere)、weaviate 等算法清单，但文本层同样不含具体召回率或 QPS 数值 [^3]。更关键的是，ANN-Benchmarks README 已明确声明不再积极维护，建议改投 VIBE 等其他基准 [^2]。Qdrant 自建基准页声明在同一台机器运行所有引擎、内存限制 25GB、代码开源，但**其比较的引擎为 Qdrant、Milvus、Weaviate、Elasticsearch、Redis，正文未提及 pgvector 或 pgvectorscale** [^23]；该页的定性结论（Qdrant 在几乎所有场景取得最高 RPS 与最低延迟、Milvus 索引时间最快、Weaviate 改进最小）为厂商自述，且具体数值同样以交互式图表呈现、抓取文本未包含 recall@10 与 p95 延迟数字 [^23]。pgvectorscale README 的「99% recall 下 p95 延迟低 28 倍、吞吐高 16 倍、成本低 75%」主张只对比 Pinecone 的 storage-optimized (s1) 索引，且为厂商自述、无第三方复现 [^14]。因此：**任何「pgvector 比专用库慢 N 倍/快 N 倍」的说法都不应基于本次证据给出。**

**（2）两条 100M 规模基准仓库属不可信证据，已剔除。** jhondados/vector-database-benchmark 声称在 100M 向量、1536 维、HNSW 条件下对比 7 个数据库，结果表列出 Qdrant Recall@10 0.991 / 4,200 QPS / 12ms P99、pgvector 0.987 / 1,800 QPS / 28ms P99、Weaviate 0.989 / 5,100 QPS / 9ms P99 [^25]；nathaniel-gordon/vectorbench 声称在 100M 规模、HNSW 与 IVF-PQ 下评测 7 个引擎 [^26]。但两者均为 0 star、0 fork、个人署名、无实验环境与原始结果链接，license 为 NOASSERTION，且 README 徽章与描述结构高度雷同，疑似模板化生成内容 [^25][^26]；其结论方向（Weaviate 5,100 QPS 领先）与 Qdrant 官方基准（Qdrant 领先）相反 [^23]。**本报告不采用这两条来源的任何数值做横向比较。**

**（3）迁移案例与规模阈值：实质无证据。** 关于「2025–2026 年从专用向量库迁回 Postgres/pgvector 或反向迁移的案例及规模阈值（向量条数、QPS、团队规模）」，本次证据中没有任何具名企业案例、迁移原因、迁移前后指标或规模阈值。仅有 Qdrant 定价页 FAQ 一句「可从 Qdrant OSS 迁移到 Qdrant Cloud」[^17] 与 Zilliz 的「zero downtime migration」营销表述 [^18]，两者都是同一产品线内的迁移而非跨产品迁移；两个 benchmark 仓库明确不含迁移内容 [^25][^26]。**该子问题现有公开资料不足以确认。**

**（4）Weaviate 混合检索实现细节缺失。** 仓库 topics 含 hybrid-search、generative-search [^11]，但给定证据未展开其融合算子、公式与参数，无法与 Qdrant 的 rrf/dbsf [^30] 或 Milvus 的 BM25 Function + 多向量混合 [^33][^34] 做机制级对比。

**（5）pgvectorscale 的基准主张无法独立核验。** 其 README 的 28x/16x/75% 主张为厂商自述、只对比 Pinecone、无第三方复现 [^14]；仓库元数据可确认存在与活跃度 [^24]，但本次抓取 README 正文为 unavailable，无法据仓库正文复核其数据集、对比引擎与 recall/延迟数值 [^24]。

**（6）HNSW 论文的 SOTA 主张属作者自述。** 「显著优于此前的开源 SOTA 纯向量方法」是论文作者的主张，需交叉验证 [^1][^22]；本次证据中没有独立的第三方复现该结论。

**（7）版本与时间戳的时效性。** 证据中 pgvector CHANGELOG 含 2026 年的多个补丁版本（0.8.2–0.8.7）与 unreleased 的 0.9.0 [^12]，仓库 last push 为 2026-10-24 [^4][^15]；Qdrant/Milvus/Weaviate 文档中的版本号（如 Qdrant v1.20.0、Weaviate v1.40）表明功能仍在快速迭代 [^28][^37]。本报告的参数默认值与功能可用版本以抓取时点为准，实际选型前需按目标版本复核。

## 参考来源

[^1]: Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs (HNSW) — https://arxiv.org/abs/1603.09320v4（arxiv, 2016-03-30 (v1); v4 revised 2018-08-14）
[^2]: erikbern/ann-benchmarks — README (Status of ANN-Benchmarks) — https://github.com/erikbern/ann-benchmarks（github, created 2015-05-28; last push 2026-07-10）
[^3]: ANN-Benchmarks 官方结果站首页（数据集与算法清单） — https://ann-benchmarks.com/（web）
[^4]: pgvector/pgvector — README（官方仓库正文） — https://github.com/pgvector/pgvector（github）
[^5]: Qdrant Documentation — Indexing（向量索引与 payload 索引） — https://qdrant.tech/documentation/concepts/indexing/（web）
[^6]: Qdrant Documentation — Distributed Deployment — https://qdrant.tech/documentation/scaling/distributed_deployment/（web）
[^7]: Milvus Documentation — Architecture Overview — https://milvus.io/docs/architecture_overview.md（web）
[^8]: Weaviate Documentation — Vector Index（HNSW / Flat / Dynamic / HFresh） — https://weaviate.io/developers/weaviate/concepts/vector-index（web）
[^9]: qdrant/qdrant — 仓库元数据与主题标签 — https://github.com/qdrant/qdrant（github）
[^10]: milvus-io/milvus — 仓库元数据与主题标签 — https://github.com/milvus-io/milvus（github）
[^11]: weaviate/weaviate — 仓库元数据与主题标签 — https://github.com/weaviate/weaviate（github）
[^12]: pgvector CHANGELOG（0.6.0–0.9.0 unreleased 版本演进） — https://github.com/pgvector/pgvector/blob/master/CHANGELOG.md（github, 2026-10-01）
[^13]: pgvector README（类型维度上限、HNSW 参数默认值、并行构建配置） — https://github.com/pgvector/pgvector/blob/master/README.md（github, 2026-10-24）
[^14]: timescale/pgvectorscale README（StreamingDiskANN、SBQ、label 过滤与 50M 基准主张） — https://github.com/timescale/pgvectorscale/blob/main/README.md（github）
[^15]: GitHub 仓库检索快照：主流向量库 stars 横向对比（2026-10-25 前后） — https://github.com/search?q=vector+database&s=stars（github, 2026-10-25）
[^16]: pgvector LICENSE 文件正文（许可条款） — https://github.com/pgvector/pgvector/blob/master/LICENSE（github）
[^17]: Qdrant Cloud Pricing — 托管分层与计费模式 — https://qdrant.tech/pricing/（web）
[^18]: Zilliz Cloud Pricing（Milvus 商业托管）— 按集群类型给出的向量容量/QPS/单价 — https://zilliz.com/pricing（web）
[^19]: Weaviate Cloud Pricing — 免费层配额 — https://weaviate.io/pricing（web）
[^20]: ANN-Benchmarks — sift-128-euclidean (k = 10) 结果页 — https://ann-benchmarks.com/sift-128-euclidean_10_euclidean.html（web）
[^21]: ANN-Benchmarks — glove-100-angular (k = 10) 结果页 — https://ann-benchmarks.com/glove-100-angular_10_angular.html（web）
[^22]: Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs — https://arxiv.org/abs/1603.09320（arxiv, 2016-03-30 (v1); latest v4 2018-08-14）
[^23]: Vector Search Benchmarks — Qdrant — https://qdrant.tech/benchmarks/（web, 2024-06）
[^24]: timescale/pgvectorscale (repository metadata only — README unavailable) — https://github.com/timescale/pgvectorscale（github, 2026-10-14）
[^25]: jhondados/vector-database-benchmark — Vector Database Benchmark 2026 (README) — https://github.com/jhondados/vector-database-benchmark（github, 2026-08-05）
[^26]: nathaniel-gordon/vectorbench — VectorBench 100M Scale Vector Database Benchmark Suite (README) — https://github.com/nathaniel-gordon/vectorbench（github, 2026-09-01）
[^27]: weaviate/weaviate LICENSE (repository root license file) — https://raw.githubusercontent.com/weaviate/weaviate/main/LICENSE（github）
[^28]: Quantization - Qdrant Documentation — https://qdrant.tech/documentation/guides/quantization/（web）
[^29]: Vectors - Qdrant Documentation — https://qdrant.tech/documentation/concepts/vectors/（web）
[^30]: Hybrid Queries - Qdrant Documentation — https://qdrant.tech/documentation/concepts/hybrid-queries/（web）
[^31]: IVF_SQ8 - Milvus Documentation — https://milvus.io/docs/ivf-sq8.md（web）
[^32]: Index Vector Fields - Milvus Documentation — https://milvus.io/docs/index-vector-fields.md（web）
[^33]: Full Text Search - Milvus Documentation — https://milvus.io/docs/full-text-search.md（web）
[^34]: Multi-Vector Hybrid Search - Milvus Documentation — https://milvus.io/docs/multi-vector-search.md（web）
[^35]: Implement Multi-tenancy — Milvus Documentation (v3.0.x, Concepts) — https://milvus.io/docs/multi_tenancy.md（web）
[^36]: Bitset — Milvus Documentation (v3.0.x, Concepts) — https://milvus.io/docs/bitset.md（web）
[^37]: Filtering — Weaviate Documentation (Concepts) — https://weaviate.io/developers/weaviate/concepts/filtering（web）
[^38]: Multi-tenancy — Weaviate Documentation (Manage collections) — https://weaviate.io/developers/weaviate/manage-collections/multi-tenancy（web）
[^39]: Cluster architecture — Weaviate Documentation (Replication architecture) — https://weaviate.io/developers/weaviate/concepts/replication-architecture/cluster-architecture（web）