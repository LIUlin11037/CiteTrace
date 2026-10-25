# 生产环境 LLM Agent 可观测性：一条 trace 应该记录什么

## 摘要

生产环境 LLM Agent 的可观测性目前呈现「传输层已收敛、语义层仍在分裂」的格局。传输层事实标准是 OpenTelemetry：OpenAI/Anthropic 等提供商调用、检索、工具执行、embedding 被建模为不同 span 类型并配以 `gen_ai.*` 属性 [^1]，而 OpenTelemetry 官方的 GenAI 语义约定已迁出主站、进入独立仓库快速演进，尚未 Stable [^2][^42]。在此之上，LangSmith、Langfuse、Datadog、Braintrust 各自定义了 trace/span 数据模型，字段命名与层级并不一致 [^22][^23][^25][^26][^34][^39]。一条生产 trace 的最小可用信息集可归纳为六层：身份与归属、层级结构、时序与延迟、输入输出、token 与成本、状态与错误；在此之上按需叠加检索文档、工具参数、评估分数与反馈、脱敏标记。成本与保留期是 schema 设计的硬约束而非运维细节：Langfuse 按 traces+observations+scores 计费 [^30]，LangSmith 按 base/extended 保留期分档计费并存在自动升级机制 [^32]，采样决策在 trace 级别一次性做出 [^31]。学术侧，MAST 用 150 条 trace 构建了 14 种失败模式的分类体系 [^20]，RITE 证明静默工具错误可使任务成功率减半 [^21]，但「从 trace 自动定位根因」的方法类证据仍然缺失。

---

## 一、传输与语义标准层：trace 的骨架由谁定义

**OpenTelemetry GenAI 语义约定是当前唯一跨厂商的字段规范来源，但处于 Development 状态。** 规范定义的 GenAI span 类型包括 Inference、Embeddings、Retrievals、Fetch response、Memory、Execute tool [^1]。以 Embeddings span 为例，其 `gen_ai.operation.name` SHOULD 为 `embeddings`，span name SHOULD 为 `{gen_ai.operation.name} {gen_ai.request.model}`，span kind SHOULD 为 CLIENT；Required 属性为 `gen_ai.operation.name` 与 `gen_ai.provider.name`，Conditionally Required 为 `error.type`（出错时）、`gen_ai.request.model`（可用时）、`server.port`（设置了 `server.address` 时），Recommended 包括 `gen_ai.embeddings.dimension.count`、`gen_ai.request.encoding_formats`、`gen_ai.response.model`、`gen_ai.usage.input_tokens`、`server.address` [^1]。Retrievals span 的 span name SHOULD 为 `{gen_ai.operation.name} {gen_ai.data_source.id}`，Required 仅 `gen_ai.operation.name`，`gen_ai.retrieval.documents`（检索到的文档内容）为 Opt-In，需显式开启才记录 [^1]。

**Agent 生命周期被建模为独立 operation 类型。** `gen_ai.operation.name` 的已知取值包含 `chat`、`create_agent`、`create_memory` 等 [^1]，这意味着「创建 agent」本身是一个可被单独观测的操作，而非仅作为父 span 存在。

**采样相关属性必须在 span 创建时提供。** 规范明确：对采样决策重要的属性 SHOULD 在 span 创建时（at span creation time）提供，包括 `gen_ai.operation.name`、`gen_ai.provider.name`、`gen_ai.request.model`、`server.address`、`server.port` [^1]。这条约束直接决定了 SDK 实现顺序——不能等响应回来再补属性，否则基于属性的采样器拿不到决策依据。

**token 用量属性存在层级包含关系，且计费口径优先。** 细分用量属性是总量与聚合计数的子集：输入 100 文本 token（其中 40 为缓存）+ 200 图像 token 时，`gen_ai.usage.input_tokens` 记为 300，`gen_ai.usage.cache_read.input_tokens` 记为 40 [^1]。当供应商同时暴露计费单位与 token 数（如 Cohere 的 `usage.billed_units` 与 `usage.tokens`）时，instrumentation SHOULD 上报计费口径的计数，以匹配客户实际被计费的单位 [^1]。

**该规范尚未稳定，且已迁出主站。** opentelemetry.io 上的 GenAI 语义约定页面（含 agent spans 页面）已标记为迁移，不再在该仓库维护，正文迁至独立的 OpenTelemetry GenAI semantic conventions 仓库 [^2][^42]。Langfuse 也据此说明其 OTel 属性映射需支持 GenAI 生态中的额外属性，理由是 GenAI Semantic Conventions 仍在演进 [^28]。工程含义：不要把 `gen_ai.*` 字段名当作长期稳定的存储 schema，应在摄入层做映射而非直接落库。

**生态侧已有多个「OTel 之上的约定层」。** OpenInference 自述为与 OpenTelemetry 互补的约定与插件，规范以 markdown 维护在 spec 目录，设计目标是提供对 LLM 调用及周边上下文（向量库检索、外部工具使用）的洞察，且与传输/文件格式无关 [^9]。OpenLLMetry 自述为构建在 OpenTelemetry 之上的扩展，包含面向 LLM 提供商与向量数据库的标准 OTel instrumentations，SDK 仍输出标准 OTel 数据，并声明其语义约定已成为 OpenTelemetry 的一部分 [^8]。OpenInference 还提供 span processors 做跨 instrumentation 库的归一化，其中包括 `openinference-instrumentation-openllmetry`（用于 OpenLLMetry/Traceloop traces）[^9]——这说明「多套约定并存、需要转换层」是当前生态的常态而非例外。

---

## 二、一条 trace 应该记录哪些信息：六层最小集

综合 OTel 规范与三家平台的数据模型，可把一条生产 trace 的字段需求分为六层。

**第 1 层：身份与归属。** 这是把 trace 关联到用户、会话、环境、版本的字段。Langfuse 的 trace 级属性包括 `user_id`、`session_id`、`tags`、`metadata`，且这些属性存在于该 trace 内每个 observation 上，由 SDK 自动传播 [^25]；此外可添加 Environments（区分 production/staging/development）、Tags、User、Metadata（自定义 key-value）、Releases & Versions [^25]。LangSmith 的 trace 级富化信息包括 feedback（tag + score，绑定 run ID，可为连续或离散）、tags（字符串）、metadata（key-value 对）[^22]。Datadog 侧，Sensitive Data Scanner 的扫描范围明确覆盖 LLM workflow metadata [^36]，说明 metadata 是被平台当作可能含敏感信息的一等字段对待的。

**第 2 层：层级结构。** 需要能表达「trace → 多个 run/observation → 嵌套子步骤」。LangSmith 中 agent 执行的每个工作单元（模型调用、工具调用、信息检索）记录为一个 run，单次操作的所有 run 归入一个 trace；run 对应 OpenTelemetry 中的 span，通过唯一 trace ID 绑定到 trace [^22]。Langfuse 中 observation 是应用的单个步骤（LLM 调用、工具调用、检索步骤等），可嵌套以反映应用结构，trace 是共享同一 `trace_id` 的所有 observation 的逻辑分组 [^25]。AgentOps 的 trace 数据模型体现为 span 层级：session span 作为所有其他 span 的根，另有 agent span、operation/task span、workflow span，并支持嵌套 [^11]。

层级还需要一个**可排序的物化表示**。LangSmith 的 `dotted_order` 是分层可排序键，格式为 `<run_start_time>Z<run_uuid>.<child_run_start_time>Z<child_run_uuid>…`；`id` 等于 `dotted_order` 最后 36 字符，`trace_id` 等于 `dotted_order` 中第一个 UUID，`parent_run_id` 为倒数第二个 UUID [^23]。这是一个值得借鉴的设计：用单一字符串同时编码父子关系与时间顺序，避免重建树时依赖多次查询。

**第 3 层：时序与延迟。** 基础字段是 `start_time`、`end_time` [^23]。LLM 场景需要额外的首 token 时间：LangSmith 的 `first_token_time` 仅对 `run_type='llm'` 且启用 streaming 的 run 有效，记录模型输出首个 token 生成时间 [^23]。Langfuse 的 observation 行包含 `latency` 字段 [^25]。这一层是区分「模型慢」与「工具慢」的唯一依据。

**第 4 层：输入输出。** LangSmith 的 run 记录字段包含 `inputs`（object）与 `outputs`（object）；`run_type='llm'` 时 inputs 通常包含发送给模型的 messages 数组，outputs 通常包含模型返回的 messages 数组 [^23]。Langfuse 的 retriever 类型示例中，input 记录 `{query, topK}`，output 记录检索结果，metadata 记录 `{vectorStore: 'pinecone', similarity: 'cosine'}` [^26]。OTel 侧，检索到的文档内容通过 `gen_ai.retrieval.documents` 记录，且为 Opt-In [^1]——这是一个重要的默认值选择：文档正文体积大且最可能含 PII，默认不采集。

**第 5 层：token 与成本。** LangSmith 的 token 用量字段为 `total_tokens`、`prompt_tokens`、`completion_tokens`（均为 integer），成本字段为 `total_cost`、`prompt_cost`、`completion_cost`（均为 decimal），另有 `price_model_id` 标识所用定价模型 [^23]。Langfuse 的做法更细：对每次 LLM generation 记录 usage details（每种 usage type 消耗的单位数）与 cost details（每种 usage type 的美元成本），两者都记录在 type 为 generation 和 embedding 的 observation 上，且各自按 usage type 拆分；两者均可为 ingested（从 LLM 响应发送）或 inferred（依据 model 参数与含各 usage type 价格的 model definition 推算），同时存在时 ingested 优先 [^27]。usage type 示例包括 input、output，以及随 provider 而异的更细类型如 `cached_tokens`、`audio_tokens` [^27]。

这里有一个容易踩的坑：**推理模型的成本可能无法推断**。Langfuse 文档明确，对 OpenAI o1 等 reasoning 模型，若未 ingest token 数，Langfuse 无法推断成本，因为无法看到 reasoning tokens [^27]。因此「让 SDK 自动算成本」在 reasoning 模型上不成立，必须由 instrumentation 上报。

**第 6 层：状态与错误。** LangSmith 的 `status` 字段取值示例为 `'error'`、`'pending'`、`'success'`，`error` 字段记录错误消息 [^23]。OTel 侧对应 `error.type`（出错时 Conditionally Required）与 span status [^1]。这一层看似简单，但 RITE 的研究表明它恰恰是最不可靠的一层：静默错误不产生显式失败信号 [^21]（详见第五节）。

**在此之上按需叠加的字段：**

- **observation 类型标签。** Langfuse 支持 10 种 observation 类型：event、span、generation、agent、tool、chain、retriever、evaluator、embedding、guardrail [^26]。其中 generation 记录 AI 模型的生成，包含 prompts、token usage 和 costs；agent 决定应用流程（如在 LLM 指导下使用工具）；tool 表示单个动作；chain 是不同应用步骤之间的链接（如把上下文从 retriever 传给 LLM 调用）；retriever 表示只做查找而不改变状态的数据检索步骤；evaluator 表示评估 LLM 输出相关性/正确性/有用性的函数；guardrail 是防御恶意内容或越狱的组件 [^26]。LangSmith 的 `run_type` 枚举共 8 种：chain、llm、embedding、prompt、tool、retriever、parser [^23]。两套枚举的交集（llm/generation、tool、retriever、chain、embedding）可作为跨平台的最小公共类型集；Langfuse 多出的 agent、evaluator、guardrail、event 反映的是更细的职责划分。
- **评估分数与反馈。** LangSmith 的 `feedback_stats` 为该 run 的反馈统计聚合（如 `{correctness: {n, avg}}`）[^23]；Langfuse 把 scores 列为计费实体之一 [^30]。
- **流式事件。** LangSmith 的 run 字段包含 `events`（array，流式场景相关）[^23]。
- **大对象外置。** LangSmith 的 run 字段包含 `inputs_s3_urls`、`outputs_s3_urls` [^23]，即大体积输入输出以 URL 引用而非内联。

---

## 三、成本与保留期：schema 设计的硬约束

**计费口径直接由字段数量决定。** Langfuse 的计费公式为 `Units = Count of Traces + Count of Observations + Count of Scores`，且由 Langfuse 自身功能（LLM-as-a-Judge、Annotation Queues、experiments）产生的 trace/observation/score 同样计入可计费 units [^30]。官方示例量级为 20,070 traces + 119,500 observations + 561 scores = 140,131 units/月，即**单条 trace 平均约 6 个 observation** [^30]。这个数字对 schema 设计有直接含义：把每个细粒度步骤都建成 observation 会让成本随埋点密度线性上升，因此「哪些步骤值得成为独立 observation」是一个成本决策而非纯粹的技术决策。

**保留期是分档的，且存在自动升级陷阱。** LangSmith 按数据保留期把 trace 分为两档：Base 保留 14 天，Extended 保留 180 天，Extended 单价更高 [^32]。自 2026 年 9 月 14 日起，SaaS 客户的最长长期 trace 保留期上限改为 180 天，Enterprise 客户可在该上限内按 workspace 自定义 extended 保留期；保留期变更只对新 trace 生效，已有 trace 保持原档位 [^32]。关键陷阱在于：online evaluators、automation rules、以及显式传 `extend_trace_retention=true` 的 API/SDK feedback 会把 base trace 升级为 extended，从而同时提高保留期与成本；新创建的 online evaluator 与 automation rule 默认开启升级，可显式 opt out [^32]。也就是说，**开启在线评估这个「可观测性最佳实践」会默认推高存储成本**，二者必须在同一张账单上权衡。

Langfuse 侧的保留期按档位区分：Hobby 30 天、Core 90 天、Pro 3 年（含 data retention management）、Enterprise 可定制 [^29]。LangSmith SaaS 保留 trace 数据 180 天，之后永久删除，仅保留有限 metadata 用于用量统计；加入 dataset 的数据可永久保留 [^22]。Datadog 的通用 APM 计价按 Indexed Spans 的保留期分档（7 天 $1.27/1M spans/月、15 天 $1.70/1M、30 天 $2.50/1M，年付口径），但该定价页正文中未出现 LLM Observability / Agent Observability 的独立计价条目 [^35]——现有公开资料不足以确认 Datadog Agent Observability 的独立计价口径。

**采样必须在 trace 级别一次性决策。** Langfuse 的采样在客户端完成，采样率通过 `LANGFUSE_SAMPLE_RATE` 环境变量或 SDK 构造参数 `sample_rate`/`sampleRate` 配置，取值 0–1，默认 1（全量）；采样决策发生在 trace 级别：一条 trace 被采样则其内部所有 observations 与 scores 一并被采样，未被采样则整条 trace 的 observations 与 scores 都不发送 [^31]。Langfuse 同时尊重 OpenTelemetry 的采样决策，可在 OTEL SDK 中配置 sampler（如 `TraceIdRatioBasedSampler(0.2)` 只发送 20% trace）[^31]。这与 OTel 规范「采样相关属性在 span 创建时提供」的要求 [^1] 相互印证：采样是创建期决策，不是导出期过滤。

**批量发送意味着短生命周期进程会丢数据。** Langfuse 不同步发送 trace，而是在本地批量缓冲后后台发送；短生命周期应用必须在退出前显式调用 `flush()`，否则队列中未发送的 trace 会丢失 [^25]。OpenLLMetry 提供 `Traceloop.init(disable_batch=True)` 关闭批量发送以便本地立即看到 traces [^8]。

---

## 四、敏感数据：trace 是 PII 的集中地

**LLM trace 的 inputs/outputs 天然含敏感信息。** Datadog 的 Sensitive Data Scanner 扫描范围明确包含 Agent Observability traces，可扫描 LLM 应用的 inputs 与 outputs，用于防止 prompt、completion 与 LLM workflow metadata 中的 PII、API key、专有信息泄露 [^36]。对 Agent Observability 数据源，支持的动作是 Redact、Partially redact、Hash 三种，Mask 动作在 Agent Observability 列标注为 No（Mask 仅支持 Logs、APM、RUM）[^36]。四种动作语义：Redact 用自选 token 替换整个匹配值；Partially redact 只替换匹配值中的特定部分；Hash 用不可逆唯一标识符替换整个匹配值；Mask 为混淆，持有 Data Scanner Unmask 权限的用户可反混淆查看原文 [^36]。

**执行位置决定合规边界。** 云端扫描时数据先离开客户环境提交到 Datadog 后端，在后端处理阶段完成扫描与脱敏，之后才被索引并展示在 UI；客户环境内扫描则通过 Observability Pipelines 的 Sensitive Data Scanner processor 在日志离开本地前完成脱敏 [^36]。Agent Observability 的扫描采用托管配置模型：组织首次访问 Agent Observability Settings 页面时会自动创建一个默认扫描组 [^36]。云端扫描可为每个产品设置 10% 到 99% 之间的采样率以控制成本，但文档注明扫描采样数据时无法选择会混淆数据的动作 [^36]——即「降采样」与「可脱敏」存在冲突。

**客户端脱敏的时机与副作用。** Langfuse 的 masking 定位是让用户在 trace 数据离开应用之前脱敏 [^37]。Python SDK 提供两个钩子：`mask_otel_spans`（推荐，新项目应优先使用）与 `mask`（legacy）。`mask_otel_spans` 在导出阶段运行，时机在 Langfuse 决定本客户端应导出哪些 OTel span 之后、且在 Langfuse media 处理之后，覆盖范围包括 Langfuse SDK span 与由该客户端导出的第三方 instrumentation 的原始 OTel span attributes；legacy 的 `mask` 参数在 SDK attributes 创建时同步运行，只作用于通过 `start_observation()`、`update()` 等 Langfuse SDK API 写入的数据，不会检查第三方 instrumentation 产生的最终原始 OTel span attributes [^37]。

`mask_otel_spans` 的能力边界必须被显式认知：它接收一个 OTel 导出批次（`params.spans`），批次不保证包含完整 trace、请求或 Langfuse observation 树；返回稀疏 patch，`OtelSpanPatch` 先执行 `delete_attributes` 再应用 `set_attributes`，同一 key 同时出现时 `set_attributes` 生效；该钩子只能修改 span attributes，不能修改 span name、IDs、父子关系、resource attributes、events、links 或 instrumentation scope [^37]。它是同步函数，通常运行在 OTel batch span processor 的 worker 线程上，文档明确建议避免长耗时工作、无界重试、请求级状态、当前 active span 与异步 I/O，因为慢速 masking 会堵塞导出队列并延迟 span 导出 [^37]。失败语义：若 `mask_otel_spans` 抛异常或返回无效的 `MaskOtelSpansResult`，Langfuse 会丢弃整个导出批次；若单个 `OtelSpanPatch` 无效，则只丢弃该 span；若返回的 attribute 值无效，则只删除受影响的 attribute [^37]。

**多后端场景下脱敏不自动传播。** masking 只影响经由该 Langfuse 客户端导出的 span；若同一批 OTel span 通过另一个 span processor 或 exporter 发往其他可观测性后端，该后端会收到自己那份未修改的副本，需单独配置脱敏 [^37]。这与 Langfuse「构建在 OpenTelemetry 之上、可同时把 trace 发往多个目的地」的设计 [^25] 是同一枚硬币的两面。

---

## 五、trace 能回答什么、不能回答什么

**trace 是失败分类体系的构建材料。** MAST 基于 150 条 trace 的严格分析构建，由专家人工标注指导，共识别 14 种失败模式，归为 3 大类：(i) system design issues、(ii) inter-agent misalignment、(iii) task verification；标注一致性人类标注者间 kappa = 0.88 [^20]。论文发布 MAST-Data 数据集：跨 7 个主流 MAS 框架收集的 1600+ 条带标注 trace，并开发了 LLM-as-a-Judge 流水线以支持规模化标注，与人工标注达成高一致性 [^20]。这说明 trace 的字段设计至少要能支撑「失败模式标注」这一下游用途——若 trace 不记录 agent 间的消息传递与任务验证步骤，inter-agent misalignment 与 task verification 两类失败就无法被标注。

**静默错误是显式错误字段的盲区。** RITE 关注工具调用中的静默错误：工具调用的期望输出可能在 agent 开发与执行之间发生偏移，返回一个看似合理但实际错误的数值，且不产生显式失败信号；这类静默错误会沿后续决策传播到最终答案 [^21]。RITE 构建的基准来自人工编写的工业工作流与生产日志派生的输入，使用语义错误分类体系向固定工作流前缀注入错误，冻结已记录的工具行为，再让 agent 自由继续；数据集规模为 9 个电商工作流、167 个匹配单元、共 501 个实例（含静默错误、正确值、显式失败三类）[^21]。实验结论（摘要层面）：在 10 个近期 LLM 上，静默错误大约使任务成功率减半 [^21]。

对 schema 的直接含义：**仅记录 `status='success'` 与 `error` 字段不足以发现最有害的失败**。要检出静默错误，trace 需要保留工具调用的实际返回值（而非仅记录调用成功与否），以及该返回值在后续决策中被如何引用。Braintrust 把「静默失败检测」描述为其 Topics 功能的用途之一——捕获不会触发显式检查的质量问题 [^39]，这是平台侧对该缺口的产品化回应。

**单条 trace 存在原理性观测上限。** 一篇 arXiv 论文（仅通过搜索页摘要获得，未读正文）区分了 single-trace safety 与 relational k-safety 两类性质，并给出不可能性结论：relational purpose violations 属于 2-safety hyperproperty，不存在单条 trace 上的 blame 位置，因此逐动作监控器无法检出；该论文称在所调查的六个 guardrail 中，没有一个实现 cross-execution monitoring [^3]。按取证纪律，该来源不能作为结论依据，但它指出的方向与 LangSmith 的 thread 概念一致：thread 表示多轮会话的 trace 序列，通过传入 `thread_id` metadata key 将 trace 归组 [^22]。也就是说，跨 trace 的性质需要 trace 之上的分组键才能表达。

**结构性问题可被规则化检测。** 一个针对 agent 运行记录的确定性 linter 自述读取执行 trace、标记结构性缺陷（被忽略的错误、schema 违规、循环）并附带证据，不使用 LLM judge [^19]。该描述间接表明 agent trace 中存在可被规则化检测的「schema 违规」与「循环」等结构性问题类别，但未读取 README 正文，无法获得具体字段定义 [^19]。

---

## 六、平台数据模型的差异与互操作

**LangSmith：run 为中心，字段最完整。** run 记录字段包含 `id`、`name`、`inputs`、`run_type`、`start_time`、`end_time`、`extra`、`error`、`outputs`、`events`、`tags`、`trace_id`、`dotted_order`、`status`、`child_run_ids`、`direct_child_run_ids`、`parent_run_ids`、`parent_run_id`、`feedback_stats`、`reference_example_id`、`session_id`、`in_dataset`、`serialized`、`inputs_s3_urls`、`outputs_s3_urls`、`price_model_id`、`app_path`、`last_queued_at`、`share_token` [^23]。单个 trace 最多 25,000 个 run，超过后 LangSmith 拒绝该 trace 的后续 run [^22]。trajectory 是 thread 内 trace 的投影，为扁平有序的消息列表（human/AI/tool 消息各出现一次，去除 run 嵌套）[^22]——这是一个有用的派生视图：调试时看嵌套树，评估时看扁平轨迹。

**Langfuse：observation 为中心，存储反规范化。** Langfuse 把 span 称为 observation，同时 span 也是具体的 observation 类型之一 [^25]。存储上 Langfuse 只有一张 observations 表，每行保存 observation 级数据加一份 trace 级属性副本，以加速查询与聚合；observation 行包含的字段示例为 `id`、`type`、`name`、`latency`、`trace_id`、`trace_name`、`user_id`、`session_id` [^25]。这个设计解释了为什么 trace 级属性必须传播到每个 span：不传播就无法在该表上做过滤与聚合。Langfuse 在 OTel 接入文档中明确：使用 OTel 时，若需按 `userId`、`sessionId`、`metadata`、`version`、`release`、`tags` 过滤与聚合，必须把这些 trace 级属性传播到 trace 内每个 span，而非仅根 span；推荐用 OpenTelemetry Baggage 配合 `BaggageSpanProcessor` 传播 [^28]。

**Datadog：trace/span 语义与 APM 对齐。** Datadog 的 LLM 可观测产品现名为 Agent Observability；应用中每个被处理的请求在 Datadog 中表示为一条 trace，一条 trace 可代表单次 LLM 推理（含 tokens、错误信息、延迟）、预定义的 LLM 工作流，或由 LLM agent 执行的动态工作流；每条 trace 包含若干 span，span 表示 agent 做出的每个选择或工作流的每个步骤；trace 还可包含输入输出、延迟、隐私问题、错误等 [^34]。Datadog 将 LLM/Agent 可观测性归入 APM Tracing 体系，与 Logs、Profiler、Monitors and Alerting、Sensitive Data 等模块并列 [^15]。

**Braintrust：logs 与 experiments 同构。** Braintrust 将每次请求捕获为 trace；日志与实验使用同一数据结构，同一套埋点代码同时适用于 logging 与 evaluation，生产与测试捕获的 trace 数据一致；Scores 与 feedback 同时作用于 logs 和 experiments，生产数据可直接转为评测数据集 [^39]。Logs 页面以可搜索、可过滤的表格展示所有 trace，每一行代表一条完整 trace 及其 root span；支持按 metadata 或 tags 对相关 trace 分组，并可创建自定义列以展示重要字段值 [^39]。

**互操作的现实路径是 OTLP + 映射层。** LangSmith 支持 OTel tracing：OTel SDK 产生 span，经 OTLP 送到 Collector，Collector 批量分发到一个或多个后端，LangSmith 在其 OTLP endpoint 接收 span 并展示为 trace；非 LangChain 应用可用标准 OTel client 手动埋点，需设置 `langsmith.span.kind=LLM`、`langsmith.metadata.user_id`、`gen_ai.system`、`gen_ai.request.model`、`llm.request.type=chat` 等属性 [^24]。Langfuse 在 `/api/public/otel` 接收 trace，目前支持 OTLP over HTTP（HTTP/JSON 与 HTTP/protobuf），尚不支持 gRPC [^28]。属性映射方面，Langfuse 定义 `langfuse.user.id` 或 `user.id` 映射 userId、`langfuse.session.id` 或 `session.id` 映射 sessionId、`langfuse.trace.metadata.*` 映射 metadata、`langfuse.version` 映射 version、`langfuse.release` 映射 release、`langfuse.trace.tags` 映射 tags、`langfuse.trace.name` 映射 trace_name [^28]。

一个具体的接入陷阱：`OTEL_EXPORTER_OTLP_ENDPOINT` 是 base URL，OTLP/HTTP exporter 会自行追加 `/v1/traces`；若把 base endpoint 写成 `.../otel/v1/traces` 会导致重复路径并返回 404 [^24]。Langfuse 侧，直接通过 OTel 发送 span 时需带 `x-langfuse-ingestion-version: 4` header，否则直接摄入的 OTel 数据可能延迟最多 10 分钟才出现在 v4 数据模型与 v2 Observations/Metrics API 中 [^28]。

---

## 矛盾与不确定性

**1. 语义约定层没有单一权威，且权威本身在迁移中。** OTel GenAI 语义约定处于 Development 状态 [^1]，官方页面已迁出主站 [^2][^42]；OpenLLMetry 声明其语义约定已成为 OpenTelemetry 的一部分 [^8]，OpenInference 则维护一套独立的、与 OTel 互补的约定 [^9]，并提供 processor 在两者之间转换 [^9]。这意味着「一条 trace 应该记录哪些字段」目前不存在唯一正确答案，只存在「以哪套约定为主、如何映射到其他后端」的选择。跨平台字段名（如 token 用量）在 OTel、LangSmith、Langfuse 三处命名不同 [^1][^23][^27]，任何长期存储 schema 都应把映射层与存储层分离。

**2. 采样与脱敏、采样与诊断之间存在张力。** Langfuse 的采样是 trace 级一次性决策 [^31]，Datadog 的云端扫描采样率为 10%–99% 且文档注明扫描采样数据时无法选择会混淆数据的动作 [^36]。前者意味着未被采样的 trace 完全不可用于事后诊断，后者意味着为控成本而降采样会削弱脱敏能力。现有证据未给出二者的联合最优策略。

**3. 静默错误使得「status 字段」的可信度被高估。** RITE 证明静默错误不产生显式失败信号且使任务成功率减半 [^21]，而各平台 schema 的错误表达仍以显式 `status`/`error` 为主 [^1][^23]。Braintrust 把静默失败检测列为 Topics 的用途 [^39]，但这是产品功能描述而非可复用的字段规范。**现有公开资料不足以确认**业界已有通行的「静默错误」字段级表达方式。

**4. 学术侧对 trace schema 与自动根因定位的实质证据缺失（blocking）。** 本报告检索到的 agent trace schema 相关 GitHub 仓库（RagaAI-Catalyst [^4]、traccia-py [^5]、Tracekit [^6]、future-agi [^7]、witness [^17]、coding-agent-tracing [^18]、tracelint [^19]）全部仅有元数据、未读 README 正文，按取证纪律不得作为结论依据；arXiv 侧，2610.27620 仅有搜索页摘要且明确标注未读正文 [^3]，cs.SE 列表页明确不能作为字段定义证据且其可见条目中无任何标题直接涉及 agent trace schema 或 observability data model [^16]。因此**「agent trace schema 与数据模型的研究进展」这一子问题目前没有任何可引用的实质证据**。同样，「基于 trace 的自动诊断与根因定位方法」仅有 MAST（失败分类体系）[^20] 与 RITE（韧性评估）[^21] 两篇，二者均未覆盖从 trace 自动定位根因的方法类工作（trace 级诊断、因果归因、LLM-as-judge 诊断流水线），独立来源不足 2 个，亦属 blocking。LangSmith 文档提到其 Engine 可自动检测 traces 中反复出现的问题并诊断根因 [^13]，Braintrust 提到 Loop 可用自然语言调查日志、Patterns 按计划在 trace 积压上排查反复出现的问题并给出证据与修复建议 [^39]，但这些都是厂商产品描述，不构成方法学证据。

**5. Datadog 与 W&B 的推荐字段与告警配置无可用证据（blocking）。** Datadog Agent Observability SDK Reference 与 Monitoring 页面均只抓到站点导航，未返回正文，无法提取推荐字段或属性清单，也无法取得 LLM Observability 专属指标或告警配置示例 [^15][^41]；W&B Weave 的 trace objects 页面返回访问码墙，未获得任何关于 Op 字段、attributes 或 trace 对象结构的正文 [^40]。因此本报告关于 Datadog 的字段级结论仅来自其产品概览页 [^34] 与 Sensitive Data Scanner 文档 [^36]，关于告警配置则**现有公开资料不足以确认**。Datadog 的 LLM Observability 独立计价条目在其定价页正文中亦未出现 [^35]。

**6. 保留期与成本的数字随档位和日期变化，不可当作常量。** LangSmith 的 Base 14 天 / Extended 180 天分档 [^32] 与 2026 年 9 月 14 日的上限变更 [^32] 是时点性事实；Langfuse 的 Hobby 30 天 / Core 90 天 / Pro 3 年 [^29] 同样按档位区分。LangSmith 的 trace 用量按 "base traces" 计量，官方注明 "Actual LSU consumption varies with the work your agents perform" [^33]。任何容量规划都应读取当前档位文档而非引用本报告数字。

**7. 单条 trace 的观测上限存在理论争议但证据不足。** 关于 relational purpose violations 不存在单条 trace 上 blame 位置、逐动作监控器无法检出的结论，仅来自一篇未读正文的 arXiv 摘要 [^3]，不能作为结论依据。可确认的只有工程事实：LangSmith 用 thread 表达多轮会话的 trace 序列 [^22]，Langfuse 用 session 表达同类分组 [^25]，即跨 trace 性质需要 trace 之上的分组键——但这是产品设计事实，不是对上述不可能性结论的验证。

## 参考来源

[^1]: Semantic conventions for generative client AI spans (OpenTelemetry GenAI semantic conventions) — https://raw.githubusercontent.com/open-telemetry/semantic-conventions-genai/main/docs/gen-ai/gen-ai-spans.md（web）
[^2]: Generative AI semantic conventions（原 opentelemetry.io 规范页，已迁移公告） — https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/（web）
[^3]: Local Allow, Global Deny: Why Per-Action Authorization Does Not Compose in LLM Agents (arXiv:2610.27620) — https://arxiv.org/search/?searchtype=all&query=LLM+agent+observability+trace&start=0（arxiv, 2026-10-20）
[^4]: raga-ai-hub/RagaAI-Catalyst — Python SDK for Agent AI Observability, Monitoring and Evaluation — https://github.com/raga-ai-hub/RagaAI-Catalyst（github, 2026-02-11）
[^5]: traccia-ai/traccia-py — OpenTelemetry-native SDK for AI agent observability — https://github.com/traccia-ai/traccia-py（github, 2026-10-21）
[^6]: Cygnux-Labs/Tracekit — Tamper-evident tracing, policy gates and cross-checks for AI agents — https://github.com/Cygnux-Labs/Tracekit（github, 2026-10-24）
[^7]: future-agi/future-agi — end-to-end platform for evaluating, observing and improving LLM/agent apps — https://github.com/future-agi/future-agi（github, 2026-10-24）
[^8]: traceloop/openllmetry — Open-source observability for your LLM application — https://github.com/traceloop/openllmetry（github）
[^9]: Arize-ai/openinference — Conventions and plugins complementary to OpenTelemetry for tracing AI applications — https://github.com/Arize-ai/openinference（github）
[^11]: AgentOps-AI/agentops — Observability and DevTool platform for AI Agents — https://github.com/AgentOps-AI/agentops（github）
[^13]: LangSmith Observability — 文档首页（Docs by LangChain） — https://docs.smith.langchain.com/administration/data_retention（web）
[^15]: Datadog Docs — Agent Observability SDK Reference（仅取得站点导航结构） — https://docs.datadoghq.com/llm_observability/instrumentation/sdk/（web）
[^16]: arXiv cs.SE recent submissions listing (Fri, 23 Oct 2026) — https://arxiv.org/list/cs.SE/recent（arxiv, 2026-10-23）
[^17]: aevyraai/witness — Execution trace schema for AI agent pipelines — https://github.com/aevyraai/witness（github, 2026-07-20）
[^18]: shashwat134/coding-agent-tracing — normalize coding agents into a shared LangSmith trace schema — https://github.com/shashwat134/coding-agent-tracing（github, 2026-07-21）
[^19]: AshwinUgale/tracelint — deterministic linter for agent execution traces — https://github.com/AshwinUgale/tracelint（github, 2026-10-21）
[^20]: Why Do Multi-Agent LLM Systems Fail? (MAST) — https://arxiv.org/abs/2503.13657（arxiv, 2025-03-17 (v1); v3 2025-10-26）
[^21]: RITE: Evaluating End-to-End Resilience to Silent Tool Errors in E-Commerce Agentic Systems — https://arxiv.org/abs/2610.27777（arxiv, 2026-10 (announced)）
[^22]: LangSmith Observability concepts（runs / traces / threads / trajectories） — https://docs.smith.langchain.com/observability/concepts（web）
[^23]: LangSmith Run data format（run 字段清单与 run types） — https://docs.smith.langchain.com/reference/data_formats/run_data_format（web）
[^24]: LangSmith Trace with OpenTelemetry（OTLP 接入） — https://docs.langchain.com/langsmith/trace-with-opentelemetry（web）
[^25]: Langfuse Data Model（observations / traces / sessions） — https://langfuse.com/docs/observability/data-model（web）
[^26]: Langfuse Observation Types（observation 类型枚举） — https://langfuse.com/docs/observability/features/observation-types（web）
[^27]: Langfuse Token & Cost Tracking（usage details / cost details） — https://langfuse.com/docs/observability/features/token-and-cost-tracking（web）
[^28]: Langfuse OpenTelemetry (OTLP) 接入 — https://langfuse.com/integrations/native/opentelemetry（web）
[^29]: Langfuse Pricing (official pricing page) — https://langfuse.com/pricing（web）
[^30]: Langfuse Docs — Billable Units — https://langfuse.com/docs/administration/billable-units（web）
[^31]: Langfuse Docs — Sampling — https://langfuse.com/docs/observability/features/sampling（web）
[^32]: LangSmith Docs — Usage and billing (data retention & billing model) — https://docs.langchain.com/langsmith/usage-and-billing（web）
[^33]: LangSmith Pricing (official pricing page) — https://www.langchain.com/pricing-langsmith（web）
[^34]: Datadog Docs — Agent Observability (LLM Observability) overview — https://docs.datadoghq.com/llm_observability.md（web）
[^35]: Datadog Pricing list (official pricing page) — https://www.datadoghq.com/pricing/list/（web）
[^36]: Datadog Sensitive Data Scanner (official docs, markdown) — https://docs.datadoghq.com/security/sensitive_data_scanner.md（web）
[^37]: Langfuse Docs — Masking sensitive LLM data — https://langfuse.com/docs/observability/features/masking（web）
[^39]: Braintrust Docs — Observe logs (Logs / tracing overview) — https://www.braintrust.dev/docs/guides/logs（web）
[^40]: W&B Weave Docs — Trace objects（访问受限，未取得正文） — https://docs.wandb.ai/weave/guides/tracking/trace-objects（web）
[^41]: Datadog Docs — LLM Observability Monitoring（仅返回站点导航，未取得正文） — https://docs.datadoghq.com/llm_observability/monitoring/（web）
[^42]: Generative AI semantic conventions have moved (opentelemetry.io 迁移公告页) — https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/（web）