# vLLM 等 LLM 推理引擎如何提升服务吞吐：PagedAttention 与 continuous batching 的分工

## 摘要

LLM 自回归解码是访存密集型负载，GPU 算力在逐 token 生成时大量闲置，提升吞吐的主路径是增大 batch；而 batch 能开多大，取决于每个请求的 KV cache 内存能否被高效管理[^2]。vLLM 的两项核心机制正好切在这条链的两端：**PagedAttention** 解决「内存怎么放」——把 KV cache 从按请求预分配的连续大块改为固定大小 block 的非连续组织，消除碎片并支持块级共享[^1][^3]；**continuous batching（iteration-level scheduling）** 解决「批次怎么排」——以单次模型迭代为调度粒度，已完成请求即时退出、新请求即时进入，而非等整批结束[^12]。二者不是替代关系：continuous batching 决定「可以同时跑多少请求」，PagedAttention 决定「这些请求的 KV cache 放不放得下、能共享多少」。在此之上，chunked prefill、抢占/重算、prefix caching、prefill-decode 分离等机制进一步调节吞吐与延迟的权衡[^13][^6][^20]。需要强调：本报告能引用的量化吞吐倍数均来自 2023–2024 年的论文与官方博客，不能代表 2026 年当前版本的横向水平（见「矛盾与不确定性」）。

---

## 1. 吞吐瓶颈的成因：为什么必须管理 KV cache

**负载性质。** 自回归逐 token 生成使推理呈 memory-bound，GPU 算力未被充分利用，从而限制服务吞吐；提升吞吐需增大 batch，而这依赖对每请求内存的高效管理[^2]。

**KV cache 的量级。** 以 OPT-13B 为例，单个 token 的 KV cache 需 800 KB（2×(key,value)×5120 隐藏维×40 层×2 字节 FP16），2048 token 序列的单请求可达 1.6 GB[^2]。在 A100 40GB 上跑 13B 模型时，约 65% 内存给模型权重，近 30% 用于请求的动态状态（即 KV cache）[^2]。这决定了「并发数」直接受显存约束。

**连续内存假设带来的浪费。** 现有系统因主流深度学习框架要求张量连续，把每个请求的 KV cache 存放在连续内存空间；而 KV cache 随生成动态增长/收缩，生命周期与长度无法预先知晓[^1]。为把 KV cache 存进连续空间，系统按请求最大长度（如 2048 tokens）预分配一整块连续内存[^2]。由此产生两类低效：内部与外部内存碎片，以及无法利用内存共享机会[^1]。profiling 显示，现有系统中仅 **20.4%–38.2%** 的 KV cache 内存真正用于存储实际 token 状态，其余被碎片与预留浪费[^2]。

**调度侧的浪费。** 在 Orca 之前，推理服务系统因调度机制不灵活而表现不佳：批内已完成的请求无法返回客户端，新到达请求必须等当前批次完全结束[^12]。这造成批次尾部的算力空转。

这两类浪费对应两个正交的解法。

---

## 2. PagedAttention：解决 KV cache 的内存碎片与共享问题

### 2.1 定义（权威出处）

PagedAttention 是 Kwon 等在 SOSP 2023 论文中提出的、**受操作系统经典虚拟内存与分页（paging）技术启发的 attention 算法**[^1]。其核心是把 KV cache 的存储单位从「每请求一段连续内存」改为「固定大小的 block」，attention 计算按 block 索引取 key/value，而非按序列连续寻址。

vLLM 官方设计文档给出了 block 的完整定义：key/value cache 数据被切分为 blocks，**每个 block 存储固定数量（BLOCK_SIZE）个 token 在单个 head 上的数据**；一个 block 可能只包含整个 context 的一部分[^3]。容量示例：若 block size 为 16、head size 为 128，则单个 head 的一个 block 可存 16×128 = 2048 个元素[^3]。

内存布局上，kernel 签名中 k_cache 形状为 `[num_blocks, num_kv_heads, head_size/x, block_size, x]`，v_cache 形状为 `[num_blocks, num_kv_heads, head_size, block_size]`，即**按 block 索引组织而非按序列连续组织**[^3]。计算侧，warp 一次处理一个 query token 与一个完整 block 的 key token 的计算[^3]。

术语上需注意：vLLM 的 paged attention block 与 GPU thread block 是两个不同概念，文档明确将前者称为 "block"、后者称为 "thread block"[^3]。另需注意该设计文档的时效性声明——它是 "historical document based on the original paper for vLLM"，已不再描述当前 vLLM 代码；当前实现使用自研 multi-head query attention kernel（`csrc/attention/attention_kernels.cu`）[^3]。

### 2.2 它解决的具体问题

1. **碎片与预留浪费**：按最大长度预分配连续内存是浪费根源[^2]；改为 block 粒度后，论文声称实现 KV cache 内存近乎零浪费（near-zero waste）[^1]。官方博客给出的对应数字是：PagedAttention 下内存浪费仅发生在序列最后一个 block，实际浪费低于 4%[^15]。
2. **共享机会无法利用**：parallel sampling、beam search 等高级解码算法会为单个请求生成多个输出序列，这些序列可部分共享 KV cache[^2]。PagedAttention 支持请求内部与跨请求的 KV cache 灵活共享[^1]。量化上：parallel sampling 场景下 prompt 部分的 KV cache 约占总 KV cache 的 12%，可被共享；beam search 场景下不同 beam 可共享更大比例，最高节省 55% 内存，且共享模式随解码推进而变化[^2]。官方博客称该共享可将并行采样/beam search 的内存占用最多降低 55%，并转化为最高 2.2x 吞吐提升[^15]。

### 2.3 共享机制在工程上的落地：prefix caching

vLLM 的自动前缀缓存把「块级共享」推广到跨请求：采用基于哈希的方法，对每个 KV cache block 用「块内 token」加上「该块之前的 prefix token」共同计算哈希，从而识别可复用的前缀块[^4]。块哈希为 `hash(tuple[components])`，components 包括父块哈希值（parent hash）、块内 token 元组（用于降低哈希碰撞）、以及额外哈希（LoRA ID、多模态输入哈希、多租户隔离用的 cache salt）[^4]。复用粒度受限：文档明确「We only cache full blocks」，即只缓存完整的块[^4]。

生命周期管理靠引用计数：块数据结构 KVCacheBlock 含 block_id（不可变）、block_hash（块填满时赋值、被驱逐时重置）、ref_cnt（当前使用该块的请求数）、以及构成空闲队列双向链表的 prev_free_block / next_free_block 指针[^4]；分配已计算块时对其 ref_cnt 加一并将其移出空闲队列，以避免这些可复用块被驱逐[^4]。初始化时一次性分配全部 KVCacheBlock 构成 block pool，避免 Python 对象创建开销；管理器组件包括 Block Pool、Free Block Queue、Cache blocks（哈希→块 ID 映射）、Request blocks（请求 ID→已分配块 ID 映射）[^4]。安全隔离上，支持 per-request cache_salt 注入首块哈希，使只有相同 salt 的请求才能复用缓存块，用于防止基于时延的缓存内容推断攻击[^4]。哈希算法自 v0.11 起默认使用 sha256，可通过 `--prefix-caching-hash-algo` 选择 sha256 / sha256_cbor / xxhash / xxhash_cbor[^4]。

---

## 3. Continuous batching：解决批次的调度粒度问题

### 3.1 定义（权威出处）

continuous batching 的学术原型是 Orca 提出的 **iteration-level scheduling**：以 iteration（而非 request）为粒度进行调度，调度器只让执行引擎在当前批次上运行模型的单次 iteration[^12]。与之配套的是 **selective batching**：为在 Transformer 模型上同时使用 batching 与 iteration-level scheduling，仅对选定的一组算子（a selected set of operations）应用 batching[^12]。Orca 基于这两项技术实现了分布式服务系统，并针对数千亿参数模型的可扩展性做了额外设计[^12]。

机制效果上，它直接消除了第 1 节所述的调度浪费：批内请求完成后无需等待整批结束即可退出，新请求可在下一次 iteration 加入[^12]。

### 3.2 vLLM 中的调度职责划分

vLLM V1 采用多进程架构，engine core 进程负责运行 scheduler、管理 KV cache，并协调各 GPU worker 上的模型执行；engine core 运行一个 busy loop，持续调度请求并把工作分发给 GPU worker[^14]。每个 data parallel rank 对应 1 个 engine core 进程（例如 `--data-parallel-size 4` 时有 4 个 engine core 进程）[^14]。LLMEngine 的职责包括 input processing、model execution、scheduling 与 output processing，其中 Scheduling 被描述为「选择每一步处理哪些请求」（Chooses which requests are processed in each step）[^14]。每个 GPU 由一个专用 worker 进程管理，worker 负责加载模型权重、执行前向计算并管理 GPU 显存[^14]。

需说明：本次抓取到的 vLLM README 片段仅含项目标识与链接，未包含调度器（continuous batching / chunked prefill / 抢占）相关正文，故不能据 README 得出调度语义结论[^5]；调度语义以上述官方文档与论文为准。

### 3.3 与 PagedAttention 的耦合点：抢占与重算

continuous batching 把 batch 开大后，KV cache 空间可能不够，此时 vLLM 会抢占（preempt）请求，为其他请求释放 KV cache 空间；被抢占的请求在重新获得足够空间后会被 recompute（重算）[^13]。文档给出的告警示例为：`'Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space.'`[^13]。在 vLLM V1 中，默认抢占模式是 RECOMPUTE 而非 SWAP，原因是 recompute 在 V1 架构下开销更低[^13]。抢占与重算会损害端到端延迟；频繁抢占时的缓解手段包括提高 `gpu_memory_utilization`、降低 `max_num_seqs` 或 `max_num_batched_tokens`、提高 `tensor_parallel_size`、提高 `pipeline_parallel_size`[^13]。可通过 Prometheus 指标监控抢占次数，并可通过设置 `disable_log_stats=False` 记录累计抢占次数[^13]。

这正是两个机制的分工边界：**PagedAttention 让内存利用率接近上限，continuous batching 则在这个上限内尽可能填满 batch；当填充超过内存上限时，由抢占机制回退。**

---

## 4. 吞吐-延迟权衡的进一步机制

### 4.1 Chunked prefill

vLLM 文档定义：chunked prefill 让 vLLM 把大的 prefill 拆成更小的 chunk，并与 decode 请求合批处理，从而同时改善吞吐与延迟，更好地平衡 compute-bound（prefill）与 memory-bound（decode）操作[^13]。在 V1 中，chunked prefill 在可能的情况下默认启用（enabled by default whenever possible）[^13]。

调度策略：启用 chunked prefill 时优先处理 decode 请求——先批入所有待处理的 decode 请求，再在 `max_num_batched_tokens` 预算还有剩余 token 时调度待处理的 prefill；若某个待处理 prefill 放不进预算，则自动对其进行切块[^13]。两点收益：优先 decode 可改善 inter-token latency (ITL)；把 compute-bound 的 prefill 与 memory-bound 的 decode 放进同一 batch 可提升 GPU 利用率[^13]。

参数权衡：较小的 `max_num_batched_tokens`（如 2048）获得更好的 ITL，因为拖慢 decode 的 prefill 更少；较大值获得更好的 TTFT，因为单批可处理更多 prefill token[^13]。为获得最优吞吐，文档建议设置 `max_num_batched_tokens > 8192`，尤其是在大 GPU 上跑较小模型时[^13]。若该值等于 `max_model_len`，则几乎等价于 V0 的默认调度策略（区别是它仍然优先 decode）[^13]；当 chunked prefill 被禁用时，`max_num_batched_tokens` 必须大于 `max_model_len`，否则 vLLM 可能在服务启动时崩溃[^13]。文档将更多细节指向 arXiv 2401.08671 与 arXiv 2308.16369[^13]。

学术侧的对应工作是 Sarathi-Serve：它提出 chunked-prefills，把一个 prefill 请求切分为大小近似相等的多个 chunk，使调度器能构造 stall-free 调度，在不暂停进行中 decode 的前提下把新请求加入批次；stall-free 调度使系统可用大 batch size 提升吞吐，同时把 batching 对延迟的影响降到最小[^8]。论文还指出均匀批次可缓解迭代间的不均衡，从而减少流水线气泡（pipeline bubbles）[^8]。

### 4.2 Prefill-decode 分离

Splitwise 基于实测刻画指出：LLM 推理请求包含两个阶段——计算密集的 prompt computation 与访存密集的 token generation，二者在延迟、吞吐、内存与功耗特征上各不相同；token generation 阶段即使采用 SOTA 的 batching 与调度，仍会低设计算资源利用率，且不需要最新 GPU 的算力，可用更低功耗与成本的硬件运行[^7]。Splitwise 将两阶段拆分到不同机器上，为每个阶段选用适配的硬件并独立配置资源；跨机器拆分需把 prompt computation 机器的状态传输到 token generation 机器，论文利用 GPU 集群的高速背板互联实现并优化该状态传输[^7]。

DistServe 进一步指出，共置（colocate）方案不仅带来强烈的 prefill-decode 干扰，还把两个阶段的资源分配与并行策略耦合在一起；它以应用的 TTFT 与 TPOT 需求为输入，为每个阶段分别协同优化资源分配与并行策略，并依据服务集群带宽决定两阶段的放置位置以最小化拆分带来的通信开销[^6]。

在 vLLM 生态中，该路径已进入产品化文档：production-stack 文档目录列有独立的 "Disaggregated Prefill" 用例章节，证明其已被纳入正式文档化用例而非仅存在于代码分支[^19]。该用例页说明 disaggregated prefill 将模型执行拆分为 prefill 与 decode 两阶段并分布到不同服务器，目的是通过分离初始处理与 token 生成来提升吞吐与资源利用率[^20]；部署路径为 Kubernetes + Helm，用 `values-16-disagg-prefill.yaml` 与 `helm install pd helm/ -f tutorials/assets/values-16-disagg-prefill.yaml` 一次性部署 prefill server、decode server 与协调二者的 router 三个组件[^20]。KV cache 由 prefill 侧生产（`kvRole: "kv_producer"`、`nixlRole: "sender"`）、经 NIXL 传输至 decode 侧消费（`kvRole: "kv_consumer"`、`nixlRole: "receiver"`）[^20]。值得注意的是，该示例配置中 prefill 与 decode 节点均显式设置 `enableChunkedPrefill: false` 与 `enablePrefixCaching: false`[^20]——说明分离式部署与 chunked prefill / prefix caching 在该参考配置中是互斥取舍，而非叠加。

### 4.3 其他引擎的对照

SGLang 的运行时通过 RadixAttention 实现 KV cache 复用，并通过压缩有限状态机（compressed FSM）加速结构化输出解码[^9]；其仓库 README 致谢中说明设计与代码复用了 vLLM、LightLLM、FlashInfer、Guidance、Outlines、LMQL 等项目[^10]。但当前版本 README 未以 RadixAttention 作为主要卖点，也未给出 prefix caching 的机制描述或性能数字[^10]。TensorRT-LLM 的 README 自述通过常见算子的专用 kernel、高效运行时以及可定制扩展的 pythonic 框架来优化 LLM 与视觉生成模型的推理[^11]。这些属于工程路线差异，现有证据不足以做机制层面的逐项对比。

---

## 5. 效果数据与其时效边界

**PagedAttention 论文主张**：相比 FasterTransformer 与 Orca 等 SOTA 系统，在同等延迟下吞吐提升 2–4×；序列更长、模型更大、解码算法更复杂时提升更明显[^1]。Orca 一侧的基线数据是：在 GPT-3 175B 上，ORCA 在相同延迟水平下相对 NVIDIA FasterTransformer 取得 36.9× 吞吐提升[^12]。

**vLLM 首发博客（2023-06-20）**：相比 HuggingFace Transformers 最高 24x 吞吐提升，且无需修改模型架构；实验为 LLaMA-7B 单张 A10G、LLaMA-13B 单张 A100 (40GB)，请求输入/输出长度从 ShareGPT 采样[^15]。相对 TGI 最高 3.5x[^15]。分场景：单输出补全下比 HF 高 14x–24x、比 TGI 高 2.2x–2.5x；每请求三个并行输出下比 HF 高 8.5x–15x、比 TGI 高 3.3x–3.5x[^15]。LMSYS 内部早期微基准中 vLLM 后端相比初始 HF 后端最高 30x 吞吐，上线后将服务用 GPU 数量削减了 50%[^15]。

**v0.6.0 性能博客（2024-09-05）**：vLLM v0.6.0 相对 v0.5.3 在 Llama 8B 上实现 2.7x 吞吐提升与 5x 更快 TPOT，在 Llama 70B 上实现 1.8x 吞吐提升与 2x 更低 TPOT；硬件为 1xH100 / 4xH100，ShareGPT 500 prompts，TPOT 在 32 QPS 下测量；整体提升区间 1.8–2.7x[^16]。横向对比对象为 TensorRT-LLM r24.07、SGLang v0.3.0、LMDeploy v0.6.0a0，其他引擎用默认设置，vLLM 开启 `--num-scheduler-steps 10` 多步调度[^16]。结论：H100 上服务 Llama-3 时 vLLM 在 ShareGPT 与 Decode-heavy 上取得最高吞吐[^16]；但 Llama 3 8B 在 1xA100 上吞吐 TensorRT-LLM 最高、vLLM 在 ShareGPT 与 decode-heavy 上第二[^16]；Llama 3 8B 与 70B 在 H100 上的 Prefill-heavy 场景 vLLM 吞吐较低[^16]。机制侧归因：多步调度使 Llama 70B 在 4xH100 上吞吐提升 28%，异步输出处理使 TPOT 改善 8.7%，对象缓存使端到端吞吐提升 24%[^16]。性能诊断显示 Llama 3 8B 在单张 H100 上 HTTP API server 占总执行时间 33%、调度占 29%、GPU 实际执行仅占 38%[^16]。已知代价：多步调度导致 inter-token 延迟不平滑（成批返回 token），且低请求率下 TTFT 更高（`--num-scheduler-steps` 越大越明显）[^16]。

**其他引擎的论文主张**：Sarathi-Serve 报告 Mistral-7B 单张 A100 上服务容量相比 vLLM 提升 2.6 倍、Yi-34B 双 A100 上最高 3.7 倍、Falcon-180B 配合流水线并行时端到端服务容量最高提升 5.6 倍[^8]。DistServe 报告相比当时 SOTA 可服务 7.4 倍更多请求，或同等条件下支持 12.6 倍更紧的 SLO，同时 >90% 请求满足延迟约束[^6]。Splitwise 报告相比当时设计可实现 1.4 倍吞吐且成本低 20%，或同等成本与功耗预算下实现 2.35 倍吞吐[^7]。SGLang 报告在 agent control、逻辑推理、few-shot 学习基准、JSON 解码、RAG 流水线与多轮对话等任务上相比当时 SOTA 最高 6.4 倍吞吐[^9]。

**仓库热度（工程落地信号）**：vLLM 95,287 stars、23,880 forks、9,381 open issues，Apache-2.0，创建于 2023-02-09，最近 push 2026-10-24[^5]；SGLang 37,685 stars、9,856 forks、6,142 open issues，20,856 次提交，创建于 2024-01-08，最近 push 2026-10-24[^10]；TensorRT-LLM 14,935 stars、2,861 forks、1,787 open issues，创建于 2023-08-16，最近 push 2026-10-24[^11]。production-stack 创建于 2025-01-21、最近 push 2026-10-23，说明该项目在 2025–2026 年持续活跃维护[^18]。

---

## 矛盾与不确定性

**1. 24x / 3.5x 不能当作当前水平。** vLLM 相比 HF Transformers 最高 24x、相比 TGI 最高 3.5x 的唯一来源是 2023-06-20 官方首发博客，硬件为 A10G / A100-40GB、模型 LLaMA-7B/13B，距今约三年半且为厂商自测[^15]。相对 2026-10-24 的时效要求，该数字不能作为当前版本吞吐倍数的支撑。

**2. 2–4× 的基线已停止演进。** PagedAttention 论文的 2–4× 主张出自 SOSP 2023 论文 v1（2023 年），基线 FasterTransformer 与 Orca 均已停止演进，无法代表 2024–2026 的对比格局[^1][^12]；直接引用会被误读为当前水平。

**3. 缺少 V1 架构下的当代横向基准。** 现有证据只有 v0.6.0 相对 v0.5.3 的自比数据（2024-09），以及该博客附录中 2024 年的横向对比（TensorRT-LLM r24.07、SGLang v0.3.0、LMDeploy v0.6.0a0）[^16]。缺少 2025–2026 年、V1 架构下的横向基准。博客索引页所列 2026 年博文以新硬件与新模型为主（Vera Rubin NVL72、GB200/GB300 NVL72、H200、MI355X、TPU 等），未出现以 H100/A100 对比 HF Transformers / TGI / TensorRT-LLM 的基准博文标题[^17]。因此「vLLM 当前版本相对 HF / TGI / TensorRT-LLM / SGLang 的吞吐与延迟位置」这一问题，**现有公开资料不足以确认**。

**4. 并发序列数完全空缺。** 本课题要求给出「支持的并发序列数 / 最大 batch 规模」的量化事实，但 20 条证据中无任何一条给出具体数值（如 `max_num_seqs` 默认值、单卡可并发序列数实测）。文档只提到可通过降低 `max_num_seqs` 缓解频繁抢占[^13]，未给默认值或实测上限。该子项**现有公开资料不足以确认**。

**5. 设计文档与当前代码的错位。** vLLM 的 Paged Attention 设计页自述为基于原始论文的 historical document，已不再描述当前代码，当前实现使用自研 multi-head query attention kernel[^3]。因此第 2.1 节的 kernel 内存布局与 warp 计算语义应理解为「论文时期的设计」，而非 2026 年 vLLM 的实现细节。

**6. 分离式部署与 chunked prefill / prefix caching 的关系未澄清。** production-stack 的 disaggregated prefill 示例配置显式关闭了 chunked prefill 与 prefix caching[^20]，但证据中没有任何一条解释这是示例简化、镜像版本限制，还是架构上的必然取舍。此外，该文档中 "Sharing KV Cache Across Instances" 路径的性能收益基准仍标注为 "Work in Progress"[^19]，说明跨实例 KV 共享的收益尚无定论。

**7. 厂商自测的系统性偏差。** 第 5 节所有倍数（含 Sarathi-Serve、DistServe、Splitwise、SGLang 的主张）均为各自作者在自选基线与配置下的报告[^6][^7][^8][^9]，彼此基线不同、年份不同，不可横向相加或直接排序。v0.6.0 博客中 vLLM 开启多步调度而其他引擎用默认设置[^16]，也构成配置不对等。

## 参考来源

[^1]: Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., SOSP 2023) — https://arxiv.org/abs/2309.06180v1（arxiv, 2023-09-12）
[^2]: Efficient Memory Management for Large Language Model Serving with PagedAttention — 正文（引言/背景章节，arXiv HTML） — https://arxiv.org/html/2309.06180v1（arxiv, 2023-09-12）
[^3]: vLLM 官方设计文档：Paged Attention — https://docs.vllm.ai/en/latest/design/paged_attention.html（web）
[^4]: vLLM 官方设计文档：Automatic Prefix Caching — https://docs.vllm.ai/en/latest/design/prefix_caching.html（web）
[^5]: vllm-project/vllm — repository metadata and README header — https://github.com/vllm-project/vllm（github, 2026-10-24）
[^6]: DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving — https://arxiv.org/abs/2401.09670v3（arxiv, 2024-01-18）
[^7]: Splitwise: Efficient generative LLM inference using phase splitting — https://arxiv.org/abs/2311.18677v2（arxiv, 2023-11-30）
[^8]: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve — https://arxiv.org/abs/2403.02310（arxiv, 2024-03-04）
[^9]: SGLang: Efficient Execution of Structured Language Model Programs — https://arxiv.org/abs/2312.07104v2（arxiv, 2023-12-12）
[^10]: sgl-project/sglang (GitHub repository) — https://github.com/sgl-project/sglang（github, 2024-01-08）
[^11]: NVIDIA/TensorRT-LLM (GitHub repository) — https://github.com/NVIDIA/TensorRT-LLM（github, 2023-08-16）
[^12]: Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI '22 官方论文页) — https://www.usenix.org/conference/osdi22/presentation/yu（web, 2022-07）
[^13]: vLLM Documentation — Optimization and Tuning (Preemption / Chunked Prefill) — https://docs.vllm.ai/en/latest/configuration/optimization.html（web）
[^14]: vLLM Documentation — Architecture Overview (V1 Process Architecture / LLM Engine) — https://docs.vllm.ai/en/latest/design/arch_overview.html（web）
[^15]: vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention (vLLM 官方首发博客) — https://blog.vllm.ai/2023/06/20/vllm.html（web, 2023-06-20）
[^16]: vLLM Performance Update: v0.6.0 release (官方性能博客) — https://blog.vllm.ai/2024/09/05/perf-update.html（web, 2024-09-05）
[^17]: vLLM Blog 索引页（2026 年博文列表，用于确认 2024–2026 官方性能博客覆盖范围） — https://blog.vllm.ai/（web）
[^18]: vllm-project/production-stack — vLLM 生产部署参考栈仓库 — https://github.com/vllm-project/production-stack（github, 2025-01-21）
[^19]: vLLM Production Stack 官方文档首页 — https://docs.vllm.ai/projects/production-stack/en/latest/（web）
[^20]: Disaggregated Prefill — vLLM Production Stack 文档用例页 — https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/disaggregated-prefill.html（web）