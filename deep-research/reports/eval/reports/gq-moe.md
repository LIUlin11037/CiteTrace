# 混合专家模型（MoE）在大模型中的主流做法与关键工程挑战

## 摘要

混合专家模型（Mixture-of-Experts, MoE）的核心思想是**条件计算**：不再对所有输入复用同一套参数，而是为每个输入样本选择不同的参数子集，从而得到参数量极大但计算成本近似恒定的稀疏激活模型[^3]。该范式可溯源至 Jacobs et al. 1991、Jordan and Jacobs 1994，现代稀疏门控版本由 Shazeer et al. 2017 提出[^1][^3]。当前主流做法高度收敛：以 Transformer 的 FFN 层替换为「N 个专家 + top-k 路由」结构，配合专家并行（EP）与 all-to-all 通信完成分布式训练与推理。代表模型覆盖 7B 到 1T 以上参数区间，激活比例从约 1/7 到约 1/4 不等[^4][^5][^11][^32]。工程挑战集中在三处：**复杂度、通信成本、训练不稳定**[^3]，并延伸出负载均衡、容量因子与 token 丢弃、专家并行通信后端选择、低精度训练与推理等具体问题。

---

## 一、MoE 的标准定义与原始出处

### 1.1 稀疏门控 MoE 层的定义

Shazeer et al. 2017 给出的原始定义是：MoE 层由多达数千个前馈子网络（专家）组成；门控网络（gating network）是可训练的，为每个样本决定使用哪些专家的稀疏组合[^1]。其动机是条件计算（conditional computation）——网络的一部分按样本激活，从而在不按比例增加计算量的前提下大幅提升模型容量[^1]。该工作将 MoE 应用于语言建模与机器翻译，最大模型达 137B 参数，置于堆叠 LSTM 层之间，并声称在 GPU 集群上实现超过 1000 倍的模型容量提升而计算效率损失很小[^1]。

### 1.2 现代 MoE 的核心定义

Switch Transformer 给出的定义更为凝练：模型不再对所有输入复用同一套参数，而是为每个输入样本选择不同的参数，得到稀疏激活模型——参数量极大但计算成本恒定；稀疏性来自为每个输入样本只激活神经网络权重的一个子集[^3]。DeepSpeed 官方文档同样将 MoE 模型定义为稀疏激活模型，其计算成本相对于参数量呈次线性[^29]。DeepSpeed-MoE 论文亦指出 MoE 架构可实现相对于模型参数的次线性（sub-linear）计算需求，从而在不增加训练成本的情况下提升模型质量[^28]。

### 1.3 路由机制的定义

以 Mixtral 8x7B 为例，其路由机制的官方表述为：对每个 token、在每一层，一个 router network 选择两个专家处理当前状态并组合其输出，即 top-2 路由；尽管每个 token 只经过两个专家，被选中的专家在每个时间步可以不同[^4][^21]。Mixtral 8x7B 的架构与 Mistral 7B 相同，区别在于每一层（each layer）都由 8 个前馈块（即专家）组成——即 MoE 替换发生在所有层，而非部分层[^4][^21]。

---

## 二、主流架构做法

### 2.1 专家粒度：从粗粒度到细粒度

传统 MoE 架构（如 GShard，从 N 个专家中激活 top-K 个）在保证专家专业化（expert specialization，即每个专家获得不重叠且聚焦的知识）方面面临挑战[^18]。DeepSeekMoE 提出两条策略：

- **fine-grained expert segmentation（细粒度专家切分）**：将专家细分为 mN 个，并从中激活 mK 个，从而实现更灵活的激活专家组合[^18]。
- **shared expert isolation（共享专家隔离）**：隔离出 K_s 个专家作为共享专家（shared experts），旨在捕获公共知识并缓解路由专家（routed experts）中的冗余[^18]。

DBRX 采用细粒度 MoE 架构，总参数量 132B，任一输入激活 36B 参数；它有 16 个专家、每个 token 选择 4 个，而 Mixtral 和 Grok-1 为 8 个专家选 2 个[^10]。Kimi K2 的官方规格表给出 61 层、384 专家、每 token 选 8 专家、1 共享专家、上下文长度 128K、MLA 注意力[^32]。

### 2.2 路由粒度与激活比例对照

| 模型 | 总参数 | 激活参数 | 专家配置 | 来源 |
|---|---|---|---|---|
| Mixtral 8x7B | 约 46.7B（safetensors 实测） | 13B | 8 专家选 2（num_experts_per_tok = 2） | [^7][^4] |
| Mixtral 8x22B | 约 140.6B（safetensors 实测） | — | 8 专家选 2（num_experts_per_tok = 2） | [^8] |
| Qwen1.5-MoE-A2.7B | 14.3B | 2.7B | — | [^9] |
| DBRX | 132B | 36B | 16 专家选 4 | [^10] |
| Grok-1 | 314B | 激活 25% 权重 | — | [^11] |
| DeepSeek-V2 | 236B | 21B | — | [^6] |
| DeepSeek-V3 | 671B | 37B | — | [^5] |
| OLMoE-1B-7B | 7B | 1B | — | [^30] |
| Kimi K2 | 1T | 32B | 384 专家选 8 + 1 共享专家 | [^32] |

Mixtral 8x7B 每个 token 可访问 47B 参数，但推理时仅使用 13B 活跃参数[^4][^21]。Grok-1 在给定 token 上激活 25% 的权重[^11]。

### 2.3 注意力与配套架构

DeepSeek-V2 与 V3 均采用 Multi-head Latent Attention (MLA) 与 DeepSeekMoE[^5][^6]。MLA 通过将 KV cache 压缩为 latent 向量实现高效推理[^6]。DeepSeek-V2 相比 DeepSeek 67B 节省 42.5% 训练成本，KV cache 减少 93.3%，最大生成吞吐提升至 5.76 倍[^6]。DBRX 使用 RoPE、GLU、GQA，并使用 GPT-4 tokenizer（tiktoken）[^10]。

### 2.4 训练目标与负载均衡

DeepSeek-V3 在基础架构之外实施两项额外策略：auxiliary-loss-free 负载均衡，以及 multi-token prediction 训练目标[^20]。auxiliary-loss-free 策略引用自 Wang et al., 2024a，其目标是减少为鼓励负载均衡而对模型性能产生的负面影响[^20]。multi-token prediction 训练目标被观察到可提升评测基准上的整体表现[^20]。DeepSeek-V3 首创 auxiliary-loss-free（无辅助损失）负载均衡策略[^5]。

### 2.5 训练规模与稳定性

DeepSeek-V3 预训练语料为 14.8 万亿 token，之后经 SFT 与 RL 阶段；完整训练仅需 2.788M H800 GPU 小时；训练过程高度稳定：全程未出现不可恢复的 loss spike，也未执行任何回滚[^5]。DeepSeek-V3 训练支持 FP8 混合精度，并针对训练框架做了全面优化[^20]。Switch Transformer 首次展示大型稀疏模型可用低精度 bfloat16 格式训练[^3]。

Kimi K2 在 15.5T tokens 上预训练，采用 MuonClip 优化器，使用 Muon 优化器训练[^32]。OLMoE-1B-7B 在 5 trillion（5T）tokens 上预训练[^30]。DBRX 预训练使用 12T token 的文本与代码数据[^10]。

### 2.6 加速效果

Switch Transformer 基于 T5-Base/T5-Large，在相同计算资源下获得最高 7 倍预训练加速；在 C4 语料上预训练至万亿参数，相对 T5-XXL 获得 4 倍加速[^3]。Qwen1.5-MoE-A2.7B 由稠密模型 Qwen-1.8B 升级（upcycled）而来，性能与 Qwen1.5-7B 相当，但仅需 25% 的训练资源，推理速度是 Qwen1.5-7B 的 1.74 倍[^9]。DeepSeekMoE 16B 达到与 LLaMA2 7B 相当的性能，而仅使用约 40% 的计算量[^18]；初步扩展至 145B 参数验证了相对 GShard 架构的显著优势，性能与 DeepSeek 67B 相当，仅使用 28.5%（甚至可能 18.2%）的计算量[^18]。

---

## 三、关键工程挑战

### 3.1 三大障碍：复杂度、通信成本、训练不稳定

Switch Transformer 论文明确指出，MoE 广泛采用受阻于三点：复杂度、通信成本、训练不稳定（complexity, communication costs, and training instability）[^3]。这是理解后续所有工程方案的框架。

### 3.2 分布式训练：专家并行与自动分片

GShard 由一组轻量级标注 API 与 XLA 编译器扩展组成，用于表达并行计算模式；它将多语言神经机器翻译 Transformer 用 Sparsely-Gated Mixture-of-Experts 扩展到超过 6000 亿参数（automatic sharding），在 2048 块 TPU v3 上训练 4 天，用于 100 种语言到英语的翻译[^2]。论文将条件计算与自动分片作为核心工程手段[^2]。

DeepSpeed 提供 MoE 训练支持，通过 `--moe-expert-parallel-size` 参数控制专家并行度，即每张 GPU 上放置 num-experts/moe-expert-parallel-size 个专家；该参数约束为不得超过 GPU 数量，也不得超过 `--num-experts`[^22]。DeepSpeed 实验中将每层专家数 `--num-experts` 设为 128，并指出专家数越多收敛越好但存在收益递减[^22]。

DeepSpeed MoE 支持五种并行形式，并同时利用 GPU 与 CPU 内存，允许混合多种并行技术，包括 E（Expert）、E+D（Expert+Data）、E+Z（Expert+ZeRO-powered data）、E+D+M（Expert+Data+Model）、E+D+Z、E+Z-Off+M（Expert+ZeRO-Offload+Model）[^29]。MoE 层 API 现在接受 ep_size 参数（除 num_experts 外），允许每个 MoE 层拥有不同的专家数量与不同的专家并行度；参与大小为 ep_size 的专家并行组的 GPU/rank 会分摊该层指定的专家总数[^29]。MoE 层 API 为 `deepspeed.moe.layer.MoE(hidden_size=..., expert=..., num_experts=..., ep_size=...)`，其输入与输出维度相同[^29]。

MegaBlocks 自述为轻量级 MoE 训练库，核心是高效的 dropless-MoE（dMoE）与标准 MoE 层；README 称已与 Megatron-LM 集成，支持 MoE 的数据并行、专家并行与流水线并行训练[^17]。MegaBlocks README 声称 dMoE 相比 Tutel 最佳 capacity_factor 配置性能提升最高 40%（仓库自述主张，未注明第三方验证）[^17]。

### 3.3 容量因子与 token 丢弃

DeepSpeed 提供 `--moe-train-capacity-factor` / `--moe-eval-capacity-factor` / `--moe-min-capacity` 控制单个专家可处理的 token 数；容量越大收敛越好但负载更不均衡、训练更慢[^22]。`--disable-moe-token-dropping` 可完全移除单专家 token 数上限，但官方仅推荐在推理/评估时使用[^22]。

### 3.4 推理侧：专家并行与 all-to-all 通信后端

vLLM 支持 Expert Parallelism (EP)，将 MoE 模型的专家部署到不同 GPU 上，以提升 locality、efficiency 和整体 throughput；EP 通常与 Data Parallelism (DP) 配合使用，DP 可独立于 EP 使用，但 EP 与 DP 结合时效率更高[^23]。使用 EP 需安装 DeepEP（EP kernels）与 DeepGEMM 库；disaggregated serving 还需安装 gdrcopy[^23]。deepep_v2 后端要求 NCCL >= 2.30.4（CUDA 13+），而 PyTorch 自带的 NCCL 版本较旧，需先升级[^23]。vLLM 通过 `--all2all-backend` 选择 EP 通信后端，提供 allgather_reducescatter（默认，通用）、deepep_high_throughput（多节点 prefill，grouped GEMM + continuous layout）、deepep_low_latency（多节点 decode，CUDA graph + masked layout）、flashinfer_nvlink_one_sided（MNNVL 系统）等选项[^23]。

vLLM 通过多个 all2all 通信后端为 MoERunner 层实现 expert parallelism (EP)，每个后端对应一个 FusedMoEPrepareAndFinalizeModular 子类接口；不同后端在 activation 格式（standard/batched）、支持的量化方案（如 deepep_high_throughput 仅支持 block-quantized fp8）以及 async 支持上存在差异；若未指定量化类型，kernel 以 float16 和/或 bfloat16 运行；Async 后端支持 DBO（Dual Batch Overlap）[^24]。

FusedMoEModularKernel 依据输入 activation 格式分为两类：Contiguous/Standard/Non-Batched 与 Batched；输入 activation 格式完全取决于所用的 All2All Dispatch——Contiguous 变体返回形状 (M, K) 的连续张量及 TopK Ids/weights，Batched 变体返回形状 (num_experts, max_tokens, K) 的张量，并附带 expert_num_tokens 张量；Batched 与 Non-Batched 在操作上的主要差异是 Permute/Unpermute 操作，其余操作相同[^25]。

SGLang 的 EP 将 MoE 模型专家权重分布到多设备，解决内存瓶颈并支持大规模 MoE 推理扩展；通过优化的 all-to-all 通信与 grouped GEMM 降低延迟、提升吞吐、减少 GPU 空闲时间[^26]。SGLang EP 通过 `--moe-a2a-backend` 选择 all-to-all 通信后端，通过 `--moe-runner-backend` 选择 MoE 计算后端[^26]。all-to-all 后端包括：none（默认，用 All-Reduce/All-Gather 做 token dispatch，适用于 hybrid EP+TP）、deepep（大规模 EP）、mooncake（弹性推理，RDMA）、nixl（NIXL-EP，弹性 EP，原生 RDMA/NVLink）、mori（AMD ROCm 原生）、flashinfer、ascend_fuseep（Ascend NPU）、pplx（Perplexity 的 NVSHMEM-based dispatch/combine kernel，面向 Hopper 上 FP8 DeepGEMM MoE 的低延迟 decode）[^26]。DeepEP 与 Mooncake 后端支持两种 token dispatch 模式：normal 模式（面向 prefill 高吞吐）与 low_latency 模式（面向 decode 低延迟、兼容 CUDA Graph）；推荐设置 `--deepep-mode auto` 以在运行时自动切换[^26]。DeepEP、Mooncake、NIXL-EP、ascend_fuseep、pplx、MORI 目前仅支持 ep_size = tp_size；hybrid EP+TP（ep_size < tp_size）仅支持 none 后端[^26]。

### 3.5 推理成本与压缩

DeepSpeed-MoE 论文声称其新颖的 MoE 架构设计与模型压缩技术可将 MoE 模型体积最多减少 3.7x；其高度优化的推理系统相比现有 MoE 推理方案提供 7.3x 更好的延迟与成本；相比同等质量的稠密模型可提供最高 4.5x 更快、9x 更便宜的推理；对自回归语言模型可实现 5x 的训练成本节省[^27]。DeepSpeed 官方称该 MoE 方案可将训练成本降低 5 倍、MoE 模型规模缩小 3 倍（引自其 Blog）；PR-MoE（Pyramid-Residual-MoE）相比标准 MoE 将参数效率提升最多 3 倍[^22]。DeepSpeed 文档举例：Switch Transformer 拥有超过 1.6 万亿参数，但其训练所需计算量约等于一个 100 亿参数稠密模型[^29]。

### 3.6 开源工具链现状

| 仓库 | stars | 定位 | 来源 |
|---|---|---|---|
| vllm-project/vllm | 93767 | 'Easy, fast, and cheap LLM serving for everyone'，topics 含 moe、deepseek-v3、kimi、qwen3 | [^12] |
| sgl-project/sglang | 37126 | 'Fast inference for LLMs and multimodal models'，topics 含 moe、deepseek、glm | [^13] |
| NVIDIA/Megatron-LM | 18148 | 'GPU-optimized library for training transformer models at scale'，release 0.19.0 | [^14] |
| deepspeedai/DeepSpeed | 43229 | topics 含 mixture-of-experts、model-parallelism、pipeline-parallelism、zero | [^15] |
| microsoft/Tutel | 1026 | 'An Optimized Mixture-of-Experts Implementation'，MIT 许可 | [^16] |
| databricks/megablocks | 1598 | 轻量级 MoE 训练库，dropless-MoE | [^17] |

Tutel README 自述为 'An Optimized Mixture-of-Experts Implementation'，并称是首个提出 'No-penalty Parallism/Sparsity/Capacity/.. Switching' 的并行方案，面向具有动态行为的现代训练与推理；README 声称支持对 MoE 模型（GLM-5.x / DeepSeek-3.x / Kimi-2.x / Kimi-3.x / Qwen3 / Gpt-OSS）在 A100/A800/H100/MI300 上进行 NVFP4/MXFP4/BlockwiseFP8 直接推理，并给出具体部署配置示例：MI300x8/MI325x8（192GB PCIe5）可运行 Kimi K3（2.8TB），max-context-size = 1M[^16]。

### 3.7 开放性与可复现性

OLMoE 论文正文指出 MoE 在每层包含多个专家，每次仅激活其中一个子集；论文称业界前沿模型使用 MoE，包括 Gemini-1.5，以及据报道的 GPT-4；论文指出多数 MoE 模型为闭源，OLMoE 定位为最开放的 MoE 模型之一[^31]。OLMoE 采用稀疏 MoE 架构，并开源了模型权重、训练数据、代码与日志[^30]。Mixtral 基础模型与指令微调模型均以 Apache 2.0 许可发布[^4]；Grok-1 以 Apache 2.0 许可发布权重与架构[^11]。

---

## 四、矛盾与不确定性

1. **Llama-4 覆盖缺口**：本报告 32 条证据中完全没有 Llama-4（Scout/Maverick）的任何来源，无法给出其总参数、激活参数、专家数、top-k、路由粒度、训练 token 与上下文长度。现有公开资料不足以确认 Llama-4 的 MoE 配置。

2. **DeepSeek-V2/V3 专家级参数缺口**：现有 DeepSeek-V2/V3 证据只给出总参数（236B / 671B）、激活参数（21B / 37B）与训练 token 量（8.1T / 14.8T），缺少专家数量、每 token 激活专家数、路由粒度等关键字段，无法支撑完整的参数对照表。现有公开资料不足以确认 DeepSeek-V2/V3 的专家数量与 top-k 具体数值。

3. **专家卸载（expert offloading）与显存容量挑战缺口**：现有证据仅涉及 EP、all-to-all 通信后端与 grouped GEMM，没有任何关于专家卸载（如 DeepSpeed ZeRO-Offload 用于 MoE、vLLM/SGLang 的 CPU offload、MoE 显存容量瓶颈）的一手来源。现有公开资料不足以确认 MoE 专家卸载的主流工程方案。

4. **厂商主张与第三方验证的区分**：Mixtral-8x7B 在多数基准上优于 Llama 2 70B 属厂商主张[^7]；OLMoE 在相近激活参数量的模型中表现最佳、超过 Llama2-13B-Chat 与 DeepSeekMoE-16B 属论文自述主张[^30]；MegaBlocks dMoE 相比 Tutel 最佳 capacity_factor 配置性能提升最高 40% 属仓库自述主张，未注明第三方验证[^17]。这些数字应谨慎对待。

5. **MoE 早期范围局限**：DeepSpeed-MoE 论文明确指出 MoE 模型存在限制其广泛应用的挑战，其中一项为「Limited Scope」——NLP 领域 MoE 模型的范围主要局限于 encoder-decoder 模型与 sequence-to-sequence 任务[^28]。这一判断与后续 decoder-only MoE 大模型的普及形成时间上的对照，但本报告证据不足以量化该转变的节点。

6. **稠密模型成本参照**：DeepSpeed-MoE 论文以 Megatron-Turing NLG 530B 为例说明稠密模型训练成本：在 NVIDIA Selene 超算上使用 2000+ A100 GPU 训练约 3 个月，消耗超过 300 万 GPU 小时[^28]。该数据用于对比 MoE 的成本优势，但不同硬件与时期的可比性有限。

---

## 参考来源

[^1]: Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017) — https://arxiv.org/abs/1701.06538v1 (arxiv, 2017-01-23)
[^2]: GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (Lepikhin et al., 2020) — https://arxiv.org/abs/2006.16668v1 (arxiv, 2020-06-30)
[^3]: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Fedus, Zoph, Shazeer, 2021) — https://arxiv.org/abs/2101.03961v3 (arxiv, 2021-01-11)
[^4]: Mixtral of Experts (Jiang et al., Mistral AI, 2024) — https://arxiv.org/abs/2401.04088v1 (arxiv, 2024-01-08)
[^5]: DeepSeek-V3 Technical Report — https://arxiv.org/abs/2412.19437v2 (arxiv, 2024-12-27)
[^6]: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model — https://arxiv.org/abs/2405.04434v5 (arxiv, 2024-05-07)
[^7]: Model Card for Mixtral-8x7B (Hugging Face) — https://huggingface.co/mistralai/Mixtral-8x7B-v0.1 (github, 2023-12-01)
[^8]: Model Card for Mixtral-8x22B (Hugging Face) — https://huggingface.co/mistralai/Mixtral-8x22B-v0.1 (github, 2024-04-16)
[^9]: Qwen1.5-MoE-A2.7B (Hugging Face Model Card) — https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B (github, 2024-02-29)
[^10]: Introducing DBRX: A New State-of-the-Art Open LLM (Databricks Blog) — https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm (web, 2024-03-27)
[^11]: Open Release of Grok-1 (xAI Blog) — https://x.ai/blog/grok-os (web, 2024-03-17)
[^12]: vllm-project/vllm — https://github.com/vllm-project/vllm (github)
[^13]: sgl-project/sglang — https://github.com/sgl-project/sglang (github)
[^14]: NVIDIA/Megatron-LM — https://github.com/NVIDIA/Megatron-LM (github)
[^15]: deepspeedai/DeepSpeed — https://github.com/deepspeedai/DeepSpeed (github)
[^16]: microsoft/Tutel — https://github.com/microsoft/Tutel (github)
[^17]: databricks/megablocks — https://github.com/databricks/megablocks (github)
[^18]: DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models (arXiv:2401.06066, abstract page) — https://arxiv.org/abs/2401.06066 (arxiv, 2024-01-11)
[^19]: DeepSeekMoE (arXiv:2401.06066) HTML full text — abstract, Figure 1 caption, Introduction (truncated) — https://arxiv.org/html/2401.06066v1 (arxiv, 2024-01-11)
[^20]: DeepSeek-V3 Technical Report (HTML full text, Introduction) — https://arxiv.org/html/2412.19437v2 (arxiv, 2024-12-27)
[^21]: Mixtral of Experts — https://arxiv.org/abs/2401.04088 (arxiv, 2024-01-08)
[^22]: DeepSpeed Mixture of Experts (MoE) Tutorial — https://www.deepspeed.ai/tutorials/mixture-of-experts-nlg/ (web)
[^23]: Expert Parallel Deployment — vLLM Documentation — https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html (web)
[^24]: Fused MoE Kernel Features — vLLM Documentation — https://docs.vllm.ai/en/latest/design/moe_kernel_features.html (web)
[^25]: Fused MoE Modular Kernel — vLLM Documentation — https://docs.vllm.ai/en/latest/design/fused_moe_modular_kernel.html (web)
[^26]: Expert Parallelism (EP) — SGLang Documentation — https://docs.sglang.ai/advanced_features/expert_parallelism.html (web)
[^27]: DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale (arXiv:2201.05596v2, ICML 2022) — https://arxiv.org/abs/2201.05596v2 (arxiv, 2022-01-14)
[^28]: DeepSpeed-MoE (arXiv:2201.05596v2) — Introduction 正文 — https://arxiv.org/html/2201.05596v2 (arxiv, 2022-01-14)
[^29]: DeepSpeed MoE Tutorial (官方文档) — https://www.deepspeed.ai/tutorials/mixture-of-experts/ (web)
[^30]: OLMoE: Open Mixture-of-Experts Language Models — https://arxiv.org/abs/2409.02060v2 (arxiv, 2024-09-03)
[^31]: OLMoE: Open Mixture-of-Experts Language Models (HTML full text, Introduction) — https://arxiv.org/html/2409.02060v2 (arxiv, 2024-09-03)
[^32]: MoonshotAI/Kimi-K2 (official repository) — https://github.com/MoonshotAI/Kimi-K2 (github)