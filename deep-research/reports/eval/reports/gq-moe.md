## 摘要

混合专家模型（MoE）在大模型中的主流做法已从 2017 年的「稀疏门控 + top-k 路由」原型收敛为一条相对清晰的工程路线：**细粒度专家分割 + 共享专家隔离 + 无辅助损失（或轻量辅助损失）的负载均衡 + 专家并行（EP）为主的混合并行 + 低精度训练**。DeepSeek-V3 以 671B 总参数 / 37B 激活参数、14.8T token、2.788M H800 GPU 小时的配置，是当前公开披露最完整的 MoE 训练样本[^5][^18]；Mixtral 8x7B 的 top-2 路由（47B 总参数 / 13B 激活）代表了另一条更简单的路线[^7]。关键工程挑战集中在四处：负载均衡与路由崩塌、专家并行的 all-to-all 通信、训练稳定性、以及推理侧的专家调度与显存。需要说明的是，本报告所依据的证据在若干被广泛讨论的模型上存在明显缺口（Llama 4 完全缺失、Qwen3 MoE 细节缺失、GPT-4/Gemini 仅有官方否定性口径），这些在下文与末节显式标注。

---

## 一、MoE 的标准定义与计算结构

MoE 的定义性表述在 Switch Transformer 中给出得最清楚：常规深度学习对所有输入复用同一套参数，MoE 则对每个输入选择不同参数，从而得到「稀疏激活」模型——参数量极大但计算成本恒定[^2]。

具体到层结构，Shazeer et al. 2017 的定义是：MoE 层由 n 个专家网络 E_1..E_n 与一个门控网络 G 组成，G 的输出是一个稀疏的 n 维向量；MoE 模块的输出为

y = Σ_{i=1}^{n} G(x)_i · E_i(x)

当 G(x)_i = 0 时无需计算 E_i(x)，计算节省完全来自门控输出的稀疏性[^1]。该论文把 MoE 定位为「条件计算」（conditional computation）的一种通用神经网络组件：按样本激活部分网络，在不按比例增加计算的情况下大幅提升模型容量[^1]。

路由的具体步骤（以 Shazeer 2017 的 Noisy Top-K Gating 为准）：路由器变量 W_r 产生 logits h(x) = W_r · x，在 softmax 之前加入可调高斯噪声，然后只保留 top-k 个值、其余置为 −∞（使对应门控值为 0），再在 N 个专家上做 softmax 归一化，专家 i 的门控值由此给出[^1][^2]。噪声的作用是引入探索、缓解路由的早期固化；稀疏性用于节省计算[^1]。

**输入 / 步骤 / 产出**可以概括为：输入是 token 表示 x；步骤是「线性投影得 logits →（可选加噪）→ top-k 选择 → softmax 归一化 → 按门控值加权求和被选中专家的输出」；产出是与 x 同维的表示 y，以及路由决策（每个 token 去了哪些专家）这一副产品——后者正是所有负载均衡机制的统计对象[^1][^2]。

---

## 二、主流做法：从 top-k 到细粒度 + 共享专家

### 2.1 路由稀疏度：top-1 与 top-2 两条线

Switch Transformer 把 top-k 简化为 top-1（每 token 只路由到单个专家），并给出三点收益：路由器计算减少、每个专家的批大小（expert capacity）至少可减半、路由实现简化且通信成本降低[^2]。Mixtral 8x7B 则采用 top-2：每层由 8 个前馈块（专家）组成，每个 token 在每一层由路由器网络选择 2 个专家处理当前状态并合并其输出，每个 token 可访问 47B 参数但推理时仅使用 13B 激活参数，且被选中的专家在每个时间步可以不同[^7]。

### 2.2 专家粒度与共享专家：DeepSeekMoE 的两项改造

DeepSeekMoE 指出 GShard 等传统 MoE 架构（从 N 个专家中激活 top-K）在保证专家特化（每个专家获得不重叠且聚焦的知识）方面存在挑战，存在知识混杂与知识冗余问题，且常仅用 8 或 16 个专家[^6][^9]。其两项核心策略是：

1. **细粒度专家分割**：把专家切分为 mN 个并从中激活 mK 个，从而允许更灵活的激活专家组合；做法是在保持参数总量不变的前提下拆分 FFN 中间隐藏维度，并在保持计算量不变的前提下激活更多细粒度专家[^6][^9]。
2. **共享专家隔离**：隔离 K_s 个专家作为始终激活的共享专家，用于捕获跨上下文的公共知识，缓解路由专家之间的知识冗余[^6][^9]。

效果上，2B 规模的 DeepSeekMoE 达到与 GShard 2.9B 相当的性能，而后者拥有 1.5 倍的专家参数与计算量；16B 规模达到与 LLaMA2 7B 相当的性能而仅用约 40% 计算量；初步扩展到 145B 参数时，以仅 28.5%（甚至可能 18.2%）的计算量达到与 DeepSeek 67B 相当的性能[^6][^9]。DeepSeek-V2（236B 总参数 / 21B 激活，128K 上下文，8.1T token 预训练）与 DeepSeek-V3（671B / 37B）均采用「MLA + DeepSeekMoE（细粒度专家分割 + 共享专家隔离）」这一组合[^8][^5][^18]。DeepSeek-V2 相对 DeepSeek 67B 节省 42.5% 训练成本、KV cache 减少 93.3%、最大生成吞吐提升至 5.76 倍[^8]。

### 2.3 负载均衡：辅助损失 → 线性分配 → 无辅助损失

这是过去几年变化最明显的一条线。

**辅助损失路线。** Switch Transformer 为鼓励专家间负载均衡加入辅助损失，并简化了 Shazeer 2017 中分离的 load-balancing 与 importance-weighting 两项损失[^2]。该损失由两个向量构成：f_i 为批次 B 中被路由到专家 i 的 token 比例（argmax p(x) = i 的指示函数均值），P_i 为路由器分配给专家 i 的概率份额均值；均匀路由时二者均应为 1/N[^2]。该损失在均匀分布下最小化，P 向量可微、f 向量不可微，最终损失乘以专家数 N 以在专家数变化时保持损失量级恒定，系数 α 取 10^-2（在 10^-1 到 10^-5 间扫描）[^2]。

**线性分配路线。** BASE 层把 token 到专家的分配建模为线性分配问题，从而得到每个专家接收等量 token 的最优分配；该方案通过保证计算负载均衡来提升效率，且不需要任何新的超参数或辅助损失[^3]。其动机是：学习能充分利用可用专家的均衡路由函数本身很难，已有方法通常依赖路由启发式或辅助均衡损失[^3]。

**无辅助损失路线。** Loss-Free Balancing 指出较大的辅助损失会向训练引入不可忽视的干扰梯度（interference gradients），从而损害模型性能；其做法是在 top-K 路由决策之前，先对每个专家的路由分数施加一个 expert-wise bias，该 bias 依据每个专家近期的负载动态更新，从而持续维持均衡分布，且不产生任何干扰梯度[^4]。实验在最多 3B 参数、最多 200B token 训练的 MoE 模型上验证，相比传统辅助损失策略同时取得更好的性能和更好的负载均衡[^4]。DeepSeek-V3 称其首创（pioneers）无辅助损失的负载均衡策略，目的正是最小化为鼓励负载均衡而对模型性能造成的不利影响[^5][^18]。

### 2.4 容量因子与 token 丢弃

Switch Transformer 引入容量因子（capacity factor）：大于 1.0 的容量因子为 token 在专家间分布不完美均衡提供额外缓冲；若路由到某专家的 token 过多（dropped tokens），则跳过计算，token 表示经残差连接直接传到下一层[^2]。权衡在于：提高专家容量并非没有代价，高取值会造成计算与内存浪费；实验中发现保证较低的 token 丢弃率对稀疏专家模型的扩展很重要，通常丢弃率 < 1%[^2]。

### 2.5 并行策略：专家并行成为一等公民

Switch Transformer 通过高效结合数据并行、模型并行与专家并行（expert parallelism）构建最高万亿参数模型[^2]。DeepSeek-V2 训练中使用专家并行，并设计了控制通信开销与保证负载均衡的补充机制[^8]。工程框架侧，Megatron Core 提供 TP、PP、DP、EP、CP 五种并行策略，其中 EP 即面向 MoE 的专家并行，并支持 FP16/BF16/FP8/FP4 混合精度[^21]；DeepSpeed-MoE 支持五种可混合的并行形式：E、E+D、E+Z、E+D+M、E+D+Z、E+Z-Off+M，MoE 层 API 为 `deepspeed.moe.layer.MoE(hidden_size, expert, num_experts, ep_size, ...)`，ep_size 与 num_experts 作为参数传入，允许每层使用不同的专家数与专家并行度[^23]。DeepSpeed 文档还给出具体换算示例：WORLD_SIZE=4、EP_WORLD_SIZE=2、EXPERTS=[^8] 时，8 个专家分布在 4 张 GPU 上，即 4 experts/GPU 的 E+D 模式[^23]。

### 2.6 低精度训练与蒸馏回稠密

Switch Transformer 称首次证明大型稀疏模型可用 bfloat16 低精度训练，并提出缓解训练不稳定的技术[^2][^10]；同时稀疏模型可蒸馏回小型稠密模型，模型规模缩减最高 99% 并保留 30% 的质量增益[^2]。DeepSeek-V3 进一步支持 FP8 混合精度训练并针对训练框架做了全面优化[^5]。

### 2.7 规模收益的量化主张

Switch Transformer 基于 T5-Base/T5-Large 在同等算力下获得最高 7x 预训练加速；在 C4 上预训练至万亿参数模型，相对 T5-XXL 获得 4x 加速；多语言上 101 种语言全部获益，91% 的语言获得 4x+ 加速[^2][^10]。DeepSpeed 文档给出的对照是：Switch Transformer 参数超 1.6 万亿，而训练所需计算约等于一个 100 亿参数稠密模型[^23]。更早的 Shazeer 2017 声称实现 >1000x 的模型容量提升，且在现代 GPU 集群上计算效率仅有轻微损失，最大模型达 137B 参数，专家数可达数千个而每样本只评估少数几个[^1]。

---

## 三、关键工程挑战

### 3.1 负载均衡与路由崩塌

MoE 中专家负载不均衡会导致路由崩塌（routing collapse）或计算开销上升[^4]。这是 MoE 最核心的失效模式：路由是自我强化的——某专家被选得越多、训练越充分、越容易被再选。应对手段即上节三条路线（辅助损失 / 线性分配 / expert-wise bias），但每条都有代价：辅助损失引入干扰梯度[^4]，容量因子过高浪费算力与显存、过低则丢弃 token[^2]，线性分配需要求解分配问题[^3]。Switch 论文把「复杂性、通信成本、训练不稳定性」列为 MoE 广泛采用受阻的三点原因[^2][^10]。

### 3.2 通信：all-to-all 与专家并行的开销

Shazeer 2017 就已列出条件计算落地的工程障碍：GPU 更擅长算术而非分支、条件计算会缩小条件激活块的批大小、网络带宽可能成为瓶颈、可能需要额外损失项来实现稀疏与负载均衡[^1]。在专家并行下，token 需要被发送到持有目标专家的设备再送回，这一 all-to-all 交换是 MoE 训练与推理的主要通信成本；Switch 路由简化带来的收益之一正是通信成本降低[^2]。

公开的量化数据主要来自推理侧：SGLang 在 12 节点 × 8 张 H100（共 96 张 H100）上部署 DeepSeek，支持 prefill-decode（PD）分离与大规模专家并行，包含 DeepEP、DeepGEMM 与 EPLB（Expert Parallelism Load Balancer）；在 2000-token 输入序列下达到每节点 52.3k input tokens/s 与 22.3k output tokens/s，相比同等资源下的朴素张量并行（TP）将输出吞吐最高提升 5 倍[^12]。该博客还给出一个具体的并行选择理由：DeepSeek-V3 的 intermediate dimension 为 18,432，高 TP 度（如 TP32）会切分为 576 单元，无法被 H100 常用的 128 对齐边界整除，损害计算效率与显存利用率，因此 Dense FFN 采用数据并行而非张量并行；MLA 注意力层则采用 DP Attention 以消除跨设备的 KV cache 复制[^12]。其技术内容还覆盖 Two-batch Overlap（双批次重叠以隐藏通信）、EPLB 负载均衡器、Disposable Tensor、专家工作负载提取与模拟，以及针对 Two-batch Overlap / EPLB 的消融实验[^12]。

需要强调：**训练侧 all-to-all 的公开 benchmark、专家 offloading、推理时专家缓存与批处理策略、straggler 缓解，在本批证据中均无 2025–2026 年的量化来源**，仅有上述 SGLang 一篇推理侧博客提供数据。

### 3.3 训练稳定性

Switch 论文明确提出缓解训练不稳定的技术，并将其列为 MoE 三大采用障碍之一[^2][^10]。DeepSeek-V3 的报告则从正面给出了稳定性证据：预训练使用 14.8T 高质量多样化 token，预训练过程非常稳定，全程未出现不可恢复的 loss spike，也未执行任何回滚[^5][^18]。这两条证据构成对照——稳定性是可实现的，但需要专门的工程手段（V3 归因于 FP8 混合精度训练与针对训练框架的全面优化[^5]）。

### 3.4 训练成本与硬件规模

DeepSeek-V3 提供了目前最完整的成本口径：完整训练仅需 2.788M H800 GPU 小时（预训练 2664K + 上下文扩展 119K + 后训练 5K），按 $2/GPU 小时计约 5.576M 美元；预训练阶段每训练 1T token 仅需 180K H800 GPU 小时，即在 2048 张 H800 的集群上约 3.7 天[^5]。上下文长度分两阶段扩展：先扩到 32K，再扩到 128K[^5]。作为对照，DeepSeek-V2 的预训练语料为 8.1T token，相对 DeepSeek 67B 节省 42.5% 训练成本[^8]。SGLang 博客给出的推理侧折算成本为 $0.20 / 1M output tokens，约为官方 DeepSeek Chat API 成本的五分之一[^12]。

### 3.5 推理侧：显存、调度与部署形态

MoE 推理的核心矛盾是「总参数决定显存、激活参数决定算力」。Mixtral 8x7B 的 47B 总参数 / 13B 激活参数即典型体现[^7]。部署工具方面，Tutel 自我定位为面向现代训练与推理的优化 MoE 实现，提出 "No-penalty Parallelism/Sparsity/Capacity/.. Switching"，声明支持 NVFP4 / MXFP4 / BlockwiseFP8 低精度直接推理，覆盖 GLM-5.x、DeepSeek-3.x、Kimi-2.x、Kimi-3.x、Qwen3、Gpt-OSS 等模型族，硬件覆盖 A100/A800/H100/MI300（部署表另列 B200x8、RTX-6000PRO SXM、MI325x8），并提供 Docker 镜像与 OpenAI 兼容的 /v1/models、/chat 接口[^11]。**但 Tutel README 中的吞吐表与「vLLM/SGLang 为 0 t/s（OoM）」对比属项目自述、未经第三方验证，且涉及 2026 年新模型族，本报告不将其作为性能结论引用。** Megatron Core 侧则提供 Megatron MoE Model Zoo，称其提供训练 DeepSeek-V3、Mixtral、Qwen3 MoE 模型的最佳实践与优化配置，含性能基准与 checkpoint 转换工具，并称代码库可在数千 GPU 上高效训练 2B 至 462B 参数模型[^21]。

---

## 四、架构披露的边界：哪些「MoE 大模型」其实没有官方确认

这一节是本报告必须显式声明的方法论限制。

**GPT-4。** OpenAI 官方技术报告摘要将 GPT-4 描述为 "a large-scale, multimodal model which can accept image and text inputs and produce text outputs"，架构类别为 "a Transformer-based model pre-trained to predict the next token in a document"，摘要中未出现 mixture-of-experts 或任何稀疏化架构表述，也未给出任何参数量数字[^14]。报告强调的核心是 "developing infrastructure and optimization methods that behave predictably across a wide range of scales"，并称可基于不超过 GPT-4 计算量 1/1,000 的模型预测其部分性能[^14]。

**Gemini 1.5 / 2.5。** Gemini 1.5 报告第 1 节称该系列 "incorporates our latest innovations in sparse and dense scaling"，即官方承认使用了稀疏与稠密两种扩展方式，但所读章节未出现 "mixture of experts" 字样；效率提升被归因于 "major advances in training, distillation and serving infrastructure"，而非明确指向 MoE 路由结构[^15]。报告称 Gemini 1.5 Pro "surpasses Gemini 1.0 Pro and 1.0 Ultra on a wide array of benchmarks while requiring significantly less compute to train"——以更少训练算力超越更大模型，这是外界推测其为稀疏/MoE 架构的间接线索之一，但报告未在所读章节披露任何参数量数字[^15]。Gemini 2.5 报告声明包含对架构的描述，但所抓取的摘要与第 1 节正文中未出现 mixture-of-experts 表述，也未在所读章节披露参数量[^16]。

**结论**：现有证据只能支持「官方未确认 MoE 结构」这一否定性陈述；**缺少传闻侧来源（如第三方分析报道）及其可信度评估，因此无法呈现「传闻 vs 官方口径」的对照**。任何把 GPT-4 或 Gemini 当作 MoE 案例的具体参数描述都不应写入研究报告。

---

## 五、矛盾与不确定性

1. **Llama 4 完全缺失。** 本批证据中没有任何 Llama 4 来源，其总参数/激活参数、专家数、top-k、共享专家、训练 token、硬件规模均无法确认。

2. **Qwen3 MoE 细节缺失。** Qwen3 技术报告确认该系列同时包含 dense 与 MoE 两种架构、参数规模覆盖 0.6B 至 235B、支持 thinking/non-thinking 统一框架与 thinking budget、多语言从 29 种扩展到 119 种、全部以 Apache 2.0 公开[^13]；但该来源自述未给出 Qwen3-235B-A22B 的专家数量、top-k、共享专家配置或训练 token 量等具体数值[^13]。因此无法建立 Qwen3 MoE 的参数事实表。

3. **Mixtral 8x22B 无任何来源。** 仅有 Mixtral 8x7B 的 47B/13B、8 专家、top-2、32k 训练上下文、Apache 2.0 发布等信息[^7]。Mixtral 与 Switch 的训练 token 量与 GPU 数量/成本量级，除 DeepSeek-V2/V3 外均缺失。

4. **vLLM 与 Fairseq 的 MoE 支持程度无法评估。** 两条证据均为仓库元数据（vLLM ★95401、最近 push 2026-10-25、简介为高吞吐内存高效 LLM 推理与服务引擎[^24]；fairseq ★32211、最近 push 2025-09-30、简介为 Python 序列到序列工具包，同组织 fairseq2 为独立仓库且活跃度差异明显[^25]），README 正文抓取失败。用元数据推断 MoE 支持程度属于空泛证据，本报告不采用。BASE 层虽声明代码在 fairseq 公开[^3]，但该仓库当前对 BASE/MoE 的实际支持状态未获正文取证。

5. **Tutel 的性能对比不可采信。** README 中 GLM-5.3-Flash 吞吐表（MI300 单卡 6,120 tok/s/chip 输入 0% 命中、32,191 tok/s/chip 输入 90% 命中、输出 210 tok/s/chip；H200 双卡 4,900 / 25,774 / 60.9 tok/s/chip）及与 vLLM/SGLang 的对比（AMD MI300X x8 + GLM-5.3 场景 vLLM/SGL 为 0 t/s 因 OoM，Tutel 为 761 t/s）均为项目自述、未经第三方验证，且涉及 2026 年新模型族与「竞品 OoM」这类强主张[^11]。仓库元数据仅提供 stars（1029）、描述与最近提交时间，不含任何吞吐数字的验证信息[^26]。本报告仅引用其功能声明（低精度支持、硬件覆盖、部署接口），不引用其性能数字。

6. **2025–2026 年工程挑战的解决方案缺乏量化证据。** all-to-all 通信瓶颈、专家 offloading、推理时专家缓存与批处理、straggler 缓解，仅有 SGLang 一篇博客（96×H100、52.3k input / 22.3k output tok/s/node、相对 TP 最高 5x 提升、$0.20/1M output tokens）提供量化数据，且只覆盖 EP + EPLB + PD 分离[^12]。其余挑战在 2025–2026 的实际解决方案与公开 benchmark 数据，现有公开资料不足以确认。

7. **DeepSeek-R1 与 MoE 架构的关系存在证据边界。** R1-Zero 与 R1 均以 DeepSeek-V3-Base 为起点训练，即 R1 系列继承 V3 的基座权重[^17]，因此间接继承了 V3 的 MoE 结构；但 R1 论文摘要层面未给出总参数/激活参数、训练 token 量或 GPU 小时等具体数字[^19]，官方 GitHub 仓库本次仅取得元数据（约 91,954 stars、11,660 forks、最近 push 2025-06-27），README 正文抓取失败，故不得作为架构或算力结论的依据[^20]。R1 的高性能 RL 基础设施细节位于 Supplementary B.1，本次抓取未覆盖[^17]。

8. **「首创」类表述需谨慎。** DeepSeek-V3 称首创无辅助损失负载均衡策略[^5]，而 Loss-Free Balancing 论文（2024-08）已提出同一思路并在 3B/200B token 规模验证[^4]。两者的时间关系与「首创」归属，现有证据不足以判定，本报告仅并列陈述双方表述。

9. **Megatron-LM README 中的未来时间戳条目。** README 新闻条目包含 [2026/03] 的 MoE 扩展技术报告、[2026/05] 的 DeepSeek-V4 初始实现等[^21]。这些条目与仓库最近 push 时间（2026-10-24）自洽，但其中涉及的模型与报告在本批证据中无其他来源交叉验证，本报告仅作为「框架路线图」陈述，不作为模型事实。

## 参考来源

[^1]: Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017) — https://arxiv.org/html/1701.06538v1（arxiv, 2017-01-23）
[^2]: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Fedus, Zoph, Shazeer, 2021/2022, JMLR) — https://arxiv.org/html/2101.03961v3（arxiv, 2021-01-11）
[^3]: BASE Layers: Simplifying Training of Large, Sparse Models (Lewis et al., 2021) — https://arxiv.org/abs/2103.16716v1（arxiv, 2021-03-30）
[^4]: Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts (Loss-Free Balancing) — https://arxiv.org/abs/2408.15664（arxiv, 2024-08-28）
[^5]: DeepSeek-V3 Technical Report — https://arxiv.org/html/2412.19437v2（arxiv, 2024-12-27）
[^6]: DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models — https://arxiv.org/abs/2401.06066（arxiv, 2024-01-11）
[^7]: Mixtral of Experts (arXiv:2401.04088v1) — https://arxiv.org/abs/2401.04088v1（arxiv, 2024-01-08）
[^8]: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (arXiv:2405.04434v5) — https://arxiv.org/html/2405.04434v5（arxiv, 2024-05-07）
[^9]: DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models (arXiv:2401.06066v1) — https://arxiv.org/html/2401.06066v1（arxiv, 2024-01-11）
[^10]: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (arXiv:2101.03961v3) — https://arxiv.org/abs/2101.03961v3（arxiv, 2021-01-11）
[^11]: microsoft/Tutel — Tutel MoE: An Optimized Mixture-of-Experts Implementation (README) — https://raw.githubusercontent.com/microsoft/Tutel/main/README.md（github）
[^12]: SGLang — Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs (LMSYS Blog) — https://lmsys.org/blog/2025-05-05-large-scale-ep/（web, 2025-05-05）
[^13]: Qwen3 Technical Report — https://arxiv.org/abs/2505.09388v1（arxiv, 2025-05-14）
[^14]: GPT-4 Technical Report (arXiv:2303.08774, v6) — https://arxiv.org/abs/2303.08774（arxiv, 2023-03-15）
[^15]: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context (arXiv:2403.05530, v5) — https://arxiv.org/html/2403.05530v5（arxiv, 2024-03-08）
[^16]: Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities (arXiv:2507.06261, v1) — https://arxiv.org/html/2507.06261v1（arxiv, 2025-07-01）
[^17]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948v2) — https://arxiv.org/html/2501.12948v2（arxiv, 2025-01-22）
[^18]: DeepSeek-V3 Technical Report (arXiv:2412.19437v2) — https://arxiv.org/abs/2412.19437（arxiv, 2024-12-27）
[^19]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning — arXiv 摘要页元数据 (2501.12948v2) — https://arxiv.org/abs/2501.12948v2（arxiv, 2025-01-22）
[^20]: deepseek-ai/DeepSeek-R1 (官方 GitHub 仓库，仅取得元数据) — https://github.com/deepseek-ai/DeepSeek-R1（github）
[^21]: NVIDIA/Megatron-LM — Megatron-LM and Megatron Core README — https://github.com/NVIDIA/Megatron-LM（github, 2026-10-24）
[^23]: DeepSpeed Tutorial — Mixture of Experts (DeepSpeed-MoE 官方文档) — https://www.deepspeed.ai/tutorials/mixture-of-experts/（web）
[^24]: vllm-project/vllm — 仓库元数据（README 正文未取得） — https://github.com/vllm-project/vllm（github, 2026-10-25）
[^25]: facebookresearch/fairseq — 仓库元数据（README 正文未取得） — https://github.com/facebookresearch/fairseq（github, 2025-09-30）
[^26]: microsoft/Tutel repository metadata (search_repos) — https://github.com/microsoft/Tutel（github）