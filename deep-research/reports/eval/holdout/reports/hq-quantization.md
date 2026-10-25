# LLM 量化（INT8/INT4）主流方法与精度-吞吐权衡

## 摘要

LLM 量化在 2022–2024 年形成了三条清晰的技术路线：**W8A8 激活-权重量化**（以 SmoothQuant、ZeroQuant 为代表）、**低比特 weight-only 量化**（GPTQ、AWQ、QuIP#）与**旋转/等价变换类端到端 4-bit 量化**（QuaRot、SpinQuant），外加独立的 **KV cache 量化**分支（KIVI）。量化基本形式为浮点向量近似为标量 scale 乘整数向量，x̂ = s_x · x_int ≈ x[^22]；通用非对称 uniform affine 量化为 x_int = clamp(⌊x/s⌉ + z, 0, 2^b−1)，反量化为 x ≈ s(x_int − z)[^22]。

精度-吞吐权衡的核心矛盾可归结为三点：(1) **权重易量化、激活难量化**——超过 6.7B 参数后激活出现系统性大幅值 outlier，per-tensor INT8 激活量化使 OPT 系列平均精度从 FP16 的 64.9%–71.6% 崩到 32.3%–39.9%[^26][^28]；(2) **加速来源与算法解耦**——weight-only 4-bit 的加速主要来自访存带宽节省而非乘法本身，GPTQ 作者自陈因硬件缺乏 FP16×INT4 混合精度支持，方法本身不为实际乘法提供加速[^36]；(3) **INT4 正被 FP8/FP4 挤压**——TensorRT-LLM 官方 recipe 以 FP4/FP8 为主、未列 INT4/W4A16[^40]，vLLM 推荐 LLM Compressor 的 FP8/INT8/INT4[^14]。现有证据**不足以**给出按 Llama-3.x/Qwen/Mistral/DeepSeek 模型族组织的 INT8/INT4 配置与显存/吞吐对照表（见"矛盾与不确定性"）。

---

## 一、量化基本形式与术语基线

**量化与反量化的标准定义**（以 Nagel et al. 白皮书为据）：量化把浮点向量近似为标量 scale factor 乘整数向量，x̂ = s_x · x_int ≈ x，其中 s_x 为浮点 scale，x_int 为 INT8 等整数向量[^22]。通用（非对称、uniform affine）量化公式为 x_int = clamp(⌊x/s⌉ + z, 0, 2^b − 1)，⌊·⌉ 为 round-to-nearest，clamp 截断到 [0, 2^b−1]；反量化为 x ≈ x̂ = s(x_int − z)[^22]。合并两者得统一量化函数 q(x; s, z, b) = s[clamp(⌊x/s⌉ + z, 0, 2^b−1) − z][^22]。

**对称量化**是上述通用情形的简化：将 zero-point 限制为 0，从而在累加运算中省去 zero-point 偏移开销，但限制了整数域与浮点域之间的映射灵活性[^22]。有符号网格 x_int = clamp(⌊x/s⌉, −2^(b−1), 2^(b−1)−1)，无符号网格 x_int = clamp(⌊x/s⌉, 0, 2^b−1)[^22]。**power-of-two 量化**是对称量化的特例，scale 被限制为 s = 2^(−k)，可用位移实现缩放，但限制了 scale 表达能力[^22]。

**误差结构**：量化网格上下限为 q_min = −sz、q_max = s(2^b − 1 − z)，超出范围的值被裁剪产生 clipping error；增大 scale 可减小 clipping error 但会增大 rounding error，后者落在 [−s/2, s/2][^22]。权重与激活使用各自独立的 scale（s_w 与 s_x）可降低量化误差，且因每个 scale 作用于整个张量可提到求和号外，从而在定点格式下完成 MAC 运算[^22]。bias 通常以 32-bit 存储，其 scale 取决于权重与激活的 scale[^22]。

**粒度术语**在工具链中的对应：PyTorch 官方 API 中 quantize（PTQ 静态）、quantize_dynamic（weights-only）、quantize_qat（QAT）并列存在，说明 PTQ 与 QAT 在官方工具链中被区分为不同流程[^1]；quantize_dynamic 被官方描述为将 float 模型转换为 dynamic（即 weights-only）量化模型[^1]。prepare / prepare_qat 用于校准或量化感知训练准备模型副本，convert 负责替换为量化模块，体现"准备—校准/训练—转换"的流程划分[^1]。determine_qparams 被描述为"给定 min 和 max 值张量计算量化参数"，calculate_qmin_qmax 依据量化范围、observer 数据类型与是否缩减范围计算实际 qmin/qmax[^1]。DTypeWithConstraints 可为给定 dtype 指定量化值范围、scale 值范围与固定量化参数，表明 scale 取值范围本身是可约束对象[^1]。

**工具链迁移**：PyTorch 官方量化文档页已改为迁移公告，声明所有量化相关开发正集中到 torchao；Eager mode（quantize、quantize_dynamic）应迁移到 torchao 的 eager mode quantize_ API，FX graph mode（prepare_fx、convert_fx）应迁移到 torchao pt2e API（prepare_pt2e、convert_pt2e），并计划在 PyTorch 2.10 删除 torch.ao.quantization[^2]。torchao 被定义为"用于自定义数据类型与优化的库"，可对权重、梯度、优化器状态与激活进行量化与稀疏化，覆盖推理与训练场景[^3]；其快速上手示例用 Int4WeightOnlyConfig 配 group_size=32，int4_choose_qparams_algorithm 可取 "hqq"，int4_packing_format 可取 "tile_packed_to_4d"（CUDA）或 "plain_int32"（XPU），说明 4-bit 权重需打包存储且打包格式依硬件后端而异[^3]。不过 TorchAO 仓库 README 声明正在减少新特性开发、将量化开发整合到其他仓库，底层原语（dtypes、casts、gemms）将直接加入 torch[^17]。

---

## 二、W8A8 路线：激活量化是精度瓶颈

**问题诊断**。SmoothQuant 的核心观察是 weights 容易量化而 activations 难以量化；当模型超过 6.7B 参数时激活中出现系统性大幅值 outlier，幅值约为大多数激活值的 100 倍[^26]。机制上，per-tensor 量化下 outlier 主导最大幅值测量，非 outlier 通道的有效量化级数降为 2–3 级，通道 i 的有效量化级数为 2^8 · m_i/m[^26]。定量证据（OPT 6.7B–175B，WinoGrande/HellaSwag/PIQA/LAMBADA 平均）：FP16 为 64.9%/65.6%/67.9%/69.5%/71.6%；INT8 per-tensor 崩到 39.9%/33.0%/32.8%/33.1%/32.3%；INT8 per-token 为 42.5%/33.0%/33.1%/32.9%/31.7%；INT8 per-channel 为 64.8%/65.6%/68.0%/69.4%/71.4%[^26][^28]。即 **per-channel 激活量化能恢复精度，但与 INT8 GEMM kernel 不兼容**（Tensor Core MMA 序列不容忍低吞吐的转换/CUDA Core FMA 插入）[^28]。

**解法一：等价变换迁移量化难度**。SmoothQuant 用数学等价变换把量化难度从 activations 离线迁移到 weights（per-channel smoothing factor），实现 W8A8，最高 1.56× 加速、2× 内存节省，可在单节点服务 530B 模型[^26][^28]。平滑因子可离线融合进前一层参数，不引入额外 kernel 调用开销；当输入来自 residual add 时需在残差分支上加额外 scaling[^28]。覆盖 OPT、BLOOM、GLM、MT-NLG、Llama-1/2、Falcon、Mistral、Mixtral 等模型的全部矩阵乘 INT8 量化[^28]。

**解法二：细粒度 + 逐层蒸馏**。ZeroQuant 是端到端 PTQ + 推理流水线，含三个组件：面向权重与激活的细粒度且硬件友好的量化方案、无需原始训练数据的逐层知识蒸馏 LKD、高度优化的量化系统后端以消除量化/反量化开销[^29]。细粒度方案为权重 group-wise、激活 token-wise 动态量化[^29]。W8A8 下 BERT 与 GPT-3 风格模型可无损降到 INT8，相对 FP16 在 BERT-base / GPT-3-350M 上分别最高 5.19× / 4.16× 加速[^29]。但精度细节揭示权衡：GPT-3 350M 上 W16A16 平均 38.9 → W8A8 37.8，Wikitext-2 perplexity 21.5 → 26.2；W8A16 为 38.5 / 22.1；W16A8 为 37.8 / 24.6[^29]。论文明确指出 **INT8 激活量化（W16A8）是主要精度损失来源**；权重进一步压到 INT8（W8A8）不改变 zero-shot 精度但让困惑度变差，说明生成任务比 zero-shot 评测更敏感[^29]。ZeroQuant + LKD 可做 INT4/INT8 混合精度（全连接权重 INT4、注意力权重 INT8、激活 INT8），相对 FP16 实现 3× 内存降低[^29]；可扩展到 GPT-J-6B（相对 FP16 加速 3.67×）与 GPT-NeoX-20B（推理 GPU 需求从 2 降到 1，延迟 65ms→25ms，合计 5.2× 系统效率提升）[^29]。ZeroQuant 还报告 W4/8A16（MHSA 权重 INT8 + FFC 权重 INT4）在 GPT-3 350M 上严重失效：平均 28.9，Wikitext-2 PPL 1.76e5，Lambada 0.00[^29]。

**离群值的独立现象**。LLM.int8() 针对 transformer 的 FFN 与 attention projection 层实现 INT8 矩阵乘法，将推理显存减半同时保持全精度性能[^7]。其依据是 transformer 中"系统性涌现特征"：在约 6.7B 参数处发生相变，所有层与 75% 序列维度受极端幅值特征影响；6.7B 规模下每个序列出现 150,000 个 outlier，但整个 transformer 中只集中在 6 个 feature 维度上[^24]。这些维度约占全部输入特征的 0.1%，置零会使 top-1 attention softmax 概率质量下降超过 20%、验证集困惑度退化 600–1000%，而移除同等数量随机特征最多只使概率下降 0.3%、困惑度退化约 0.1%[^24]。方法上先做 vector-wise 量化（每个内积使用独立归一化常数），再对离群维度做混合精度分解到 16-bit 矩阵乘，仍有超过 99.9% 的数值以 8-bit 相乘[^7]；对不超过 13B 的 transformer，outlier 维度数 |O| ≤ 7，混合精度分解只额外消耗约 0.1% 内存[^24]。实证可在最高 175B 参数 LLM 上推理而无性能退化，对 BLOOM-176B 内存占用降低 1.96 倍[^7][^24]。

**注意两类离群值不是同一现象**。Massive activations 研究指出：LLM 中存在极少数幅值远大于其他的激活（例如大 100,000 倍），LLaMA2-70B 中绝对值可超过 15,000；在数千万个激活中通常少于 10 个，LLaMA2-7B 单个 hidden state 约 40,000 个激活中最多只有 4 个[^25]。通道位置固定（LLaMA2-7B 位于 feature 维度 1415 与 2533；LLaMA2-13B 位于 2100 与 4743；Mixtral-8x7B 位于 2070 与 3398）[^25]。其数值基本与输入无关、近似常数，起到不可或缺的 bias 项作用，出现在中间层绝大多数层中，在初始层涌现、最后几层减弱[^25]。关键结论：用 Dettmers et al.(2022) 的定义在 LLaMA2-7B/13B 中分别识别出 10 个和 25 个 outlier features，但**其中没有一个对应 massive activations 的通道**——说明 massive activations 与 LLM.int8() 的 outlier features 是两类不同现象[^25]。

---

## 三、低比特 weight-only 路线：GPTQ / AWQ / QuIP#

**GPTQ** 是 one-shot 权重量化方法，基于近似二阶（Hessian）信息做逐层误差补偿[^4]。方法为逐层求解 argmin ||WX − ŴX||₂² 的重建问题，量化粒度与 group 划分属于该框架下的可调设置[^36]。可在约 4 GPU 小时内量化 175B 参数 GPT 模型，权重位宽降至 3 或 4 bit，相对未压缩基线精度退化可忽略[^4]；相比此前 one-shot 方法压缩收益提升一倍以上，首次使 175B 模型可在单 GPU 上做生成式推理[^4]。极端区间（2-bit 甚至三值量化）仍能保持合理精度[^4][^36]。论文明确指出压缩在语言建模精度、bit-width 与原始模型规模之间存在 non-trivial tradeoffs[^36]。**关键限定**：作者自陈因主流硬件缺乏混合精度（FP16×INT4）支持，方法本身不为实际乘法提供加速，加速来自访存带宽节省；实测端到端相对 FP16 加速约 3.25×（A100）与 4.5×（A6000）[^36]。

**AWQ** 是面向硬件的低比特 weight-only 量化（INT3/INT4）[^5]。核心观察：并非所有权重同等重要，仅保护约 1% 的显著权重即可大幅降低量化误差；显著权重通道的识别应依据激活分布而非权重本身[^5][^27]。为避免硬件低效的混合精度量化，论文推导出"放大显著通道"可降低量化误差，并用等价变换缩放显著权重通道，缩放系数由离线收集的激活统计量确定[^5]。AWQ 不依赖反向传播或重建，因此可泛化到不同领域与模态，不易过拟合校准集；作者指出 GPTQ 的重建过程会过拟合校准集、在分布外领域扭曲已学特征[^5][^27]。

AWQ 的定量证据（INT3-g128，WikiText perplexity）：OPT-6.7B FP16 12.29、RTN 43.16、保留 1% 显著权重为 FP16 后 13.02；OPT-1.3B 16.41 / 206.5 / 18.51；OPT-13B 11.50 / 45.37 / 12.04[^37]。LLaMA-7B 在 w3-g128 下 FP16 9.49、RTN 12.10、搜索缩放后 10.82[^37]。AWQ 全文采用 group-wise 量化，默认 group size = 128[^37]；聚焦 INT4/INT3 的理由是这两档"基本能保留 LLM 性能"[^37]。按权重范数挑选显著通道几乎无改善（与随机选择相当），仅 0.1%–1% 的显著权重（按激活幅度挑选）对量化损失起决定作用[^37]。加速方面：配套 TinyChat 在桌面/移动 GPU 上相对 HuggingFace FP16 获得 3.2–3.3× 平均加速，可在单张 Jetson Orin（64GB）上部署 Llama-2-70B[^27]；AWQ 报告相对 GPTQ 1.45×、相对带 reordering 的 GPTQ 2× 加速，比 cuBLAS FP16 快 1.85×[^37]。官方仓库 mit-han-lab/llm-awq 定位 INT3/INT4，提供 AWQ search 与预计算 model zoo（Llama-1/2/3、OPT、CodeLlama、StarCoder、Vicuna、VILA、LLaVA），提供 PyTorch 内存高效 4-bit Linear 与支持 prefill/decoding 两阶段的 CUDA kernel，3654 stars、最近 push 2025-07-17[^6]。

**QuIP#** 是 weight-only PTQ，定位 ≤ 4 bits per weight 的极端压缩区间[^33]。三项改进：用 randomized Hadamard transform 替代 QuIP 的 incoherence processing（更快且理论性质更好）；用向量量化利用非相干权重的球状 sub-Gaussian 分布，提出基于 E8 格的硬件友好 codebook（该格实现最优的 8 维单位球堆积）；用 fine-tuning 提升对原始模型的保真度[^33]。incoherence processing 被定义为一种有原则的离群值抑制形式，使权重矩阵近似服从高斯分布，让被量化对象与重要量化方向（Hessian）在任何坐标上都不至于过大，从而使量化误差可被证明有界[^34]。推理流程为：先做 Hadamard 正交变换 Had(S_V ⊙ x)，再 decompress_multiply 与 codebook 相乘，最后再做一次 Hadamard 变换 Had(S_U ⊙ y)[^34]。使用 BlockLDLQ 块自适应舍入配合 E8 格 codebook；LDLQ 在线性反馈自适应舍入类中最优，误差率可证明优于 nearest 或 stochastic rounding[^34]。论文指出分组保留 FP16 缩放因子会带来额外开销：每 64 个权重用一个 16-bit scale 相当于额外 0.25 bits/weight，在极端压缩区间这一开销显著[^34]。论文声称 QuIP# 是首个 3-bit 模型比 4-bit 模型扩展性更好的 PTQ 方法，直接反驳 Dettmers & Zettlemoyer (2023) 关于 4-bit 模型"最优"的论断[^33]；CUDA 概念验证在 RTX 4090 上达到超过 50% 峰值内存带宽；Llama 2 70B 在 16-bit 下需 140GB 显存、量化到 2 bits 后只需不到 20GB[^33]。

---

## 四、旋转类端到端 4-bit：从 QuaRot 到 SpinQuant

**QuaRot** 基于旋转实现端到端量化，将权重、激活与 KV cache 全部量化为 4 bit[^8]。通过旋转在不改变输出的前提下从 hidden state 中移除离群值（利用计算不变性），旋转应用于 hidden state（残差）、FFN 激活、attention 机制部分以及 KV cache[^8]。量化后所有矩阵乘法均以 4 bit 执行，无需保留任何通道于更高精度[^8]。4-bit 量化的 LLaMa2-70B 在 WikiText-2 上 perplexity 损失最多 0.47，保留 99% 零样本性能；还能在无任何校准数据、仅用 round-to-nearest 的情况下给出无损的 6-bit 与 8-bit LLaMa2 模型[^8]。

**SpinQuant** 的核心主张是：不同随机旋转矩阵导致的量化后性能差异很大，因此应"学习"旋转矩阵而非使用随机旋转[^35]。在 LLaMA-2 7B 的 W4A4 设置下做 100 次随机旋转试验，最好的随机旋转比最差的在下游零样本推理平均准确率上高出 13 分；随机 Hadamard 矩阵整体优于随机浮点旋转矩阵，但随机 Hadamard 旋转之间仍存在最高约 6 分的性能方差[^35]。方法用 Cayley SGD 在 Stiefel 流形上优化旋转矩阵以最小化量化网络最终损失，权重参数保持固定，该优化不改变全精度网络输出[^35]。定义两个可融合旋转矩阵（R1、R2）产生旋转不变的全精度网络，另加两个 online Hadamard 旋转（R3、R4）处理 MLP 块内部与 KV cache 的激活 outlier[^35]。结果：SpinQuant_had 在 LLaMA-2 7B 的极端 W4A4KV4 设置下达到平均准确率 64.0，与全精度仅差 2.9 分，而 LLM-QAT 在相同精度条件下差距为 22.0 分；SpinQuant_no had 在 Mistral-7B 的 W4A8KV8 零样本常识推理上把量化性能差距从 12.1 缩小到 1.6，其 W4A8 在 LLaMA-2 上达到与 QuIP#、OmniQuant 等 SOTA 权重-only 方法相当的性能；相对 QuaRot 在 LLaMA-3 8B 这类难量化模型上把与全精度的差距最多缩小 45.1%（相对值）[^35]。

**OmniQuant** 提供另一条可学习等价变换路线：Learnable Weight Clipping (LWC) 优化 clipping 阈值调节权重极值，Learnable Equivalent Transformation (LET) 学习数学等价变换把量化挑战从激活转移到权重；整体在可微分框架下做 block-wise 量化误差最小化，只引入少量可学习量化参数、冻结原始全精度权重，量化后所有可学习参数可被消除（不增加推理开销）[^30]。与 SmoothQuant / Outlier Suppression+ 依赖手工设定的 migration strength 或网格搜索 scaling 不同，OmniQuant 用梯度优化学习这些参数，并把等价变换扩展到注意力机制从而支持 K/V cache 量化[^30]。覆盖 W4A4、W6A6、W4A16、W3A16、W2A16[^30]。效率上 LLaMA-2 全家族（7B–70B）可在单张 A100-40G 上用 128 个样本在 1–16 小时内完成量化[^30]。W2A16 下 LLaMA-13B 量化后困惑度 13.21，而 GPTQ 升至 3832[^30]。

**Atom** 则强调吃满低比特算子：定位面向 LLM serving 的低比特 weight-activation 量化，重点是利用 INT4 Tensor Core 而非仅做 weight-only[^31]。三项保精度设计：混合精度量化保留少量显著激活与权重在高精度；权重和激活都采用细粒度 group 量化；激活采用动态量化[^31]。配套设计包括重排激活与权重以维持混合精度下的规则内存访问、把量化与重排融合进已有算子、把离群值进一步量化到 8-bit、把 KV-cache 量化到低比特[^31]。精度：4-bit（权重与激活均 4-bit）时平均 zero-shot 精度下降 1.4%，Llama-65B 的 WikiText2 困惑度仅增加 0.3[^31]。吞吐：相同延迟目标下端到端吞吐相对 FP16 最高 7.73×、相对 W4A16 最高 5.5×、相对 W8A8（INT8）最高 2.53×[^31]。硬件动机数据：A100 的 INT4 为 1248 TOPS、INT8 为 624 TOPS，而 FP16 Tensor Core 仅 312 TFLOPS[^31]。论文同时指出 OmniQuant、QLLM 等 4-bit weight-activation 方法相比 FP16 基线仍有明显的困惑度上升[^31]。

---

## 五、KV cache 量化：独立于权重量化的第二战场

KIVI 指出随 batch size 与上下文长度增长，KV cache 成为速度与显存的新瓶颈[^9]。KIVI 是免调优的非对称 2-bit KV cache 量化：key cache 按 channel 维度分组量化，value cache 按 token 维度量化[^9][^42]。在 Llama、Falcon、Mistral 上保持几乎相同质量的同时将峰值显存（含模型权重）降低 2.6×，使 batch size 可扩大至 4×，在真实推理负载上带来 2.35×–3.47× 吞吐提升[^9][^42]。论文将 KV cache 内存增长归因于大 batch 与长上下文，并指出 KV cache 加载会导致计算核心空闲、限制推理速度[^42]。官方仓库 jy-yuan/KIVI 为 MIT 许可、436 stars，README 自述该工作"largely inspires the HuggingFace Transformers KV Cache quantization"，2025-01-18 更新加入支持 GQA 的 KIVI 实现并兼容 transformers 4.43[^43]。

**产业侧 KV cache 量化已明显转向 FP8/FP4**。vLLM 的 KV cache 量化以 FP8 为主：kv_cache_dtype 可选 auto、fp8_e4m3（CUDA 11.8+ 与 ROCm）、fp8_e5m2（CUDA 11.8+）[^11]。FP8 KV cache 支持 per-tensor（Q/K/V 各一个 scale）与 per-attention-head（q_scale=[num_heads]，k/v_scale=[num_kv_heads]）两种粒度；per-attention-head 目前仅在 Flash Attention 后端可用且需要 llm-compressor 的校准流程；默认无校准配置下所有 scale 设为 1.0，官方推荐用数据集校准以获得最高精度[^11]。使用 Flash Attention 3 后端配合 FP8 KV cache 时 query 也会被量化为 FP8，注意力运算在量化域内执行；提供 --kv-cache-dtype-skip-layers 参数可将 sliding-window 等敏感层保留在模型原生 dtype[^11]。TensorRT-LLM 的 KV cache recipe 只有 FP8 KV Cache 与 NVFP4 KV Cache 两项，**文档未列出 INT8 KV cache recipe**[^40]；NVFP4 KV cache 需先用 ModelOpt 离线量化，且当前启用时只支持 FP8 权重/激活量化[^40]。SGLang 官方 Quantization 页则未列出任何 KV cache 量化方法[^41]。

---

## 六、吞吐收益的真实来源：算法 vs kernel

这是本课题最容易被误读的部分。**weight-only 4-bit 的加速本质是带宽收益，不是算力收益**。Marlin 给出定量论证：现代 GPU 的 FLOP-to-byte 比值约为 100–200，因此只要每个 4-bit 量化权重对应的 tensor core 乘加次数少于 25–50 次，理论上就能维持相对 FP16 权重接近理想的 4× 加速[^39]。由此推论 weight-only 量化的全部性能收益原则上可延伸到比现有 kernel 大 4–8 倍的 batchsize，但实际实现极具挑战，因为必须同时充分利用 global memory、L2 cache、shared memory、tensor core 与 vector core 全部资源[^39]。dequant 开销与 tensor core 的流水线重叠是核心工程问题，README 明确要求"carefully order dequantization and tensor core instructions to ensure that both GPU pipelines are well saturated and do not bottleneck each other"[^39]。实测（A10、groupsize 128、大矩阵）：现有 kernel 在 batchsize 1 时接近最优的 3.87× 加速（3.87× 而非 4× 是因为 group scales 带来 0.125 bit 存储开销），但随输入数增加性能快速退化；Marlin 在 batchsize 约 16–32 以内基本保持理想的 3.87× 加速[^39]。硬件约束：需要 CUDA ≥ 11.8、compute capability ≥ 8.0（Ampere 或 Ada），README 注明 Marlin 尚未针对 Hopper 优化；ECC 开启时（如 A10）可达最大显存带宽比官方规格低 10–15%[^39]。

**因此三组加速数字不可直接并列**：GPTQ 的 3.25×/4.5× 是端到端相对 FP16，且作者明确加速来自访存带宽节省[^36]；AWQ 的 3.2–3.3× 是 TinyChat 相对 HuggingFace FP16 实现（含框架开销差异）[^27]，其相对 GPTQ 的 1.45×/2× 则混合了算法与 kernel 实现差异[^37]；Marlin 的 3.87× 是纯 kernel 级、且限定 batchsize 16–32[^39]。三者对加速来源（算法 vs kernel 实现）与量级的归因不一致。

**W8A8 的加速口径同样受限**。SmoothQuant 报告最高 1.56× 加速与 2× 内存节省[^26][^28]；ZeroQuant 报告 BERT-base / GPT-3-350M 上最高 5.19× / 4.16× 加速、GPT-J-6B 相对 FP16 加速 3.67×[^29]。Atom 报告相对 W8A8（INT8）最高 2.53×、相对 FP16 最高 7.73×[^31]，但这是其自研混合精度 kernel + serving 框架的端到端结果。

---

## 七、生态与部署现状

**服务框架的支持矩阵**。vLLM 官方支持的量化格式包括 AutoAWQ、BitsAndBytes、GPTQModel、Intel Neural Compressor、LLM Compressor、FP8 W8A8、INT4 W4A16、INT8 W4A8、INT8 W8A8、NVIDIA Model Optimizer、AMD Quark、TorchAO 等[^10][^14]。vLLM 将 INT4 权重仅量化标注为 W4A16（4-bit 权重、16-bit 激活），INT8 标注为 W8A8[^10]。硬件兼容矩阵：AWQ 支持 Turing/Ampere/Ada/Hopper 与 Intel GPU、x86 CPU，但不支持 Volta、AMD GPU、Arm CPU；GPTQ 覆盖 Volta 至 Hopper 及 Intel GPU、x86 CPU，不支持 AMD GPU 与 Arm CPU；Marlin 内核（覆盖 GPTQ/AWQ/FP8/FP4）支持 Turing（不含 MXFP4）至 Hopper，不支持 Volta、AMD GPU、Intel GPU 与 CPU；llm-compressor INT8 (W8A8) 支持 Turing/Ampere/Ada/Hopper 及 x86 CPU、Arm CPU；llm-compressor INT8 (W4A8) 仅在 Arm CPU 上标记为支持，所有列出的 GPU 平台均不支持；bitsandbytes 在 Volta 至 Hopper 上支持；GGUF 在 Volta 至 Hopper 及 AMD GPU 上支持[^14]。vLLM 提供 --linear-backend 与 linear_backend_per_quant 参数可为不同量化方案分别指定线性层后端（如 NVFP4 W4A16 用 Humming、其余用 CUTLASS），并支持通过 @register_quantization_config 装饰器注册 out-of-tree 自定义量化方法[^14]。文档注明兼容性表会随 vLLM 演进变化，最新支持情况应以 vllm/model_executor/layers/quantization 代码为准[^14]。

**工具链更替**。AutoAWQ 已官方标记为 deprecated、不再维护，最后测试配置为 Torch 2.6.0 与 Transformers 4.51.3；累计 2M+ 下载量、HuggingFace 上 7000+ 个模型、2.1k stars，说明 AWQ 格式在开源模型生态中的实际部署规模；其能力已被 vLLM 项目接管，迁移至 vllm-project/llm-compressor；MLX-LM 已支持在 Mac 设备上使用 AWQ[^12]。llm-compressor 采用 Apache-2.0 许可，约 3.9k stars、712 forks，最近提交 2026-10-24（活跃维护），定位为将模型优化为可在 vLLM 中部署的量化格式[^13]。GPTQModel 自述为可扩展的 LLM 量化、验证与部署平台，支持 GPTQ、AWQ、ParoQuant、GGUF、FP8、EXL3、QQQ 等格式，支持 NVIDIA CUDA、AMD ROCm、Huawei Ascend、Intel XPU 与 CPU，可与 Transformers、vLLM 和 SGLang 集成[^18]。bitsandbytes 的 topics 包含 qlora，显示其与 QLoRA 微调生态的关联[^19]。

**量化也用于训练侧**。QLoRA 通过冻结的 4-bit 量化预训练模型向 LoRA 反向传播梯度，可在单张 48GB GPU 上微调 65B 参数模型，同时保持完整 16-bit 微调的任务性能[^32]。三项显存创新：4-bit NormalFloat (NF4)（对正态分布权重在信息论上最优的新数据类型）、double quantization（对量化常数本身再量化）、paged optimizers[^32]。作者微调了超过 1,000 个模型，最佳模型族 Guanaco 在 Vicuna 基准上达到 ChatGPT 性能水平的 99.3%，仅需单 GPU 24 小时微调[^32]。

**端侧与多后端**。llama.cpp 自述支持 1.5-bit 至 8-bit 整数量化，提供 NVIDIA GPU 自定义 CUDA kernel，通过 HIP 支持 AMD GPU、通过 MUSA 支持 Moore Threads GPU，支持 CPU+GPU 混合推理以部分加速超出总 VRAM 容量的模型，后端包括 BLAS、BLIS、CANN(Ascend NPU)、CUDA、HIP、Hexagon、IBM zDNN、MUSA、Metal、OpenCL、OpenVINO(进行中)、RPC、SYCL 等，在 x86 上支持 AVX/AVX2/AVX512/AMX，在 Apple Silicon 上通过 ARM NEON、Accelerate 与 Metal 优化[^16]。torchao 官方首页给出量化收益口径示例：将 Llama-3-8B 量化为 int4 可获得 1.89× 推理加速与 58% 内存下降；并称 QAT 可在 Gemma3-4B 上恢复 67% 的量化精度损失[^3]。

---

## 八、INT4 与 FP8/FP4 的竞争关系

这是 2025–2026 年权衡格局的关键变化，现有证据指向产业侧重心已从 INT4 转向 FP8/FP4：

- TensorRT-LLM 官方 Quantization 文档列出的 recipe 为 FP4、FP8 Per Tensor、FP8 Block Scaling、FP8 Rowwise、FP8 KV Cache、NVFP4 KV Cache、W4A16 GPTQ、W4A8 GPTQ、W4A16 AWQ、W4A8 AWQ[^40]。INT4 仅以 GPTQ/AWQ 的 W4A16/W4A8 形式出现，无独立 INT4 recipe。
- TensorRT-LLM 官方性能页列出的量化模型族为 FP4（Llama-3.3-70B-Instruct-FP4、Llama-3.1-405B-Instruct-FP4）与 FP8（Llama-3.1-8B、Llama-3.3-70B、Llama-3.1-405B、Llama-4-Maverick-17B-128E），**
**未列出 INT4/W4A16 模型，也未给出 BF16 基线列**[^38]。因此该页的吞吐数字（如 Llama 3.3 70B FP4 在 B200 TP=1 上 128/128 为 10,613.84 tokens/s、128/4096 为 6,276.85、20000/2000 为 636.36 tokens/s；Llama 3.1 8B FP8 在 H100 TP=1 上 128/128 为 26,401.48 tokens/s、20000/2000 为 1,340.69 tokens/s）只能作为 FP4/FP8 的参考点，**不能用于推导 INT4 相对 BF16 的加速比**[^38]。官方亦声明这些数据仅作为验证观测性能的参考点，不应视为可交付的峰值性能[^38]。
- vLLM 推荐 LLM Compressor 作为面向 vLLM 部署的量化优化库，支持 FP8、INT8、INT4 等格式[^10][^14]；其 KV cache 量化只提供 FP8 方案[^11]。
- SGLang 的平台兼容性表同时列出 fp8、mxfp4、mxfp8、mxfp_w4a8、blockwise_int8、w8a8_int8、w8a8_fp8、awq、gptq、nvfp4_online、petit_nvfp4、bitsandbytes 等，int8 相关条目（blockwise_int8、w8a8_int8）均为权重/激活量化方法[^41]。文档建议优先使用离线量化而非在线量化，并提示使用预量化模型时不要同时加 --quantization 开启在线量化；量化模型必须经基准测试验证，以防量化精度异常回退[^41]。

由此可得的权衡判断是：**INT8 的定位是"近乎无损的通用档"**——W8A8 在 per-channel 或等价变换后可基本恢复 FP16 精度（SmoothQuant 的 per-channel 数据 64.8%–71.4% 对 FP16 的 64.9%–71.6%[^26][^28]；ZeroQuant 的 W8A8 在 zero-shot 上 37.8 对 W16A16 的 38.9[^29]），代价是加速幅度有限（1.56×–5.19× 视模型与实现而定[^26][^29]）。**INT4 weight-only 的定位是"显存/带宽优先档"**——精度在 group size 128 下可接受（AWQ INT3-g128 在 OPT-6.7B 上 13.02 对 FP16 12.29[^37]），加速上限受 batchsize 与 kernel 质量约束（Marlin 在 batchsize 16–32 内维持 3.87×[^39]）。**INT4 weight-activation（W4A4）的定位是"算力优先档"**——需要旋转/等价变换类方法压住激活 outlier（QuaRot 的 LLaMa2-70B perplexity 损失 ≤0.47[^8]；SpinQuant_had 在 W4A4KV4 下与全精度差 2.9 分[^35]；Atom 平均 zero-shot 下降 1.4%[^31]），换来相对 FP16 最高 7.73× 的端到端吞吐[^31]。而 FP8/FP4 正在侵蚀 INT4 的部署位：TensorRT-LLM 的 recipe 与性能页均以 FP4/FP8 为主[^38][^40]，vLLM 的 KV cache 量化只有 FP8[^11]。

---

## 矛盾与不确定性

**1. 缺少按模型族组织的 INT8/INT4 配置与显存/吞吐对照表。** 课题需要的 Llama-3.x、Qwen2.5/3、Mistral、DeepSeek 在 INT8/INT4 下的典型 group size、bit-width、KV cache 位宽、显存占用（GB）、压缩倍数、单卡可容纳参数量、tokens/s 与延迟加速比区间，现有证据**不足以确认**。证据只提供零散单点：AWQ 默认 group size = 128[^37]、torchao 示例 group_size=32[^3]、QuIP# 的 Llama2-70B 140GB→<20GB[^33]、KIVI 的 2.6× 显存与 2.35–3.47× 吞吐[^9][^42]。没有任何一条证据给出 Qwen 或 DeepSeek 系列的量化配置，也没有给出 Llama-3.x 的 INT8 显存数字。因此本报告无法给出课题核心的"精度损失 vs 吞吐收益"定量曲线，只能给出方法层面的权衡结构。

**2. TensorRT-LLM 性能页与 INT8/INT4 课题口径错配。** 该页只列 FP4 与 FP8 模型，未列 INT4/W4A16，也未给 BF16 基线列[^38]。用它论证 INT8/INT4 吞吐收益属于口径错配，本报告已避免此用法。

**3. weight-only 4-bit 的"3–4× 加速"归因不一致。** GPTQ 自陈加速来自访存带宽节省而非乘法本身[^36]；AWQ 的 3.2–3.3× 是 TinyChat 相对 HuggingFace FP16 实现[^27]，其相对 GPTQ 的 1.45×/2× 混合了算法与 kernel 差异[^37]；Marlin 的 3.87× 是纯 kernel 级且限定 batchsize 16–32[^39]。三者不可不加区分地并列，本报告已在第六节分列口径。

**4. 缺少 2025–2026 年的系统性精度-吞吐评测。** 现有最新方法证据止于 SpinQuant（2024-05）与 QuIP#（2024-02）[^33][^35]，没有 2025–2026 年 arXiv 上覆盖多模型族、多量化配置的统一评测。同时产业侧已明显转向 FP8/FP4（TensorRT-LLM recipe 以 FP4/FP8 为主[^40]、vLLM 推荐 LLM Compressor 的 FP8[^14]），INT4 的长期部署地位存在不确定性。

**5. 工具链处于迁移期，API 稳定性存疑。** PyTorch 计划在 2.10 删除 torch.ao.quantization[^2]，而 TorchAO 自身 README 又声明正在减少新特性开发、将量化开发整合到其他仓库[^17]。迁移终点（torch 内建原语 + 各仓库内联工作流）尚未定型，依赖旧 API 的量化流程面临重写成本。

**6. 两类离群值现象的边界尚未在工程上统一。** LLM.int8() 的 outlier features 与 massive activations 在 LLaMA2-7B/13B 上无通道重叠[^25]，但现有量化方法（SmoothQuant 的 per-channel scaling[^26]、QuaRot/SpinQuant 的旋转[^8][^35]、AWQ 的激活感知缩放[^5]）是否分别针对这两类现象、各自的适用边界在哪，证据未给出直接对照实验，**现有公开资料不足以确认**。

**7. 精度评测口径不统一。** 各方法分别报告 WikiText perplexity[^37]、zero-shot 平均准确率[^29][^31]、下游任务均值[^26][^28]、Vicuna 基准[^32]，且 ZeroQuant 明确指出生成任务（困惑度）比 zero-shot 评测更敏感[^29]。跨方法的精度数字不能直接横向比较，本报告引用时均保留了各自的评测口径。

## 参考来源

[^1]: Quantization API Reference — torch.ao.quantization (PyTorch 2.14 docs) — https://docs.pytorch.org/docs/2.14/quantization-support.html（web, 2026-05-11）
[^2]: Quantization — PyTorch 2.14 docs (migration notice to torchao) — https://docs.pytorch.org/docs/2.14/quantization.html（web, 2026-05-11）
[^3]: Welcome to the torchao Documentation (PyTorch-Native Training-to-Serving Model Optimization) — https://docs.pytorch.org/ao/stable/index.html（web, 2026-03-25）
[^4]: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers — https://arxiv.org/abs/2210.17323v2（arxiv, 2022-10-31）
[^5]: AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration — https://arxiv.org/abs/2306.00978v6（arxiv, 2023-06-01）
[^6]: mit-han-lab/llm-awq — AWQ 官方开源仓库 — https://github.com/mit-han-lab/llm-awq（github, 2023-06-01）
[^7]: LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale — https://arxiv.org/abs/2208.07339v2（arxiv, 2022-08-15）
[^8]: QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs — https://arxiv.org/abs/2404.00456v2（arxiv, 2024-03-30）
[^9]: KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache — https://arxiv.org/abs/2402.02750v2（arxiv, 2024-02-05）
[^10]: vLLM Documentation — Quantization (supported formats & hardware matrix) — https://docs.vllm.ai/en/latest/features/quantization/（web）
[^11]: vLLM Documentation — Quantized KV Cache (FP8 KV cache schemes) — https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html（web）
[^12]: casper-hansen/AutoAWQ — README (deprecation notice) — https://github.com/casper-hansen/AutoAWQ（github, 2023-08-25）
[^13]: vllm-project/llm-compressor — README — https://github.com/vllm-project/llm-compressor（github, 2024-06-20）
[^14]: vLLM 官方文档：Quantization（支持格式清单与硬件兼容矩阵） — https://docs.vllm.ai/en/latest/features/quantization/index.html（web）
[^16]: ggml-org/llama.cpp（仓库元数据与 README 正文） — https://github.com/ggml-org/llama.cpp（github, 2026-10-24）
[^17]: pytorch/ao（TorchAO 仓库元数据与 README 正文） — https://github.com/pytorch/ao（github, 2026-10-22）
[^18]: ModelCloud/GPTQModel（仓库元数据与 README 正文） — https://github.com/ModelCloud/GPTQModel（github, 2026-10-24）
[^19]: bitsandbytes-foundation/bitsandbytes（仓库元数据） — https://github.com/bitsandbytes-foundation/bitsandbytes（github, 2026-10-19）
[^22]: A White Paper on Neural Network Quantization (Nagel et al., 2021) — https://arxiv.org/html/2106.08295v1（arxiv, 2021-06-15）
[^24]: LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (Dettmers et al., NeurIPS 2022) — https://ar5iv.labs.arxiv.org/html/2208.07339（arxiv, 2022-08-15 (v1); v2 2022-11-10）
[^25]: Massive Activations in Large Language Models (Sun, Chen, Kolter, Liu; COLM 2024) — https://arxiv.org/html/2402.17762v2（arxiv, 2024-02-27 (v1); v2 2024-08-14）
[^26]: SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models (Xiao et al., ICML 2023) — https://ar5iv.labs.arxiv.org/html/2211.10438（arxiv, 2022-11-18 (arXiv v1); ICML 2023）
[^27]: AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration (Lin et al., MLSys 2024 Best Paper) — https://ar5iv.labs.arxiv.org/html/2306.00978（arxiv, 2023-06-01 (arXiv v1); MLSys 2024）
[^28]: SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models (arXiv:2211.10438, v7) — https://arxiv.org/abs/2211.10438（arxiv, 2022-11-18）
[^29]: ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers (arXiv:2206.01861, v1) — https://arxiv.org/abs/2206.01861（arxiv, 2022-06-04）
[^30]: OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models (arXiv:2308.13137, v3) — https://arxiv.org/abs/2308.13137（arxiv, 2023-08-25）
[^31]: Atom: Low-bit Quantization for Efficient and Accurate LLM Serving (arXiv:2310.19102, v3) — https://arxiv.org/abs/2310.19102（arxiv, 2023-10-29）
[^32]: QLoRA: Efficient Finetuning of Quantized LLMs — https://arxiv.org/abs/2305.14314v1（arxiv, 2023-05-23）
[^33]: QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks — https://arxiv.org/abs/2402.04396（arxiv, 2024-02-06）
[^34]: QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks (HTML 全文, Section 2-4) — https://arxiv.org/html/2402.04396v2（arxiv, 2024-06-04）
[^35]: SpinQuant: LLM Quantization with Learned Rotations (arXiv:2405.16406v4, ICLR 2025) — https://arxiv.org/html/2405.16406v4（arxiv, 2025-02-20）
[^36]: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (arXiv:2210.17323v2, ICLR 2023) — https://arxiv.org/html/2210.17323v2（arxiv, 2023-03-22）
[^37]: AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration (arXiv:2306.00978v1) — https://arxiv.org/html/2306.00978v1（arxiv, 2023-06-01）
[^38]: TensorRT-LLM Performance Overview (official docs) — https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html（web）
[^39]: IST-DASLab/marlin — Mixed Auto-Regressive Linear kernel (FP16xINT4 matmul kernel for LLM inference) — https://github.com/IST-DASLab/marlin（github, 2024-01-17）
[^40]: TensorRT-LLM Docs — Quantization (recipes, FP8 KV Cache, NVFP4 KV Cache, support matrices) — https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html（web）
[^41]: SGLang Docs — Quantization (platform compatibility table) — https://docs.sglang.ai/advanced_features/quantization.html（web）
[^42]: KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (ICML 2024) — https://arxiv.org/abs/2402.02750（arxiv, 2024-02-05）
[^43]: jy-yuan/KIVI — official implementation repository — https://github.com/jy-yuan/KIVI（github, 2024-02-02）