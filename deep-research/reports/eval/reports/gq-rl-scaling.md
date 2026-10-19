# 推理模型的 RL Scaling 与预训练 Scaling 的关系，及 Test-Time Compute 的重要性

## 摘要

本报告基于 21 条证据，考察 o1/R1 路线所代表的「强化学习训练推理模型」与经典预训练 scaling law 之间的关系，并解释 test-time compute 为何成为独立且重要的扩展维度。

核心结论有四条：

1. **两条 scaling 路线在文献中被明确并置，但从未被同一套 scaling law 拟合。** 预训练侧有 Kaplan 与 Chinchilla 的幂律与等比例缩放结论 [^19][^20]；推理侧有「用更多 test-time compute 改进输出」的实证收益 [^2][^3][^21]。但没有任何证据给出 RL 训练算力/环境规模的 scaling law，也没有直接比较 RL compute 与 pretrain compute 的边际收益——这是本课题的核心证据缺口。
2. **RL 的定位在证据中更接近「激励/解锁」而非「扩展能力」。** DeepSeek-R1 主张纯 RL 可激励出推理能力 [^1][^12]，但 pass@k 研究显示大 k 下基座模型反超 RLVR 模型，推理能力受基座模型约束 [^4][^14]；真正扩展能力的是蒸馏 [^4][^14]。
3. **Test-time compute 之所以重要，是因为它在 FLOPs 匹配下可替代参数量。** compute-optimal 策略相比 best-of-N 提升 test-time compute 扩展效率 4 倍以上，并可使小模型超过大 14 倍的模型 [^2][^3][^21]。
4. **RL 训练本身存在可预测的天花板。** 策略熵崩溃被形式化为 R = -a·e^H + b，性能是从熵中「换取」的，上限可预测 [^16]。

---

## 一、两条 scaling 路线的原始表述

### 1.1 预训练 scaling：幂律与等比例缩放

Kaplan et al. (2020) 给出语言模型交叉熵损失随模型规模、数据集规模、训练算力呈幂律缩放，趋势跨越超过七个数量级；网络宽度或深度等架构细节在较宽范围内影响极小 [^19]。该工作还给出过拟合对模型/数据规模依赖的方程、训练速度对模型规模依赖的方程，并指出这些关系可用于确定固定算力预算下的最优分配 [^19]。其结论之一是：更大的模型显著更具样本效率，因此最优的算力高效训练是「用相对适中的数据训练非常大的模型，并显著早于收敛就停止训练」[^19]。

Hoffmann et al. (2022, Chinchilla) 修正了这一结论的方向：当前大型语言模型显著训练不足，原因是近期聚焦于扩大模型规模而保持训练数据量不变 [^20]。通过训练 400 多个语言模型（70M 到 160 亿+参数，5 到 5000 亿 token），该研究得出 compute-optimal 训练下模型规模与训练 token 数应等比例缩放——模型规模每翻倍，训练 token 数也应翻倍 [^20]。验证模型 Chinchilla（70B 参数，4 倍数据，与 Gopher 相同算力）在大量下游任务上一致且显著超越 Gopher(280B)、GPT-3(175B)、Jurassic-1(178B)、Megatron-Turing NLG(530B)，在 MMLU 上达到 67.5% 平均准确率 SOTA，比 Gopher 提升超过 7%；且因模型更小，微调与推理使用显著更少算力 [^20]。

**这两条是「预训练 scaling law」的标准定义来源**：输入是模型参数量 N、训练 token 数 D、训练算力 C；产出是交叉熵损失 L 的预测值；方法是拟合幂律并求固定 C 下的最优 (N, D) 分配 [^19][^20]。

### 1.2 推理模型的 RL 路线：官方表述

OpenAI o1 System Card 的摘要将 o1 模型系列定义为「trained with large-scale reinforcement learning to reason using chain of thought」（用大规模强化学习训练以使用思维链推理）[^17]。正文第 2 节（标注日期 December 5, 2024）进一步说明 o1 大语言模型家族「trained with reinforcement learning to perform complex reasoning」，o1 在回答前会「produce a long chain of thought before responding to the user」[^18]。文档称通过训练，模型学会「refine their thinking process, try different strategies, and recognize their mistakes」[^18]。o1-mini 被描述为该模型的更快版本，尤其擅长编程 [^18]。

DeepSeek-R1 侧，论文主张 LLM 的推理能力可通过纯强化学习（pure RL）被激励出来，无需人工标注的推理轨迹 [^1]。RL 框架使模型涌现出高级推理模式，包括 self-reflection（自我反思）、verification（验证）与 dynamic strategy adaptation（动态策略适应）[^1]。经 RL 训练的模型在可验证任务（数学、编程竞赛、STEM）上表现优于用传统监督学习（基于人类示范）训练的同类模型 [^1]。论文称这些大规模模型涌现的推理模式可被系统性地用于引导和增强更小模型的推理能力 [^1]。

具体数字上，DeepSeek-R1-Zero 在 AIME 2024 上的 pass@1 分数从 15.6% 提升到 71.0%，使用 majority voting 后进一步提升到 86.7%，与 OpenAI-o1-0912 持平；R1-Zero 经数千次 RL 步骤后展现出强推理性能 [^5]。官方仓库称 R1-Zero 通过大规模 RL 训练、且不以 SFT 作为前置步骤，是「首个公开研究验证 LLM 推理能力可纯靠 RL 激发、无需 SFT」[^12]。R1-Zero 与 R1 均为 671B 总参数、37B 激活参数、128K 上下文，基于 DeepSeek-V3-Base 训练 [^12]。

R1-Zero 存在可读性差、语言混杂、无限重复等问题，这是引入 R1 多阶段训练流程的动机；DeepSeek-R1 在 RL 前引入少量 cold-start 数据（数千条）与多阶段训练流程 [^5][^12]。README 称 DeepSeek-R1 在数学、代码、推理任务上性能与 OpenAI-o1 相当 [^12]。

论文同时将 OpenAI o1 系列描述为首次引入 inference-time scaling（通过增加 Chain-of-Thought 推理长度）的模型 [^5]。Qwen3 报告则将 o3 与 DeepSeek-R1 并列为通过 RL 优化、增强 inference-time scaling 的推理模型代表 [^6]。

---

## 二、RL Scaling 与预训练 Scaling 的关系：证据能支持什么

### 2.1 证据支持的关系：RL 受基座模型约束

这是本课题中证据最扎实的一条。

《Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?》使用大 k 值下的 pass@k 作为评估指标，跨多个模型家族、RL 算法以及数学/代码/视觉推理基准，系统探测 RLVR 训练模型的推理能力边界 [^4][^14]。发现：RLVR 训练模型在小 k（如 k=1）时优于其 base model，但当 k 很大时 base model 取得更高的 pass@k 分数，即出现交叉/反超现象 [^4][^14]。coverage 与 perplexity 分析表明，所观察到的推理能力来源于 base model 并受其约束（bounded by the base model）[^4][^14]。以 base model 作为上界，定量分析显示六种流行 RLVR 算法表现相近，且都远未充分利用 base model 的潜力 [^4][^14]。相比之下，蒸馏（distillation）可以从 teacher 引入新的推理模式，真正扩展模型的推理能力 [^4][^14]。作者结论是：当前 RLVR 方法尚未实现 RL 引出全新推理能力的潜力，需要改进的 RL 范式（如持续 scaling、多轮 agent-环境交互）[^14]。

**这条证据的含义**：RL 不是预训练的替代品，而是在预训练所划定的能力空间内做「采样重加权」。RL 提升 k=1 的表现（即把概率质量集中到基座模型已经能采到的高质量解上），但不提升 k→∞ 的覆盖率（即不创造基座模型采不到的解）。

### 2.2 证据支持的关系：RL 训练存在熵瓶颈

《The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models》指出 scaling RL for reasoning 的主要障碍之一是策略熵（policy entropy）崩溃：在无熵干预的大量 RL 运行中，策略熵在训练早期急剧下降，探索能力削弱并伴随策略性能饱和 [^16]。论文提出经验变换方程 R = -a·e^H + b，将熵 H 与下游性能 R 关联，表明策略性能是从策略熵中「换取」的，并受熵耗尽的瓶颈限制，其上限可预测为 H=0 时 R = -a + b [^16]。理论推导表明策略熵的变化由动作概率与 logits 变化之间的协方差驱动，在使用 Policy Gradient 类算法时该协方差与 advantage 成正比；实证显示协方差项与熵差完全吻合，且协方差项在训练中大多保持正值，解释了策略熵为何单调下降 [^16]。论文提出 Clip-Cov 与 KL-Cov 两种技术，对高协方差 token 分别做裁剪与 KL 惩罚，实验表明可鼓励探索、帮助策略摆脱熵崩溃并获得更好的下游性能 [^16]。

**这条证据的含义**：RL 的收益不是无限可扩展的——它有一个由熵耗尽决定的、可预测的上限。这与预训练 scaling law 的「幂律无饱和」形态在结构上不同。

### 2.3 证据支持的关系：长 CoT 的涌现依赖算力与奖励塑形

《Demystifying Long Chain-of-Thought Reasoning in LLMs》通过大量 SFT 与 RL 实验，系统考察长 CoT 推理的机制，识别出使模型生成长 CoT 轨迹的关键因素 [^15]。四条发现：

- SFT 并非严格必需，但可简化训练并提升效率 [^15]。
- 推理能力倾向于随训练算力增加而涌现，但其发展并非必然，奖励塑形（reward shaping）对稳定 CoT 长度增长至关重要 [^15]。
- scaling 可验证奖励信号对 RL 至关重要；利用带过滤机制的噪声网络抽取解在 OOD 任务（如 STEM 推理）上潜力显著 [^15]。
- 纠错等核心能力本就存在于基座模型中，但通过 RL 为复杂任务有效激励这些技能需要大量算力，且衡量其涌现需要细致方法 [^15]。

**这条证据的含义**：这是最接近「RL 侧 scaling 现象」的表述——「推理能力倾向于随训练算力增加而涌现」——但它描述的是涌现趋势，不是拟合出的 scaling law，且强调「并非必然」，依赖奖励塑形 [^15]。第四条发现再次支持 2.1 的结论：核心能力本就存在于基座模型中 [^15]。

### 2.4 证据无法支持的关系（显式声明）

**现有公开资料不足以确认 RL scaling law 与预训练 scaling law 之间的定量关系。** 21 条证据中没有任何一条给出 RL 训练算力/数据/环境规模的 scaling law 拟合结果，也没有论文直接比较 RL compute 与 pretrain compute 的边际收益。现有证据（2504.13837、2502.03373、2505.22617）只讨论 RLVR 的能力边界、长 CoT 涌现与熵崩溃，属于「RL 能否扩展能力」而非「RL compute 如何 scaling」[^4][^14][^15][^16]。因此「互补 / 替代 / 权衡」的三选一在本报告的证据范围内无法判定。

唯一可确认的定性关系是：**RL 的能力上限受基座模型约束** [^4][^14]，因此两者在逻辑上是「上界—优化」关系而非平行替代关系。但这一关系是否随基座模型规模变化、是否可写成函数形式，证据未覆盖。

---

## 三、Test-Time Compute 为何重要

### 3.1 定位：通用 agent 的关键一步

《Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters》将「让 LLM 通过使用更多 test-time computation 来改进输出」定位为构建可自我改进、能处理开放式自然语言任务的通用 agent 的关键一步 [^2]。研究聚焦的问题是：若允许 LLM 使用固定但非平凡的 inference-time compute，其在困难 prompt 上的性能能提升多少；论文明确指出这关系到 LLM 预训练的未来以及 inference-time 与 pre-training compute 之间应如何权衡 [^2]。

论文动机还包含一个具体的部署含义：若预训练模型规模可与推理期额外计算做权衡，则可在设备端小模型替代数据中心级大模型 [^21]。

### 3.2 两类扩展机制（标准定义）

论文分析了两类扩展 test-time computation 的主要机制 [^2][^3]：

1. **针对稠密的、基于过程的 verifier reward model 进行搜索**（search against a dense, process-based verifier reward model）[^2][^3]。
2. **在测试时根据 prompt 自适应地更新模型对响应的分布**（adaptively updating the model's distribution over responses at test time）[^2][^3]。

**定义性说明**：第一类的输入是候选解与过程奖励模型（PRM）的打分，步骤是对候选解空间做搜索，产出是选出的解；第二类的输入是 prompt 与当前模型分布，步骤是在测试时更新该分布，产出是更新后的分布及其采样结果 [^2][^3]。

### 3.3 核心发现：有效性取决于 prompt 难度

作者发现两种机制下，扩展 test-time compute 的不同方法其有效性都关键地取决于 prompt 的难度（difficulty of the prompt）[^2][^3]。基于该观察提出 compute-optimal 扩展策略，按 prompt 自适应分配 test-time compute [^2][^3]。

量化结果：

- 相比 best-of-N 基线，compute-optimal 策略可将 test-time compute 扩展效率提升超过 4 倍 [^2][^3]。
- 在 FLOPs-matched 评估中，对于较小 base model 已能达到一定非平凡成功率的问题，test-time compute 可使其表现超过大 14 倍的模型 [^2][^3]。
- Figure 1 描述：在 revision（迭代自我修正）设置下，标准 best-of-N（'parallel'）与 compute-optimal scaling 之间的差距逐渐扩大，使 compute-optimal 能以少 4 倍的 test-time compute 超过 best-of-N [^21]。

论文同时指出此前研究对多种 test-time 策略多给出负面结果（negative results），且少有研究理解各种 test-time 推理方法的 scaling 行为 [^3]。

### 3.4 工程侧的印证：推理框架把 RL rollout 列为核心负载

Test-time compute 的重要性在工程侧有独立印证。SGLang 是面向大语言、视觉语言与扩散模型的开源推理框架，README 明确其优化目标包括 agentic workloads、大规模 serving 与 RL rollouts [^10]。README 列出 RL 与后训练生态集成项目：Miles、slime、AReaL、Tunix、verl，均集成 SGLang 用于 rollout 生成 [^10]。vLLM 侧，核心技术包括 PagedAttention 的 KV 内存管理、continuous batching、chunked prefill、prefix caching，并支持投机解码与 disaggregated prefill/decode/encode [^11]。

**注意**：这些是推理框架 README 的自述定位，属于工程实现层面的证据，不构成对 test-time compute 收益的独立学术验证。

### 3.5 产品侧的印证：thinking budget

Qwen3 将 thinking 模式（复杂多步推理）与 non-thinking 模式（快速上下文响应）整合进统一框架，可基于用户查询或 chat template 动态切换 [^6]。Qwen3 引入 thinking budget 机制，允许用户在推理时自适应分配计算资源，以在延迟与性能间权衡 [^6]。Qwen3 系列包含稠密与 MoE 两种架构，参数规模从 0.6B 到 235B；旗舰模型 Qwen3-235B-A22B 为 MoE 架构，总参数 235B，每 token 激活 22B 参数；全部模型以 Apache 2.0 许可公开 [^6]。

**这条证据的含义**：compute-optimal 分配从论文概念变成了产品 API 中的可调参数，说明「按需分配 test-time compute」已被工程化采纳。

---

## 四、支撑 RL 训练的开源基础设施

这一节回答「RL scaling 在工程上如何被执行」，属于背景性证据。

- **verl**：由字节跳动 Seed 团队发起、由 verl 社区维护的 RL 训练库，定位为「灵活、高效、生产就绪的 LLM RL 训练库」，是论文 HybridFlow: A Flexible and Efficient RLHF Framework 的开源实现 [^7]。README 主张 hybrid-controller 编程模型可用几行代码构建 GRPO、PPO 等 RL dataflow；通过模块化 API 与 FSDP、Megatron-LM、vLLM、SGLang 等现有 LLM 框架解耦集成；3D-HybridEngine 实现 actor 模型重分片，消除内存冗余并降低训练/生成阶段切换的通信开销 [^7]。README 自述具备「SOTA LLM 训练与推理引擎集成和 SOTA RL 吞吐」（营销性主张，出处为项目自身 README）[^7]。仓库历史提交数为 3,102 Commits [^7]。
- **OpenRLHF**：README 自述为「首个高性能、生产就绪的开源 RLHF 框架」，结合 Ray + vLLM 分布式架构与统一 agent-based 设计范式（营销性主张，出处为项目自身 README）[^8]。架构基础为 Ray + vLLM + DeepSpeed；支持的 RL 算法包括 PPO、REINFORCE++、GRPO、RLOO [^8]。新后端 Molt 基于 Automodel，README 称其比 DeepSpeed 更强、可将 RL 训练扩展到数千亿参数（营销性主张）[^8]。仓库历史提交数为 1,585 Commits [^8]。
- **TRL**：定位为「post-train foundation models 的综合库」，支持 SFT、GRPO、DPO 等后训练技术，提供 SFTTrainer、GRPOTrainer、DPOTrainer、KTOTrainer 等 trainer 类 [^9]。README 明确 GRPOTrainer 实现 Group Relative Policy Optimization，并称该算法比 PPO 更省内存、被用于训练 DeepSeek AI 的 R1 [^9]。README 称新特性可在同一 GPU 上支持最长 6.9× 的序列长度，SFT/DPO/KTO/GRPO/RLOO 默认使用 fused LM head 打分，8k token 时峰值显存减少 52%–82% [^9]。仓库历史提交数为 3,831 Commits [^9]。
- **Open-R1**：DeepSeek-R1 的完全开源复现项目，目标是补齐 R1 pipeline 的缺失部分 [^13]。README 顶部声明该项目已不再维护，新 issue 与 PR 不再被审查；其训练侧工作已迁移至 TRL（「For GRPO, SFT and distillation today, use TRL」）[^13]。三步计划：Step 1 复现 R1-Distill 模型；Step 2 复现 R1-Zero 的纯 RL pipeline；Step 3 展示从 base model 经多阶段训练到 RL-tuned [^13]。2025/05/26 宣布 Step 1 完成，发布 Mixture-of-Thoughts 数据集（350k 条从 R1 蒸馏的已验证推理轨迹，覆盖数学、代码、科学），并提供 OpenR1-Distill-7B 训练配方 [^13]。2025/03/11 发布 CodeForces-CoTs 数据集（10k 竞赛编程题、100k 解法），并称 7B Qwen 模型在 IOI24 上可超过 Claude 3.7 Sonnet、32B 模型可超过 R1 本身 [^13]。仓库历史提交数为 202 Commits [^13]。

**注意**：Open-R1 的「Step 2 复现 R1-Zero 的纯 RL pipeline」在其证据中未标注完成状态，现有公开资料不足以确认该步骤是否完成 [^13]。

---

## 五、矛盾与不确定性

### 5.1 核心矛盾：RL 是「激励」还是「扩展」

- DeepSeek-R1 论文与官方仓库主张纯 RL 可激励出推理能力，且 RL 训练的模型在可验证任务上优于监督学习训练的同类模型 [^1][^12]。
- pass@k 研究则显示 RLVR 模型在大 k 下被 base model 反超，推理能力受 base model 约束，真正扩展能力的是蒸馏 [^4][^14]。

两者并不完全冲突——前者讨论 k=1 或多数投票下的表现，后者讨论大 k 覆盖率——但**对「RL 是否扩展了模型能力」这一问题的回答方向相反**。本报告的证据无法裁决哪一方更接近「RL scaling 的真实形态」，因为缺少 RL compute 的 scaling law 拟合。

### 5.2 Blocking 缺口一：RL scaling law 与预训练 scaling law 的定量关系

**现有公开资料不足以确认。** 21 条证据中没有任何一条给出 RL 训练算力/数据/环境规模的 scaling law 拟合结果，也没有论文直接比较 RL compute 与 pretrain compute 的边际收益 [^4][^14][^15][^16]。这是本课题主问题的核心缺口。

### 5.3 Blocking 缺口二：产业界算力投入结构的可验证证据

**现有公开资料不足以确认。** 21 条证据中无一条来自 OpenAI/Anthropic/Google DeepMind/DeepSeek 的官方博客、财报或可信媒体报道，也没有任何算力投入结构的量化证据（如 capex 拆分、训练集群规模变化）。因此「预训练算力投入下降、推理算力上升」这一常见叙事在本报告证据范围内**无任何来源覆盖**，不予采信。

### 5.4 非 blocking 缺口：test-time compute 收益递减曲线只有单一独立来源

「pass@k / best-of-N / 多数投票 / 长 CoT 长度的收益递减形状」实际只有 1 个独立来源。相关证据中两条是同一论文（2408.03314v1 与 2408.03314）[^2][^3]，第三条 2504.13837 讨论的是 RLVR 能力边界而非 test-time compute 的饱和曲线 [^4][^14]。因此不满足 2 个独立来源要求，相关曲线形状的结论应视为**单一来源、待独立复现**。

### 5.5 其他不确定性

- verl 与 OpenRLHF 的「SOTA」「首个」「比 DeepSpeed 更强」等表述均为项目自身 README 的营销性主张，未经第三方验证 [^7][^8]。
- DeepSeek-R1 官方仓库历史提交数仅 36 Commits，为模型发布页而非训练代码库，不能作为工程活跃度指标 [^12]。
- Open-R1 已声明不再维护，其 Step 2（纯 RL pipeline 复现）完成状态不明 [^13]。

---

## 参考来源

[^1]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning — https://arxiv.org/abs/2501.12948v2 (arxiv, 2025-01-22)
[^2]: Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters — https://arxiv.org/abs/2408.03314v1 (arxiv, 2024-08-06)
[^3]: Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters — https://arxiv.org/abs/2408.03314 (arxiv, 2024-08-06)
[^4]: Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? — https://arxiv.org/abs/2504.13837 (arxiv, 2025-04-18)
[^5]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (ar5iv full text) — https://ar5iv.labs.arxiv.org/html/2501.12948 (arxiv, 2025-01-22)
[^6]: Qwen3 Technical Report — https://arxiv.org/abs/2505.09388v1 (arxiv, 2025-05-14)
[^7]: volcengine/verl — Volcano Engine Reinforcement Learning for LLMs — https://github.com/volcengine/verl (github)
[^8]: OpenRLHF/OpenRLHF — Ray + vLLM distributed RLHF framework — https://github.com/OpenRLHF/OpenRLHF (github)
[^9]: huggingface/trl — Transformers Reinforcement Learning — https://github.com/huggingface/trl (github)
[^10]: sgl-project/sglang — Fast inference for LLMs and multimodal models — https://github.com/sgl-project/sglang (github)
[^11]: vllm-project/vllm — Easy, fast, and cheap LLM serving for everyone — https://github.com/vllm-project/vllm (github)
[^12]: deepseek-ai/DeepSeek-R1 — official R1 model release — https://github.com/deepseek-ai/DeepSeek-R1 (github)
[^13]: huggingface/open-r1 — A fully open reproduction of DeepSeek-R1 — https://github.com/huggingface/open-r1 (github)
[^14]: Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? — https://arxiv.org/abs/2504.13837v5 (arxiv, 2025-04-18)
[^15]: Demystifying Long Chain-of-Thought Reasoning in LLMs — https://arxiv.org/abs/2502.03373v1 (arxiv, 2025-02-05)
[^16]: The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models — https://arxiv.org/abs/2505.22617v1 (arxiv, 2025-05-28)
[^17]: OpenAI o1 System Card (arXiv:2412.16720v2) — https://arxiv.org/abs/2412.16720v2 (arxiv, 2024-12-21)
[^18]: OpenAI o1 System Card — Section 2 Model data and training (HTML full text) — https://arxiv.org/html/2412.16720v2 (arxiv, 2024-12-05)
[^19]: Scaling Laws for Neural Language Models (Kaplan et al., 2020) — https://arxiv.org/abs/2001.08361 (arxiv, 2020-01-23)
[^20]: Training Compute-Optimal Large Language Models (Hoffmann et al., 2022 / Chinchilla) — https://arxiv.org/abs/2203.15556v1 (arxiv, 2022-03-29)
[^21]: Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (HTML full text) — https://arxiv.org/html/2408.03314v1 (arxiv, 2024-08-06)