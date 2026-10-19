# DeepSeek-R1 核心技术路线与推理模型训练范式影响

## 摘要

DeepSeek-R1 的核心技术主张是：LLM 的推理能力可以通过纯强化学习（pure RL）被激发，无需人工标注的推理轨迹[^1]。其第一代模型 DeepSeek-R1-Zero 直接以 DeepSeek-V3-Base 为基座、跳过 RL 前的常规监督微调（SFT）阶段、仅以最终答案正确性作为奖励信号进行大规模 RL，在 AIME 2024 上把 pass@1 从 15.6% 提升到 71.0%，多数投票后达 86.7%[^2][^16]。由于 R1-Zero 存在可读性差与语言混杂问题，正式版 DeepSeek-R1 引入冷启动数据与多阶段训练[^2][^3]。围绕这条路线，社区分化出两条后续脉络：一条沿 GRPO 家族做算法与工程改良（DAPO、Dr. GRPO、VAPO、GSPO 等）[^6][^7][^8][^9][^22]，另一条以 pass@k 为工具质疑 RLVR 是否真正扩展了基座模型之外的推理边界[^17][^18]。需要说明的是，本报告所依据的证据在奖励设计细节、GRPO 目标函数原文、R1 多阶段流程的中间环节、训练算力与完整基准分数、以及产业侧厂商动作等方面存在明显缺口，相关结论的强度受此限制。

---

## 一、核心技术路线

### 1.1 基座与 RL 框架

DeepSeek-R1 与 R1-Zero 均以 DeepSeek-V3-Base 为基座模型训练[^3][^10]，采用 GRPO（Group Relative Policy Optimization，Shao et al. 2024）作为 RL 框架[^2][^4][^16]。模型规模为 671B 总参数、37B 激活参数、128K 上下文[^3][^5][^10]，属 MoE 架构（架构细节论文指向 DeepSeek-V3 仓库）[^3]。

GRPO 本身的标准定义（输入、步骤、产出）在本次取证中**未被完整给出**：现有证据只说明它是 R1 采用的 RL 框架[^2][^4]，以及 TRL 的 GRPOTrainer 实现了该算法并「was used to train Deepseek AI's R1」[^13]。目标函数、优势估计公式、组内归一化的具体形式均未取证。第三方证据补充了 GRPO 的一个已知性质：它通过比较同一 prompt 多次 rollout 的奖励来更新模型[^17]，以及它采用 token-level importance ratios（这是 GSPO 与之对比的差异点）[^7]。

### 1.2 奖励设计

R1-Zero 的奖励信号**仅基于最终预测相对标准答案的正确性，不对推理过程本身施加约束**[^2][^4][^16]。这是「纯 RL」主张的技术支点：不引入过程奖励模型（PRM）、不引入人工标注的推理轨迹[^1]。

需要明确指出：现有证据**未给出**论文原文中规则化奖励（accuracy reward + format reward）的具体构成，也未给出训练模板（如要求把最终答案放入 `\boxed{}` 的格式约束）的原文表述。因此「奖励设计」这一子问题在本报告的证据范围内只能停留在上述概括层面。

### 1.3 R1-Zero：跳过 SFT 的纯 RL 路径

R1-Zero 的技术特征可归纳为三点：

1. **不做 SFT 前置**，直接对 base model 做大规模 RL[^2][^3][^16]；
2. **奖励只作用于最终答案正确性**[^2][^4]；
3. 论文明确说明绕过 SFT 阶段的理由是：人为定义的推理模式可能限制模型探索，而无约束的 RL 训练更能激发涌现[^2][^4]。

其结果是推理行为的自然涌现，包括自我反思（self-reflection）、验证（verification）与动态策略调整（dynamic strategy adaptation）[^1]，以及自验证、反思与长 CoT 行为[^3]。量化结果：AIME 2024 pass@1 从 15.6% 提升至 71.0%，多数投票后 86.7%，与 OpenAI-o1-0912 持平[^2][^16]。

代价是 R1-Zero 存在可读性差、语言混杂、无限重复等问题[^2][^3]。

### 1.4 DeepSeek-R1：冷启动 + 多阶段训练

R1 相对 R1-Zero 的改动是引入**多阶段训练**与**RL 前的冷启动数据**[^2][^16]。流程起点是收集数千条冷启动数据对 DeepSeek-V3-Base 进行微调，随后执行与 R1-Zero 类似的推理导向 RL[^2]。

官方 README 把 R1 的训练流程描述为：2 个 RL 阶段（发现推理模式、对齐人类偏好）+ 2 个 SFT 阶段（作为推理与非推理能力的种子）[^3]。

**证据缺口**：拒绝采样（rejection sampling）与全场景 RL（general/非推理任务对齐）这两个阶段在现有证据中没有任何具体描述，各阶段的顺序与数据来源亦缺失。因此「R1 相对 R1-Zero 增加了哪些阶段」这一问题只能给出上述粗粒度答案。

### 1.5 蒸馏路线

官方开源了 6 个从 DeepSeek-R1 蒸馏的稠密模型，规模为 1.5B、7B、8B、14B、32B、70B，分别基于 Qwen2.5 与 Llama3 系列[^3][^5][^10]。README 声称 DeepSeek-R1-Distill-Qwen-32B 在多项基准上超过 OpenAI-o1-mini，并称其取得稠密模型的新 SOTA[^3]，同时声称 DeepSeek-R1 在数学、代码与推理任务上达到与 OpenAI-o1 相当的性能[^3][^10]。这两条均为**官方自我主张**，本次取证未读到支撑分数表。

---

## 二、对推理模型训练范式的影响

### 2.1 直接冲击：SFT 前置不再是默认选项

R1-Zero 的范式意义在于它证明了「跳过 SFT 直接做 RL」可以work[^2][^3][^10]。官方 README 自述这是首个公开研究，验证 LLM 推理能力可纯靠 RL 激励、无需 SFT 前置步骤[^10]。论文的假设是：人为定义的推理模式可能限制模型探索[^2][^4]。这一主张把「先 SFT 再 RL」的行业默认流程变成了一个可选项而非必需项。

### 2.2 GRPO 家族：算法与工程改良的密集爆发

R1 之后，围绕 GRPO 的变体形成了清晰的技术谱系。一份策展式索引列出的 GRPO 家族变体包括 DAPO、Dr.GRPO、VAPO、CISPO、GiGPO、GSPO、GMPO、GTPO、Reinforce++、COPO、GAGPO[^22]，另列出 SIS（Selective Importance Sampling）、advantage shaping 类方法（AAPO、ProGRPO、GRPOEA、EMPG、BAPO、ASPO）、TRPO/FiberPO 理论线，以及 MoE RL Stability 作为独立主题[^22]。该索引把 Track 02 标注为「GRPO Family & Engineering Refinements (2025–2026)」[^22]。

其中几条有明确技术主张的：

- **DAPO**：提出 Decoupled Clip and Dynamic sAmpling Policy Optimization，开源大规模 RL 系统，在 Qwen2.5-32B base 上于 AIME 2024 达 50 分；论文明确指出 SOTA 推理模型（点名 OpenAI o1 blog 与 DeepSeek R1 technical report）隐藏关键技术细节，社区难以复现[^6]。
- **Dr. GRPO**：识别出 GRPO 中的优化偏置（optimization bias），会在训练中人为增加响应长度，尤其是对错误输出；提出无偏优化方法 Dr. GRPO，在保持推理性能的同时提升 token 效率；其极简 R1-Zero recipe 在 7B base 上于 AIME 2024 达 43.3%[^8]。
- **VAPO**：走 value-based 范式，基于 Qwen 32B 预训练模型在 AIME 2024 达 60.4 分，论文称比此前报告的 DeepSeek-R1-Zero-Qwen-32B 与 DAPO 结果高出 10 分以上，仅 5,000 步达 SOTA 且多次独立运行未崩溃[^9]。
- **GSPO**：把重要性比率定义在序列似然上，执行序列级裁剪、奖励与优化，与采用 token-level importance ratios 的先前算法（含 GRPO）不同，在训练效率与性能上优于 GRPO，并显著稳定 MoE 的 RL 训练；论文称其贡献带来了 Qwen3 模型的显著提升[^7]。

### 2.3 复现与工程化：从论文到可运行管线

R1 的官方仓库定位为论文与权重发布页而非训练代码库——仅含 README.md、DeepSeek_R1.pdf、figures/、LICENSE、.github/，共 36 次 commit[^10]。训练侧的空白由社区填补：

- **Open-R1** 自我定位为「DeepSeek-R1 的完全开放复现」，路线分三步：Step 1 蒸馏复现 R1-Distill；Step 2 复现生成 R1-Zero 的纯 RL pipeline；Step 3 展示从 base model 经多阶段训练到 RL-tuned[^11]。该项目已不再维护，训练侧能力迁移至 TRL[^11]。已发布产物包括 Mixture-of-Thoughts（350k 条从 R1 蒸馏并验证的推理轨迹，覆盖数学、代码、科学）[^11]、CodeForces-CoTs（10k 竞赛编程题、100k 解法）与 IOI24 基准[^11]、OpenR1-Math-220k[^11]。
- **verl** 是 ByteDance Seed 发起、社区维护的 RL 训练库，论文 HybridFlow 的开源版本，README 称可用几行代码构建 GRPO、PPO 等 RL dataflow，并与 FSDP、Megatron-LM、vLLM、SGLang 集成[^12]。
- **TRL** 把 GRPO 与 R1 直接关联，README 明确写 GRPOTrainer 实现的 GRPO「was used to train Deepseek AI's R1」，并提供 SFTTrainer、GRPOTrainer、DPOTrainer、KTOTrainer 等[^13]。README 提到针对推理模型应使用 `reasoning_accuracy_reward()` 函数，示例数据集为 trl-lib/DeepMath-103K[^13]。
- **OpenRLHF** 支持 PPO、REINFORCE++、GRPO、RLOO，其 News 记录 2025/2 的 Logic-RL 与 PRIME 工作表明 REINFORCE++ 相比 GRPO 训练更稳定、比 PPO 更快[^14]。
- **Unsloth** 提供桌面应用与 CLI/Docker 等安装方式，支持 RL、LoRA、QLoRA、全量微调、预训练、GRPO、DPO、FP8[^15]。

**证据缺口**：除官方 DeepSeek-R1 仓库（★91891、11655 forks、最近 push 2025-06-27）[^10] 外，Open-R1、verl、TRL、OpenRLHF、Unsloth 五条证据均只给出 commit 数（分别为 202、3102、3831、1585、10956 次）[^11][^12][^13][^14][^15]，**未给出 stars 量级**；「最近提交活跃度」多以 News 时间线条目代替，未给出最近一次 commit/push 的具体日期。因此无法在本报告中对这些仓库的相对热度做严格排序。

### 2.4 同期厂商路线：R1 不是唯一解

与 R1 同期或稍后的推理模型采取了不同技术路线，构成对「R1 路线是否唯一」的直接对照：

- **Kimi k1.5**：官方描述为 o1-level 多模态模型，技术路线核心是 **RL 上下文长度扩展**——把 RL 的 context window 扩展到 128k，并观察到性能随上下文长度增加而持续提升；采用 partial rollouts 复用已有轨迹的大段内容来采样新轨迹；使用在线镜像下降变体做策略优化[^23]。官方**明确表示不依赖 MCTS、value functions 和 process reward models** 等复杂技术即可取得强性能[^23]。这与 R1 路线在「是否用 value model」上形成分野——VAPO 正是走 value-based 路线的代表[^9]。
- **Qwen3**：核心卖点是 thinking 模式与 non-thinking 模式之间的无缝切换[^24]，官方称其构建在 QwQ 与 Qwen2.5 的经验基础上[^24]。后续 Qwen3-2507 拆分为 Instruct 与 Thinking 两个变体[^24]。
- **Kimi K2**：走的是另一条路——MoE 语言模型，总参数 1T、激活 32B，使用 Muon 优化器训练，提出 MuonClip 解决扩展中的不稳定问题；定位为面向 agentic 能力优化，发布变体 Kimi-K2-Instruct 被官方描述为 'reflex-grade model without long thinking'，即**不含长思考**[^25]。

Kimi K2 的评测表提供了 R1 之后开源阵营的横向坐标：Kimi K2 Instruct 在 SWE-bench Verified (Agentic Coding) 单次尝试得 65.8，高于 DeepSeek-V3-0324 的 38.8 与 Qwen3-235B-A22B (non-thinking) 的 34.4，低于 Claude Sonnet 4 的 72.7；LiveCodeBench v6 得 53.7，高于 DeepSeek-V3-0324 (46.9)、Qwen3-235B-A22B non-thinking (37.0)、GPT-4.1 (44.7)；AIME 2024 Avg@64 得 69.6，高于 DeepSeek-V3-0324 (59.4)、Qwen3-235B-A22B non-thinking (40.1)、GPT-4.1 (46.5)[^25]。

**证据缺口**：OpenAI o 系列、Llama、Gemini 三条厂商线在现有证据中**完全缺失**；Qwen 与 Kimi 证据均来自官方仓库 README（厂商自述），无第三方或媒体交叉验证；全部证据均**未涉及任何定价信息或开源策略变化的对比**。因此「R1 发布后各厂商在定价/开源策略上的变化」这一问题在本报告证据范围内无法回答。

### 2.5 范式争论：RLVR 到底扩展了什么

R1 路线的一个核心隐含主张是「RL 能激发基座模型之外的推理能力」。这一主张受到系统性挑战。

**怀疑派**的核心证据来自以 pass@k（大 k）为指标的研究：RLVR 训练模型在小 k（如 k=1）时优于基座模型，但大 k 时基座模型 pass@k 更高；覆盖度与困惑度分析显示观察到的推理能力源自基座模型并受其约束；以基座模型为上界做定量分析，六种流行 RLVR 算法表现相近，且在利用基座模型潜力方面都远未达到最优[^18]。该研究同时给出对照：**蒸馏可以从教师模型引入新的推理模式，真正扩展模型推理能力**[^18]。作者结论是当前 RLVR 方法尚未实现 RL 引出真正新颖推理能力的潜力，需要改进的 RL 范式，如持续扩展与多轮智能体-环境交互[^18]。

另一条怀疑派理论论证给出：RLVR 无法超出基座模型的表征极限（representational limits）[^17]。机制直觉是：多数 RL-for-reasoning 流程（GRPO/PPO 变体）通过比较同一 prompt 多次 rollout 的奖励来更新模型；若所有 rollout 都失败（pass@K = 0），则所有样本同样差，梯度消失[^17]。

**乐观派**的证据包括：ProRL 在基座模型表现很差的 Reasoning Gym 字母形 2D 拼图任务上扩展了推理边界[^17]；另有工作报告了在长时间近零奖励平台期后出现的 grokking 式相变（two-phase reward schedule）[^17]。该资源清单呼吁专门研究 pass@K = 0 的「hard frontier」问题族，认为数学与代码基准在异质池上取平均会把真正困难实例的信号冲淡[^17]。其标签体系区分证据强度：pass@k 从 0 到 1 为强信号，pass@k 改善为较弱信号（因基座模型已有部分成功）[^17]。

对 R1-Zero 本身的一个批判性发现值得单列：DeepSeek-V3-Base 已表现出 ''Aha moment''，而 Qwen2.5 base models 即使无 prompt templates 也展现强推理能力，提示存在**预训练偏置**[^8]。这直接削弱了「RL 从零激发推理」的叙事强度——至少部分能力可能来自预训练。

### 2.6 测试时计算：与 RL 路线并行的一条轴

R1 路线的重心在训练侧，但同期工作表明测试时计算是另一条独立且可叠加的轴：

- **Self-consistency**：先采样一组多样化的推理路径，再通过对采样路径做边缘化选出最一致的答案；在 GSM8K 上 +17.9%、SVAMP +11.0%、AQuA +12.2%、StrategyQA +6.4%、ARC-challenge +3.9%[^21]。R1-Zero 的 majority voting 结果（86.7%）正是这一思路的应用[^2][^16]。
- **Budget forcing**：通过强制终止模型思考过程，或在模型试图结束时反复追加 "Wait" 来延长思考，从而控制测试时计算量；s1-32B 在 MATH 与 AIME24 上超过 o1-preview 最多 27%，用 budget forcing 可在 AIME24 上从 50% 外推到 57%[^19]。
- **Compute-optimal 扩展**：分析两类机制——针对稠密、基于过程的 verifier 奖励模型进行搜索，以及在测试时根据 prompt 自适应更新模型对回答的分布；发现有效性关键取决于 prompt 难度；据此提出的策略相比 best-of-N 基线可将测试时计算扩展效率提升 4 倍以上；在 FLOPs 匹配的评估中，当较小基座模型已有一定成功率时，测试时计算可使其超越大 14 倍的模型[^20]。

---

## 三、矛盾与不确定性

**1. 奖励设计与 GRPO 定义的原文缺失。** 现有证据只给出「奖励信号仅基于最终预测与标准答案的正确性」这一句概括[^2][^4][^16]，缺少论文原文对规则化奖励（accuracy reward + format reward）的具体构成、GRPO 目标函数/优势估计公式、以及 R1-Zero 训练模板的原文表述。本报告中关于 GRPO 的描述均来自第三方对比性证据[^7][^17]，而非 R1 论文原文。

**2. R1 多阶段流程的中间环节缺失。** 证据仅笼统提到「多阶段训练 + cold-start 数据」[^2][^16] 与 README 的「2 个 RL 阶段 + 2 个 SFT 阶段」[^3]，但拒绝采样与全场景 RL 这两个阶段无任何具体描述，阶段顺序与数据来源亦缺失。

**3. 量化事实严重不足。** 现有证据只给出 R1-Zero 的 AIME 2024 pass@1 15.6%→71.0%（及 majority voting 86.7%）一个基准点[^2][^16]。MATH-500、Codeforces（rating/percentile）、GPQA Diamond 的具体分数全部缺失；训练算力/成本量级、RL 步数与最大生成长度无任何来源；蒸馏模型（1.5B–70B）的对应成绩表仅有 README 的定性主张[^3]，无分数支撑。因此本报告无法对「R1 路线的成本效率」做任何定量判断。

**4. 仓库热度指标不完整。** 除官方 DeepSeek-R1 仓库（★91891）[^10] 外，其余五个训练/复现仓库均只有 commit 数，无 stars 量级与最近 commit 日期[^11][^12][^13][^14][^15]。

**5. 产业侧证据基本空白。** OpenAI o 系列、Llama、Gemini 三条厂商线完全缺失；Qwen 与 Kimi 证据均为厂商自述[^23][^24][^25]，无第三方交叉验证；定价与开源策略变化无任何证据。

**6. 厂商自述与第三方验证的强度差异。** DeepSeek 官方 README 的多项性能主张（R1 对标 o1、Distill-Qwen-32B 超 o1-mini、取得稠密模型新 SOTA）[^3][^10]、Qwen3-Thinking-2507 的 SOTA 主张[^24]、Kimi k1.5 的 +550% 提升主张[^23] 均为厂商自述。本报告已逐条标注，但无法在现有证据范围内完成交叉验证。

**7. 范式争论本身尚未收敛。** 怀疑派（RLVR 受基座模型 support 限制[^17][^18]）与乐观派（ProRL 扩展边界、grokking 式相变[^17]）的证据强度不同：前者有跨模型家族、跨算法、跨基准的系统性 pass@k 证据[^18]，后者目前更多是个案与现象报告[^17]。同时 Dr. GRPO 的发现（DeepSeek-V3-Base 已有 ''Aha moment''、Qwen2.5 base 已具强推理能力）[^8] 从另一角度支持了怀疑派的「预训练偏置」解释。但该索引本身明确说明其条目背后是作者的中文深度博客解读，各 track 结尾给出的是作者本人的 bottom-line judgment 而非论文摘要[^22]，引用时需注意这一层中介。

**8. 一个未被现有证据回答的关键问题**：如果蒸馏能真正扩展推理能力而 RLVR 不能[^18]，那么 R1 开源蒸馏小模型（1.5B–70B）[^3][^10] 的成功究竟应归因于「R1 的 RL 训练」还是「从 R1 蒸馏」这一动作本身？现有证据不足以区分这两者的贡献比例。

---

## 参考来源

[^1]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948v2) — https://arxiv.org/abs/2501.12948v2 （arxiv，2025-01-22）
[^2]: DeepSeek-R1 (ar5iv HTML full text, v1) — Introduction — https://ar5iv.labs.arxiv.org/html/2501.12948 （arxiv，2025-01-22）
[^3]: deepseek-ai/DeepSeek-R1 — 官方仓库 README（模型下载表） — https://raw.githubusercontent.com/deepseek-ai/DeepSeek-R1/main/README.md （github）
[^4]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948v2) — https://arxiv.org/abs/2501.12948 （arxiv，2025-01-22）
[^5]: HuggingFace 模型卡 deepseek-ai/DeepSeek-R1（与官方 README 同源） — https://huggingface.co/deepseek-ai/DeepSeek-R1/raw/main/README.md （web）
[^6]: DAPO: An Open-Source LLM Reinforcement Learning System at Scale — https://arxiv.org/abs/2503.14476 （arxiv，2025-03-18）
[^7]: Group Sequence Policy Optimization (GSPO) — https://arxiv.org/abs/2507.18071 （arxiv，2025-07-24）
[^8]: Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO) — https://arxiv.org/abs/2503.20783 （arxiv，2025-03-26）
[^9]: VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks — https://arxiv.org/abs/2504.05118 （arxiv，2025-04-07）
[^10]: deepseek-ai/DeepSeek-R1 (official repository) — https://github.com/deepseek-ai/DeepSeek-R1 （github，2025-06-27）
[^11]: huggingface/open-r1 — A fully open reproduction of DeepSeek-R1 — https://github.com/huggingface/open-r1 （github）
[^12]: volcengine/verl — Volcano Engine Reinforcement Learning for LLMs — https://github.com/volcengine/verl （github）
[^13]: huggingface/trl — Transformers Reinforcement Learning — https://github.com/huggingface/trl （github）
[^14]: OpenRLHF/OpenRLHF — https://github.com/OpenRLHF/OpenRLHF （github）
[^15]: unslothai/unsloth — https://github.com/unslothai/unsloth （github）
[^16]: DeepSeek-R1 (ar5iv HTML 全文，Introduction 章节) — https://ar5iv.org/abs/2501.12948 （arxiv，2025-01-22）
[^17]: rdi-berkeley/awesome-RLVR-boundary — Awesome RLVR Reasoning Boundary — https://github.com/rdi-berkeley/awesome-RLVR-boundary （github，2025-12-12）
[^18]: Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? — https://arxiv.org/abs/2504.13837v5 （arxiv，2025-04-18）
[^19]: s1: Simple test-time scaling — https://arxiv.org/abs/2501.19393 （arxiv，2025-01-31）
[^20]: Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters — https://arxiv.org/abs/2408.03314 （arxiv，2024-08-06）
[^21]: Self-Consistency Improves Chain of Thought Reasoning in Language Models — https://arxiv.org/abs/2203.11171 （arxiv，2022-03-21）
[^22]: hscspring/rl-llm-nlp — A curated, opinionated index of post-R1 LLM × Reinforcement Learning — https://github.com/hscspring/rl-llm-nlp （github，2026-09-11）
[^23]: MoonshotAI/Kimi-k1.5 — Kimi k1.5: Scaling Reinforcement Learning with LLMs — https://github.com/MoonshotAI/Kimi-k1.5 （github，2025-01）
[^24]: QwenLM/Qwen3 — Qwen3 官方仓库 README（含 Qwen3-2504 与 Qwen3-2507 两次发布） — https://github.com/QwenLM/Qwen3 （github，2025-04）
[^25]: MoonshotAI/Kimi-K2 — Kimi K2 官方仓库 README（模型卡与评测表） — https://github.com/MoonshotAI/Kimi-K2 （github，2025-07）