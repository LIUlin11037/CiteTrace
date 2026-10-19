# 大模型知识蒸馏的主流做法与 DeepSeek-R1 蒸馏小模型效果评估

## 摘要

知识蒸馏（Knowledge Distillation, KD）最初由 Hinton、Vinyals 与 Dean 于 2015 年提出，目标是把集成模型或多个模型的知识压缩（compress）到单个更易部署的模型中[^1]。在大模型（LLM）语境下，KD 被重新定位为把领先闭源 LLM（如 GPT-4）的先进能力迁移到开源模型（如 LLaMA、Mistral）的方法论，并在开源生态中承担两类角色：压缩模型，以及以自身为教师实现自我改进[^2]。当前主流做法可归纳为三条技术脉络：基于教师生成序列的序列级监督蒸馏[^13][^14]、引入 rationale 的逐步蒸馏[^15][^16]、以及针对分布不匹配的 on-policy / 散度目标改进路线[^4][^17][^20]。DeepSeek-R1 走的是第一条路线的最直接形态——用 R1 生成的样本对 Qwen2.5 与 Llama3 系列 dense 模型做微调，开源了 1.5B 至 70B 共 6 个蒸馏模型[^5][^7][^9]。官方与论文报告的效果包括：Distill-Qwen-7B 在 AIME 2024 达 55.5%，超过 QwQ-32B-Preview；Distill-Qwen-32B 在 AIME 2024 得 72.6%、MATH-500 得 94.3%、LiveCodeBench 得 57.2%，与 o1-mini 相当[^7]。但需注意：这些数字属官方自我宣称[^5]，且 2025 年多项研究对「蒸馏 vs 从零 RL」路线提出质疑，指出蒸馏可引入新推理模式[^31]，而基于验证器的 RL 在固定算力预算下理论上更优[^28]，同时当前推理基准对实现细节高度敏感、宣称增益常被高估[^30]。本报告的核心不确定性在于：DeepSeek-R1 蒸馏训练数据规模（约 80 万条）在现有证据中无任何来源支撑；主流推理框架对 R1-Distill 系列的官方支持情况证据不足；争议双方未在同一证据面上对齐。

---

## 一、知识蒸馏的定义与经典范式

### 1.1 标准定义

知识蒸馏的经典定义来自 Hinton 等人 2015 年的工作：将集成模型（ensemble）或多个模型的知识压缩（compress）到单个更易部署的模型中[^1]。该文明确以「compress the knowledge in an ensemble into a single model」描述蒸馏目标，并称使用了一种与 Caruana 等人不同的压缩技术[^1]。论文在 MNIST 上给出结果，并将一个商用系统的大型声学模型集成蒸馏为单一模型，报告了显著改进；此外还引入了一种由完整模型与多个 specialist 模型组成的新型集成，specialist 模型学习区分完整模型易混淆的细粒度类别[^1]。

在序列生成任务上，Kim & Rush（EMNLP 2016）给出了更形式化的方法定义：压缩方法分为两类——(1) 剪枝（pruning）与 (2) 知识蒸馏；蒸馏是让较小的 student 网络通过最小化与 teacher 输出的损失（通常 L2 或交叉熵）来模仿 teacher[^14]。标准 word-level 蒸馏（word-level KD）在神经机器翻译（NMT）上直接套用多类交叉熵蒸馏：对每个位置最小化 student 分布与 teacher 分布之间的交叉熵，同时也可与 one-hot 数据分布做插值[^14]。蒸馏损失常与 NLL 损失插值：L(θ,θT) = (1-α)L_NLL(θ) + α L_KD(θ,θT)，α 为混合系数[^14]。

### 1.2 LLM 语境下的重新定位

Xu et al. 2024 的综述将 LLM 语境下的知识蒸馏定位为把领先闭源 LLM（如 GPT-4）的先进能力迁移到开源模型（如 LLaMA、Mistral）的方法论[^2]。该综述指出 KD 在开源 LLM 生态中承担两类角色：压缩模型（compressing these models）与以自身为教师实现自我改进（self-improvement by employing themselves as teachers）[^2]。其组织框架由三大支柱构成：algorithm（算法）、skill（技能）、verticalization（垂直化）[^2]，并强调数据增强（data augmentation, DA）与 KD 的交互，将 DA 视为 KD 框架内增强 LLM 性能的范式[^2]。该综述配有对应的 GitHub 仓库，将 KD 拆解为 Knowledge Elicitation（知识引出）与 Distillation Algorithms（蒸馏算法）两部分，并探讨 LLM 的 Skill & Vertical Distillation[^3]。

---

## 二、大模型知识蒸馏的主流技术路线

### 2.1 序列级蒸馏（Sequence-Level KD）

Kim & Rush 指出，NMT 与以往多类分类任务的区别在于：NMT 虽在 word-level 做多类预测，但任务目标是「在给定先前决策的条件下预测完整序列输出」[^14]。据此提出 sequence-level 蒸馏的核心做法：让 student 在一个新生成的数据集上训练，该数据集由 teacher 网络运行 beam search 产生，从而近似匹配 teacher 的 sequence-level（而非 word-level）分布[^14]。论文提出两种变体：sequence-level KD（student 训练于 teacher beam search 中得分最高的输出）与 sequence-level interpolation（student 训练于与目标序列 sim 最高的 teacher beam search 输出）[^14]。温度参数 τ 在 {1, 1.5, 2} 中测试后，发现 τ=1 效果最好[^14]。

效果方面：作者称 sequence-level 蒸馏「似乎消除了对 beam search 的需求」，即使应用在原始 teacher 模型上也是如此；最佳 student 模型比其 SOTA teacher 快 10 倍，性能损失很小[^13]。相比不使用知识蒸馏训练的 baseline，student 在 greedy decoding / beam search 下分别提升 4.2 / 1.7 BLEU；在知识蒸馏之上再做权重剪枝，可得到参数量比原 teacher 少 13 倍的 student，BLEU 仅下降 0.4[^13]。实验设置上将 4×1000 LSTM 压缩为 2×500 LSTM，大致匹配完整系统性能[^14]。

### 2.2 逐步蒸馏（Distilling Step-by-Step）：引入 rationale

Hsieh et al.（Findings of ACL 2023）提出 Distilling step-by-step：从 LLM 中提取 rationale（自然语言推理理由）作为额外监督信号，在多任务框架下训练小模型[^15]。方法分两步：先用 CoT prompting 让 LLM 生成标签及 rationale，再用这些 rationale 加任务标签训练下游小模型[^15]。

该工作明确了一个核心视角转变：从「把 LLM 视为噪声标签来源」转为「把 LLM 视为能推理的 agent」，即 LLM 可生成为其预测标签辩护的自然语言 rationale[^16]。论文将 LLM 生成 rationale 视为「任务知识」的载体，认为 rationale 中包含如「Area = length × width」这类原本需要大量数据才能学到的任务知识[^15]。

效果上，论文在 4 个 NLP benchmark 上得出三点结论：相比 finetuning 与 distillation，本方法用更少标注/无标注样本取得更好性能；相比 few-shot prompted LLM，用显著更小的模型取得更好性能；同时降低模型规模与所需数据量[^15]。具体数字：微调后的 770M T5 模型在某个 benchmark 上仅用 80% 可用数据即超过 few-shot prompted 的 540B PaLM，而标准微调同一 T5 模型即便用 100% 数据也难以匹敌[^15]。论文称其方法可用超过 500× 更少的模型参数超越 LLM，平均减少超过 50% 训练样本（最高超过 85% 减少）；在仅有未标注数据时，仅用 11B T5 模型即超过 540B PaLM 的性能[^15]。

### 2.3 On-Policy 蒸馏与散度目标改进

**MiniLLM（reverse KLD）**：Gu et al. 提出用 reverse KLD 替换标准 KD 中的 forward KLD 目标，理由是 reverse KLD 更适合生成式语言模型的蒸馏，可防止学生模型高估教师分布中的低概率区域[^17]。论文论证标准 KD 目标本质是最小化近似的 forward KLD KL[p||q_θ]，它迫使学生分布覆盖教师分布的所有 mode；在开放式文本生成中，由于学生容量有限，forward KLD 会使学生给教师分布的 void 区域分配不合理的高概率，并在自由生成时产生教师分布下极不可能的样本[^18]。MiniLLM 改为最小化 reverse KLD KL[q_θ||p]，其 mode-seeking 行为使学生聚焦教师分布的主要 mode、对 void 区域分配低概率[^18]。reverse KLD 目标的梯度通过 Policy Gradient Theorem 推导，采用 on-policy 训练；为稳定和加速训练提出三项技巧：(1) single-step decomposition 降低方差，(2) teacher-mixed sampling 缓解 reward hacking，(3) length normalization 消除长度偏差[^18]。论文将 MiniLLM 与 sequence-level KD (SeqKD) 对比：SeqKD 最小化 forward KLD，强迫学生记忆教师生成的所有样本；MiniLLM 不强迫学生拟合教师分布的所有样本，而是鼓励学生在自身能力范围内生成教师偏好的样本[^18]。实验覆盖 GPT-2、GPT-J、OPT 等生成式模型家族，规模 120M–13B，使用 5 个数据集，评估指标包括 Rouge-L、GPT-4 feedback 和人工判断[^18]。该论文发表于 ICLR 2024[^17]，官方实现位于 microsoft/LMOps 的 minillm 子目录，已被 HuggingFace TRL 支持（trl/experimental/minillm）[^19]。

**GKD（Generalized Knowledge Distillation）**：该论文针对自回归序列模型 KD 中「训练时见到的输出序列」与「学生模型推理时生成的序列」之间的分布不匹配（distribution mismatch）问题[^4]。GKD 不依赖固定的输出序列集合，而是让学生在自生成的输出序列上训练，并利用教师模型对这些序列的反馈[^4]。与监督式 KD 不同，GKD 允许在学生与教师之间采用替代的损失函数，论文指出这在学生表达能力不足以模仿教师分布时有用[^4]。论文将 on-policy 蒸馏形式化为：给定学生 p_S 与教师 p_T，在 token 级分布上最小化散度 D(p_T || p_S)，训练数据为学生自生成序列[^33]。论文强调学生-教师间最优散度（forward KL / reverse KL / JSD）具有任务依赖性[^33]。GKD 支持将蒸馏与 RL 微调（RLHF）无缝结合，这一组合此前未被探索[^4][^33]。论文在摘要中声明在摘要生成、翻译、算术推理任务上验证了 GKD 蒸馏自回归语言模型的有效性，并做了面向指令微调的任务无关蒸馏[^4]。

**DistiLLM（skew KLD + adaptive off-policy）**：Ko et al.（ICML 2024）提出 DistiLLM，由两个组件构成：(1) 一种新的 skew Kullback-Leibler divergence loss，论文揭示并利用其理论性质；(2) 一种 adaptive off-policy 方法，用于提升利用 student-generated outputs 的效率[^20]。论文针对现有自回归序列模型 KD 方法的两点问题：缺乏标准化目标函数，以及使用学生生成输出导致计算成本大幅上升[^20]。论文指出传统 KLD 因不对称性会导致 mode averaging / mode collapse：学生分布要么过度平滑无法覆盖教师分布，要么过度集中于高概率区域[^21]。Skew KLD 的设计目标：解决 KLD 优化中潜在梯度爆炸导致的不稳定性，以及缺乏对泛化性与收敛性的强调；声称具有稳定梯度与最小近似误差，经验上带来更快收敛与更优性能[^21]。Adaptive off-policy 模块用于自适应且高效地利用 student-generated outputs（SGO），因为持续生成 SGO 会显著增加训练时间且难以确定最优使用比例[^21]。论文声称 DistiLLM 在指令跟随、文本摘要等生成任务上取得学生 LM 的 state-of-the-art 性能，并相比 Gu et al. 2024 与 Agarwal et al. 2024 等近期 KD 技术实现 2.5~4.3 倍训练加速[^21]。官方实现基于 HuggingFace Transformers 的特定 commit，并沿用 MiniLLM 的实现[^22]。

### 2.4 蒸馏的边界条件：Scaling Law 视角

Distillation Scaling Laws 提出用算力预算及师生算力分配来估计蒸馏模型性能，支持计算最优的师生算力分配[^29]。论文给出两种场景的计算最优蒸馏配方：教师已存在时，以及教师也需训练时[^29]。关键边界条件：在涉及多个学生或已有教师的场景下，蒸馏在某个算力水平内优于监督学习，该水平随学生规模可预测地扩展；反之，若只蒸馏单个学生且教师也需从头训练，则监督学习通常更可取[^29]。

---

## 三、DeepSeek-R1 蒸馏小模型：做法与效果

### 3.1 蒸馏做法

DeepSeek-R1 论文摘要指出：大模型涌现出的推理模式可被系统性地用于引导和增强更小模型的推理能力（即蒸馏路线）[^6]。论文正文称，蒸馏做法是使用 DeepSeek-R1 生成的推理数据（reasoning data generated by DeepSeek-R1）对若干社区常用 dense 模型做微调[^7]。官方 README 进一步明确：蒸馏模型的训练方式是——基于开源模型微调，使用 DeepSeek-R1 生成的样本（samples generated by DeepSeek-R1），官方对蒸馏模型的 config 与 tokenizer 做了轻微改动，并建议使用官方设置运行[^5][^9]。

官方开源了 6 个从 DeepSeek-R1 蒸馏的 dense 模型，参数量分别为 1.5B、7B、8B、14B、32B、70B，基于 Qwen2.5 与 Llama3 系列[^5][^7]。具体基座对应关系为：DeepSeek-R1-Distill-Qwen-1.5B 基于 Qwen2.5-Math-1.5B；Qwen-7B 基于 Qwen2.5-Math-7B；Llama-8B 基于 Llama-3.1-8B；Qwen-14B 基于 Qwen2.5-14B；Qwen-32B 基于 Qwen2.5-32B；Llama-70B 基于 Llama-3.3-70B-Instruct[^5][^9]。

作为对照，R1 主模型为 MoE 架构，671B 总参数 / 37B 激活参数，上下文 128K，基于 DeepSeek-V3-Base 训练[^9]。R1-Zero 与 R1 均采用 GRPO 作为 RL 框架；R1-Zero 的 RL 训练超参为：学习率 3e-6，KL 系数 0.001，rollout 采样温度 1；每问采样 16 个输出，8.2k step 前最大长度 32,768 tokens、之后 65,536 tokens；共训练 10,400 步，约 1.6 个 epoch；每步 32 个不同问题，训练 batch size 为 512；每 400 步用最新策略模型替换参考模型[^8]。R1-Zero 使用基于规则的奖励，由 accuracy reward 与 format reward 等权相加构成，format reward 要求把推理过程放在 <think> 与 </think> 标签内[^8]。Nature 版正文确认 DeepSeek-R1 采用多阶段框架，整合拒绝采样、RL 与监督微调[^23]。

### 3.2 官方报告的效果

论文正文称：以 Qwen2.5-32B 为基座时，直接从 DeepSeek-R1 蒸馏的效果优于在该基座上直接做 RL，说明大基座模型发现的推理模式对提升推理能力至关重要[^7]。官方明确表述：大模型的推理模式可以被蒸馏进小模型，且效果优于在小模型上通过 RL 自行发现的推理模式[^9]。

具体基准数字（均来自论文正文，属官方报告）[^7]：
- DeepSeek-R1-Distill-Qwen-7B 在 AIME 2024 上达到 55.5%，超过 QwQ-32B-Preview；
- DeepSeek-R1-Distill-Qwen-32B 在 AIME 2024 得 72.6%、MATH-500 得 94.3%、LiveCodeBench 得 57.2%，显著超过此前的开源模型，与 o1-mini 相当；
- 蒸馏出的 14B 模型大幅超过当时 SOTA 的开源模型 QwQ-32B-Preview，蒸馏的 32B 与 70B 模型在 dense 模型的推理基准上创下新纪录。

作为教师模型的参照：DeepSeek-R1 在 AIME 2024 上 Pass@1 为 79.8%，在 MATH-500 上为 97.3%，Codeforces Elo 为 2029（超过 96.3% 人类参赛者）[^7]。DeepSeek-R1-Zero 在 AIME 2024 上 pass@1 从初始 15.6% 升至 77.9%，自一致性解码下达 86.7%[^23]。

官方 README 称 DeepSeek-R1-Distill-Qwen-32B 在多个基准上超过 OpenAI-o1-mini，并在 dense 模型上取得新的 SOTA[^5][^9]。**需注意：这一表述属官方自我宣称**[^5]。评估设置方面，官方采用最大生成长度 32768 tokens，采样温度 0.6、top-p 0.95，每 query 生成 64 个响应估计 pass@1[^9]。

### 3.3 第三方复现与对比

**Open-R1（HuggingFace）**：目标是复现 R1 流水线，分三步：Step 1 复现 R1-Distill 模型（从 R1 蒸馏高质量语料）；Step 2 复现 R1-Zero 的纯 RL 流水线；Step 3 实现 base model 经多阶段训练到 RL-tuned[^10]。2025/05/26 宣布 Step 1 完成：发布 Mixture-of-Thoughts，350k 条从 R1 蒸馏并经验证的推理轨迹，覆盖数学、代码、科学，并提供训练 OpenR1-Distill-7B 的配方，称其复现了 DeepSeek-R1-Distill-Qwen-7B 的推理能力[^10]。2025/03/11 发布 CodeForces-CoTs（10k 竞赛编程题、100k 从 R1 蒸馏的解法）与 IOI24 基准；称 7B 模型在 IOI24 上可超过 Claude 3.7 Sonnet，32B 模型可超过 R1 本身[^10]。2025/02/10 发布 OpenR1-Math-220k（220k 条从 R1 蒸馏的轨迹，基于新版 NuminaMath），称在该数据集上训练的模型可匹配 DeepSeek 蒸馏模型的性能[^10]。该仓库 README 顶部声明项目已不再维护，GRPO、SFT 与蒸馏的训练侧工作已迁移到 TRL[^10]。

**Open Thoughts（Bespoke Labs 与 DataComp 社区）**：首要目标是策展推理数据集以训练超越 DeepSeek-R1-Distill-Qwen-32B 和 DeepSeek-R1-Distill-Qwen-7B 的小型推理模型[^11]。README 结果表显示：OpenThinker3-7B 在 AIME24 得 69.0、AIME25 得 53.3、AMC23 得 93.5、MATH500 得 90.0、GPQA-D 得 53.7、JEEBench 得 72.4；对比 DeepSeek-R1-Distill-Qwen-32B 分别为 51.3 / 38.0 / 92.0 / 88.0 / 33.2 / 50.4[^11]。OpenR1-Distill-7B 在表中 AIME24 为 57.7、AIME25 为 39.7、MATH500 为 88.0、GPQA-D 为 58.9[^11]。值得注意的是，OpenThoughts3-1.2M 的推理轨迹改用 QwQ-32B 生成（而非 R1 标注），是 1000+ 次实验的结果[^11]。该项目声明完全开源：模型权重、数据集、数据生成代码、评估代码、训练代码均公开[^11]。其数据策展工具 Bespoke Curator 支持 LiteLLM、vLLM 及主流 batch API 等多种推理后端[^12]。

**s1**：用仅 1000 条精选问题-推理轨迹（s1K）对 Qwen2.5-32B-Instruct 做监督微调（即蒸馏式 SFT），并配合 budget forcing 控制测试时算力[^32]。s1-32B 在竞赛数学题（MATH、AIME24）上超过 o1-preview 最多 27%；通过 budget forcing 扩展，s1-32B 在 AIME24 上从 50% 提升到 57%，超出无测试时干预时的性能[^32]。该工作是「少量高质量蒸馏数据即可获得强推理」的代表性证据[^32]。

---

## 四、蒸馏 vs 从零 RL：路线争议

### 4.1 支持蒸馏的证据

Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? 用大 k 值的 pass@k 作为评测指标，跨多个模型家族、RL 算法及数学/代码/视觉推理基准，系统探测 RLVR 训练模型的推理能力边界[^31]。发现当前训练设置并未引出根本性的新推理模式：RLVR 模型在小 k（如 k=1）时优于基座模型，但大 k 时基座模型 pass@k 更高；覆盖度与困惑度分析表明，观察到的推理能力源自基座模型并受其约束；六种主流 RLVR 算法表现相近，且远未充分利用基座模型潜力[^31]。对比之下，该研究认为蒸馏可以从教师引入新的推理模式，真正扩展模型的推理能力[^31]。

DeepSeek-R1 论文自身的核心主张也支持蒸馏路线：以 Qwen2.5-32B 为基座时，直接从 DeepSeek-R1 蒸馏的效果优于在该基座上直接做 RL[^7]；官方明确表述大模型的推理模式可以被蒸馏进小模型，且效果优于在小模型上通过 RL 自行发现的推理模式[^9]。

### 4.2 质疑蒸馏路线的证据

**Scaling Test-Time Compute Without Verification or RL is Suboptimal** 明确把社区争议表述为两条路线：一是蒸馏成功的搜索/思考轨迹（verifier-free, VF），二是用验证器（0/1 结果奖励、奖励模型或 verifier）引导 RL 与搜索（verifier-based, VB）[^28]。理论证明：在固定算力/数据预算下，基于验证器的 RL 或搜索方法远优于基于蒸馏或克隆搜索轨迹的无验证器方法；当基座 LLM 对正确解轨迹呈异质分布（长度、风格不同）且奖励分布非尖锐时，随测试时算力（输出 token 长度）与训练数据扩展，VF 方法的次优性比 VB 恶化得更快[^28]。该结论意味着 VB 方法渐近扩展性更好，且随测试时预算增大，VB 与 VF 的性能差距会扩大；在 3B/8B/32B 预训练 LLM 的教学与数学推理问题上做了实证验证[^28]。

**SFT Memorizes, RL Generalizes** 直接对比 SFT（含蒸馏式监督微调）与 RL 在泛化与记忆上的差异，覆盖文本规则变体与视觉变体两类任务[^27]。结论：RL（尤其使用基于结果的奖励）能在文本与视觉规则变体上泛化；SFT 倾向于记忆训练数据，难以泛化到分布外（OOD）场景；分析显示 RL 提升了模型底层的视觉识别能力，从而增强视觉域泛化[^27]。但该研究同时指出：尽管 RL 泛化更优，SFT 仍是有效 RL 训练的前提——SFT 稳定输出格式，使后续 RL 能取得性能增益[^27]。

**A Sober Look at Progress in Language Model Reasoning** 指出当前数学推理基准对细微实现选择高度敏感，包括解码参数、随机种子、提示格式，甚至硬件与软件配置；近期研究报告的性能增益常依赖不清晰的比较或未报告的方差来源[^30]。用标准化评测框架重新评估后发现：多数 RL 方法仅带来适度改进，远低于此前宣称，且容易过拟合，尤其在小规模基准（如 AIME'24）上；相反，在其研究的设定中，监督微调（SFT）方法表现出持续更强的泛化能力[^30]。

---

## 五、矛盾与不确定性

### 5.1 关键事实缺失：R1 蒸馏数据规模

本报告无法确认 DeepSeek-R1 蒸馏训练数据的具体规模（如「约 80 万条样本」）。在现有 33 条证据中，无任何一条给出该数字或可推导该数字的来源——R1 论文正文[^7]、官方 README[^5][^9]、Nature 版[^23]均未在已收集证据中出现相关表述。**该数字属无来源支撑的关键事实，不应被引用。**

### 5.2 推理框架支持情况：证据不足

关于主流推理框架（vLLM / SGLang / llama.cpp）对 DeepSeek-R1-Distill 系列的支持与实测性能，现有证据不足以确认。vLLM 官方文档抓取正文在模型清单（List of Text-only Language Models）之前即被截断，未包含 DeepSeek-R1-Distill 系列的具体条目、推荐配置或性能数据[^24]；另两条证据为 0 star 与 1 star 的个人仓库，明确被标注「不构成官方支持情况的证据」[^25][^26]。SGLang 与 llama.cpp 零证据。**该子问题实际独立来源数为 0–1，不满足 ≥2 要求，无法给出可靠结论。**

### 5.3 路线争议：双方未在同一证据面对齐

已收集的 4 篇争议论文（SFT Memorizes, RL Generalizes[^27]；Scaling Test-Time Compute Without Verification or RL is Suboptimal[^28]；A Sober Look at Progress in Language Model Reasoning[^30]；Does RL Really Incentivize Reasoning Capacity[^31]）与 s1[^32] 均为 2025 年 1–4 月，且未包含 R1 论文自身「Qwen2.5-32B 上直接蒸馏优于在该基座上做 RL」这一原始主张[^7]与上述反驳之间的直接对照证据。**争议双方未在同一证据面上对齐，无法支撑「争议现状」的完整叙述。** 换言之，现有证据只能分别呈现「支持蒸馏」与「质疑蒸馏」两类主张，但无法判定哪一方在 R1 的具体设定下更成立。

### 5.4 官方宣称与第三方复现的差异

DeepSeek-R1-Distill-Qwen-32B 的「超过 o1-mini、dense 模型新 SOTA」属官方自我宣称[^5]。第三方复现中，Open Thoughts 的 OpenThinker3-7B 在 AIME24（69.0 vs 51.3）、AIME25（53.3 vs 38.0）、GPQA-D（53.7 vs 33.2）、JEEBench（72.4 vs 50.4）上超过 DeepSeek-R1-Distill-Qwen-32B，但在 AMC23（93.5 vs 92.0）与 MATH500（90.0 vs 88.0）上仅小幅领先[^11]。**需注意：OpenThoughts3-1.2M 的推理轨迹改用 QwQ-32B 生成而非 R1 标注**[^11]，因此该对比并非纯粹的「R1 蒸馏 vs 其他蒸馏」对照，其结论不能直接用于评估 R1 蒸馏路线的优劣。

### 5.5 评测方法学风险

A Sober Look at Progress in Language Model Reasoning 的发现对上述所有基准数字构成方法学警示：当前数学推理基准对解码参数、随机种子、提示格式乃至硬件与软件配置高度敏感，近期研究报告的性能增益常依赖不清晰的比较或未报告的方差来源[^30]。因此，本报告中引用的 AIME 2024、MATH-500 等具体数字，在跨论文比较时应保持谨慎。

## 参考来源

[^1]: Distilling the Knowledge in a Neural Network (Hinton, Vinyals, Dean 2015) — https://arxiv.org/abs/1503.02531v1（arxiv, 2015-03-09）
[^2]: A Survey on Knowledge Distillation of Large Language Models (Xu et al. 2024) — https://arxiv.org/abs/2402.13116v4（arxiv, 2024-02-20）
[^3]: Tebmer/Awesome-Knowledge-Distillation-of-LLMs — https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs（github, 2025-03-09）
[^4]: On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD) — https://arxiv.org/abs/2306.13649v3（arxiv, 2023-06-23）
[^5]: deepseek-ai/DeepSeek-R1 — README (Model Downloads: DeepSeek-R1-Distill Models) — https://raw.githubusercontent.com/deepseek-ai/DeepSeek-R1/main/README.md（github, 2025-01-22）
[^6]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948v2) — 摘要与引言 — https://arxiv.org/abs/2501.12948v2（arxiv, 2025-01-22）
[^7]: DeepSeek-R1 (ar5iv HTML full text) — Introduction / Contributions: Distillation — https://ar5iv.labs.arxiv.org/html/2501.12948（arxiv, 2025-01-22）
[^8]: DeepSeek-R1 (arXiv HTML v2) — Section 2: DeepSeek-R1-Zero 训练配置（GRPO） — https://arxiv.org/html/2501.12948v2（arxiv, 2025-01-22）
[^9]: deepseek-ai/DeepSeek-R1 (official repository README) — https://github.com/deepseek-ai/DeepSeek-R1（github）
[^10]: huggingface/open-r1 — A fully open reproduction of DeepSeek-R1 — https://github.com/huggingface/open-r1（github）
[^11]: open-thoughts/open-thoughts — Curating the best open reasoning datasets — https://github.com/open-thoughts/open-thoughts（github）
[^12]: bespokelabsai/curator — Bulk Inference and Scalable Data Curation for Post-Training — https://github.com/bespokelabsai/curator（github）
[^13]: Sequence-Level Knowledge Distillation (Kim & Rush, EMNLP 2016) — https://arxiv.org/abs/1606.07947（arxiv, 2016-06-25）
[^14]: Sequence-Level Knowledge Distillation — 方法章节（ar5iv 全文） — https://ar5iv.org/abs/1606.07947（arxiv, 2016-06-25）
[^15]: Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes (Hsieh et al., Findings of ACL 2023) — https://arxiv.org/abs/2305.02301（arxiv, 2023-05-03）
[^16]: Distilling Step-by-Step — 引言与相关工作（ar5iv 全文） — https://ar5iv.org/abs/2305.02301（arxiv, 2023-05-03）
[^17]: MiniLLM: On-Policy Distillation of Large Language Models (arXiv:2306.08543v6) — https://arxiv.org/abs/2306.08543v6（arxiv, 2023-06-14）
[^18]: MiniLLM 论文正文（方法章节，arXiv HTML v6） — https://arxiv.org/html/2306.08543v6（arxiv, 2023-06-14）
[^19]: microsoft/LMOps — minillm README — https://raw.githubusercontent.com/microsoft/LMOps/main/minillm/README.md（github）
[^20]: DistiLLM: Towards Streamlined Distillation for Large Language Models (arXiv:2402.03898v2) — https://arxiv.org/abs/2402.03898v2（arxiv, 2024-02-06）
[^21]: DistiLLM (arXiv:2402.03898v2) — 正文：方法动机与贡献 — https://arxiv.org/html/2402.03898v2（arxiv, 2024-07-03）
[^22]: jongwooko/distillm — Official PyTorch implementation of DistiLLM (ICML 2024) — https://github.com/jongwooko/distillm（github）
[^23]: DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning (Nature 645:633-638) — https://www.nature.com/articles/s41586-025-09422-z（web, 2025-09-17）
[^24]: vLLM Supported Models (official documentation) — https://docs.vllm.ai/en/latest/models/supported_models.html（web）
[^25]: jbrcoleman/eks-deepseek-model-inference — https://github.com/jbrcoleman/eks-deepseek-model-inference（github）
[^26]: ai-xcode/vllm-blackwell-nvfp4 — https://github.com/ai-xcode/vllm-blackwell-nvfp4（github）
[^27]: SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training — https://arxiv.org/abs/2501.17161v2（arxiv, 2025-01-28）
[^28]: Scaling Test-Time Compute Without Verification or RL is Suboptimal — https://arxiv.org/abs/2502.12118v2（arxiv, 2025-02-17）
[^29]: Distillation Scaling Laws — https://arxiv.org/abs/2502.08606v2（arxiv, 2025-02-12）
[^30]: A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility — https://arxiv.org/abs/2504.07086v2（arxiv, 2025-04-09）
[^31]: Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? — https://arxiv.org/abs/2504.13837v5（arxiv, 2025-04-18）
[^32]: s1: Simple test-time scaling — https://arxiv.org/abs/2501.19393v3（arxiv, 2025-01-31）
[^33]: On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (arXiv:2306.13649v3) — full text — https://arxiv.org/html/2306.13649v3（arxiv, 2023-06-23）