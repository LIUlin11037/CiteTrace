## 摘要

大模型知识蒸馏的主流做法已从「在固定数据集上最小化 forward KL」演进为三条并行的技术路线：其一，以 reverse KL / generalized JSD 替代 forward KL 并配合 on-policy 学生自生成样本，缓解训练-推理分布不匹配 [1][2]；其二，以 skew KL 加自适应 off-policy 调度降低自生成样本带来的算力开销 [4]；其三，在数据侧用强模型生成的推理链（rationale / trace）做序列级 SFT 式蒸馏，把「能力迁移」而非「分布拟合」作为主要目标 [3][24][25]。DeepSeek-R1 走的是第三条路线的极端形态：不做 logit 级蒸馏，而是用 R1 生成的样本微调开源底座 [7]，6 个 dense 蒸馏模型覆盖 1.5B–70B [6]，其中 Qwen2.5-32B 底座上「直接蒸馏优于在该底座上直接做 RL」[6]，DeepSeek-R1-Distill-Qwen-32B 在 AIME 2024 达 72.6%、MATH-500 达 94.3%、LiveCodeBench 达 57.2%，与 o1-mini 相当 [6][7]。但蒸馏训练数据规模与 SFT 超参在现有证据中均无出处，工程热度证据亦严重不足，相关结论须谨慎。

## 一、大模型知识蒸馏的主流做法

### 1.1 问题设定与两条传统路线

LLM 蒸馏的既有格局可先用一组二分描述：早期工作把 KD 的适用对象分为白盒分类模型与黑盒模型 API（如 ChatGPT），并指出把白盒 LLM 的知识蒸馏进小模型仍属欠探索 [1]。在自回归序列模型上，既有方法被明确表述为两条路线——要么需要 teacher 生成固定的输出序列集合（成本高），要么使用固定数据集并由 teacher 赋予 token 级概率标注 [2]。前者即 SeqKD（用 teacher 生成的序列），后者即 supervised KD（用 ground-truth 序列 + teacher token 概率），二者共同的缺陷是：训练时所见序列与 student 推理时自回归生成的序列之间存在分布不匹配 [2]。综述层面，LLM 时代的 KD 被归纳为三类作用——以 GPT-4 等专有模型为 teacher 提升开源模型能力、传统压缩、以及用模型自身作 teacher 的自我改进——并以 algorithm、skill、verticalization 三个支柱组织文献 [3]。

### 1.2 目标函数：从 forward KL 到 reverse KL 与 JSD(β)

标准 KD 使用 forward Kullback-Leibler divergence（KLD）作为目标 [1]。其问题在于 student 的表达能力可能不足以拟合 teacher 分布，从而生成 teacher 不太可能生成的样本 [2]。MiniLLM 的做法是把 forward KLD 替换为 reverse KLD，理由是防止 student 高估 teacher 分布中的低概率区域 [1]。GKD 则给出更一般的族：允许在 student 与 teacher 之间使用替代损失，包括 reverse KL 与 generalized JSD(β)，其中 JSD(β) 在 0<β<1 之间插值 forward 与 reverse KL，且其梯度在 β 接近 0 与接近 1 时分别表现得类似 forward KL 与 reverse KL [2]。序列级散度的定义被显式写作各 token 位置散度的平均：D(p_T||p_S)(y|x) = (1/L_y) Σ_n D(p_T(·|y_<n,x) || p_S(·|y_<n,x)) [2]。DistiLLM 进一步指出，自回归序列模型的现有 KD 方法缺少标准化目标函数，并提出 skew Kullback-Leibler divergence 损失及其理论性质 [4]。

### 1.3 数据来源：on-policy 学生自生成样本

GKD 把自回归序列模型的 KD 建模为「带交互式专家的模仿学习」问题 [2]，并区分三类训练数据来源：supervised KD 与 FT 使用 ground-truth 输出序列；SeqKD 使用 teacher 生成的输出序列；on-policy GKD 使用从 student 采样的输出序列 [2]。MiniLLM 同样提出 on-policy 优化方法来学习 reverse KLD 目标，学生模型覆盖 120M 到 13B 参数 [1]。效果上，相对初始 student 的性能提升，on-policy GKD 在摘要任务上为基线 KD 方法的 2.1 倍、机器翻译 1.7 倍、算术推理 1.9 倍；task-agnostic 蒸馏在 BBH 与 MMLU 上分别取得 2% 与 1% 绝对准确率提升 [2]。MiniLLM 在 instruction-following 设定下报告了精确性、整体质量、更低 exposure bias、更好校准与长文本生成上的优势 [1]。

### 1.4 效率修正：自适应 off-policy

on-policy 路线自身带来新成本：DistiLLM 明确指出，近期使用学生自生成输出以缓解训练-推理不匹配的做法显著抬高了计算成本 [4]。其对策包含两部分——(1) 新的 skew KL 损失并揭示与利用其理论性质；(2) 一种自适应 off-policy 方法，用于提升学生自生成输出的利用效率 [4]。在包含 instruction-following 的实验中，DistiLLM 相比近期 KD 方法最高实现 4.3 倍加速，同时构建出高性能学生模型 [4]。

### 1.5 数据侧路线：rationale 蒸馏与自生成指令

与上述「分布对齐」路线并列的是「数据构造」路线。Distilling Step-by-Step 的核心是从 LLM 中抽取 rationale（推理链）作为额外监督信号，在 multi-task 框架下训练小模型，即同时学习「预测标签」与「生成理由」两个任务 [24]；在 4 个 NLP 基准上，相比标准 finetuning 与标准 distillation，该机制用显著更少的标注/无标注训练样本即取得更好性能，且用体积小得多的模型即超过 few-shot prompting 的大模型 [24]。更早的 Self-Instruct 则确立了「自生成 + 筛选」范式：由语言模型自举生成 instruction、input、output 样本，过滤掉无效或高度相似的样本后用于微调原模型，被作者描述为「几乎无需标注」的对齐方法 [25]。综述把这一交织关系表述为：数据增强与 KD 相互交织，用 DA 生成富含上下文的、技能特定的训练数据，使开源模型逼近专有模型的行为；其关键词包含 supervised fine-tuning，即把 SFT 式（序列级）蒸馏纳入 LLM KD 框架内讨论 [3]。

## 二、DeepSeek-R1 蒸馏小模型：路线与效果

### 2.1 路线依据：蒸馏与纯 RL 并列

R1 论文主张 LLM 的推理能力可通过纯强化学习激发，无需人工标注的推理轨迹，且自我反思、验证、动态策略调整等高级推理模式自发涌现 [5]。论文同时明确指出：大模型涌现出的推理模式可被系统地用于指导和增强更小模型的推理能力 [5]，并把蒸馏作为独立于纯 RL 的另一条能力迁移路径，与 R1-Zero 的 RL 路线并列呈现 [11]。这一区分有第三方支撑：有研究用大 k 值 pass@k 系统探测 RLVR 的推理边界，发现当前训练设置并未引出根本性的新推理模式，RLVR 模型在小 k（如 k=1）时优于 base model，但 k 较大时 base model 的 pass@k 更高，观察到的推理能力源自 base model 并受其上限约束；该论文明确对比指出，蒸馏可以从 teacher 引入新的推理模式，真正扩展模型的推理能力 [12]。

### 2.2 具体做法：用 R1 生成样本微调开源底座

官方 README 明确说明：DeepSeek-R1-Distill 系列是在开源模型基础上微调，使用的是 DeepSeek-R1 生成的样本 [7]。这与 1.5 节的序列级 SFT 式蒸馏同构，而非 logit 级分布对齐。底座对应关系为：DeepSeek-R1-Distill-Qwen-7B 基于 Qwen2.5-Math-7B；DeepSeek-R1-Distill-Qwen-32B 基于 Qwen2.5-32B；DeepSeek-R1-Distill-Llama-8B 基于 Llama-3.1-8B（README 后文注明为 Llama3.1-8B-Base）；DeepSeek-R1-Distill-Llama-70B 基于 Llama-3.3-70B-Instruct [7]。论文开源了 6 个 dense 蒸馏模型，规模为 1.5B、7B、8B、14B、32B、70B，底座来自 Qwen 与 Llama 两个系列 [6]。

### 2.3 效果：蒸馏模型评测数据

评测配置为：所有模型最大生成长度 32,768 tokens；需要采样的 benchmark 使用 temperature 0.6、top-p 0.95，每问题生成 64 个回答以估计 pass@1 [7]。蒸馏模型成绩（AIME 2024 pass@1 / cons@64 / MATH-500 pass@1 / GPQA Diamond pass@1 / LiveCodeBench pass@1 / CodeForces rating）如下 [7]：

- Qwen-1.5B：28.9 / 52.7 / 83.9 / 33.8 / 16.9 / 954
- Qwen-7B：55.5 / 83.3 / 92.8 / 49.1 / 37.6 / 1189
- Qwen-14B：69.7 / 80.0 / 93.9 / 59.1 / 53.1 / 1481
- Qwen-32B：72.6 / 83.3 / 94.3 / 62.1 / 57.2 / 1691
- Llama-8B：50.4 / 80.0 / 89.1 / 49.0 / 39.6 / 1205
- Llama-70B：70.0 / 86.7 / 94.5 / 65.2 / 57.5 / 1633

对照基线为：o1-mini 63.6 / 80.0 / 90.0 / 60.0 / 53.8 / 1820；QwQ-32B-Preview 44.0 / 60.0 / 90.6 / 54.5 / 41.9 / 1316；GPT-4o-0513 9.3 / 13.4 / 74.6 / 49.9 / 32.9 / 759；Claude-3.5-Sonnet-1022 16.0 / 26.7 / 78.3 / 65.0 / 38.9 / 717 [7]。教师模型 DeepSeek-R1 自身对应分数为 AIME 2024 Pass@1 79.8、MATH-500 97.3、GPQA-Diamond 71.5、LiveCodeBench (Pass@1-COT) 65.9、Codeforces Rating 2029（Percentile 96.3）[7]；论文正文另给出 GPQA Diamond 71.5、MMLU 90.8、MMLU-Pro 84.0，以及 Codeforces 2029 Elo 超过 96.3% 人类参赛者 [6]。R1 与 R1-Zero 均为 671B 总参数、37B 激活参数的 MoE 模型，上下文长度 128K [7]。

论文正文给出的关键结论包括：以 Qwen2.5-32B 为底座时，直接从 DeepSeek-R1 蒸馏的效果优于在该底座上直接做 RL [6]；DeepSeek-R1-Distill-Qwen-7B 在 AIME 2024 上达到 55.5%，超过 QwQ-32B-Preview [6]；DeepSeek-R1-Distill-Qwen-32B 在 AIME 2024 得 72.6%、MATH-500 得 94.3%、LiveCodeBench 得 57.2%，显著超过此前开源模型，与 o1-mini 相当 [6]。作为 RL 路线的参照，R1-Zero 在 AIME 2024 上 pass@1 从 15.6% 提升到 71.0%，配合多数投票（cons@64）进一步提升到 86.7% [6]；另一处记录为 pass@1 从 15.6% 升至 77.9%，self-consistency 解码下达 86.7% [11]。R1-Zero 的 RL 配置为学习率 3e-6、KL 系数 0.001、rollout 采样温度 1，每题采样 16 个输出，8.2k step 前最大长度 32,768 tokens、之后 65,536 tokens，共训练 10,400 steps（约 1.6 个 epoch），每步 32 个不同问题、batch size 512，每 400 步用最新策略模型替换 reference model [11][14]。

### 2.4 第三方复现与生态

open-r1 把 R1 pipeline 拆成三步复现，其中 Step 1 即用从 DeepSeek-R1 蒸馏的高质量语料复现 R1-Distill 模型 [9]。2025/05/26 宣布 Step 1 完成，发布 Mixture-of-Thoughts——一个从 R1 蒸馏的 350k 条已验证推理轨迹数据集，覆盖数学、代码与科学，并提供训练 OpenR1-Distill-7B 的 recipe，称其复现了 deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 的推理能力 [9]。更早的 2025/02/10 发布 OpenR1-Math-220k（220k 条从 R1 蒸馏的 trace，基于新版 NuminaMath），README 称在该数据集上训练的模型与 DeepSeek 的蒸馏模型性能相当 [9]。2025/03/11 发布 CodeForces-CoTs（10k 竞赛编程题、100k 条从 R1 蒸馏的解）与 IOI24 基准，README 称在 CodeForces-CoTs 上训练的 7B Qwen 模型可在 IOI24 上超过 Claude 3.7 Sonnet，32B 模型可超过 R1 本身 [9]。README 另给出把蒸馏 recipe 迁移到更小 base model 的示例：以 Qwen/Qwen3-0.6B-Base 为底座训练 OpenR1-Distill-0.6B [9]。需注意该仓库已声明不再维护，并指引 GRPO/SFT/蒸馏训练迁移到 TRL 库 [9]。

与蒸馏路线形成对照的是小模型 RL 的可行性边界：TinyZero 在 countdown 与 multiplication 任务上复现 R1-Zero，README 称通过 RL，3B base LM 能自行发展出自我验证与搜索能力，成本低于 30 美元；同时明确记录规模边界——单 GPU 配置适用于 ≤1.5B 模型，但对 Qwen2.5-0.5B base「已知无法学会推理」，3B+ 模型才能发展出复杂推理技能 [10]。simpleRL-reason 从 Qwen2.5-Math-7B base 直接做 RL，无 SFT、无 reward model，仅用 MATH 数据集的 8K 样本，其 SimpleRL-Zero 平均 48.8 对比 base 的 30.2；同数据 SFT 对照（8K MATH SFT）AIME 2024 仅 3.3、平均 26.5，低于 base 的 30.2 [20]。1-shot RLVR 进一步显示，仅用一条训练样本即可在 Qwen2.5-Math-1.5B 上把 MATH500 从 36.0% 提升到 73.6%，并在包括 DeepSeek-R1-Distill-Qwen-1.5B 在内的模型上观察到类似大幅提升 [21]。这些结果说明「小模型 + RL」并非不可行，但任务复杂度与底座规模存在明确门槛 [10]。

厂商侧的公开表述较为间接：Qwen3 技术报告摘要写明「借助旗舰模型的知识，显著降低了构建小规模模型所需的算力，同时保证其性能具有高度竞争力」，即把「用强模型知识构建小模型」作为既定路线写入摘要 [22]；Qwen3 系列覆盖 0.6B–235B，全部以 Apache 2.0 公开权重 [22][23]，且 dense base 模型在更低参数规模上达到上一代更高参数规模的性能 [22]。但 QwenLM/Qwen3 README 未在抓取到的正文中给出「蒸馏 vs 直接 RL」的路线对比表述或相关实验数据，其 Training 一节标注后训练（含 SFT 与 RLHF）复现说明在抓取时仍为 TODO [23]。

## 矛盾与不确定性

**（1）蒸馏训练数据规模与 SFT 超参无出处。** 常被引用的「约 80 万条 R1 生成样本」这一数字在本次 25 条证据中无任何一条给出出处；相反，证据明确记录「论文正文中未在此处给出蒸馏 SFT 阶段的对应超参数」[14]。因此本报告不采用该数字，蒸馏阶段的学习率、epoch、batch、max length 等均属**现有公开资料不足以确认**。可确认的仅是 README 层面的定性描述——蒸馏模型使用 DeepSeek-R1 生成的样本微调 [7]。

**（2）工程落地热度无法从现有证据推断。** 热度侧证据仅来自 4 个 1–258 星的小型个人仓库（r1-ktransformers-guide 258 星、last push 2025-03-03 [15]；Deepseek-R1-qwen1.5B 29 星、2025-01-24 [16]；deepseek-server 2 星、2025-02-24 [17]；IA_2025 1 星、2025-02-08 [18]；DeepSeekQwen_AIO 1 星、2025-01-25 [19]），且均停在 2025 年 1–3 月。核心的 deepseek-ai/DeepSeek-R1 官方仓库只取了 README 内容，未记录 stars/commits/最近提交时间 [7]；unsloth 虽显示 12,050 次 commits 并声称微调「2× faster with 70% less VRAM with no accuracy loss」（属厂商自述性能主张，需第三方验证）[13]，但未覆盖 unsloth DeepSeek-R1-GGUF 的 HuggingFace 下载量。以这些样本推断「工程落地热度」会严重失真。

**（3）「蒸馏 vs 小模型 RL」的厂商表态与争议缺乏证据。** 仅有 Qwen3 技术报告一句「leveraging the knowledge from the flagship models」的间接表述 [22]，且 QwenLM/Qwen3 README 明确未给出路线对比 [23]。Llama/Meta 侧表态、以及第三方独立 benchmark 对 R1-Distill 分数的复现验证完全缺失——open-r1 的「性能相当」「复现推理能力」均为项目自述 [9]，不构成独立验证。因此「2025–2026 年存在厂商路线争议」这一判断**现有公开资料不足以确认**。

**（4）R1-Zero 分数在证据间存在口径差异。** 一处记录 pass@1 从 15.6% 到 71.0%、cons@64 达 86.7% [6]，另一处记录升至 77.9%、16 样本自洽解码达 86.7% [8][11]。差异可能源于版本或评测口径，但证据未给出解释。另需注意 alphaXiv 页面指出：GRPO 与 PPO 的对比实验使用的是 MATH 数据集与 DeepSeek-Coder-V2-Lite（16B 总参数、2.4B 激活参数），并非 671B 的 DeepSeek-R1-Zero [8]——引用该类对比时不可张冠李戴。论文版本方面，v1 于 2025-01-22 提交、v2 于 2026-01-04 修订，且已发表于 Nature（DOI 10.1038/s41586-025-09422-z）[8][11]，引用具体数字时应标注版本。

**（5）蒸馏「扩展能力」的结论依赖单一来源。** 「蒸馏可引入新推理模式、真正扩展推理能力」这一判断出自一篇 RLVR 批判性研究 [12]，其对照论证有力但仍是单一论文的主张；本报告未找到第二个独立来源对该结论做交叉验证。

## 参考来源

[^1]: MiniLLM: On-Policy Distillation of Large Language Models — https://arxiv.org/abs/2306.08543v6（arxiv, 2023-06-14）
[^2]: On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD) — https://arxiv.org/abs/2306.13649v3（arxiv, 2023-06-23）
[^3]: A Survey on Knowledge Distillation of Large Language Models — https://arxiv.org/abs/2402.13116v4（arxiv, 2024-02-20）
[^4]: DistiLLM: Towards Streamlined Distillation for Large Language Models — https://arxiv.org/abs/2402.03898v2（arxiv, 2024-02-06）
[^5]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning — https://arxiv.org/abs/2501.12948v2（arxiv, 2025-01-22）
[^6]: DeepSeek-R1 论文正文（ar5iv HTML 版）：引言中的蒸馏结论与 DeepSeek-R1 主模型分数 — https://ar5iv.labs.arxiv.org/html/2501.12948（arxiv, 2025-01-22）
[^7]: deepseek-ai/DeepSeek-R1 官方仓库 README（模型清单与蒸馏模型评测表） — https://raw.githubusercontent.com/deepseek-ai/DeepSeek-R1/main/README.md（github）
[^8]: alphaXiv 论文页（arXiv:2501.12948）——版本与发表信息 — https://www.alphaxiv.org/abs/2501.12948（web）
[^9]: huggingface/open-r1 — A fully open reproduction of DeepSeek-R1 — https://github.com/huggingface/open-r1（github, 2025-05-26）
[^10]: Jiayi-Pan/TinyZero — Minimal reproduction of DeepSeek R1-Zero — https://github.com/Jiayi-Pan/TinyZero（github）
[^11]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948v2) — https://arxiv.org/abs/2501.12948（arxiv, 2025-01-22）
[^12]: Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (arXiv:2504.13837v5) — https://arxiv.org/abs/2504.13837（arxiv, 2025-04-18）
[^13]: unslothai/unsloth — README（Finetune/Run 能力与支持平台） — https://github.com/unslothai/unsloth（github）
[^14]: DeepSeek-R1 (arXiv HTML v2，第 2 节 DeepSeek-R1-Zero 训练配置) — https://arxiv.org/html/2501.12948v2（arxiv, 2025-01-22）
[^15]: ubergarm/r1-ktransformers-guide — https://github.com/ubergarm/r1-ktransformers-guide（github）
[^16]: fabiomatricardi/Deepseek-R1-qwen1.5B — https://github.com/fabiomatricardi/Deepseek-R1-qwen1.5B（github）
[^17]: rybruscoe/deepseek-server — https://github.com/rybruscoe/deepseek-server（github）
[^18]: Elixx007/IA_2025 — https://github.com/Elixx007/IA_2025（github）
[^19]: fabiomatricardi/DeepSeekQwen_AIO — https://github.com/fabiomatricardi/DeepSeekQwen_AIO（github）
[^20]: simpleRL-reason (hkust-nlp/simpleRL-reason) — README — https://github.com/hkust-nlp/simpleRL-reason（github, 2025-01-25）
[^21]: Reinforcement Learning for Reasoning in Large Language Models with One Training Example (1-shot RLVR) — https://arxiv.org/abs/2504.20571（arxiv, 2025-04-29）
[^22]: Qwen3 Technical Report — https://arxiv.org/abs/2505.09388（arxiv, 2025-05-14）
[^23]: QwenLM/Qwen3 (official GitHub repository README) — https://github.com/QwenLM/Qwen3（github）
[^24]: Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes — https://arxiv.org/abs/2305.02301v2（arxiv, 2023-05-03）
[^25]: Self-Instruct: Aligning Language Models with Self-Generated Instructions — https://arxiv.org/abs/2212.10560v2（arxiv, 2022-12-20）