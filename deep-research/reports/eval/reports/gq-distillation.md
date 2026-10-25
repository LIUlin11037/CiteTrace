## 摘要

大模型知识蒸馏（KD）的主流做法在 2023 年后分化为两条主线：一条延续 Hinton 等人奠定的「教师输出/中间表征/关系」三类知识划分[^1][^2]，在自回归序列模型上依次演化出序列级蒸馏[^3]、on-policy 蒸馏（GKD、MiniLLM）[^4][^6]与改进散度目标的 DistiLLM[^7]；另一条是「蒸馏推理轨迹」——把大模型生成的思维链（rationale）当作额外监督信号微调小模型，代表工作为 Distilling Step-by-Step[^20]，其思路与 STaR 的自我 bootstrapping 不同，后者不依赖外部教师[^5]。DeepSeek-R1 走的是第二条路线：用 R1 生成的样本对 Qwen/Llama 系列底座做微调，开源 6 个 1.5B–70B 的 dense 模型[^9][^12]。效果上，Distill-Qwen-32B 在 AIME 2024 达 72.6%、MATH-500 达 94.3%、LiveCodeBench 达 57.2%，官方称与 o1-mini 相当并超过之[^9][^12]；Distill-Qwen-7B 在 AIME 2024 达 55.5%，超过 QwQ-32B-Preview[^9]。但「蒸馏优于在小模型上直接 RL」这一核心主张目前只有 DeepSeek 自身的论文与 README 支撑，缺少同底座、同基准的独立第三方对照，是本报告最大的不确定性。

## 一、知识蒸馏的标准定义与知识类型划分

知识蒸馏的标准定义可拆为三要素。按 Gou 等人的综述，一个 KD 系统由 knowledge（知识）、distillation algorithm（蒸馏算法）、teacher-student architecture（师生架构）三部分组成，其目标表述为「从一个大的教师模型有效地学习一个小学生模型」[^2]。该综述把教师知识分为三类：response-based knowledge，即教师模型最后一层输出层的神经响应，核心思想是让学生直接模仿教师的最终预测，综述评价其「简单但有效」[^2]；feature-based knowledge，即中间层的激活、神经元或特征[^2]；relation-based knowledge，即不同激活之间、神经元之间或样本对之间的关系[^2]。此外教师模型的参数或层间连接构成另一类知识[^2]。vanilla KD 的具体做法是用大模型的 logits 作为教师知识[^2]。

这一框架的源头是 Hinton、Vinyals、Dean 的工作（arXiv:1503.02531，标注为 NIPS 2014 Deep Learning Workshop），其动机是把集成或多个大模型中的知识压缩进单个更易部署的模型以降低部署计算开销，并在 MNIST 上报告「surprising results」，还把商用系统声学模型的集成蒸馏进单一模型取得显著改进[^1]。综述把 KD 的思想更早追溯到 Bucilua 等人 2006 年的 model compression 与 Urner 等人 2011 年用无标注数据做师生知识迁移，而「从小模型学大模型」由 Hinton 等人正式普及化[^2]。

## 二、面向自回归语言模型的蒸馏方法演进

**序列级蒸馏。** Kim 与 Rush 把此前在其它领域成功的 KD 方法应用到神经机器翻译的模型压缩，证明标准 KD 用在词级预测上对 NMT 有效，并进一步提出两种序列级 KD 变体[^3]。其报告的反直觉发现是：序列级蒸馏似乎消除了对 beam search 的需求，即使应用在原始教师模型上也是如此；最佳学生模型比 SOTA 教师快 10 倍而性能损失很小，相比不用 KD 的基线在 greedy decoding / beam search 下分别提升 4.2 / 1.7 BLEU，再叠加权重剪枝可得到参数量比教师少 13 倍、BLEU 仅降 0.4 的学生[^3]。

**分布不匹配与 on-policy 化。** GKD 指出自回归序列模型的 KD 存在分布不匹配：训练时学生见到的输出序列与推理时学生自己生成的序列不一致；其解法是不再只依赖一组固定输出序列，而是让学生在自生成的序列上训练并利用教师对这些序列的反馈，且允许师生之间采用替代损失函数，在学生表达能力不足以模仿教师分布时有用，还可与 RLHF 无缝结合，在摘要、翻译、算术推理与指令微调任务上验证[^4]。MiniLLM 走的是另一条修正路径：把标准 KD 的 forward KL 目标替换为 reverse KL，理由是 reverse KL 更适合生成式语言模型蒸馏、可防止学生高估教师分布中的低概率区域，并为该目标推导了 on-policy 优化方法；在 instruction-following 设定下，MiniLLM 相比基线生成更精确的回复、整体质量更高、exposure bias 更低、校准更好、长文本生成更强，方法可扩展到 120M 至 13B 参数[^6]。

**目标函数与效率。** DistiLLM 指出自回归序列模型的现有 KD 方法缺少标准化目标函数，且近期使用学生自生成输出虽缓解训练-推理不匹配却显著抬高计算成本；其方案由 skew Kullback-Leibler divergence 损失与提升学生自生成输出利用效率的 adaptive off-policy 方法两部分组成，在 instruction-following 实验中取得更优学生模型，同时相比近期 KD 方法最高 4.3 倍加速[^7]。

**蒸馏推理轨迹（rationale 蒸馏）。** Distilling Step-by-Step 明确把自身与常规蒸馏区分开：常规蒸馏把 LLM 视为噪声标签来源，该文改为把 LLM 视为能推理的 agent，蒸馏的不只是标签还有教师的 rationale[^20]。其机制分两步：先用 CoT prompting 让 LLM 对无标注数据生成标签及为其辩护的自然语言 rationale，再用这些 rationale 连同任务标签在多任务框架下训练下游小模型（输入前加 task prefix，使模型同时承担标签预测与 rationale 预测）[^20]。论文指出知识蒸馏的一个常见局限是依赖大量无标注数据来构造有用的噪声训练集，而蒸馏 rationale 可降低这一数据需求；实验覆盖 4 个 NLP benchmark，相比 finetuning 与 distillation 平均减少 50% 以上训练样本（最多减少 85% 以上）仍取得更好性能，770M T5 仅用某 benchmark 80% 数据即超过 few-shot prompted 540B PaLM，仅有未标注数据时 11B T5 即可达到或超过 540B PaLM[^20]。该文发表于 Findings of ACL 2023，代码开源于 github.com/google-research/distilling-step-by-step[^20]。

需要区分的是，STaR 虽然同样使用 rationale 微调，但其机制是模型从自己生成的推理中自我改进（self-training / rationale bootstrapping），并非从外部教师模型蒸馏[^5]；Distilling Step-by-Step 也把 STaR 归入「把推理步骤作为额外微调数据以 self-improve LLM」的一类工作，并指出这类工作改进的仍是 LLM 本身，受其庞大体积限制[^20]。Automate-CoT 则属于 prompt 工程侧工作——从少量标注数据自动增广并筛选 rationale chains 以构造 CoT prompt，而非把教师 rationale 蒸馏进学生模型参数[^21]。

**一个反复出现的经验限制。** 综述指出实证上「更大的模型未必是更好的教师」，原因是 model capacity gap[^2]。

## 三、DeepSeek-R1 的蒸馏做法

DeepSeek-R1 论文表明 LLM 的推理能力可通过纯强化学习激发，无需人工标注的推理轨迹，该 RL 框架使自我反思、验证、动态策略调整等高级推理模式自发涌现[^8]。作为对照的蒸馏路线则把这一能力迁移到小模型：官方仓库开源 6 个从 DeepSeek-R1 蒸馏的 dense 模型，即 Qwen-1.5B、Qwen-7B、Llama-8B、Qwen-14B、Qwen-32B、Llama-70B，学生底座分别为 Qwen2.5-Math-1.5B、Qwen2.5-Math-7B、Llama-3.1-8B、Qwen2.5-14B、Qwen2.5-32B、Llama-3.3-70B-Instruct[^12]。蒸馏方式表述为：用 DeepSeek-R1 生成的样本对开源模型做微调，并小幅修改 config 与 tokenizer[^12]。论文层面确认开源了这 6 个规模的 dense 模型、底座基于 Qwen 与 Llama 系列，且蒸馏的教师模型是 DeepSeek-R1 完整版（含 cold-start + 多阶段训练），而非纯 RL 的 DeepSeek-R1-Zero[^9]。R1 主模型为 671B 总参数 / 37B 激活参数的 MoE，基于 DeepSeek-V3-Base，上下文 128K[^12]；Nature 版本确认 R1 为整合 rejection sampling、RL 与 supervised fine-tuning 的多阶段学习框架[^15]。

从方法分类看，这属于第二节所述的「rationale / 样本蒸馏 + SFT」路线，而非 logits 级 KD 或 on-policy 蒸馏。官方主张大模型的推理模式可以被蒸馏进小模型，且效果优于在小模型上直接做 RL 所发现的推理模式[^12]；论文给出的依据是：以 Qwen2.5-32B 为底座时，从 DeepSeek-R1 直接蒸馏的效果优于在该底座上直接做 RL，作者据此认为大模型发现的推理模式对小模型推理能力提升至关重要[^9]。

社区侧，huggingface/open-r1 定位为 R1 的完全开源复现，三步计划为：用从 R1 蒸馏的高质量语料复现 R1-Distill 模型、复现 R1-Zero 的纯 RL pipeline、打通 base model 经多阶段训练到 RL 调优[^13]。其 2025/05/26 宣布 Step 1 完成，发布 Mixture-of-Thoughts（350k 条经校验的 R1 蒸馏推理轨迹，覆盖数学、代码、科学）并提供 OpenR1-Distill-7B 训练配方，称其复现了 DeepSeek-R1-Distill-Qwen-7B 的推理能力[^13][^16]；此前 2025/02/10 发布 OpenR1-Math-220k（220k 条基于新版 NuminaMath 的 R1 蒸馏轨迹），称在该数据上训练的模型达到 DeepSeek 蒸馏模型的表现[^13]。SFT 蒸馏配方示例为：底座 open-r1/Qwen2.5-Math-7B-RoPE-300k、数据 Mixture-of-Thoughts、lr 4.0e-5、5 epochs、max_seq_length 32768、per_device_batch 2、bf16、Liger kernel、8×H100[^13]。该仓库 README 顶部已声明项目不再维护、新 issue/PR 不再审阅，GRPO、SFT 与蒸馏的训练侧能力迁至 huggingface/trl[^13][^16]。

## 四、蒸馏小模型的效果

官方评测（pass@1，最大生成长度 32768 tokens，采样类基准 temperature 0.6、top-p 0.95、每 query 生成 64 条估计 pass@1）[^12]：

| 学生模型 | AIME 2024 | MATH-500 |
|---|---|---|
| Distill-Qwen-1.5B | 28.9 | 83.9 |
| Distill-Qwen-7B | 55.5 | 92.8 |
| Distill-Qwen-14B | 69.7 | 93.9 |
| Distill-Qwen-32B | 72.6 | 94.3 |
| Distill-Llama-8B | 50.4 | 89.1 |
| Distill-Llama-70B | 70.0 | 94.5 |

（数据出自官方仓库 README[^12]；论文 v1 对 32B 给出 AIME 2024 72.6%、MATH-500 94.3%、LiveCodeBench 57.2%，并称其显著超越此前开源模型并与 o1-mini 相当[^9]。）

关键对照点：

- **小模型超越大模型。** Distill-Qwen-7B 在 AIME 2024 达 55.5%，超过开源模型 QwQ-32B-Preview[^9]；蒸馏的 14B 模型大幅超过当时 SOTA 开源模型 QwQ-32B-Preview，32B 与 70B 模型在 dense 模型范围内刷新推理基准记录[^9]。
- **对 o1-mini。** 官方称 Distill-Qwen-32B 在多项基准上超过 OpenAI-o1-mini（o1-mini AIME2024 63.6、MATH-500 90.0、LiveCodeBench 53.8）[^12]。
- **与教师差距。** 教师 DeepSeek-R1 自身 AIME 2024 为 79.8% Pass@1、MATH-500 97.3%、Codeforces Elo 2029、GPQA Diamond 71.5%、MMLU 90.8%、MMLU-Pro 84.0%[^9]。32B 学生在 AIME 上距教师约 7 个百分点，MATH-500 距教师约 3 个百分点。
- **第三方引用的基线数值。** 后续工作报告 DeepSeek-R1-Distill-Qwen-7B 在 AIME 2025 为 30.0%、Minerva 37.6%[^17]；DeepSeek-R1-Distill-Qwen-1.5B 在五个数学推理基准上平均 39.9%[^18]。这些数值可作为外部工作使用该模型时的参照基线。

## 五、矛盾与不确定性

**1. 「蒸馏优于小模型直接 RL」缺独立验证（核心不确定性）。** 该结论目前仅有 DeepSeek-R1 论文（v1 正文与官方 README）单一来源，属作者自证[^9][^12]。hkust-nlp/simpleRL-reason 只证明小模型纯 RL 可行——在 Qwen2.5-Math-7B base 上仅用 8K 条 MATH 数据做 RL（无 SFT、无奖励模型）得到 AIME 33.3%、AMC 62.5%、MATH 77.2%，平均 50.9，并称与 Eurus-2-7B-PRIME（48.9）、rStar-Math-7B 相当[^14]——但它并未在相同底座、相同基准上与 R1 蒸馏做平行对照。另有工作自述其研究的是 SFT 蒸馏小模型输出冗长、无法判断最优停止点的问题，而非比较「从 R1 蒸馏」与「直接 RL」两条路线的优劣，且明确未在相同底座与基准上把二者作为平行路线做直接对照，因此不构成对该主张的独立验证或反驳[^19]。故「蒸馏 vs RL 对比」目前缺独立第三方证据。

**2. 对 R1 蒸馏范式的批评（是否只模仿表层 CoT、长度泛化）无直接证据。** 现有两条相关证据均在关键点中明确自述不涉及「R1 蒸馏是否仅学到表层 CoT 风格」这一问题[^17][^18]；其中一条只讨论 SFT 蒸馏模型的冗长与停止点问题，并提出 Temperature scaling 控制 thinking 阶段停止点与 trace 长度、TLDR（基于 GRPO 的长度正则化 RL，支持短/中/长多级 trace 长度控制），报告 TLDR 在相对 SFT 基线几乎不损失精度的情况下将 token 效率提升约 50%[^19]，但这不能替代对「长度泛化」批评的独立来源。R1 蒸馏范式在 2025–2026 年产业界的采纳与批评，现有公开资料不足以确认。

**3. 蒸馏数据规模无出处。** 有说法称 R1 蒸馏训练数据约 80 万 SFT 样本，但该数字在现有证据中无任何出处；官方 README 仅表述为「using samples generated by DeepSeek-R1」，未给样本量[^12]。该数字究竟是 R1 全流程的 SFT 数据量还是蒸馏专用数据量，现有公开资料不足以确认。

**4. 版本差异。** DeepSeek-R1 论文首次提交于 2025-01-22（v1），v2 修订于 2026-01-04[^11]；v2 中 R1-Zero 的 AIME 2024 pass@1 为 15.6%→77.9%、自一致性解码 86.7%，并给出完整 RL 配置（学习率 3e-6、KL 系数 0.001、采样温度 1、每题 16 个输出、8.2k step 前最大长度 32,768 token 之后 65,536、共 10,400 step 约 1.6 epoch、batch size 512、每 400 step 替换 reference model）[^10]。引用具体数值时需注意所依据的版本。MiniLLM 的 arXiv 版本亦持续更新至 v6（2026-01-31），说明该工作到 2026 年仍在被维护或修订[^6]。

**5. 蒸馏模型评测口径。** 官方评测使用最大生成长度 32768 tokens、每 query 64 条采样估计 pass@1[^12]，与部分第三方工作采用的评测设置可能不同，跨来源比较 AIME 等基准数值时需谨慎。

## 参考来源

[^1]: Distilling the Knowledge in a Neural Network (Hinton, Vinyals, Dean) — https://arxiv.org/abs/1503.02531（arxiv, 2015-03-09）
[^2]: Knowledge Distillation: A Survey (Gou, Yu, Maybank, Tao) — IJCV 2021 — https://arxiv.org/html/2006.05525v7（arxiv, 2021-05-20）
[^3]: Sequence-Level Knowledge Distillation (Kim & Rush) — EMNLP 2016 — https://arxiv.org/abs/1606.07947（arxiv, 2016-09-22）
[^4]: On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD) — ICLR 2024 — https://arxiv.org/abs/2306.13649（arxiv, 2024-01-17）
[^5]: STaR: Bootstrapping Reasoning With Reasoning (Zelikman, Wu, Mu, Goodman) — https://arxiv.org/abs/2203.14465（arxiv, 2022-05-20）
[^6]: MiniLLM: On-Policy Distillation of Large Language Models — https://arxiv.org/abs/2306.08543（arxiv, 2023-06-14 (v1); v6 2026-01-31）
[^7]: DistiLLM: Towards Streamlined Distillation for Large Language Models — https://arxiv.org/abs/2402.03898（arxiv, 2024-02-06 (v1); v2 2024-07-03）
[^8]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning — https://arxiv.org/abs/2501.12948（arxiv, 2025-01-22 (v1); v2 2026-01-04）
[^9]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (v1, 2025-01-22) — https://arxiv.org/html/2501.12948v1（arxiv, 2025-01-22）
[^10]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (v2, rev. 2026-01-04) — https://arxiv.org/html/2501.12948v2（arxiv, 2025-01-22）
[^11]: DeepSeek-R1 arXiv abstract page (2501.12948v2 metadata) — https://arxiv.org/abs/2501.12948v2（arxiv, 2025-01-22）
[^12]: deepseek-ai/DeepSeek-R1 — 官方仓库 README（蒸馏模型清单与蒸馏主张） — https://github.com/deepseek-ai/DeepSeek-R1（github, 2025-06-27）
[^13]: huggingface/open-r1 — 社区复现仓库 README（三步计划、蒸馏数据集与现状） — https://github.com/huggingface/open-r1（github, 2026-10-02）
[^14]: hkust-nlp/simpleRL-reason — 小模型纯 RL 复现 R1-Zero/R1 的 README — https://github.com/hkust-nlp/simpleRL-reason（github, 2025-12-23）
[^15]: DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning (Nature 645, 633-638) — https://www.nature.com/articles/s41586-025-09422-z（web, 2025 (Nature, volume 645, pages 633-638)）
[^16]: Open R1 README — 停止维护公告与迁移至 TRL 的说明 — https://raw.githubusercontent.com/huggingface/open-r1/main/README.md（github）
[^17]: Reflective Recovery: A Self-Supervised Method for Reasoning by Learning from Mistakes — https://arxiv.org/abs/2609.19156（arxiv, 2026-09）
[^18]: MADA-RL: Multi-Agent Debate-Aware Reinforcement Learning for Parameter-Efficient Reasoning in Compact Models — https://arxiv.org/abs/2607.18006（arxiv, 2026-07）
[^19]: Making Small Language Models Efficient Reasoners: Intervention, Supervision, Reinforcement (arXiv:2505.07961) — https://arxiv.org/abs/2505.07961（arxiv, 2025-05-12）
[^20]: Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes — https://arxiv.org/abs/2305.02301v2（arxiv, 2023-05-03）
[^21]: Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data (Automate-CoT) — https://arxiv.org/abs/2302.12822（arxiv, 2023-02-24）