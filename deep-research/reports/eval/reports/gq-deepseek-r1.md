# DeepSeek-R1 的核心技术路线及其对推理模型训练范式的影响

## 摘要

DeepSeek-R1 的核心主张是：LLM 的推理能力可以由纯强化学习激发，无需人工标注的推理轨迹[^1][^6]。技术路线上，它以 DeepSeek-V3-Base 为基座、以 Group Relative Policy Optimization（GRPO）为优化器，奖励信号只检查最终答案的正确性（accuracy reward）与格式合规（format reward），不对推理过程施加内容约束[^2][^4]。由此得到的 R1-Zero 在 AIME 2024 上 pass@1 从 15.6% 升至 77.9%，配合自一致性解码（cons@16）达 86.7%[^4]。由于 R1-Zero 存在可读性差、语言混杂等问题，正式版 R1 改为「冷启动 SFT → 推理导向 RL → 拒绝采样重建 SFT 数据 → 全场景 RL」的四阶段流程[^3][^17]，并把 R1 的推理轨迹蒸馏到 1.5B–70B 的 Qwen/Llama 稠密模型上[^3][^5]。对训练范式的影响分三层：一是 GRPO + 规则式可验证奖励成为开源推理 RL 的默认起点，被 TRL、OpenRLHF、verl 等框架直接内建[^10][^11][^9]；二是围绕「RL 是扩展还是仅放大基座能力」形成明确争论，怀疑派（大 k 下 base model pass@k 反超）[^14][^16]与乐观派（ProRL 在 pass@k=0 任务上取得突破）[^15]各有实证；三是工程侧出现 DAPO、Open-Reasoner-Zero 等对 GRPO 的修正与替代，并暴露出 R1 论文训练细节不足、复现困难的问题[^29][^8]。

---

## 一、核心技术路线

### 1.1 优化器：GRPO 的完整定义

GRPO（Group Relative Policy Optimization）是 R1 的 RL 框架，论文引用 Shao et al., 2024，提出动机是简化训练流程并降低 PPO（Schulman et al., 2017）的资源消耗[^2]。其算法定义可拆为输入、步骤、产出三部分：

- **输入**：一个问题 $q$、旧策略 $\pi_{\theta_{old}}$、参考策略 $\pi_{ref}$、超参 $\varepsilon$（裁剪范围）与 $\beta$（KL 系数）。
- **步骤**：(1) 对每个 $q$ 从 $\pi_{\theta_{old}}$ 采样一组 $G$ 个输出 $\{o_1,\dots,o_G\}$；(2) 用组内奖励的标准化计算每个样本的优势 $A_i = (r_i - \text{mean}(\{r_1..r_G\})) / \text{std}(\{r_1..r_G\})$（式3）；(3) 最大化目标 $J_{GRPO}(\theta)$（式2），其结构为对组内 $G$ 个样本取平均，每项为 $\min(\text{比率}\times A_i,\ \text{clip}(\text{比率},1-\varepsilon,1+\varepsilon)\times A_i)$ 减去 $\beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$[^2]。
- **产出**：更新后的策略参数 $\theta$。

关键设计在于**没有独立的 value network / critic**：优势完全由同一问题下组内样本的相对奖励给出，因此不需要训练价值函数[^2]。这一点后来成为争议焦点（见 §3.2）。

### 1.2 奖励设计：只验证结果，不约束过程

奖励信号仅基于最终预测与标准答案的正确性，不对推理过程本身施加约束[^2][^4]。具体分为两类基于规则的奖励[^4]：

- **accuracy rewards**：数学要求把最终答案写在 box 内；代码用编译器跑预定义测试用例。
- **format rewards**：约束输出结构（`<think>...</think>` 与 `<answer>...</answer>` 标签）。Nature 版明确说明「我们有意把约束限制在这一结构格式上，避免任何内容层面的偏见」[^18]。

这一选择直接决定了「纯 RL」的可行性：不需要过程标注、不需要 PRM（process reward model），也就不受 PRIME 所指出的过程标签获取成本与 reward hacking 问题困扰[^26]。

### 1.3 R1-Zero：无 SFT 冷启动的纯 RL

R1-Zero 完全依赖 RL，不做 SFT 冷启动[^4]，是「大规模 RL、且不以 SFT 作为前置步骤」的模型[^3]。其训练配置为：学习率 3e-6，KL 系数 0.001，rollout 采样温度 1；每题采样 16 条输出；8.2k 步之前单条最大长度 32,768 tokens，之后为 65,536 tokens；总训练 10,400 步，约 1.6 个 epoch；每步含 32 道不同题目，训练 batch size 为 512；每 400 步用最新策略模型替换 reference model；每次 rollout 生成 8,192 条输出，随机切分为 16 个 mini-batch，仅训练 1 个 inner epoch[^2][^4]。

结果上，AIME 2024 平均 pass@1 从初始 15.6% 提升到 77.9%，cons@16 达 86.7%；性能与响应长度在 8.2k 步处同时出现显著跃升[^4]。训练中还观察到「aha moment」——模型在反思时突然增多使用 "wait" 一词，标志推理模式转变[^4]。论文摘要将自我反思、验证、动态策略适应等列为 RL 框架促成的涌现行为[^1][^6]。

### 1.4 R1：四阶段流程

R1-Zero 的缺陷是明确的：可读性差、语言混杂（单条 CoT 中英混杂），且规则式 RL 阶段只覆盖推理任务，导致写作与开放域 QA 受限[^18]。R1 因此在 RL 之前引入少量冷启动数据与多阶段训练[^3]：

1. 收集数千条（thousands of）冷启动数据，微调 DeepSeek-V3-Base[^3][^17]。
2. 执行与 R1-Zero 相同的 reasoning-oriented RL[^3]。
3. 在 RL 接近收敛时，对 RL checkpoint 做拒绝采样（rejection sampling）生成新的 SFT 数据，并混入 DeepSeek-V3 在写作、事实问答、自我认知等领域的监督数据，然后重新训练 DeepSeek-V3-Base[^3][^17]。
4. 用新数据微调后的 checkpoint 再进行一次 RL，此次纳入所有场景的 prompt，得到 DeepSeek-R1[^3][^17]。

R1 的奖励模型训练集共 66,000 个偏好对；最终偏好分为四个独立判断的平均，仅保留分差 Δ>1 的样本对，并控制 chosen/rejected 长度可比以减少长度偏差[^4]。

### 1.5 蒸馏：大模型发现的推理模式迁移到小模型

蒸馏实验的核心结论是：以 Qwen2.5-32B 为基座，直接从 DeepSeek-R1 蒸馏优于在其上直接做 RL，说明大基座模型发现的推理模式对提升推理能力至关重要[^3][^17]。开源蒸馏模型覆盖 Qwen 与 Llama 系列，规模为 1.5B、7B、8B、14B、32B、70B[^3]，具体基座为 Qwen2.5-Math-1.5B / Qwen2.5-Math-7B / Llama-3.1-8B / Qwen2.5-14B / Qwen2.5-32B / Llama-3.3-70B-Instruct，蒸馏方式是用 R1 生成的样本对开源模型微调，并对其 config 与 tokenizer 做小幅修改[^5]。

效果上，蒸馏 7B 在 AIME 2024 达 55.5%，超过 QwQ-32B-Preview；蒸馏 32B 在 AIME 2024 得 72.6%、MATH-500 得 94.3%、LiveCodeBench 得 57.2%[^3]。完整分数表（AIME2024 pass@1 / cons@64 / MATH-500 / GPQA-Diamond / LiveCodeBench / CodeForces rating）为：Qwen-1.5B 28.9/52.7/83.9/33.8/16.9/954；Qwen-7B 55.5/83.3/92.8/49.1/37.6/1189；Qwen-14B 69.7/80.0/93.9/59.1/53.1/1481；Qwen-32B 72.6/83.3/94.3/62.1/57.2/1691；Llama-8B 50.4/80.0/89.1/49.0/39.6/1205；Llama-70B 70.0/86.7/94.5/65.2/57.5/1633[^5]。

### 1.6 主模型规格与评测

R1-Zero 与 R1 均为 671B 总参数、37B 激活参数、128K 上下文长度，均基于 DeepSeek-V3-Base 训练[^5]。R1 的基准分数：MMLU 90.8、MMLU-Redux 92.9、MMLU-Pro 84.0、DROP 92.2、IF-Eval 83.3、GPQA-Diamond 71.5、SimpleQA 30.1、FRAMES 82.5、AlpacaEval2.0 87.6、ArenaHard 92.3；代码类 LiveCodeBench (Pass@1-COT) 65.9、Codeforces percentile 96.3、Codeforces rating 2029、SWE Verified 49.2、Aider-Polyglot 53.3；数学类 AIME 2024 pass@1 79.8、MATH-500 pass@1 97.3、CNMO 2024 pass@1 78.8；中文 CLUEWSC 92.8、C-Eval 91.8、C-SimpleQA 63.7[^5]。对照 OpenAI o1-1217 为 AIME 2024 79.2、MATH-500 96.4、GPQA-Diamond 75.7、LiveCodeBench 63.4、Codeforces rating 2061；DeepSeek V3 为 AIME 2024 39.2、MATH-500 90.2、GPQA-Diamond 59.1、Codeforces rating 1134[^5]。评测设置为所有模型最大生成长度 32,768 tokens，需采样的基准用 temperature 0.6、top-p 0.95，每 query 生成 64 条响应估计 pass@1[^5]。

---

## 二、对推理模型训练范式的影响

### 2.1 把「可验证奖励 + 组相对优势」确立为默认起点

R1 之后，GRPO 与规则式可验证奖励被主流开源框架直接内建，这是最直接的范式扩散证据：

- **TRL** 的 README 直接写明 GRPOTrainer 实现 Group Relative Policy Optimization，「more memory-efficient than PPO and was used to train Deepseek AI's R1」，并提供 `trl.rewards` 中的 `accuracy_reward` 与面向推理模型的 `reasoning_accuracy_reward()`，示例数据集为 `trl-lib/DeepMath-103K`[^10]。
- **OpenRLHF** 支持 PPO、REINFORCE++、GRPO、RLOO，并记录了复现生态：2025/1 HKUST 用其在小模型上复现 R1-Zero 与 R1 的训练；2025/2 LMM-R1 作为其 fork 在多模态任务上复现 R1[^11]。
- **verl** 主张可用几行代码构建 GRPO、PPO 等 RL dataflow，并与 FSDP、Megatron-LM、vLLM、SGLang 集成[^9]；其仓库现位于 verl-project/verl，24,008 stars、4,808 forks，最近 push 2026-10-23[^31]。verl 已成为社区二次开发的基础设施层，衍生出 EasyR1（多模态，★5209）、Search-R1（reasoning 与 search engine calling 交错，★5536）、verl-agent（agent RL，★2399）等项目[^13]。
- **Unsloth** 把 RL、GRPO 与 LoRA/QLoRA、full fine-tuning 并列列为完整支持项[^12]。

换言之，R1 之前「推理 RL」是少数团队的专门流水线，R1 之后它变成了框架的一个 trainer 类。

### 2.2 蒸馏与 RL 的分工被重新界定

R1 论文给出的经验结论是：小模型上「从强模型蒸馏」优于「在弱基座上直接做 RL」[^3][^17]。这一结论被后续工作以两种方式继承与修正：

- **Qwen3** 对小模型采用 strong-to-weak 蒸馏，同时利用 off-policy 与 on-policy 知识迁移，并明确表述来自先进教师模型的蒸馏在性能与训练效率上显著优于强化学习（该结论针对小模型构建路径）[^20]。Qwen3 的后训练本身也沿用「长 CoT 冷启动微调 → 数学与代码聚焦 RL → 推理/非推理数据合并微调 → 通用领域 RL」的多阶段结构[^20]。
- **Open-Reasoner-Zero** 把 ORZ recipe 迁移到 DeepSeek-R1-Distill-Qwen-14B，得到 ORZ-R1-Distill-Qwen-14B：AIME2024 75.2、AIME2025 60.0、MATH500 95.6、GPQA 60.4，超过更大的 DeepSeek-R1-Distill-Qwen-32B（72.6/60.0/94.3/62.1）[^7]。这说明蒸馏产物本身仍是可继续 RL 的起点，而非终点。

同时，怀疑派论文明确把蒸馏与 RL 对立起来：蒸馏可以从教师模型引入新的推理模式、真正扩展模型推理能力，而当前 RLVR 设置并未引出根本性的新推理模式[^14]。这把「蒸馏 vs RL」从工程选择升级为能力边界问题。

### 2.3 对 GRPO 的修正与替代路线

R1 的 GRPO 配置并非终局，围绕它的失效机制出现了多条修正路线：

- **DAPO** 报告其最初的朴素 GRPO 训练在 AIME 上只得到 30 分，低于 DeepSeek 的 RL 结果（47 分）[^29]；DAPO 用四项技术（Clip-Higher 解耦 $\varepsilon_{low}/\varepsilon_{high}$ 以缓解熵坍缩、Dynamic Sampling 过滤准确率为 1 和 0 的 prompt 组、Token-Level Policy Gradient Loss、Overlong Reward Shaping）在 Qwen2.5-32B 上达到 AIME 2024 50 分，超过 DeepSeek-R1-Zero-Qwen-32B 的 47 分，且只用了 50% 的训练步数[^29][^28]。DAPO 还去掉了 KL 惩罚项，理由是长 CoT 推理训练中模型分布可显著偏离初始模型，该限制不必要[^29]。
- **Open-Reasoner-Zero** 明确选择 PPO 而非 GRPO，理由是 learned critic 提供 token 级 value estimation，能识别并压低重复等有害模式（credit assignment），advantage 估计比 GRPO 更稳健；论文直接指出 GRPO 缺少独立 value network 时难以区分「真正正确的回答」与「出现在负面模式（如重复循环）中的回答」，会误导强化信号，导致训练不稳定乃至 collapse[^8]。其做法是 vanilla PPO + GAE（λ=1, γ=1）+ 简单 rule-based reward、完全不用 KL 正则，在 0.5B/1.5B/7B/32B 四档上复现了 R1-Zero 式的 scaling 现象[^8]。
- **PRIME** 走的是另一条路：用 implicit process rewards 仅凭 policy rollouts 与 outcome labels 在线更新 PRM，无需专门奖励模型训练阶段，并称可普遍增强 RLOO、REINFORCE、PPO、GRPO 等多种算法[^26]。这实际上是对 R1「只给结果奖励」选择的一种回应——保留结果奖励的可扩展性，同时补回过程信号。

### 2.4 训练细节不透明成为公认的复现障碍

DAPO 作者指出社区在复现 DeepSeek 结果时遇到类似困难，暗示 R1 论文可能省略了关键训练细节[^29]。这一判断有版本层面的旁证：v1 引言对 R1-Zero 训练时长的表述是模糊的 "After thousands of RL steps"，未在该段给出具体步数[^17]；arXiv abs 页面的 Submission history 仅列出两个版本的时间与文件大小，未提供任何文字性 change log 或修订说明[^17]。具体超参（10,400 步、batch 512、每 400 步重置 reference model 等）是在 v2 的 §2.1 中才完整呈现的[^2][^4]。

### 2.5 从「单模型推理」到「推理预算与模式切换」

R1 把推理能力做成 RL 的直接产物，也把「推理要花多少 token」变成可调项。Qwen3 把 thinking mode 与 non-thinking mode 整合进单一模型，并引入 thinking budget 机制允许用户在推理阶段自适应分配计算资源[^19]；报告观察到增加 thinking token 的 thinking budget 会在多种任务上带来一致的性能提升[^20]。这条线索与 R1 的响应长度随训练步数跃升[^4]是同一现象的两端：训练侧长度是涌现结果，推理侧长度成为可控参数。更早的 test-time compute 研究已给出理论支撑——compute-optimal 地分配推理时算力相比 best-of-N 可将效率提升 4 倍以上，在 FLOPs 匹配下可使小模型超过大 14 倍的模型[^21]；s1 则用 1,000 条数据加 budget forcing 在 Qwen2.5-32B-Instruct 上把 AIME24 从 50% 提到 57%[^22]。

---

## 三、矛盾与不确定性

### 3.1 RL 是扩展推理边界还是仅放大基座能力

这是 R1 范式最核心的未决问题，证据直接对立：

- **怀疑派**：系统探测 RLVR 训练后 LLM 的推理能力边界，覆盖多模型家族、多 RL 算法与数学/代码/视觉推理基准，以大 k 值 pass@k 为指标，发现 RLVR 模型在小 k（如 k=1）上优于 base model，但 k 很大时 base model 取得更高 pass@k；作者结论是当前训练设置并未引出根本性的新推理模式，覆盖度与困惑度分析显示推理能力源自 base model 并受其边界约束；六种主流 RLVR 算法表现相近且都远未充分利用 base model 潜力[^14]。该文发表于 NeurIPS 2025（Oral）[^14]。其机制直觉是：若同一 prompt 的所有 rollout 都未解出任务（pass@K=0），则每个样本同样差，梯度消失[^16]。
- **乐观派**：ProRL 提出包含 KL 散度控制、参考策略重置与多样化任务套件的训练方法，实证显示 RL 训练后模型在广泛 pass@k 评估中一致优于 base model，包括 base model 无论尝试多少次都完全失败（pass@k=0）的场景；推理边界提升与 base model 任务胜任度及训练时长强相关[^15]。另有工作报告在两阶段奖励调度下，经长时间近零奖励平台期后出现类 grokking 相变[^16]。

值得注意的是，怀疑派论文本身承认蒸馏可以引入新的推理模式[^14]，这与 R1 论文「大模型推理模式可系统性用于指导小模型」的主张[^1]并不冲突——冲突点在于「纯 RL 能否做到同样的事」。现有公开资料不足以在这一争论上给出定论；社区提出的判别标准是专门研究「hard frontier」（pass@K=0 的问题族），并要求投稿提供 pass@k 改进或 pass@k 从 0 到 1 的直接证据（K 通常 ≥128）[^16]。

### 3.2 GRPO 是否足够稳健

R1 用 GRPO 取得了结果[^2][^4]，但多个独立工作报告朴素 GRPO 复现失败或表现显著偏低：DAPO 的朴素 GRPO 基线在 AIME 上只有 30 分[^29]；Open-Reasoner-Zero 从机制上论证 GRPO 缺少独立 value network 会导致 credit assignment 失效与训练 collapse[^8]。这与 R1 论文将 GRPO 描述为「简化流程、降低资源消耗」的正面表述[^2]并不矛盾，但说明 R1 的成功可能依赖论文未完全披露的实现细节[^29]。现有公开资料不足以确认 R1 的 GRPO 实现与开源复现之间的具体差异。

### 3.3 版本与数值的不一致

同一篇论文的不同版本存在数值差异：v1 引言称 R1-Zero 在 AIME 2024 上 pass@1 从 15.6% 升至 71.0%，多数投票后达 86.7%[^3][^17]；v2 则称平均 pass@1 从 15.6% 提升到 77.9%，cons@16 达 86.7%[^4]。抓取到的 Nature 版正文在 "jumping from an initial value of" 处被截断，未能读到该版本的具体数值与训练步数原文[^18]，因此无法判断 Nature 版对应哪一组数字。此外，v1 摘要称 R1 "achieves performance comparable to OpenAI-o1-1217 on reasoning tasks"[^17]，而 README 的对照表中 R1 与 o1-1217 互有胜负（如 AIME 2024 79.8 vs 79.2、Codeforces rating 2029 vs 2061）[^5]，「comparable」的强度需按具体基准解读。

### 3.4 官方与第三方的自我主张需交叉验证

以下均为发布方自述，尚未在给定证据内获得独立验证：README 称 DeepSeek-R1-Distill-Qwen-32B 在多项基准上超过 OpenAI-o1-mini、取得 dense 模型新 SOTA[^5]；verl README 称 DAPO 是「基于 Qwen2.5-32B base 达到 AIME 2024 50 分的开源 SOTA RL 算法，超过此前 DeepSeek GRPO 的 SOTA」[^9]；Unsloth 称微调「2× faster with 70% less VRAM with no accuracy loss」[^12]；Qwen3 报告称在代码生成、数学推理、agent 任务上取得 SOTA[^19]。其中 DAPO 的 50 分有论文与项目主页双重来源[^28][^30]，但 Open-Reasoner-Zero 用统一评测口径复测得 DAPO-Qwen-32B* 为 AIME2024 48.3、MATH500 71.8、GPQA 16.0[^8]，与 DAPO 自报的 50 分及 MATH500 表现存在明显差距，说明跨论文的分数不可直接比较。

### 3.5 工程侧的可复现性证据强度不均

Open-Reasoner-Zero 提供了较细的规模门槛证据（ORZ-0.5B 单张 A800/H800 可跑通，1.5B 需 2 节点、7B 需 4 节点、32B 需 16 节点，基于 OpenRLHF + vLLM + DeepSpeed + Ray）[^7]，但其仓库活跃度偏低（31 commits，last push 2025-06-02，stars 2.1k）[^7]。DAPO 的训练配置较完整（DAPO-Math-17k 约 17k 条数学题、AIME 2024 验证集、字符串归一化规则式 verifier、128 张 H20、火山引擎平台）[^30]。相比之下，R1 主模型的本地部署支持有限：README 明确 Hugging Face Transformers 尚未直接支持 R1 主模型，本地部署示例是 vLLM 启动 Distill-Qwen-32B[^5]。官方还观察到 R1 系列对某些 query 会跳过思考模式（输出 `"<think>\n\n</think>"`），建议强制模型以 `"<think>\n"` 开头以保证充分推理[^5]——这一行为细节与「纯 RL 已内化推理习惯」的叙事之间存在张力，现有公开资料不足以确认其成因。

### 3.6 过程监督路线的位置尚未被取代

R1 路线刻意放弃过程监督[^2][^4]，但过程监督的实证优势仍然成立：process supervision 在 MATH 上显著优于 outcome supervision，其过程监督模型解决了 MATH 测试集一个有代表性子集中 78% 的问题[^24]；Math-Shepherd 用自动构造的过程级监督把 Mistral-7B 的 GSM8K 从 77.9% 提到 84.1%、MATH 从 28.6% 提到 33.0%[^25]；PRIME 从 Qwen2.5-Math-7B-Base 出发相比仅用结果奖励的 RL 取得 2.5 倍样本效率提升与 6.9% 的性能提升[^26]。PRIME 同时指出当前业界领先模型主要依赖可验证的结果奖励，尚未在稠密奖励上展示出有意义的进展[^26]。因此「结果奖励足够」目前是 R1 路线在特定任务分布（数学、代码等可验证任务）上的经验结论，而非已被证明的普遍结论。

## 参考来源

[^1]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948v2) — https://arxiv.org/abs/2501.12948v2（arxiv, 2025-01-22）
[^2]: DeepSeek-R1 (arXiv:2501.12948v2) — §1 Introduction / §2.1 Group Relative Policy Optimization — https://www.arxiv.org/html/2501.12948v2（arxiv, 2026-01-04）
[^3]: DeepSeek-R1 (arXiv:2501.12948, ar5iv 渲染版 v1) — §1 Introduction 对四阶段流程与蒸馏的表述 — https://ar5iv.labs.arxiv.org/html/2501.12948（arxiv, 2025-01-22）
[^4]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948v2) — https://arxiv.org/html/2501.12948v2（arxiv, 2025-01-22 (v1); v2 revised 2026-01-04）
[^5]: deepseek-ai/DeepSeek-R1 — 官方仓库 README（模型清单、上下文长度、评测结果表） — https://github.com/deepseek-ai/DeepSeek-R1（github, last push 2025-06-27）
[^6]: arXiv:2501.12948 摘要页（版本与发表信息元数据） — https://arxiv.org/abs/2501.12948（arxiv, 2025-01-22 (v1); v2 2026-01-04）
[^7]: Open-Reasoner-Zero/Open-Reasoner-Zero (GitHub README) — https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero（github, 2025-06-03）
[^8]: Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model (arXiv:2503.24290v2) — https://arxiv.org/html/2503.24290v2（arxiv, 2025-07-05）
[^9]: volcengine/verl — Volcano Engine Reinforcement Learning for LLMs (GitHub README) — https://github.com/volcengine/verl（github, 2026-08-01）
[^10]: huggingface/trl — TRL: Transformers Reinforcement Learning (GitHub README) — https://github.com/huggingface/trl（github）
[^11]: OpenRLHF/OpenRLHF (GitHub README) — https://github.com/OpenRLHF/OpenRLHF（github）
[^12]: unslothai/unsloth (GitHub README) — https://github.com/unslothai/unsloth（github）
[^13]: GitHub 仓库检索结果：基于 veRL 的社区扩展项目（Search-R1 / EasyR1 / verl-agent） — https://github.com/hiyouga/EasyR1（github）
[^14]: Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? — https://arxiv.org/abs/2504.13837（arxiv, 2025-04-18）
[^15]: ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models — https://arxiv.org/abs/2505.24864（arxiv, 2025-05-30）
[^16]: Awesome RLVR Reasoning Boundary (rdi-berkeley/awesome-RLVR-boundary) — https://github.com/rdi-berkeley/awesome-RLVR-boundary（github, 2025-12-12）
[^17]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948v1) — https://arxiv.org/abs/2501.12948v1（arxiv, 2025-01-22）
[^18]: DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning (Nature 645, 633–638) — https://www.nature.com/articles/s41586-025-09422-z（web）
[^19]: Qwen3 Technical Report (arXiv:2505.09388, v1, 14 May 2025) — https://arxiv.org/abs/2505.09388（arxiv, 2025-05-14）
[^20]: Qwen3 Technical Report — 后训练流程与蒸馏 vs RL（arXiv HTML 全文） — https://arxiv.org/html/2505.09388v1（arxiv, 2025-05-14）
[^21]: Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters — https://arxiv.org/abs/2408.03314v1（arxiv, 2024-08-06）
[^22]: s1: Simple test-time scaling — https://arxiv.org/abs/2501.19393（arxiv, 2025-01-31）
[^24]: Let's Verify Step by Step — https://arxiv.org/abs/2305.20050v1（arxiv, 2023-05-31）
[^25]: Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations — https://arxiv.org/abs/2312.08935（arxiv, 2023-12-14）
[^26]: Process Reinforcement through Implicit Rewards (PRIME) — https://arxiv.org/abs/2502.01456v2（arxiv, 2025-02-03）
[^28]: DAPO: An Open-Source LLM Reinforcement Learning System at Scale (arXiv:2503.14476v2) — https://arxiv.org/abs/2503.14476v2（arxiv, 2025-03-18）
[^29]: DAPO 论文正文（HTML 版，Section 1-3） — https://arxiv.org/html/2503.14476v2（arxiv, 2025-05-20）
[^30]: DAPO 项目主页（dapo-sia.github.io） — https://dapo-sia.github.io/（web）
[^31]: verl-project/verl（原 volcengine/verl）— GitHub 仓库元数据 — https://github.com/verl-project/verl（github, 2026-10-23）