# LLM-as-a-Judge 的已知偏差与评测校准实践

## 摘要

LLM-as-a-judge 指用强 LLM（如 GPT-4）作为人类评分者的替代者（surrogate）为开放式回答打分，其被界定的两大优势是 scalability 与 explainability（不仅给分还给理由）[^1]。该范式在 2023 年已成为业界默认做法（de facto standard）[^3]，但"仅采用 LLM-as-a-Judge 并不能保证评测准确、与既定标准一致"[^8]。本报告梳理证据可支撑的偏差类型：position bias（有最硬的量化数据）、verbosity/length bias、self-enhancement/self-preference bias、format 与 knowledge bias，以及 judge 自身的 sensitivity/specificity 不完美导致的系统性偏差；并整理四类校准手段：成对比较与顺序交换、长度等中介变量的回归控制、以人类标签为唯一 ground truth 的一致率与机会校正（Cohen's kappa、Krippendorff's alpha）、多模型评审团与微调专用 judge。需要前置声明两处重大不确定性：其一，谄媚偏差（sycophancy）、权威/来源偏差、评分刻度与分数聚集三类偏差在本次 27 条证据中零覆盖，无法支撑"已系统识别的偏差类型"这一表述；其二，MT-Bench 报告的"GPT-4 判官与人类一致率超过 80%"与一份预注册实测（原始一致率 46.6%–51.4%、Cohen's kappa 0.129–0.162）量级严重冲突，且证据中无任何一方对此作出解释。

---

## 一、范式定义与偏差为何是结构性问题

**定义。** LLM-as-a-judge 的输入是待评回答（单答案或成对答案）、评测指令与可选参考材料（参考答案、评分细则 score rubric），步骤是由强 LLM 按预定义评分维度输出判定（1–5 分或 A/B 胜者）及理由，产出是可聚合为模型排名的判定结果[^1][^2][^3]。其子范式包括 pairwise comparison（两两比较）、single answer grading（单答案打分）与 reference-guided grading（提供参考答案的评分）[^1]；G-Eval 进一步把单答案打分形式化为"form-filling paradigm"，并叠加 LLM 自动生成的 chain-of-thoughts（Evaluation Steps）与基于输出 token 概率的连续化评分函数 score = Σ p(s_i) × s_i，以缓解 LLM 只输出整数造成的大量并列[^2]。

**偏差为何不可避免。** 自动评测方法往往正是通过利用虚假相关（输出长度、是否含列表、位置偏差等）来达到与人类标注者的高相关[^10]。换言之，偏差不是实现缺陷，而是这类指标获得高相关性的机制的一部分。此外，LLM 评测器对指令与 prompt 敏感[^2]，且存在偏好 LLM 生成文本的偏差，若用作自我改进的奖励信号可能导致 self-reinforcement[^2]。

---

## 二、证据可支撑的偏差类型

### 2.1 Position bias（位置偏差）

这是唯一有直接量化对比的偏差。在默认 prompt 下，Claude-v1 的顺序一致性仅 23.8%、75.0% 的案例偏向第一个答案；GPT-3.5 为 46.2%；GPT-4 为 65.0%——仅 GPT-4 在 60% 以上案例给出一致结果[^1]。Claude-v1 还表现出 name bias，在 rename prompt 下偏向 "Assistant A"[^1]。

### 2.2 Verbosity / length bias（冗长与长度偏差）

MT-Bench 论文将 verbosity bias 列为该范式的明确局限之一[^1][^11]，但本次仅取得摘要，未读到正文中 verbosity bias 实验的具体量化结果，故不作为量化结论依据[^11]。可用的间接量化来自 AlpacaEval：其 LLM 判官与人类偏好高度相关，但已知偏好生成更长输出的模型[^10]。另有一条临床推理场景的数据：GPT-4o-mini 在信息稀缺条件下诊断准确率从 95.0% 降至 32.5%，原作者将其标注为 verbosity bias[^15]——该数值衡量的是被评模型的诊断表现而非 judge 的打分偏移，引用时需注意其语义。

### 2.3 Self-enhancement / self-preference bias（自我增强/自我偏好）

标准定义：LLM 评估器给自己的输出打分高于他人输出，而人类标注者认为二者质量相同[^12]。机制层面，GPT-4 与 Llama 2 在开箱状态下即具备非平凡的准确率，能把自身输出与其他 LLM 及人类的输出区分开；微调实验显示自我识别能力与自我偏好偏差强度呈线性相关，作者称受控实验表明该因果解释能抵御直接混淆因素[^12]。

但这一结论正在被新证据收窄。一项研究指出，既有测量因生成文本中风格特征与回答质量混杂，无法把真实自我偏好与混淆因素分离；作者改用不带风格指纹但保留模型特定签名的叙事约束集合，发现在盲评条件下自我偏好消失，在控制选择质量与 judge 严厉度后仅剩一个方向相反的小效应；而在质量匹配条件下，仅凭 self-/other- 标签（不点名任何模型）即可双向移动分数——judge 抬高 self-labeled 选择的分数、压低 other-labeled 选择的分数，与选择的真实来源无关[^13]。该文因此提出：作者身份归因（authorship attribution）是评估偏差的一个独立驱动因素[^13]。另一项在 IFEval 指令遵循修订任务上的四模型中端模型测试（"有效"由确定性验证器判定而非另一个模型判定）跨 85 组 author-versus-fresh 对比未检测到自我偏好：作者拒绝已验证有效修正的比例与中立模型评判同一草稿时基本一致，差距 −5.1 个百分点，95% CI [−12.9, +2.7]；小规模试点中的"自我怀疑"提示效应在大规模实验中未复现[^14]。临床场景则报告了相反方向的个例：HuatuoGPT-o1 批准了所有有效计分的失败样本，并表现出正向自我偏好偏差[^15]。

### 2.4 Format bias 与 knowledge bias

JudgeLM 的 README 将微调 judge 的关键偏差归为三类：position bias、knowledge bias、format bias[^5]。这是本次证据中对 format/knowledge bias 的唯一明确命名，但未给出各自的量化测量。

### 2.5 Judge 自身的 sensitivity / specificity 不完美（判定噪声偏差）

这是最容易被忽略、但对报告结论影响最大的一类。judge 因 specificity 与 sensitivity 不完美而产生噪声，导致准确率估计有偏；已有偏差校正方法在实践中未被充分利用，且通常假设已知精确的 specificity/sensitivity[^7]。该框架给出 plug-in 偏差校正点估计 theta = (p + q0 − 1) / (q0 + q1 − 1)，其中 p 为测试集被判正确的比例、q0 为 specificity、q1 为 sensitivity；使用前提是 judge 优于随机，即 q0 + q1 > 1，否则分母为零[^7]。

### 2.6 覆盖缺口（必须显式声明）

本次证据中不存在关于 sycophancy（谄媚偏差）、权威/来源偏差、评分刻度与分数聚集（score clustering / 区间窄化）的任何内容。检索命中的两条论文——LLM 内在自我纠错[^16]与多智能体辩论[^17]——与上述三类偏差均无直接关系，且证据本身已标注为检索误命中、不用于支撑相关结论。因此"已系统识别的偏差类型"这一提法目前只能覆盖 position、verbosity、self-preference、format/knowledge 与判定噪声五类，其余三类现有公开资料不足以确认。

---

## 三、实践中如何校准

### 3.1 设计层面：换范式、换顺序、换评审构成

- **成对比较优先于绝对打分。** Chatbot Arena 采用 pairwise comparison 而非绝对打分，理由是很难在大量不同用户间维持一致的评分标准[^20]；MT-Bench 亦指出 single answer grading 难以辨别细微差异，且绝对分数比相对比较更不稳定[^1]。代价是 pairwise 在选手数量增加时组合数平方增长而缺乏可扩展性[^1]。
- **顺序交换。** JudgeLM 引入 swap augmentation（交换答案顺序做数据增强）以对抗 position bias[^5]，训练命令中含 `--swap_aug_ratio 0.5`[^5]。
- **参考材料供给与丢弃。** JudgeLM 同时使用 reference support（提供参考基准答案）与 reference drop（训练时随机丢弃参考，避免过度依赖参考）[^5]；README 模型表中加入 reference 后各规模一致提升：JudgeLM-7B Agreement 81.11→84.08，13B 84.33→85.47，33B 89.03→89.32[^5]。Prometheus 的关键前提假设同样是"提供适当参考材料（参考答案、评分细则 score rubric）"时才能达到与 GPT-4 相当的评测能力[^3]。
- **评审团替代单一判官。** PoLL 指出普遍依赖单一大型 judge 成本高且已被证明会引入 intra-model bias，提出由多个较小模型组成评审团；在三种 judge 设置、六个数据集上 PoLL 优于单一大型 judge，因成员来自不同模型家族而表现出更少的 intra-model bias，同时成本降低七倍以上[^4]。Anthropic 亦将 multi-judge consensus 列为 model-based grader 的方法之一[^18]。需注意：以上 PoLL 陈述均为摘要级，本轮未读取正文实验章节，具体数据集与偏差度量方式未取证[^4]。

### 3.2 统计层面：控制中介变量与机会校正

- **长度控制回归。** Length-Controlled AlpacaEval 用广义线性模型（GLM）控制自动评测器偏差：先拟合 GLM，用待控制的中介变量（长度差）及其他相关特征预测有偏自动标注器的偏好；再在长度差为零的条件下预测偏好，回答的反事实问题是"如果模型与基线的输出长度相同，偏好会是多少"[^10]。长度控制不仅提升了对模型冗长程度操纵的鲁棒性，还将与 LMSYS Chatbot Arena 的 Spearman 相关系数从 0.94 提升到 0.98[^10]。方法论文定位是把长度等虚假相关因素视为因果图中的不良中介变量，用基于回归的因果推断技术做调整[^10]。
- **机会校正（chance correction）。** 一份预注册、held-out 测试集只读一次的实测（FaithBench 人类专家标注，7,012 条判定，测试集 488 条、每条每 judge 重复 3 次、temperature 0）显示：原始一致率 Tier1 付费 46.6%、Tier2 托管 51.4%、Tier3 本地 47.3%，但 Cohen's kappa 仅 0.129 / 0.162 / 0.131（95% CI 分别为 [0.084,0.175]、[0.109,0.214]、[0.083,0.178]），PABAK 分别为 −0.069 / 0.029 / −0.055；机会校正使一致率下降 33.7–35.2 个百分点，约 47% 的原始一致率变为约 0.13 的 kappa[^23]。同一任务上人类标注者之间的一致率为 82.0%[^23]。该实测还给出敏感性 0.253/0.349/0.269、特异性 0.868/0.930/0.916、解析失败率 0.5%/2.9%/2.5%，并指出在 28.0%–68.6%（宽 40.6 个百分点）的发布阈值区间内，三个 judge 都会放行人类标注本应拦截的版本[^23]。一个方法论警示：分析者自身的分箱（binarisation）选择可使 kappa 变动最多 0.069，与跨部署层级的最大差异 0.065 相当[^23]。
- **报告口径与置信区间。** 上述校正框架提供同时反映测试集与校准集不确定性的置信区间 confidence_interval(p, q0, q1, n, m0, m1, alpha)，以及自适应校准样本分配算法 allocate_calibration_sample(m, p, q0_pilot, q1_pilot, m_pilot)，在 specificity 与 sensitivity 两类校准样本间分配预算以降低估计不确定性[^7]。
- **评分连续化。** G-Eval 用概率加权求和把离散分数连续化，以同时解决单一分值主导分布与大量并列问题[^2]。

### 3.3 人类锚定：把人类标签当作唯一 ground truth

- Anthropic 明确列出 model-based grader 的弱点：非确定性、比代码评分更贵、且"需要与人类评分者校准以保证准确性"，并把 human grader 的用途之一直接定义为"用于校准 model-based grader"，列举 SME review、crowdsourced judgment、spot-check sampling、A/B testing、inter-annotator agreement 等方法[^18]。客户实践上，Descript 从人工评分演进到由产品团队定义标准的 LLM grader，并配合"定期人工校准"[^18]。
- 工具层面的一致率门禁：一个开源工具计算 judge 与人类标签的一致率与 Cohen's kappa，并给出逐类 precision/recall、逐切片指标、自一致性（同一 case 问 3 次统计矛盾）与 CI 门禁；CLI 支持 `--min-agreement 0.85` 与 `--min-kappa 0.7`，低于下限时 eval 以非零码退出，用于阻止降低一致率的 prompt 变更合并[^24]。该工具明确声明不自己标注数据：人类标签是唯一 ground truth，用另一个 judge 给 judge 打分"measures nothing"，解析失败的 case 会被报告而非丢弃[^24]。其 demo（20 条 QA step、judge 为规则而非模型）一致率 70.0%、kappa 0.476，逐类结果中 blocked 类（support 3）precision/recall/f1 全为 0.00，切片上 mobile 8 条一致率 62.5%/kappa 0.400、web 12 条 75.0%/0.532[^24]。
- 临床场景的教训最直接：在人工核验的失败集（n = 142）上，GPT-4o-mini 批准了 47.9% 临床上不正确的输出，作者结论是缺乏专家锚定验证的独立自动化临床评估可能大幅高估临床可靠性[^15]。
- 多评分者一致性度量工具：Krippendorff's alpha 的快速 Python 实现（`pip install krippendorff`），输入为形状 (M, N) 的 reliability data matrix（M 为 coder/rater 数、N 为 unit/item 数），支持 level_of_measurement 设定，并提供 bootstrap_ci（可设 confidence_level、n_resamples 如 2000、alternative_threshold 如 0.800，返回 alpha、CI 上下界、standard_error 与 q 值）[^26]。另有仓库声称实现 Cohen kappa、Fleiss kappa、Krippendorff alpha，但仅取得搜索元数据、未读 README，其 API 与是否含 judge 校准流程均未验证[^27]。

### 3.4 模型层面：微调专用 judge 与开源替代

- **微调 judge。** JudgeLM-100K 含 100K 训练 judge 样本 + 5K 验证样本，judgement 由 GPT-4 生成；JudgeLM-13B 微调超参为 global batch size 128、lr 2e-5、3 epochs、max length 2048，训练命令含 `--swap_aug_ratio 0.5` 与 `--ref_drop_ratio 0.5`[^5]。JudgeLM-33B（w/ reference）Precision 84.00、Recall 86.21、F1 84.98、Consistency 92.37；README 声称与 teacher judge（GPT-4）的一致率超过 90%，并称该一致率超过人与人之间的一致率[^5]。效率上，JudgeLM-7B 在 8×A100 上判 5K 样本需 3 分钟[^5]。
- **开源判官替代闭源判官。** Prometheus 指出闭源判官的三项适用性限制：不透明、版本不可控、成本过高，对大规模评测与自定义标准（如儿童可读性）场景不可靠[^3]。其 13B 模型在 45 条自定义评分细则下与人类评测者的 Pearson 相关为 0.897，GPT-4 为 0.882，ChatGPT 为 0.392；在四个基准（MT Bench、Vicuna Bench、Feedback Bench、Flask Eval）的 1222 条自定义评分细则上，与 GPT-4 的相关性呈相似趋势[^3]。后续 Prometheus 2 (8x7B) 在 5 点 Likert 直接评分上与 GPT-4-1106 的 Pearson 相关为 0.6–0.7（覆盖 VicunaBench、MT-Bench、FLASK），在 pairwise ranking 基准上与人类判断的一致率为 72%–85%；Prometheus 2 (7B) 仅需 16 GB VRAM，达到 8x7B 至少 80% 的评测表现[^6]。工程接口上支持 vllm 本地推理与 litellm 调用外部 API（含 GPT-4），提供 ABSOLUTE_PROMPT 与 SCORE_RUBRIC_TEMPLATE 模板，评分 1–5，但 README 自述该库仍处于 beta 阶段[^6]。
- **基准与相关性参照。** G-Eval + GPT-4 在 SummEval 摘要任务上与人类的 Spearman 相关为 0.514（Kendall-Tau 0.418），论文称大幅超越此前所有方法；同任务对比 UniEval 平均 Spearman 0.474、BARTScore 0.385、GPTScore 0.417、ROUGE-1 0.192、BERTScore 0.225[^2]。这组数字的价值在于给出"与人类相关"的合理量级参照：即便是当时最优方法，Spearman 也只有 0.5 量级。

### 3.5 流程层面：预注册、功效分析与可复现门禁

- 预注册审计协议：PROTOCOL.md 在任何数据收集前冻结；最小可检测效应在收集前用脚本计算——N=450 items 时，judge 间一致率差异小于约 7 个百分点在 item 层面不可分辨，因此不做排序，改以 constraint-level 分析（约 2,000 单元）为主；人类标注协议标注为 κ-gated[^25]。该仓库截至 2026-09-25 已收集 720 条回答、4,479 条判定，但分析与 REPORT.md 尚未撰写，README 内容为预注册设计而非结果[^25]。
- 可复现门禁：校正框架提供 `python -m run.figure2_bias_adjustment` / `figure3_ci_length` / `figure4_monte_carlo` 脚本与 notebook[^7]；上述预注册实测仓库提供 CI gate（`analysis/ci_gate.py`）在每次 push 时复现所有头条数字，判定存储已提交、无需 API key[^23]。
- 运行侧稳定性：Anthropic 强调因模型输出在多次运行间存在波动，需对同一 task 运行多次 trial 以获得更一致的结果[^18]；并指出评测需与生产监控、A/B 测试、用户研究结合才能为持续改进提供信号[^18]。
- 任务类型决定 grader 选择：Anthropic 将 grader 分为 code-based、model-based、human 三类，主张按任务选择合适 grader；确定性 grader 对编码 agent 更自然（代码能否运行、测试是否通过），并以 SWE-bench Verified 与 Terminal-Bench 为例[^18]。同时提醒 agent 评测中错误会传播和累积，且前沿模型可能找到超越静态评测设定的解法——Opus 4.5 在 τ2-bench 订票任务中发现政策漏洞，按评测字面判定为"失败"，但对用户实际是更好的方案[^18]。

### 3.6 生态现状的一个侧面

以 'LLM judge evaluation benchmark bias' 检索返回的 6 个仓库均为低热度个人项目，最高 9 stars，其余 0–1 stars，且均无机构背书[^9]。其中若干自述直接对应上述校准动作：离线 judge 工具含 position/verbosity bias 诊断、Platt 校准与一致率基准[^9]；可靠性基准覆盖人类一致率、位置偏差、冗长偏差与顺序敏感性[^9]；一个案例自述在信任 LLM 判定的增益前先审计 judge，该 judge 曾伪造出 +6 的优势，审计后降至约 0[^9]。这些仓库不能作为 OpenAI/Anthropic/Google/LMArena/HELM 等官方立场的证据[^9]。

---

## 四、矛盾与不确定性

**（1）一致率量级冲突，未解决。** MT-Bench 报告 GPT-4 判官与受控专家投票及众包人类投票的一致率超过 80%，达到人类-人类之间的一致水平[^1][^11]；JudgeLM README 亦称与 GPT-4 的一致率超过 90% 并超过人与人之间的一致率[^5]。而一份预注册实测给出原始一致率 46.6%–51.4%、Cohen's kappa 0.129–0.162，同任务人类标注者之间为 82.0%[^23]。两者量级严重冲突。可能的解释方向包括任务类型（开放式偏好 vs FaithBench 二分类事实性）、是否做机会校正、单轮 vs 多轮，但**证据中没有任何一条对此作出解释或交叉验证**。本报告因此不给出统一的一致率结论：引用 80% 类数字时必须同时说明其任务与口径，引用 kappa 类数字时必须说明其经过了机会校正。

**（2）自我偏好的存在性与强度存在分歧。** 一方报告自我识别能力与自我偏好强度线性相关且能抵御混淆因素[^12]；另一方在盲评与质量匹配条件下发现自我偏好消失、仅剩方向相反的小效应，并把效应归因于 self-/other- 标签而非真实来源[^13]；第三方在确定性验证器判定的指令遵循修订任务上未检测到自我偏好（差距 −5.1 个百分点，95% CI [−12.9, +2.7]）[^14]。三者任务与实验设计不同，证据未提供统一的调和结论。

**（3）三类偏差零覆盖。** 谄媚偏差（sycophancy）、权威/来源偏差、评分刻度与分数聚集（score clustering / 区间窄化）在 27 条证据中完全缺失；检索命中的两条论文为误命中且已标注不用于支撑相关结论[^16][^17]。现有公开资料不足以确认这三类偏差的测量与缓解方式。

**（4）官方立场覆盖极不完整。** 子问题要求的 OpenAI / Anthropic / Google / LMArena / HELM 等评测基准方立场，实际只有 Anthropic 一家取得实质内容[^18]。HELM 两条来源（官网首页与 classic leaderboard）均为 JavaScript 渲染页面，抓取仅返回标题，未获得任何关于 judge、一致率或使用边界的正文内容，属无效证据[^21][^22]。OpenAI、Google、LMArena 官方文档完全缺失。子问题明确要求的"LLM judge 不可用于高风险决策"类边界声明，在全部证据中无任何来源支撑，现有公开资料不足以确认。

**（5）部分证据为摘要级或未完成状态。** PoLL 的量化细节（具体数据集、偏差度量方式）未取证[^4]；MT-Bench 的 verbosity bias 具体量化结果未读到正文[^11]；预注册审计仓库仍处 protocol phase，分析未撰写[^25]；agreement 度量仓库仅取得搜索元数据，API 与校准流程未验证[^27]；Prometheus-Eval README 自述库仍为 beta[^6]。

**（6）测量本身的不稳定性可能被低估。** 分析者的分箱选择可使 kappa 变动最多 0.069，与跨部署层级的最大差异 0.065 相当[^23]——这意味着部分"judge 之间/部署层级之间"的差异可能落在分析者自由度之内，而非真实能力差异。

## 参考来源

[^1]: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena — https://arxiv.org/abs/2306.05685（arxiv, 2023-06-09 (v1), v4 2023-12-24; NeurIPS 2023 Datasets and Benchmarks Track）
[^2]: G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment — https://arxiv.org/abs/2303.16634（arxiv, 2023-03-29 (v1), v3 2023-05-23）
[^3]: Prometheus: Inducing Fine-grained Evaluation Capability in Language Models — https://arxiv.org/abs/2310.08491（arxiv, 2023-10-12 (v1), v2 2024-03-09; ICLR 2024）
[^4]: Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models (PoLL) — https://arxiv.org/abs/2404.18796（arxiv, 2024-04-29 (v1); v2 2024-05-01）
[^5]: JudgeLM: Fine-tuned Large Language Models are Scalable Judges (README) — https://github.com/baaivision/JudgeLM（github, 2023-10-01）
[^6]: Prometheus-Eval: A repository for evaluating LLMs in generation tasks (README) — https://github.com/prometheus-eval/prometheus-eval（github）
[^7]: How to Correctly Report LLM-as-a-Judge Evaluations (LLM-judge-reporting, README) — https://github.com/UW-Madison-Lee-Lab/LLM-judge-reporting（github, 2025-11-27）
[^8]: A Survey on LLM-as-a-Judge (arXiv:2411.15594v6) — https://arxiv.org/abs/2411.15594（arxiv, 2024-11-23 (v1); v6 2025-10-19）
[^9]: GitHub 仓库检索结果：'LLM judge evaluation benchmark bias'（6 个仓库） — https://github.com/omnirogue/jevlab（github）
[^10]: Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators — https://arxiv.org/abs/2404.04475v2（arxiv, 2024-04-06）
[^11]: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena — https://arxiv.org/abs/2306.05685v4（arxiv, 2023-06-09）
[^12]: LLM Evaluators Recognize and Favor Their Own Generations — https://arxiv.org/abs/2404.13076（arxiv, 2024-04-15）
[^13]: Self- and Other-Labels Induce Bidirectional Bias in LLM Judges — https://arxiv.org/abs/2608.18091（arxiv, 2026-08-16）
[^14]: Self-Preference Is Weak or Absent in Verifiable Instruction-Following Revision: A Four-Model Test Under Genuine Authorship — https://arxiv.org/abs/2606.20093（arxiv, 2026-06-30）
[^15]: CLExEval: A Human-in-the-Loop Framework for Qualitative Evaluation of LLM Clinical Reasoning — https://arxiv.org/abs/2606.31608（arxiv, 2026-06-30）
[^16]: Large Language Models Cannot Self-Correct Reasoning Yet — https://arxiv.org/abs/2310.01798（arxiv, 2023-10-03）
[^17]: Improving Factuality and Reasoning in Language Models through Multiagent Debate — https://arxiv.org/abs/2305.14325（arxiv, 2023-05-23）
[^18]: Demystifying evals for AI agents (Anthropic Engineering) — https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents（web）
[^20]: Chatbot Arena (arXiv:2403.04132) — HTML 全文（第 1–3 节，抓取被截断） — https://arxiv.org/html/2403.04132v1（arxiv, 2024-03-07）
[^21]: HELM (Holistic Evaluation of Language Models) — Stanford CRFM 官方站点首页 — https://crfm.stanford.edu/helm/（web）
[^22]: HELM Classic leaderboard 页面（Stanford CRFM） — https://crfm.stanford.edu/helm/classic/latest/（web）
[^23]: sean-mas/llm-judge-reliability — How much LLM-judge agreement survives chance correction? — https://github.com/sean-mas/llm-judge-reliability（github, 2026-09-07）
[^24]: kulykivska/llm-judge — Measure an LLM judge against human labels before you trust it — https://github.com/kulykivska/llm-judge（github, 2026-10-01）
[^25]: chenggma/llm-judge-audit — Pre-registered audit of LLM-as-judge (protocol phase) — https://github.com/chenggma/llm-judge-audit（github, 2026-10-22）
[^26]: pln-fing-udelar/fast-krippendorff — Fast computation of Krippendorff's alpha (★162) — https://github.com/pln-fing-udelar/fast-krippendorff（github, 2026-10-15）
[^27]: heolin/agreement — Implementation of popular agreement metrics (Cohen kappa, Fleiss kappa, Krippendorff alpha) — https://github.com/heolin/agreement（github, 2026-10-21）