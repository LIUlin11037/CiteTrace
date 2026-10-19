# LLM 幻觉的主流缓解手段及其有效性边界

## 摘要

幻觉在 NLG 与 LLM 语境下被定义为模型生成「非预期文本」[^1]，或更具体地定义为「与来源冲突、或无法被事实知识验证的内容」[^2]；Huang et al. 的综述进一步将其描述为「生成看似合理但不符合事实的内容」[^19]。现有证据显示幻觉的严重程度具有任务与模型依赖性：ChatGPT 在特定主题上约 19.5% 的回复包含编造的不可验证信息[^2]，ChatGPT 在人物传记生成上的 FActScore 仅 58%[^3][^23]，TruthfulQA 上最佳被测模型仅 58% 真实而人类为 94%[^4][^22]，HalluQA 评测的 24 个中文模型中 18 个非幻觉率低于 50%[^21]。主流缓解手段可归为五类：(1) 训练与对齐类（DPO 等偏好优化）[^10]；(2) 解码与推理时干预类（DoLa、ITI、Self-Consistency、CoVe）[^8][^11][^12][^9]；(3) 检索与接地类（Google Search grounding、RAG 相关性判断）[^17][^6]；(4) 引用归因与验证类（ALCE、AttributionBench、Anthropic 引文验证）[^24][^7][^15]；(5) 工程护栏与评估类（Guardrails、Ragas、Evals、HHEM）[^13][^14][^16][^31]。各类手段的有效性边界差异显著：解码类方法在 TruthfulQA 上可带来 12–17 个百分点提升[^8]或 32.5%→65.1% 的提升[^11]，但 ITI 明确存在真实性与有用性的权衡[^11]；引用类方法即使最好模型在 ELI5 上仍有 50% 时间缺乏完整引用支撑[^24]；自动归因评估对 SOTA LLM 仍困难，微调 GPT-3.5 二分类 macro-F1 仅约 80%[^7]；Anthropic 官方明确声明这些技术「能显著减少幻觉，但不能完全消除」[^15]。本报告同时标注三处证据不足与矛盾：2025–2026 年 RAG 归因进展无证据、Vectara 榜单内部排序反常且模型命名无法交叉验证、部分工程框架的 stars 与活跃度指标缺失。

---

## 一、幻觉的定义与度量基线

### 1.1 定义

不同来源给出的定义侧重点不同，但共享「与事实/来源不一致」这一内核：

- 综述 [1] 将幻觉定义为 NLG 模型生成「unintended text」（非预期文本），其后果是降低系统性能并无法满足真实场景用户预期[^1]。
- HaluEval 将幻觉定义为「content that conflicts with the source or cannot be verified by the factual knowledge」（与来源冲突、或无法被事实知识验证的内容）[^2]。
- Huang et al. 综述将幻觉描述为「generating plausible yet nonfactual content」（生成看似合理但不符合事实的内容），并将 factuality 与 faithfulness 作为其分类体系的核心维度[^19]。
- HHEM 模型卡从判定操作层面给出定义：判定需要 hypothesis（待判文本）与 evidence/premise（给定证据）两段文本，「hallucinated / factually inconsistent」定义为 hypothesis 不被 premise 支持[^31]。

HHEM 模型卡特别强调「factual but hallucinated」类型：给定 premise「The capital of France is Berlin」，hypothesis「The capital of France is Paris」被判为幻觉，尽管后者在世界知识中为真[^31]。该模型卡还指出幻觉检测是「asymmetric / 非交换」的：hypothesis「I visited Iowa」相对 premise「I visited the United States」被判为幻觉，反向则一致[^31]。这意味着「幻觉」在工程判定中并不等同于「世界知识错误」，而是相对于给定证据的一致性判定。

### 1.2 度量基线

主要度量工具及其报告数值：

| 基准 | 度量方式 | 关键数值 |
|---|---|---|
| HaluEval [2] | 生成+人工标注的幻觉样本集合，评估 LLM 识别幻觉能力 | ChatGPT 约 19.5% 回复含编造的不可验证信息[^2] |
| FActScore [3][^23] | 将生成文本拆为 atomic facts，计算被可靠知识源支持的百分比 | ChatGPT 人物传记 FActScore 58%[^3][^23]；自动估计误差率低于 2%[^3][^23] |
| TruthfulQA [4][^22] | 817 问题、38 类别，问题设计为诱导错误信念 | 最佳模型 58% 真实，人类 94%[^4][^22] |
| SimpleQA [5] | 短式事实问题，答案分为 correct/incorrect/not attempted | 设计目标为「知道自己知道什么」[^5] |
| HalluQA [20][^21] | 450 道对抗性问题，GPT-4 作评判者，指标为非幻觉率 | 24 个中文模型中 18 个非幻觉率低于 50%[^21] |
| HHEM 榜单 [18][^30] | 摘要任务幻觉率 | 头部模型 1.8%–4.1% 幻觉率[^18] |

TruthfulQA 的一个关键发现是：最大的模型通常最不真实，与其他 NLP 任务中性能随模型规模提升的趋势相反[^4][^22]。作者建议仅靠扩大模型规模对提升真实性不如使用非「模仿网络文本」的训练目标进行微调更有前景[^4][^22]。这一发现为后续所有缓解手段提供了动机：幻觉不是单纯靠 scale 能解决的问题。

---

## 二、训练与对齐类手段

### 2.1 DPO（Direct Preference Optimization）

DPO 提出对 RLHF 中奖励模型的新参数化，可闭式解出对应最优策略，从而仅用简单分类损失即可求解标准 RLHF 问题[^10]。其工程优势在于：稳定、性能好且计算轻量，微调时无需从 LM 采样，也无需大量超参调优[^10]。实验显示 DPO 在情感控制上超过基于 PPO 的 RLHF，在摘要与单轮对话上匹配或提升回答质量[^10]。论文将 RLHF 描述为复杂且常不稳定的过程（先拟合奖励模型再用 RL 微调）[^10]。

**有效性边界**：证据仅覆盖情感控制、摘要与单轮对话三类任务，未提供 DPO 对幻觉率或真实性基准（如 TruthfulQA、FActScore）的直接量化改善数据。因此「DPO 能降低幻觉」这一主张在现有证据下属于推断而非直接实证。

### 2.2 训练目标的取向

TruthfulQA 作者的建议构成训练类手段的方向性边界：使用非「模仿网络文本」的训练目标进行微调，比扩大模型规模更有前景[^4][^22]。但证据未给出具体训练目标设计及其幻觉降低幅度的量化结果。

---

## 三、解码与推理时干预类手段

此类手段的共同特征是不依赖外部检索、不修改模型权重（或仅做轻量推理时干预），因而部署成本低。

### 3.1 DoLa（Decoding by Contrasting Layers）

DoLa 是一种无需检索外部知识、也无需额外微调的预训练 LLM 解码策略[^8]。方法通过对比后层与前层投影到词表空间的 logits 差异来获得下一 token 分布，利用事实知识通常局部化于特定 transformer 层这一特性[^8]。在多项选择题与开放式生成任务上均提升真实性，例如将 LLaMA 系列模型在 TruthfulQA 上的表现提升 12–17 个绝对百分点[^8]。

**有效性边界**：证据仅报告 TruthfulQA 与多项选择/开放式生成任务的提升[^8]，未覆盖长文本生成的事实精度（如 FActScore 类指标）或多语言场景。

### 3.2 ITI（Inference-Time Intervention）

ITI 通过在推理时沿一组方向移动有限数量注意力头的激活来提升 LLM 真实性[^11]。在指令微调的 LLaMA（Alpaca）上，ITI 将真实性从 32.5% 提升到 65.1%（TruthfulQA）[^11]。其工程特性是侵入性小、计算开销低且数据高效：仅用数百个样本即可定位真实方向，而 RLHF 等方法需要大量标注[^11]。

**有效性边界**：论文明确指出真实性（truthfulness）与有用性（helpfulness）之间存在权衡，并说明可通过调节干预强度来平衡该权衡[^11]。这是本报告中唯一被来源显式标注的「缓解手段自身引入的代价」——即提升真实性可能以牺牲有用性为代价。

### 3.3 Self-Consistency

Self-consistency 是一种替代贪心解码的解码策略，用于 chain-of-thought 提示：先采样多条多样化推理路径，再通过对采样路径边缘化选出最一致的答案[^12]。其直觉是复杂推理问题通常存在多种思考路径但指向唯一正确答案[^12]。在算术与常识推理基准上大幅提升 CoT 表现：GSM8K +17.9%、SVAMP +11.0%、AQuA +12.2%、StrategyQA +6.4%、ARC-challenge +3.9%[^12]。

**有效性边界**：证据覆盖的是推理基准的准确率提升[^12]，而非幻觉率或事实性基准的直接降低。Self-consistency 的适用前提是「存在唯一正确答案」，对开放式生成或事实性无唯一答案的任务，其边缘化机制的有效性在现有证据中未获确认。

### 3.4 CoVe（Chain-of-Verification）

CoVe 让模型对自身回答进行审慎核查以纠正错误，分四步：先起草初始回答；再规划验证性问题以核查草稿；随后独立回答这些问题以免被其他回答带偏；最后生成经核验的最终回答[^9]。实验显示 CoVe 在多种任务上降低幻觉，包括来自 Wikidata 的列表式问题、闭卷 MultiSpanQA 以及长文本生成[^9]。

**有效性边界**：证据未给出 CoVe 相对于基线的具体幻觉降低百分比，仅报告「降低幻觉」这一方向性结论[^9]。其四步流程中「独立回答验证性问题」的设计意图是避免被初始草稿带偏[^9]，但证据未量化该设计相对非独立验证的增益。

### 3.5 推理时干预的补充：INSIDE 与 KLE

INSIDE 将既有幻觉检测方法归纳为两类：logit-level uncertainty estimation 与 language-level self-consistency evaluation，并指出二者在 token-decoding 过程中都会丢失语义信息[^25]。该论文提出 EigenScore 指标，利用响应协方差矩阵的特征值在稠密嵌入空间中度量语义一致性/多样性[^25]；并提出 test-time feature clipping 方法，通过截断内部状态中的极端激活来降低过度自信生成[^25]。

Kernel Language Entropy (KLE) 是 NeurIPS'24 的官方实现，构建于 Semantic Uncertainty 代码库之上，KLE 及其工具位于 ./kle/ 目录[^26]。其硬件要求为：7B 模型可用单张 24GB 桌面 GPU，13B 模型需要 A100 级别 GPU，70B 模型需要两张 A100 80GB[^26]。

**有效性边界**：这两项工作属于检测/不确定性量化范畴，证据未给出其作为「缓解手段」的端到端幻觉降低数值。

---

## 四、检索与接地类手段

### 4.1 Google Search Grounding

Google 官方文档说明：Grounding with Google Search 将 Gemini 模型连接至实时网页内容，支持所有可用语言，使 Gemini 能提供更准确答案并引用可验证来源[^17]。文档列出接地带来的三项能力：提高事实准确性（通过基于真实世界信息减少模型幻觉）、访问实时信息、提供引文以建立用户信任[^17]。启用 google_search 工具后，模型自动处理搜索、处理与引用信息的整个工作流[^17]。

**有效性边界**：官方文档给出的是能力声明（「减少模型幻觉」）[^17]，未提供量化幻觉降低幅度。且该文档未说明在检索失败、检索结果冲突或问题超出检索覆盖范围时的行为。

### 4.2 RAG 场景中的相关性判断：NoMIRACL

NoMIRACL 是面向 RAG 场景的多语言相关性评估数据集，用于评估 LLM 对 query 与 passage 相关性的判断能力[^6]。其核心设定是：检索系统返回的 passage 子集可能相关也可能不相关，LLM 作为生成器应判断相关性——若子集中存在相关 passage 则作答，否则应弃答（abstain）[^6]。数据集覆盖 18 种语言，用于评估 LLM 在 RAG 中面对 first-stage retrieval errors 的鲁棒性[^6]。数据采样为 50% 相关样本 + 50% 不相关样本，每类上限 250[^6]。

**有效性边界**：NoMIRACL 提供的是评估能力而非缓解手段本身[^6]。其价值在于暴露「检索器召回不足」这一 RAG 幻觉来源——当检索返回的 passage 全部不相关时，模型应弃答而非编造[^6]。但证据未给出各模型在该数据集上的弃答准确率数值。

### 4.3 RAG 抗幻觉的已知局限

Huang et al. 综述明确涵盖「检索增强 LLM 对抗幻觉的局限」[^19]，但证据摘要未展开具体局限内容。

---

## 五、引用归因与验证类手段

### 5.1 ALCE（Automatic LLMs' Citation Evaluation）

ALCE 是首个面向 LLM 引用质量自动评估的基准，要求系统端到端地检索支撑证据并生成带引用的答案[^24]。其覆盖三类问题：ASQA（Wikipedia 21M passages，factoid ambiguous）、QAMPARI（Wikipedia 21M，factoid list）、ELI5（Sphere 899M，why/how/what）[^24]。ALCE 提出沿 fluency、correctness、citation quality 三个维度的自动指标，并声称这些指标与人工判断有强相关性[^24]。

论文主张引用机制的两点收益：(1) 用户可借助引用核验 LLM 的论断；(2) LLM 可生成忠实于被引段落的文本，有望提升正确性并缓解幻觉[^24]。

**有效性边界**：论文核心结论是当前系统仍有很大改进空间——在 ELI5 数据集上，即使最好的模型也有 50% 的时间缺乏完整的引用支撑（lack complete citation support 50% of the time）[^24]。论文指出未来方向包括更好的检索器、长上下文 LLM、以及提升从多来源综合信息的能力[^24]。

### 5.2 AttributionBench

AttributionBench 关注生成式搜索引擎中「引用归因」的评估问题：即生成回答中的每条 claim 是否被其引用的证据完全支持[^7]。实验显示自动归因评估即使对 SOTA LLM 也很困难：微调后的 GPT-3.5 在二分类设定下 macro-F1 仅约 80%[^7]。对 300+ 错误案例的分析表明，多数失败源于模型无法处理细微信息（nuanced information），以及模型可访问的信息与人类标注者可访问信息之间的差异[^7]。

**有效性边界**：该基准衡量的是「自动评估归因」的难度，而非「归因机制降低幻觉」的幅度[^7]。其 80% macro-F1 意味着即使作为检测器，归因评估本身仍有约 20% 的误判空间。

### 5.3 Anthropic 官方推荐的引文验证流程

Anthropic 官方文档承认：即使最先进的语言模型（如 Claude）有时也会生成事实错误或与给定上下文不一致的文本，即「幻觉」[^15]。其推荐的基础策略包括：

- 允许 Claude 说「我不知道」——明确给予其承认不确定性的许可，可大幅减少虚假信息[^15]。
- 对长文档（>20k tokens）任务，要求 Claude 先逐字提取引文再执行任务，以将回答接地于实际文本[^15]。
- 用引文验证——让 Claude 为每条主张引用引文与来源；若找不到支持性引文则必须撤回该主张[^15]。

高级技术包括：Chain-of-thought verification（用 thinking display: summarized 审查推理块）、Best-of-N verification（多次运行同一 prompt 比较输出以发现不一致）、Iterative refinement、External knowledge restriction（明确指示仅使用所提供文档）[^15]。

**有效性边界**：文档明确声明这些技术能显著减少幻觉，但不能完全消除；高风险决策仍需人工验证关键信息[^15]。这是本报告中来自厂商官方文档的最明确边界声明。

---

## 六、工程护栏与评估类手段

### 6.1 Guardrails

Guardrails 是一个 Python 框架，通过两项核心功能构建可靠 AI 应用：(1) 在应用中运行 Input/Output Guards，检测、量化并缓解特定类型风险；(2) 帮助从 LLM 生成结构化数据[^13]。Guardrails Hub 是预构建的「validators」集合，多个 validator 可组合成 Input/Output Guards 拦截 LLM 输入输出[^13]。

README 公告（2025-02-12）称发布 Guardrails Index，自称是首个对比 24 个 guardrails 在 6 个最常见类别上性能与延迟的基准[^13]。README 公告（2026-07-06）称 Guardrails validators 正在迁移为可直接 pip 安装的标准 PyPI 包，Guardrails 将停止其托管的远程推理服务，计划截止日期为 2026-08-25[^13]。

**有效性边界与不确定性**：仓库页面显示 commit 总数为 3,306，但页面未显示 star 数[^13]。因此无法用 stars 与最近提交活跃度评估其工程落地热度——这正是子问题的核心指标之一。此外，远程推理服务停止计划[^13]可能影响依赖该服务的部署方案。

### 6.2 Ragas

Ragas 是评估与优化 LLM 应用的 toolkit，提供「客观指标、智能测试生成、数据驱动洞察」，用于替代耗时主观评估[^14]。关键特性包括：LLM-based 与传统指标、自动生成覆盖多场景的测试数据集、与 LangChain 及主流可观测性工具集成、利用生产数据构建反馈闭环[^14]。README 提供 `ragas quickstart rag_eval` 模板用于评估 RAG 系统，并列出 agent_evals、benchmark_llm、prompt_evals、workflow_eval 为「Coming Soon」[^14]。

**有效性边界与不确定性**：仓库页面显示 commit 总数为 1,147，但页面未显示 star 数[^14]。检索 explodinggradients/ragas 未直接命中主仓库，返回的是同名 fork（stars 0-1）；按 stars 排序检索时最高星仓库为 vibrantlabsai/ragas，stars 16023、forks 1761、主语言 Python、最近一次 push 时间为 2026-02-24[^27]。**explodinggradients/ragas 与 vibrantlabsai/ragas 的仓库归属关系未澄清，存在同一项目两个仓库的混淆风险。**

### 6.3 OpenAI Evals

OpenAI 官方文档将 evals 定义为测试模型输出是否满足指定风格与内容标准的机制，用于在升级或尝试新模型时理解 LLM 应用表现[^16]。构建与运行 eval 的三步流程：将任务描述为 eval、用测试输入运行 eval、分析结果并迭代改进 prompt[^16]。

**有效性边界**：文档披露 OpenAI 正在弃用 Evals 平台；现有 evals 内容在过渡窗口内仍可用，2026-10-31 起对现有用户变为只读，平台计划于 2026-11-30 关闭[^16]。文档建议新用户改用 Datasets 作为更迭代的实验环境[^16]。这意味着该评估路径的长期可用性受限。

### 6.4 HHEM 与幻觉排行榜

HHEM-2.1-Open 是 Vectara HHEM 系列中最新开源版本，用于检测 LLM 幻觉[^31]。其典型应用场景是 RAG 或 Agentic 工作流：一组事实（facts/context）被 LLM 摘要，HHEM 衡量该摘要与事实在事实一致性上的程度[^31]。HHEM 输出包括 verdict、0-1 一致性分数以及逐句分数（per-sentence scores）[^31]。模型卡提到配套的 VHC（Vectara Hallucination Corrector）用于纠正幻觉，与 HHEM 检测能力互补[^31]。HHEM-2.1-Open 基于 T5-base 基础模型[^31]。

排行榜由 HHEM 计算，评测目标是「LLM 在摘要一份文档时引入幻觉的频率」[^30]。榜单列出的指标为：Hallucination Rate、Factual Consistency Rate、Answer Rate、Average Summary Length (Words)[^30]。README 标注 Last updated on September 22, 2026[^30]。

榜单头部数据（幻觉率 / 事实一致性率）：antgroup/finix_s1_32b 1.8% / 98.2%；openai/gpt-5.4-nano-2026-03-17 3.1% / 96.9%；google/gemini-2.5-flash-lite 3.3% / 96.7%；microsoft/Phi-4 3.7% / 96.3%；meta-llama/Llama-3.3-70B-Instruct-Turbo 4.1% / 95.9%[^18]。其他模型示例：google/gemini-2.5-pro 7.0% / 93.0%；openai/gpt-6-astra 8.7% / 91.3%；openai/gpt-5.4-2026-03-05 7.0% / 93.0%；deepseek-ai/DeepSeek-V3.2-Exp 5.3% / 94.7%[^18]。

**有效性边界与不确定性**：该榜单只能作为单一来源的摘要任务指标，不能作为产业幻觉率结论（详见第七节）。

---

## 七、矛盾与不确定性

### 7.1 2025–2026 年 RAG 与归因进展无证据

本报告第四节与第五节所依据的 q2 证据全部为 2023–2024 年工作：ALCE 为 EMNLP 2023[^24]、AttributionBench 为 2024-02[^7]、NoMIRACL 为 EMNLP 2024 Findings[^6]。**没有任何 2025–2026 年的 RAG 幻觉缓解进展证据**（如 self-RAG/CRAG 之后的迭代、长上下文 RAG、多跳归因、检索器召回不足的量化边界）。因此，本报告无法就「RAG 与引用归因类方法在 2023–2026 年的学术进展」给出完整时间跨度结论；该子问题的时间跨度主张无证据支撑。

### 7.2 Vectara 榜单的交叉验证失败与内部排序反常

榜单出现 openai/gpt-6-astra、openai/gpt-6-sol、openai/gpt-5.5 等命名[^30]，与已收集的 OpenAI 官方文档[^16]、Wikipedia 条目[^29]均无法交叉验证。且榜单内部排序反常：gpt-6-astra 幻觉率 8.7% 高于 gpt-5.4-nano 的 3.1%[^18]；gemini-2.5-pro 7.0% 高于 gemini-2.5-flash-lite 3.3%[^18]。这与「模型越强幻觉越低」的直觉及 Google/Anthropic 官方「接地降低幻觉」的叙事相矛盾[^17][^15]。榜单同时混用带日期后缀（如 gpt-5.4-nano-2026-03-17）与不带日期后缀（如 gpt-6-sol、gpt-6-astra、gpt-5.5）两种命名风格[^30]。

**在矛盾未解决前，该榜单只能作为单一来源的摘要任务指标，不能作为产业幻觉率结论。** 此外，榜单 README 提供两个历史版本：基于 HHEM-1.0 的首版，以及基于旧数据集的 hhem-2.3-old-dataset 版本[^30]，说明榜单口径随版本变化，跨版本比较需谨慎。

### 7.3 工程框架活跃度指标缺失

证据明确记录 guardrails-ai/guardrails 与 explodinggradients/ragas 的「页面未显示 star 数」，仅给出 commit 总数（3,306 / 1,147）[^13][^14]。这无法支撑「stars 与最近提交活跃度」这一子问题核心指标。且 explodinggradients/ragas 与 vibrantlabsai/ragas 的仓库归属关系未澄清，后者 stars 16023、last push 2026-02-24[^27]，存在同一项目两个仓库的混淆风险。

### 7.4 厂商发布页缺乏幻觉量化指标

Anthropic 的 Claude 3.5 Sonnet 发布页列出的评测为 GPQA、MMLU、HumanEval 及内部 agentic coding 评测，未给出任何幻觉率或 groundedness 的量化数值[^28]。页面称 Claude 3.5 Sonnet 在内部 agentic coding 评测中解决了 64% 的问题，Claude 3 Opus 为 38%[^28]。页面将安全评测细节指向 model card addendum（PDF），正文本身不含幻觉/真实性指标[^28]。Wikipedia 的 GPT-4o 条目记录 MMLU 得分为 88.7（GPT-4 为 86.5），但未提及任何官方幻觉率或 groundedness 量化指标，也未引用 OpenAI system card 中的幻觉/真实性评测数值[^29]。这意味着厂商官方发布材料在幻觉维度上的披露不足，第三方榜单（如 HHEM）成为主要量化来源，而该来源本身存在 7.2 所述问题。

### 7.5 缓解手段有效性的共同边界

综合各来源，可归纳出三条被显式声明的边界：

1. **不能完全消除**：Anthropic 官方明确声明这些技术能显著减少幻觉，但不能完全消除；高风险决策仍需人工验证关键信息[^15]。
2. **真实性与有用性的权衡**：ITI 论文明确指出真实性（truthfulness）与有用性（helpfulness）之间存在权衡，并说明可通过调节干预强度来平衡该权衡[^11]。
3. **引用支撑的覆盖率不足**：ALCE 在 ELI5 上即使最好的模型也有 50% 的时间缺乏完整的引用支撑[^24]；AttributionBench 上微调 GPT-3.5 二分类 macro-F1 仅约 80%[^7]。

此外，TruthfulQA 的发现——最大的模型通常最不真实[^4][^22]——意味着「换更大模型」本身不构成缓解手段，反而可能加剧问题。

---

## 参考来源

[^1]: Survey of Hallucination in Natural Language Generation — https://arxiv.org/abs/2202.03629 (arxiv, 2022-02-08)
[^2]: HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models — https://arxiv.org/abs/2305.11747 (arxiv, 2023-05-19)
[^3]: FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation — https://arxiv.org/abs/2305.14251 (arxiv, 2023-05-23)
[^4]: TruthfulQA: Measuring How Models Mimic Human Falsehoods — https://arxiv.org/abs/2109.07958 (arxiv, 2021-09-08)
[^5]: Measuring short-form factuality in large language models (SimpleQA) — https://arxiv.org/abs/2411.04368 (arxiv, 2024-11-07)
[^6]: NoMIRACL: A Multilingual Relevance Assessment Dataset for RAG Applications — https://github.com/project-miracl/nomiracl (github, 2024-11-29)
[^7]: AttributionBench: How Hard is Automatic Attribution Evaluation? — https://arxiv.org/abs/2402.15089 (arxiv, 2024-02-23)
[^8]: DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models — https://arxiv.org/abs/2309.03883 (arxiv, 2023-09-07)
[^9]: Chain-of-Verification Reduces Hallucination in Large Language Models — https://arxiv.org/abs/2309.11495 (arxiv, 2023-09-20)
[^10]: Direct Preference Optimization: Your Language Model is Secretly a Reward Model — https://arxiv.org/abs/2305.18290 (arxiv, 2023-05-29)
[^11]: Inference-Time Intervention: Eliciting Truthful Answers from a Language Model — https://arxiv.org/abs/2306.03341 (arxiv, 2023-06-06)
[^12]: Self-Consistency Improves Chain of Thought Reasoning in Language Models — https://arxiv.org/abs/2203.11171 (arxiv, 2022-03-21)
[^13]: guardrails-ai/guardrails — Python framework for reliable AI applications (Input/Output Guards) — https://github.com/guardrails-ai/guardrails (github)
[^14]: explodinggradients/ragas — LLM application evaluation toolkit — https://github.com/explodinggradients/ragas (github)
[^15]: Anthropic Docs — Reduce hallucinations (Strengthen guardrails) — https://docs.anthropic.com/en/docs/test-and-evaluate/strengthen-guardrails/reduce-hallucinations (web)
[^16]: OpenAI Platform Docs — Evaluations (Evals API) — https://platform.openai.com/docs/guides/evals (web)
[^17]: Google Gemini API Docs — Grounding with Google Search — https://ai.google.dev/gemini-api/docs/grounding (web)
[^18]: vectara/hallucination-leaderboard — Public LLM hallucination leaderboard (HHEM) — https://github.com/vectara/hallucination-leaderboard (github, 2026-09-22)
[^19]: A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions (Huang et al.) — https://arxiv.org/abs/2311.05232 (arxiv, 2023-11-09)
[^20]: OpenMOSS/HalluQA — Dataset and evaluation code for measuring hallucinations in Chinese large language models — https://github.com/OpenMOSS/HalluQA (github, 2024-06-05)
[^21]: Evaluating Hallucinations in Chinese Large Language Models (arXiv:2310.03368v4) — https://arxiv.org/abs/2310.03368 (arxiv, 2023-10-25)
[^22]: TruthfulQA: Measuring How Models Mimic Human Falsehoods — https://arxiv.org/abs/2109.07958v2 (arxiv, 2021-09-08)
[^23]: FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation — https://arxiv.org/abs/2305.14251v2 (arxiv, 2023-05-23)
[^24]: Enabling Large Language Models to Generate Text with Citations (ALCE) — https://arxiv.org/abs/2305.14627 (arxiv, 2023-05-24)
[^25]: INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection — https://arxiv.org/abs/2402.03744 (arxiv, 2024-02-06)
[^26]: Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs (NeurIPS'24) — official implementation — https://github.com/AlexanderVNikitin/kernel-language-entropy (github)
[^27]: vibrantlabsai/ragas (原 explodinggradients/ragas) — https://github.com/vibrantlabsai/ragas (github)
[^28]: Claude 3.5 Sonnet (Anthropic 官方发布公告) — https://www.anthropic.com/news/claude-3-5-sonnet (web, 2024-06-21)
[^29]: GPT-4o (Wikipedia) — https://en.wikipedia.org/wiki/GPT-4o (web)
[^30]: vectara/hallucination-leaderboard (README) — https://raw.githubusercontent.com/vectara/hallucination-leaderboard/main/README.md (github, 2026-09-22)
[^31]: vectara/hallucination_evaluation_model (HHEM-2.1-Open model card, Hugging Face) — https://huggingface.co/vectara/hallucination_evaluation_model (web)