## 摘要

2025 年开源权重（open-weight）大模型的格局呈现三个可被一手证据支撑的结构性特征。第一，**头部阵营由「超大 MoE + 稀疏激活」主导**：DeepSeek-V3 以 671B 总参数 / 37B 激活参数、14.8T 预训练 token 树立了效率标杆[^3]，Kimi K2 把总参数推到 1T / 32B 激活、15.5T token[^4]，GLM-4.5 为 355B / 32B 激活、23T token[^6]，MiniMax-M1 为 456B / 45.9B 激活[^9]，形成「总参数膨胀、激活参数稳定在 30–46B 区间」的收敛形态。第二，**「开放权重」与「开源」在定义上被明确切开**：OSI 的 OSAID 1.0 规定 "Open Source models" 和 "Open Source weights" 必须包含推导这些参数所用的数据信息与代码，仅开放权重不足以称为 Open Source weights[^1]；而 2025 年的主流发布中，只有 OLMo 2 一类项目同时放出权重、完整训练数据、训练代码与配方、日志及数千中间检查点[^26]。第三，**许可证光谱高度分化**：Apache 2.0 被 Qwen3[^5][^7]、gpt-oss[^7] 采用，Llama 4 则采用带 7 亿 MAU 商用门槛、署名义务与蒸馏模型强制 "Llama" 前缀的社区许可[^2]，两者在权利结构上不可类比[^2][^16]。

需要前置声明：本报告的证据覆盖存在明显缺口——Mistral、Nvidia Nemotron 无任何一手证据，Llama 4 只有许可证而无模型规格，Gemma 与 DeepSeek 的许可证条款原文缺失，2025 年产业事件（R1 市场反应、Llama 4 反响、Qwen 衍生模型数量）零证据。凡涉及这些处，正文均显式标注。

---

## 一、判定框架：什么叫「开源权重」

讨论格局之前必须先固定术语，否则「开源」在 2025 年会被滥用为「权重可下载」的同义词。

**OSAID 1.0 的定义**：Open Source AI 是「以授予以下自由的方式提供的 AI 系统」——(1) 为任何目的使用系统且无需请求许可；(2) 研究系统如何工作并检查其组件；(3) 为任何目的修改系统，包括改变其输出；(4) 为任何目的共享系统供他人使用，无论是否修改[^1]。这四项自由同时适用于「完整可运行的系统」与「系统的离散组成元素」（system、model、weights and parameters 等），判定标准相同[^1]。行使自由的前提是能获得「进行修改的首选形式」（preferred form to make modifications）[^1]。

对机器学习系统，preferred form 必须包含三类要素[^1]：

- **Data Information**：足够详细的训练数据信息，使有技能者能构建实质等同的系统；须完整描述所有训练数据（含不可共享数据）并披露来源、范围与特征、获取与选择方式、标注流程、数据处理与过滤方法；列出所有公开可得的训练数据及获取位置；列出所有可从第三方（含付费）获得的训练数据及获取位置[^1]。
- **Code**：训练与运行系统的完整源代码，含数据处理/过滤、训练参数与设置、验证测试、tokenizer 等支撑库、推理代码、模型架构[^1]。
- **Parameters**：模型参数如权重或其他配置[^1]。

OSAID 同时给出组件级定义：「AI model」由模型架构、模型参数（含权重）和运行模型的推理代码组成；「AI weights」是叠加在模型架构上、由给定输入产生输出的那组学习到的参数[^1]。关键界定句是：『"Open Source models" and "Open Source weights" must include the data information and code used to derive those parameters.』[^1]

OSI 在 FAQ 中解释了为何不能照搬软件开源定义：Open Source Definition 第 2 条要求提供源代码且须为程序员修改程序的「首选形式」，但此前无人能回答「修改 AI 系统的首选形式是什么」[^21]；机器学习系统与数据、配置选项、文档以及权重与偏置等新产物交织在一起，因此需要独立于 OSD 的定义[^21]。OSI 明确否认「训练数据等于源代码」的类比，称其为常见误解，理由是 AI 系统并非以传统方式被编程，其能力在训练过程中自主获得[^21]；训练过程即便对系统创造者而言也难以可靠复现，这是 LLM 与 Linux 内核等可复现软件的关键差异[^21]。OSI 还承认其 Data Information 要求不强制完全可复现与完全透明，而是「使其成为可能」[^21]。

**这一框架的直接后果**：按 OSAID 判定，2025 年绝大多数被媒体称为「开源」的模型严格来说只是「开放权重」。本报告后续使用「开放权重」指代仅放出参数与推理代码的发布，使用「开源（OSAID 意义）」指代同时满足 Data Information + Code + Parameters 三要素的发布。

---

## 二、主要玩家与代表性模型

### 2.1 DeepSeek：效率路线与纯 RL 推理路线

DeepSeek-V3 是 MoE 语言模型，总参数 671B，每个 token 激活 37B；采用 Multi-head Latent Attention (MLA) 与 DeepSeekMoE 架构；预训练语料 14.8 万亿 tokens；全量训练仅需 2.788M H800 GPU 小时；checkpoint 发布于 GitHub[^3]。

DeepSeek-R1 的方法论贡献在于提出纯强化学习即可激励 LLM 的推理能力，无需人工标注的推理轨迹；该 RL 框架使自我反思、验证、动态策略调整等高级推理模式自发涌现；在数学、编程竞赛、STEM 等可验证任务上，纯 RL 训练的模型超过用人类示范做常规监督学习的对照模型[^8]。训练细节：R1-Zero 以 DeepSeek-V3-Base 为基座，采用 GRPO，学习率 3e-6、KL 系数 0.001、采样温度 1，共训练 10,400 步（约 1.6 个 epoch）；每个问题采样 16 个输出，8.2k 步之前最大长度 32,768 tokens、之后为 65,536 tokens；每 400 步用最新策略模型替换参考模型[^32]。训练中出现「aha moment」：模型自发学会重新审视并修正解题步骤，反思词 "wait" 的使用频率突增[^32]；思考长度随训练稳定增长[^32]。R1-Zero 在 AIME 2024 上 pass@1 从初始 15.6% 提升至 77.9%，配合自一致性解码可达 86.7%[^31]。由于 R1-Zero 存在可读性差与中英语言混杂问题，DeepSeek-R1 改用整合拒绝采样、强化学习与监督微调的多阶段流程[^31]，其基于模型的奖励使用 66,000 条偏好数据对训练奖励模型，并采用长度可比的数据构造以降低长度偏置[^32]。该工作已发表于 Nature volume 645, pages 633-638 (2025)[^31]。

**许可证缺口**：DeepSeek-V3/R1 的技术报告与 GitHub 链接均未给出许可证名称与条款，现有公开资料不足以确认其权重许可类型与商用/蒸馏条款。

### 2.2 月之暗面 Kimi K2：把总参数推到 1T

Kimi K2 是 MoE 模型，总参数 1 万亿，激活参数 32B；提出 MuonClip 优化器（Muon + QK-clip）以解决训练不稳定；预训练 15.5 万亿 tokens，全程零 loss spike；发布 base 与 post-trained 两类 checkpoint[^4]。基准成绩（均为非思考模式）：Tau2-Bench 66.1、ACEBench (En) 76.5、SWE-Bench Verified 65.8、SWE-Bench Multilingual 47.3、LiveCodeBench v6 53.7、AIME 2025 49.5、GPQA-Diamond 75.1、OJBench 27.1[^4]。作者自述其在开源非思考模型中达到 SOTA，此为主张性表述，需交叉验证[^4]。

### 2.3 阿里 Qwen：全尺寸覆盖 + Apache 2.0

Qwen3 系列同时包含 dense 与 MoE 架构，参数规模覆盖 0.6B 至 235B[^5]；具体尺寸为 0.6B、1.7B、4B、8B、14B、32B 以及 MoE 的 30B-A3B、235B-A22B[^19]。技术上，Qwen3 将 thinking 模式与非 thinking 模式统一到同一框架，并引入 thinking budget 机制[^5]；多语言支持从 Qwen2.5 的 29 种扩展到 119 种语言和方言[^5]。全部 Qwen3 模型以 Apache 2.0 许可公开[^5]。时间线上，Qwen3 首发于 2025.04.29，2025.07-08 陆续发布更新版 Qwen3-2507（Instruct / Thinking 两个变体，尺寸 235B-A22B、30B-A3B、4B）[^19]。

Qwen 系列的长上下文与多模态支线亦有独立证据：Qwen2.5-1M 将上下文扩展到 100 万 token，采用长数据合成、渐进式预训练与多阶段监督微调，并开源含长度外推方法的推理框架（无需额外训练即可把上下文扩展至少 4 倍甚至更多），配合稀疏注意力与 chunked prefill 在 100 万 token 场景实现 3x 到 7x 的 prefill 加速；开源模型为 Qwen2.5-7B-Instruct-1M 与 Qwen2.5-14B-Instruct-1M[^28]。Qwen2.5-VL 引入动态分辨率处理与绝对时间编码，可处理不同尺寸图像与长达数小时视频并支持秒级事件定位，从零训练原生动态分辨率 ViT 并引入 Window Attention；旗舰 Qwen2.5-VL-72B 在文档与图表理解上表现突出，论文称其与 GPT-4o、Claude 3.5 Sonnet 等 SOTA 模型相当[^30]。

**许可证不一致（须注意）**：Qwen3 技术报告称 Apache 2.0[^5]，但 QwenLM/Qwen3 README 正文未包含任何许可证章节或条款说明[^19]；QwenLM/Qwen2.5 的 README 已被 Qwen3 内容完全覆盖，无法提供 Qwen2.5 时期 Qwen License 的任何条款信息[^20]。因此「Qwen2.5 及更早为 Qwen License、Qwen3 转为 Apache 2.0」这一常见说法中，Qwen License 原文（含 MAU 门槛、再分发/命名限制）在现有证据中完全缺失，无法完成权利对照。

### 2.4 智谱 GLM：ARC 定位与多模态并行

GLM-4.5 是开源 MoE 模型，总参数 355B，激活参数 32B；采用混合推理方法，同时支持 thinking 与直接回复两种模式；多阶段训练共使用 23T tokens；基准成绩 TAU-Bench 70.1%、AIME 24 91.0%、SWE-bench Verified 64.2%；同时发布 GLM-4.5（355B）与紧凑版 GLM-4.5-Air（106B）[^6]。

多模态方向，GLM-4.5V 与 GLM-4.1V-Thinking 提出以推理为中心的训练框架：先通过大规模预训练构建能力强的视觉基座模型（作者认为这一步基本决定最终性能上限），再用 Reinforcement Learning with Curriculum Sampling (RLCS) 释放潜力，在 STEM 解题、视频理解、内容识别、编程、grounding、GUI agent、长文档解读等任务上全面提升；在 42 个公开基准上评测，GLM-4.5V 在同规模开源模型中近乎全部任务达到 SOTA，在 Coding 与 GUI Agent 等困难任务上可与 Gemini-2.5-Flash 竞争甚至更优；更小的 GLM-4.1V-9B-Thinking 在 29 个基准上优于更大的 Qwen2.5-VL-72B[^10]。论文另介绍 GLM-4.6V 系列，具备原生工具调用与 128K 上下文窗口[^10]。

### 2.5 OpenAI gpt-oss：首次以 Apache 2.0 放出权重

OpenAI 发布 gpt-oss-120b 与 gpt-oss-20b 两个开放权重推理模型；采用高效 MoE transformer 架构，训练使用大规模蒸馏与强化学习；针对 agentic 能力优化，包括深度研究浏览、python 工具调用、支持开发者自定义函数；模型权重、推理实现、工具环境与 tokenizer 均以 Apache 2.0 许可发布[^7]。这一发布的意义在于：OpenAI 此前不在此阵营，且其选择的 Apache 2.0 是无 MAU 门槛、无命名限制、无使用场景限制的许可[^16]。

### 2.6 MiniMax：混合注意力 + 超长上下文

MiniMax-M1 自称首个开源权重的、大规模 hybrid-attention 推理模型，架构为混合 MoE 加 lightning attention；基于 MiniMax-Text-01，总参数 456B，每 token 激活 45.9B；原生支持 100 万 token 上下文，为 DeepSeek R1 上下文长度的 8 倍；lightning attention 使 test-time compute 可高效扩展[^9]。算法上提出 CISPO：裁剪重要性采样权重而非 token 更新，优于其他竞争性 RL 变体；结合 hybrid-attention 与 CISPO，完整 RL 训练在 512 张 H800 上仅需三周，租用成本 534,700 美元；发布 40K 与 80K thinking budget 两个版本，其中 40K 版本是 80K 训练的中间阶段[^9]。

### 2.7 Google Gemma：开放权重但许可条款缺失

Gemma 是 Google DeepMind 的开放权重大语言模型家族，基于 Gemini 的研究与技术；官方 JAX 实现仓库列出 Gemma 1、Gemma 2、Gemma 3、Gemma 3n、Gemma 4 各版本的技术报告/模型卡链接，表明该家族已迭代至 Gemma 4；可运行在 CPU、GPU 和 TPU 上，GPU 场景下 2B checkpoint 建议 8GB+ 显存，7B checkpoint 使用 24GB+ 显存[^18]。

Gemma 3 是该家族的多模态新增成员，参数规模覆盖 1B 到 27B；引入视觉理解能力、更广的语言覆盖，以及至少 128K tokens 的更长上下文；架构上通过提高局部注意力层相对全局注意力层的比例、并保持局部注意力跨度较短，来降低长上下文下易膨胀的 KV-cache 内存；使用蒸馏方式训练，预训练版与指令微调版性能均优于 Gemma 2[^24]。报告称 Gemma3-4B-IT 可与 Gemma2-27B-IT 竞争，Gemma3-27B-IT 在多项基准上可与 Gemini-1.5-Pro 相比（属作者自述主张，需交叉验证）[^24]。

**许可证缺口**：Gemma 3 技术报告与 google-deepmind/gemma README 均未包含许可条款、使用限制、命名限制或 MAU 门槛信息[^18][^24]。Gemma Terms of Use 的商用、再分发、命名限制及 Prohibited Use Policy 在现有证据中无一手来源，因此无法与 Llama 社区许可做权利对照。

### 2.8 Meta Llama：只有许可证，没有规格

现有证据中关于 Llama 4 的唯一一手材料是 Llama 4 Community License Agreement，生效日 2025 年 4 月 5 日[^2]。其条款结构值得完整记录，因为它定义了「带条件的开放权重」这一类型：

- **授权性质**：非独占、全球、不可转让、免版税的有限许可，允许使用、复制、分发、拷贝、创作衍生作品及修改 Llama Materials[^2]。
- **再分发署名义务**：若分发 Llama Materials 或其衍生作品，或包含它们的任何产品或服务（含另一个 AI 模型），须随附协议副本，并在相关网站、用户界面、博客文章、about 页面或产品文档中显著展示 "Built with Llama"[^2]。
- **蒸馏模型命名限制**：若使用 Llama Materials 或其输出/结果来创建、训练、微调或以其他方式改进一个被分发或提供的 AI 模型，须在该 AI 模型名称的**开头**加上 "Llama"[^2]。
- **商用门槛**：若在 Llama 4 版本发布日，被许可方或其关联方提供的产品或服务的月活跃用户数在前一个自然月超过 7 亿，则必须向 Meta 申请许可，由 Meta 自行决定是否授予；在 Meta 明确授予前不得行使协议下任何权利[^2]。
- **其他**：须遵守通过引用并入的可接受使用政策；协议不授予商标许可（仅授予为遵守署名条款所必需的 "Llama" 标记使用许可）；被许可方对自行创作的衍生作品与修改拥有所有权；若被许可方对 Meta 提起诉讼主张 Llama Materials 或其输出侵犯其知识产权，则授予该被许可方的所有许可自起诉之日起终止，且须就第三方索赔对 Meta 进行赔偿；管辖法律为美国加利福尼亚州法律[^2]。

**规格缺口**：Llama 4（Scout / Maverick / Behemoth）的总参数、激活参数、上下文长度、训练 token 在现有证据中完全缺失——许可证据不能替代模型规格证据。Llama 4 的社区反响亦无来源。

### 2.9 Microsoft Phi：数据质量与合成数据路线

phi-4 是 140 亿参数的语言模型，训练配方以数据质量为核心；与多数以网页内容或代码等有机数据为主的预训练不同，phi-4 在整个训练过程中策略性地引入合成数据；Phi 家族此前模型主要蒸馏 GPT-4 的能力，phi-4 在 STEM 类问答能力上大幅超越其教师模型；对 phi-3 架构改动极小，性能提升主要来自改进的数据、训练课程与后训练方案创新[^25]。该来源未给出上下文长度、训练 token 总量或许可证信息[^25]。

### 2.10 AllenAI OLMo：唯一接近 OSAID 意义的发布

OLMo 2 是 AllenAI 的下一代完全开放语言模型，包含 7B、13B、32B 三个规模的稠密自回归模型；发布物完全开放：模型权重、完整训练数据、训练代码与配方、训练日志，以及数千个中间检查点[^26]。预训练数据混合引入新的专用数据配方 Dolmino Mix 1124，通过晚期课程训练（预训练退火阶段使用专用数据）显著提升下游基准表现；OLMo 2-Instruct 借鉴 Tülu 3 最佳实践，侧重宽松许可数据，并将最终阶段强化学习扩展为可验证奖励的 RLVR[^26]。作者称 OLMo 2 base 模型位于性能-训练算力的帕累托前沿，常匹配或超过 Llama 3.1、Qwen 2.5、Gemma 2 等仅开放权重的模型，且使用更少 FLOPs（属作者自述主张）[^26]。该来源未给出上下文长度、训练 token 总量或许可证名称[^26]。

按 OSAID 的三要素检验，OLMo 2 是现有证据中唯一明确同时提供 Data Information（完整训练数据）、Code（训练代码与配方）、Parameters（权重）的发布[^26][^1]；其余模型均只满足 Parameters 一项。

### 2.11 证据缺失的玩家

- **Mistral**：现有证据仅来自一个无关第三方项目（metalyzer，★1）的仓库描述，其中出现字符串 "mistralai/Ministral-3-8B-Instruct-2512"，提示可能存在名为 Ministral 3 8B Instruct、版本标记 2512 的权重模型；该字符串非 Mistral 官方来源，未提供参数量、上下文长度或许可证信息[^23]。Mixtral、Mistral Large 等完全无证据。**现有公开资料不足以确认 Mistral 在 2025 年的模型阵容与许可安排。**
- **Nvidia Nemotron**：现有证据中无任何 Nemotron 相关一手材料。**现有公开资料不足以确认。**
- **Gemma 4**：仅由官方 JAX 仓库的链接列表间接提示其存在[^18]，无技术报告或模型卡证据。

---

## 三、许可证光谱：从 Apache 2.0 到社区许可

2025 年开放权重模型的许可安排可归为三档，权利差异是实质性的。

**第一档：Apache 2.0（无门槛、含专利授权）**。§2 授予永久的、全球的、非独占的、免费的、免版税的、不可撤销的版权许可，允许 reproduce、prepare Derivative Works of、publicly display、publicly perform、sublicense 以及 distribute[^16]。§3 授予同等性质的专利许可，但含诉讼终止条款：若被许可人对任何实体提起专利诉讼（含反诉）主张该 Work 构成直接或共同专利侵权，则其在该 Work 下获得的专利许可自起诉之日起终止[^16]。再分发条件为：提供许可证副本、修改文件带显著声明、保留版权/专利/商标/归属声明、若含 NOTICE 文件则须包含其归属声明副本[^16]。§6 明确不授予商标许可[^16]。关键点：许可证正文未出现 "model"、"weights"、"distillation" 等词，也未设任何用户规模门槛或使用场景限制[^16]。采用者：Qwen3[^5]、gpt-oss（权重、推理实现、工具环境与 tokenizer 均以 Apache 2.0 发布）[^7]。

**第二档：MIT（最简）**。免费授予任何获得副本的人不受限制地处理本软件的权利，包括使用、复制、修改、合并、发布、分发、再许可和/或销售；唯一实质条件是版权声明与许可声明须包含在所有副本或主要部分中；正文未包含商标条款、未提及专利授权、未提及模型权重或蒸馏、未设任何用户规模门槛或使用场景限制[^17]。

**第三档：社区许可（带 MAU 门槛、署名与命名限制）**。Llama 4 社区许可为代表：7 亿 MAU 商用门槛、强制 "Built with Llama" 展示、蒸馏模型名称须以 "Llama" 开头、诉讼即终止、加州法律管辖[^2]。这与 Apache 2.0 的「不可撤销 + 无场景限制」形成直接对照[^16]。

**判定要点**：OSI 强调「open source」不只是可检视或可见、可获得，而是向接收者传递永久的分叉权利且无需额外费用[^22]；并规定只有以 OSI 认可的 Open Source 许可证发布的软件才应被标称为 Open Source 软件，且开源许可证不得歧视使用领域，也不得歧视个人或群体[^22]。按此标准，带 MAU 门槛的社区许可在「不得歧视使用领域/个人或群体」这一条上存在张力——但 OSAID 本身允许许可证包含「要求修改版本以与原版相同的条款发布」这类 copyleft 式条件[^1]，且 OSI 表示不要求特定的法律机制来保证模型参数对所有人自由可得，参数可以本身自由，也可通过许可证或其他法律工具确保其自由，并称随着法律体系对开源 AI 系统的进一步回应，这一点会随时间更清晰[^1]。因此「Llama 4 是否算开源」在 OSAID 框架下并非一句话可断，但「Llama 4 不是 OSI 认可的开源许可证发布」这一点在现有证据下可以成立[^2][^22]。

**无法完成的部分**：DeepSeek 权重许可、Gemma Terms of Use、Qwen License（Qwen2.5 及更早）三项原文均缺失，故无法给出完整的跨玩家许可权利对照表。

---

## 四、技术趋势：从证据中可归纳的四条线

**（1）MoE + 稀疏激活成为头部标配，激活参数收敛**。DeepSeek-V3 671B/37B[^3]、Kimi K2 1T/32B[^4]、GLM-4.5 355B/32B[^6]、MiniMax-M1 456B/45.9B[^9]、Qwen3 MoE 235B-A22B 与 30B-A3B[^19]、gpt-oss 采用高效 MoE transformer[^7]——总参数跨度从 235B 到 1T，但每 token 激活参数集中在 22B–46B。这意味着推理成本与总参数规模脱钩，模型容量通过稀疏化扩张。

**（2）推理模式统一化**。Qwen3 把 thinking 与非 thinking 统一到同一框架并引入 thinking budget[^5]；GLM-4.5 采用混合推理同时支持 thinking 与直接回复[^6]；MiniMax-M1 发布 40K 与 80K 两档 thinking budget[^9]。三家的做法指向同一目标：让推理深度成为可调参数而非模型选型。

**（3）纯 RL 与可验证奖励成为后训练主线**。DeepSeek-R1 证明纯 RL 可激励推理且高级模式自发涌现[^8]，其 GRPO 配置与「aha moment」现象被完整记录[^32]；OLMo 2-Instruct 将最终阶段 RL 扩展为可验证奖励的 RLVR[^26]；GLM-4.5V 用 RLCS 在 42 个基准上全面提升[^10]；MiniMax-M1 提出 CISPO 并给出完整 RL 训练的算力账（512 张 H800、三周、534,700 美元）[^9]。蒸馏路线亦被明确：大规模模型涌现的推理模式可被系统性地用于指导并提升小模型[^8][^31]。

**（4）长上下文与注意力效率**。Qwen2.5-1M 把上下文推到 100 万 token 并开源含长度外推的推理框架[^28]；MiniMax-M1 原生支持 1M 上下文，称其为 R1 上下文长度的 8 倍[^9]；Gemma 3 通过调整局部/全局注意力层比例控制 KV-cache 膨胀，上下文至少 128K[^24]；GLM-4.6V 具备 128K 上下文窗口[^10]。底层方法侧，YaRN 作为 RoPE 上下文扩展方法，作者自述相比此前方法只需 1/10 的 token 量和 1/2.5 的训练步数，并可外推到微调数据集上下文长度之外[^27]；NSA 作为可原生训练的稀疏注意力，采用动态分层稀疏策略（粗粒度 token 压缩 + 细粒度 token 选择），在 64k 长度序列上 decoding、前向与反向三个环节相对 Full Attention 均有显著加速，且用 NSA 预训练的模型在通用基准、长上下文任务与指令推理上保持或超过 Full Attention 模型[^29]。

---

## 五、工程落地生态（仅作存在性说明，不作热度结论）

围绕开放权重模型的推理与接入层已形成一组被广泛引用的项目，但本报告只能确认其存在与自述定位，不能确认其 2025 年热度。

- **ollama/ollama**：README 自述定位为在本地或云端运行开源模型的运行时，强调隐私；提供 macOS / Windows / Linux / Docker 四种安装方式与 ollama-python、ollama-js 两个官方客户端库；提供 REST API（默认端口 11434，/api/chat 端点）；仓库根目录含 llama、mlx、mlxrunner、openai、harmony、thinking、tools 等子目录及 LLAMA_CPP_VERSION、MLX_VERSION、MLX_C_VERSION 版本文件，表明同时内嵌 llama.cpp 与 Apple MLX 两套推理后端；README 称可接入 Claude Code、OpenClaw、OpenCode、Codex、Copilot 等编码 agent[^11]。其 README 标题直接列出 Kimi、GLM、MiniMax、DeepSeek、gpt-oss、Qwen、Gemma 等模型族，可视为上述玩家阵容在工具层的一个侧证[^11]。
- **vllm-project/vllm**：README 称最初由 UC Berkeley 的 Sky Computing Lab 开发，由来自数十个学术机构与公司的社区维护、贡献者超过 2000 人（项目自述，非第三方认证）；核心技术包括 PagedAttention 的 KV cache 管理、连续批处理、chunked prefill、前缀缓存；支持 FP8、MXFP8/MXFP4、NVFP4、INT8、INT4、GPTQ/AWQ、GGUF、compressed-tensors、ModelOpt、TorchAO 等多种量化格式；支持分离式 prefill、decode 与 encode；集成 FlashAttention、FlashInfer、TRTLLM-GEN、FlashMLA、Triton 等注意力内核，以及 EAGLE、DFlash、n-gram、suffix 等投机解码方案[^12]。
- **llama.cpp / open-webui / litellm**：仅取得检索元数据（仓库描述与 star 数），未读取 README 正文，故不作为工程细节结论依据[^13][^14][^15]。

**热度数据不可用**：上述仓库快照的 last push 均为 2026-10-24/25，vLLM release 页面最新版本为 v0.32.0、transformers 为 v5.20.2，均明显超出 2025 年 1–10 月窗口，疑为环境时钟偏差[^11][^12][^13][^14][^15][^33][^34]。这些 star 数与提交活跃度**不能作为「2025 年热度」证据**。同理，transformers v5.20.0 说明中列出的 Muse Spark Open（自述为 Meta 的多模态 MoE 模型，816B 总参数、42B 激活参数、上下文 1,048,576 tokens）与 Ling 3.0 Flash，因版本号与日期序列无法与 2025 年发布节奏对齐，亦不能作为 2025 年窗口内的时间点证据[^34]。

---

## 矛盾与不确定性

**1. 「开源」标签的系统性误用。** OSAID 要求 Open Source weights 必须包含推导参数所用的数据信息与代码[^1]，而 2025 年主流发布中只有 OLMo 2 明确同时提供权重、完整训练数据、训练代码与配方、日志及数千中间检查点[^26]。其余（DeepSeek、Kimi、Qwen、GLM、MiniMax、gpt-oss、Gemma、Phi）在现有证据中均只确认了权重与（部分）推理代码。因此「2025 年开源大模型格局」这一提法本身需要限定为「开放权重大模型格局」。

**2. Qwen 许可证的未解决不一致。** Qwen3 技术报告称全部模型以 Apache 2.0 公开[^5]，但 QwenLM/Qwen3 README 无许可证章节[^19]，QwenLM/Qwen2.5 README 已被 Qwen3 内容覆盖、无法提供 Qwen License 条款[^20]。两来源对 Qwen 许可的表述存在未解决的不一致，且 Qwen License 原文（含 MAU 门槛、再分发/命名限制）完全缺失，无法完成许可权利对照。

**3. DeepSeek 权重许可完全缺失。** DeepSeek-V3 与 R1 的技术报告及 GitHub 链接均未给出许可证名称与条款[^3][^8][^31]，DeepSeek License 与 MIT 之争需一手确认。这是本报告许可对照表最关键的一行空缺。

**4. Gemma 许可条款完全缺失。** Gemma 3 技术报告与官方 JAX 仓库 README 均不含许可条款、使用限制、命名限制或 MAU 门槛[^18][^24]。Gemma Terms of Use 与 Prohibited Use Policy 无一手来源，无法判断 Gemma 与 Llama 社区许可在权利结构上的实际差异。

**5. Llama 4 只有许可证、没有规格。** 现有证据仅覆盖 Llama 4 社区许可条款[^2]，Scout / Maverick / Behemoth 的总参数、激活参数、上下文长度、训练 token 全部缺失，Llama 4 的社区反响亦无来源。因此本报告无法把 Llama 4 放入第 2 节的模型规格对比，也无法评估其在 2025 年格局中的实际位置。

**6. Mistral 与 Nvidia Nemotron 零可用证据。** Mistral 仅有第三方仓库描述中的 "Ministral-3-8B-Instruct-2512" 字符串，属不可用证据[^23]；Nemotron 无任何一手材料。这两个玩家在 2025 年格局中的位置，**现有公开资料不足以确认**。

**7. 2025 年产业事件零证据。** DeepSeek-R1 发布引发的市场反应、Llama 4 反响、gpt-oss 开放权重的行业影响、Qwen 衍生模型数量超越 Llama 等说法，在给定证据中没有任何 web 来源（官方博客、可信媒体、Hugging Face 衍生模型统计）覆盖。本报告因此不对产业动态作任何事实性陈述。

**8. 工程热度数据整体不可用。** 所有 GitHub 快照的 last push 为 2026-10-24/25，vLLM release 为 v0.32.0、transformers 为 v5.20.2，超出 2025 年窗口，疑为环境时钟偏差[^11][^12][^13][^14][^15][^33][^34]。llama.cpp、open-webui、litellm 仅有检索元数据、未读 README[^13][^14][^15]。因此第 5 节只能作存在性说明，不能支撑「2025 年工程落地热度」的任何排序结论。

**9. 作者自述主张需交叉验证。** Kimi K2 的「开源非思考模型 SOTA」[^4]、Gemma 3 的「27B-IT 可与 Gemini-1.5-Pro 相比」[^24]、OLMo 2 的「帕累托前沿」[^26]、Qwen3 README 的「open-weight thinking models 中 SOTA」[^19]、GLM-4.5V 的「同规模开源模型近乎全部任务 SOTA」[^10] 均为作者或团队自述，本报告未获得独立第三方复现证据。

## 参考来源

[^1]: The Open Source AI Definition – version 1.0 (Open Source Initiative) — https://opensource.org/ai/open-source-ai-definition（web, 2024-10）
[^2]: LLAMA 4 COMMUNITY LICENSE AGREEMENT (meta-llama/llama-models, models/llama4/LICENSE) — https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE（github, 2025-04-05）
[^3]: DeepSeek-V3 Technical Report — https://arxiv.org/abs/2412.19437v2（arxiv, 2024-12-27）
[^4]: Kimi K2: Open Agentic Intelligence — https://arxiv.org/abs/2507.20534v2（arxiv, 2025-07-28）
[^5]: Qwen3 Technical Report — https://arxiv.org/abs/2505.09388v1（arxiv, 2025-05-14）
[^6]: GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models — https://arxiv.org/abs/2508.06471v1（arxiv, 2025-08-08）
[^7]: gpt-oss-120b & gpt-oss-20b Model Card — https://arxiv.org/abs/2508.10925v1（arxiv, 2025-08-08）
[^8]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning — https://arxiv.org/abs/2501.12948v2（arxiv, 2025-01-22）
[^9]: MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention — https://arxiv.org/abs/2506.13585v1（arxiv, 2025-06-16）
[^10]: GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning — https://arxiv.org/abs/2507.01006v6（arxiv, 2025-07-01）
[^11]: ollama/ollama — Get up and running with Kimi, GLM, MiniMax, DeepSeek, gpt-oss, Qwen, Gemma and other models — https://github.com/ollama/ollama（github）
[^12]: vllm-project/vllm — A high-throughput and memory-efficient inference and serving engine for LLMs — https://github.com/vllm-project/vllm（github）
[^13]: ggml-org/llama.cpp — LLM inference in C/C++ (仓库检索元数据) — https://github.com/ggml-org/llama.cpp（github）
[^14]: open-webui/open-webui — User-friendly AI Interface (Supports Ollama, OpenAI API, ...) (仓库检索元数据) — https://github.com/open-webui/open-webui（github）
[^15]: BerriAI/litellm — AI Gateway, 100+ LLM APIs (仓库检索元数据) — https://github.com/BerriAI/litellm（github）
[^16]: Apache License, Version 2.0 (official text, apache.org) — https://www.apache.org/licenses/LICENSE-2.0（web, 2004-01）
[^17]: The MIT License (official text, opensource.org) — https://opensource.org/license/mit（web）
[^18]: google-deepmind/gemma — README (Gemma 官方 JAX 实现仓库) — https://raw.githubusercontent.com/google-deepmind/gemma/main/README.md（github）
[^19]: QwenLM/Qwen3 — 官方 GitHub 仓库 README — https://raw.githubusercontent.com/QwenLM/Qwen3/main/README.md（github, 2025-08-08）
[^20]: QwenLM/Qwen2.5 仓库 README（实际返回内容为 Qwen3 README） — https://raw.githubusercontent.com/QwenLM/Qwen2.5/main/README.md（github）
[^21]: Open Source AI Definition – Frequently Asked Questions (OSI) — https://opensource.org/ai/faq（web）
[^22]: OSI Frequently Asked Questions (general open source FAQ) — https://opensource.org/faq（web）
[^23]: GitHub 仓库搜索 "mistralai"（按 updated 排序）返回的第三方仓库列表 — https://github.com/aldertzomer/metalyzer（github）
[^24]: Gemma 3 Technical Report (arXiv:2503.19786v1) — https://arxiv.org/abs/2503.19786v1（arxiv, 2025-03-25）
[^25]: Phi-4 Technical Report (arXiv:2412.08905v1) — https://arxiv.org/abs/2412.08905v1（arxiv, 2024-12-12）
[^26]: 2 OLMo 2 Furious (arXiv:2501.00656v3) — https://arxiv.org/abs/2501.00656v3（arxiv, 2024-12-31）
[^27]: YaRN: Efficient Context Window Extension of Large Language Models — https://arxiv.org/abs/2309.00071v3（arxiv, 2023-08-31）
[^28]: Qwen2.5-1M Technical Report — https://arxiv.org/abs/2501.15383v1（arxiv, 2025-01-26）
[^29]: Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention — https://arxiv.org/abs/2502.11089v2（arxiv, 2025-02-16）
[^30]: Qwen2.5-VL Technical Report — https://arxiv.org/abs/2502.13923v1（arxiv, 2025-02-19）
[^31]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning — https://arxiv.org/abs/2501.12948（arxiv, 2025-01-22）
[^32]: DeepSeek-R1 (HTML 全文，第 1–3 节) — https://arxiv.org/html/2501.12948v2（arxiv, 2026-01-04）
[^33]: vllm-project/vllm Releases 页面（抓取到的版本列表与最新 release 说明） — https://github.com/vllm-project/vllm/releases（github）
[^34]: huggingface/transformers Releases 页面（抓取到的版本列表与最新 release 说明） — https://github.com/huggingface/transformers/releases（github）