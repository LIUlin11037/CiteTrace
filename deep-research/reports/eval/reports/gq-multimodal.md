# 2025年多模态大模型的主流架构路线

## 摘要

2025 年多模态大模型（MLLM）的架构演进呈现三条清晰主线：**视觉编码与语言主干如何耦合**、**理解与生成是否统一**、**稠密与稀疏（MoE）如何取舍**。以 Qwen2.5-VL/Qwen3-VL、InternVL3/3.5、DeepSeek-VL2 为代表的开源旗舰延续「视觉编码器 + 连接器 + LLM 主干」的模块化范式，但在视觉侧引入原生动态分辨率、窗口注意力、多级特征融合等改造[^2][^3][^5][^19]；以 Chameleon、Emu3、Transfusion、Janus/Janus-Pro、Show-o 为代表的统一路线则尝试用单一 transformer 同时完成理解与生成，分歧集中在「视觉 token 离散还是连续」以及「理解与生成是否共用视觉编码器」[^4][^11][^12][^13][^14][^15]；MoE 从语言侧扩展到视觉语言整体架构，成为 2025 年开源与闭源旗舰的共同选择[^8][^19][^5]。闭源厂商方面，Google 明确以「native multimodality + 长上下文 + thinking」为架构叙事[^7]，Meta 以「原生多模态 + MoE」定义 Llama 4[^8]，Anthropic 则以 hybrid reasoning 模式而非架构细节作为 Claude 4 的主要卖点[^26]。需要强调的是，本报告所引性能数字绝大多数来自论文或厂商自述，缺少独立第三方基准交叉验证，因此**不足以支撑「哪条路线更优」的结论**。

---

## 一、概念界定：什么是 MLLM

按被 National Science Review 接收的综述定义，MLLM 是「基于 LLM、能够接收、推理并输出多模态信息的模型」（LLM-based model with the ability to receive, reason, and output with multimodal information）[^1]。该综述将 MLLM 之前的传统多模态工作划分为判别式（代表 CLIP）与生成式（代表 OFA）两大范式，并指出 MLLM 按序列操作方式可归入生成式范式；其核心特征是使用强大的 LLM 作为「大脑」执行多模态任务，代表模型为 GPT-4V[^1]。该综述的分析框架覆盖架构、训练策略与数据、评估三个维度[^1]。

这一定义是本节后续所有路线讨论的基线：凡被称为「多模态大模型」的系统，其语言主干必须是 LLM，而非独立的多模态编码器组合。

---

## 二、路线一：模块化「编码器—连接器—LLM」范式及其 2025 年改造

这是证据最充分、工程落地最广的路线。其基本结构为：视觉编码器提取图像/视频特征，经视觉-语言连接器映射后送入 LLM 主干。

**奠基性结论来自 MM1 的系统消融**：在大规模多模态预训练中，混合使用 image-caption、交错图文与纯文本数据对取得跨基准 SOTA few-shot 结果至关重要；图像编码器连同图像分辨率与图像 token 数量有实质性影响，而**视觉-语言连接器的设计相对而言重要性可忽略**（comparatively negligible importance）[^17]。MM1 据此构建了最高 30B 参数、含稠密与 MoE 变体的模型系列[^17]。

**LLaVA 路线**是这一范式的最小实现：LLaVA-1.5 使用 CLIP-ViT-L-336px 作为视觉编码器、MLP 投影作为连接器，论文主张全连接视觉-语言跨模态连接器「出人意料地强大且数据高效」，最终 13B checkpoint 仅用 1.2M 公开数据，完整训练在单个 8-A100 节点上约 1 天完成[^18]。其官方仓库已停止更新（last push 2024-08-12），并说明后续代码迁移至 LLaVA-VL/LLaVA-NeXT 仓库[^22][^10]。LLaVA-NeXT 可处理 4 倍于之前的像素量，并支持 Llama-3 (8B) 与 Qwen-1.5 (72B/110B) 作为语言主干[^22]。

**2025 年的主要改造集中在视觉侧与位置编码**：

- **Qwen2.5-VL** 采用从零训练的原生动态分辨率 Vision Transformer，并引入 Window Attention 以降低计算开销同时保持原生分辨率；引入动态分辨率处理与绝对时间编码，支持可变尺寸图像与长达数小时的视频，并可做到秒级事件定位[^2]。
- **Qwen3-VL** 的 README 列出三项架构更新：Interleaved-MRoPE（在时间、宽、高维度做全频段位置编码分配，增强长时程视频推理）、DeepStack（融合多层级 ViT 特征以捕捉细粒度细节并锐化图文对齐）、Text-Timestamp Alignment（超越 T-RoPE，实现精确的时间戳锚定事件定位）[^5][^20]。
- **InternVL3** 采用「原生多模态预训练」范式：在单一预训练阶段同时从多模态数据与纯文本语料中联合获得多模态与语言能力，而非把纯文本 LLM 事后改造为 MLLM；并引入 variable visual position encoding (V2PE) 以支持扩展的多模态上下文[^3]。
- **DeepSeek-VL2** 在视觉端用 dynamic tiling（动态切块）编码策略取代前代的固定分辨率 hybrid vision encoder（384×384 与 1024×1024），以处理不同宽高比的高分辨率图像[^19]。

**训练与后训练配方**同样构成该路线的一部分：InternVL3 后训练采用 SFT 与 mixed preference optimization (MPO)，并配合测试时扩展（test-time scaling）策略[^3]；InternVL3.5 开源了 CascadeRL 训练代码，含 offline RL 与 online RL 两阶段[^6]。

**多图与交错输入能力**是这一范式的延伸：xGen-MM (BLIP-3) 发布了 4B 与 14B 模型（含预训练基座与指令微调版本），具备理解交错图文输入的能力，并在单图与多图基准上评估[^16]；MM1 因大规模预训练而具备增强的上下文学习与多图推理能力，支持 few-shot 思维链提示[^17]。

---

## 三、路线二：统一理解与生成的单一 transformer

这条路线的问题意识是：能否不用「编码器 + 扩散模型」的组合式架构，而用单个 transformer 同时完成多模态理解与生成。证据中至少可分出四种技术取向。

**(a) 完全离散 token + 纯 next-token prediction。** Emu3 将图像、文本、视频 tokenize 到离散空间，在混合多模态序列上从零训练单个 transformer，仅用 next-token prediction 训练[^11]。论文自述其在生成与感知任务上均超过若干成熟的任务专用模型，超越 SDXL 与 LLaVA-1.6 等旗舰模型，且无需 diffusion 或组合式架构；并可通过预测视频序列中的下一个 token 生成高保真视频[^11]。

**(b) 离散与连续混合：语言建模损失 + diffusion。** Transfusion 将语言建模损失（next token prediction）与 diffusion 结合，在混合模态序列上训练单个 transformer，从零预训练了多个最高 7B 参数的模型，并建立了跨单模态与跨模态基准的 scaling laws[^12]。其关键实验结论是：Transfusion 的扩展性**显著优于**「量化图像 + 在离散图像 token 上训练语言模型」的做法；引入模态专属的编码与解码层可进一步提升性能，甚至将每张图像压缩到仅 16 个 patch；扩展到 7B 参数、2T 多模态 token 时，生成图像与文本的能力可与同规模 diffusion 模型和语言模型相当[^12]。

**(c) Early-fusion 混合模态。** Chameleon 是一族基于 token 的 early-fusion 混合模态模型，可在任意序列中理解并生成图像与文本；论文描述了面向 early-fusion、token-based、mixed-modal 设定的架构参数化与稳定训练方法及对齐配方[^4]。其官方仓库默认运行 7B 参数模型，可切换至 30B，运行推理与 viewer 需要 CUDA-capable GPU，使用受 Chameleon Research License 约束[^23]。

**(d) 解耦视觉编码。** Janus 指出此前研究（如 Chameleon）对理解与生成共用单一视觉编码器，因两任务所需信息粒度不同而导致理解性能次优；Janus 将视觉编码解耦为独立路径，但仍使用单一统一的 transformer 架构处理[^13]。论文称解耦既缓解了视觉编码器在理解与生成两种角色间的冲突，也提升了框架灵活性——理解与生成组件可各自独立选择最合适的编码方式[^13]。Janus-Pro 作为增强版本，改进来自优化的训练策略、扩展的训练数据、扩展到更大模型规模三方面，含 1B 与 7B 两种规模[^15]。

**(e) 自回归与离散 diffusion 的统一。** Show-o 不同于完全自回归模型，统一了自回归与（离散）diffusion 建模，以自适应处理各种及混合模态的输入输出；支持视觉问答、文本到图像生成、文本引导的 inpainting/extrapolation、混合模态生成等任务[^14]。该论文被 ICLR 2025 接收，代码与模型已开源[^14]。

---

## 四、路线三：MoE 稀疏化

MoE 在 2025 年从语言侧扩展到完整的多模态架构，成为开源与闭源旗舰的共同选择。

**开源侧**：DeepSeek-VL2 是 MoE 视觉-语言模型系列，含 Tiny / Small / VL2 三个变体，激活参数量分别为 1.0B、2.8B、4.5B；语言组件使用 DeepSeekMoE，并采用 Multi-head Latent Attention (MLA) 将 Key-Value cache 压缩为 latent 向量以提升推理效率与吞吐[^19][^24]。三个变体序列长度均为 4096，于 2024-12-13 发布[^24]。Qwen3-VL 提供 Dense 与 MoE 两种架构，覆盖 edge 到 cloud 规模，并有 Instruct 与 reasoning-enhanced Thinking 两个版本[^5][^20]。InternVL3.5 最大模型为 InternVL3.5-241B-A28B，另提供基于 GPT-OSS-20B-A4B 的 20B-A4B 版本[^6]。

**闭源侧**：Meta 称 Llama 4 Scout 与 Maverick 是「首批开放权重的原生多模态模型」，也是 Meta 首批采用 MoE 架构的模型；Scout 为 17B 激活参数、16 个专家，声称可在单张 NVIDIA H100 GPU（Int4 量化）上运行，提供 10M 上下文窗口；Maverick 为 17B 激活参数、128 个专家；Behemoth 为 288B 激活参数、16 个专家，作为教师模型用于蒸馏，发布时仍在训练中[^8]。

值得注意的是，MM1 早在 2024 年就已构建含稠密与 MoE 变体的系列[^17]，说明 MoE 并非 2025 年才出现的新路线，而是在 2025 年被旗舰模型普遍采纳。

---

## 五、闭源厂商的架构取向

**Google**：官方明确表述 Gemini 2.5 建立在「native multimodality 与长上下文窗口」之上（'Gemini 2.5 builds on what makes Gemini models great — native multimodality and a long context window'）[^7]。Gemini 2.5 Pro 发布时提供 100 万 token 上下文窗口，并预告 200 万 token 即将推出；官方称正把 thinking 能力直接内建到所有模型中，以支持更复杂问题与更具上下文感知的 agent[^7]。其模型页当前展示 Gemini 3.1 Pro，描述为具备 advanced multimodal understanding，覆盖 text、images、video、audio 甚至 code，并列有名为 Thinking 的推理模式档位[^27]；技术页当前展示 Gemini 4 Argon，称其可将原始视频自主转换为物理精确的仿真，估计深度与物理边界并编写仿真代码[^28]。**需注意这两页内容随时间滚动更新，并非 2025 年时点的快照**[^27][^28]。

**Meta**：以「原生多模态 + MoE」为核心叙事，Llama 4 Scout/Maverick 在 llama.com 与 Hugging Face 开放下载，并通过 WhatsApp、Messenger、Instagram Direct 与 Meta.AI 网站提供 Meta AI 体验[^8]。

**Anthropic**：Claude 4（Opus 4 与 Sonnet 4，2025-05-22 发布）被官方描述为 hybrid models，提供 near-instant responses 与 extended thinking 两种模式，支持 Extended thinking with tool use（beta）、并行工具调用、更精确的指令遵循与改进的记忆能力[^26]。**该公告未披露多模态架构层面的取向**，其公开叙事集中在推理模式与工具使用[^26]。

**OpenAI**：证据中仅有 GPT-4o 的 Wikipedia 条目，描述其为多语言、多模态生成式预训练 transformer，2024 年 5 月发布，可处理并生成文本、图像与音频，与依赖其他模型处理声音的 GPT-3.5/GPT-4 不同，原生支持 voice-to-voice[^9]。该条目同时记录 GPT-4o 于 2025 年 8 月随 GPT-5 发布从 ChatGPT 移除、后对付费订阅者重新提供、2026 年 2 月 13 日正式退役但 API 仍可用[^9]。**这是二手来源，且描述的是 2024 年发布的模型，缺少 GPT-5 / GPT-5.x 系列多模态架构取向的一手来源。**

**字节**：证据中仅有 bytedance/UI-TARS-desktop 一个仓库，自述为「The Open-Source Multimodal AI Agent Stack」，★39274，last push 2026-10-09[^10]。**缺少豆包/Doubao 或 Seed 系列多模态模型的一手架构披露。**

---

## 六、矛盾与不确定性

**（一）「跨模态注意力深度融合」路线在证据中完全空缺。** 本课题要求覆盖的四条路线中，Flamingo 类 gated cross-attention、Idefics、以及 Qwen-VL 第一代使用的 cross-attention adapter 均无对应证据条目。现有 28 条证据中没有任何一条直接支撑这一路线。Qwen-VL 第一代仓库 README 仅说明 Qwen-VL-Plus 与 Qwen-VL-Max 支持百万像素级超高分辨率与极端宽高比图像输入，并给出 DocVQA 93.1%、ChartQA 79.8%、AI2D 79.3%、TextVQA 79.5%、MMMU 51.4%、MathVista 51.0%、MM-Bench-CN 75.1% 等基准数字，但**未描述其视觉-语言连接机制**[^21]。这是一处覆盖度缺口，本报告无法就该路线给出任何有证据支撑的论述。

**（二）性能数字无法横向比较。** 现有性能数字分散在不同论文与厂商的自述主张中：InternVL3-78B 在 MMMU 上取得 72.2 分，作者称其为开源 MLLM 的新 SOTA[^3]；Janus-Pro-7B 在 MMBench 上得分 79.2，超过 Janus（69.4）、TokenFlow（68.9）与 MetaMorph（75.2），在 GenEval 上得分 0.80，超过 Janus（0.61）、DALL-E 3（0.67）与 Stable Diffusion 3 Medium[^15]；Qwen2.5-VL-72B 官方称在文档与图表理解上可与 GPT-4o、Claude 3.5 Sonnet 等 SOTA 模型匹敌[^2]。这些数字**评测基准不同、模型规模不同、且均为自述**，没有任何独立第三方基准（如 OpenCompass、MMMU 官方榜、LMArena）或综述性对比来交叉验证。因此本报告**无法就「哪条架构路线更优」给出结论**。

**（三）自述 SOTA 主张的普遍性。** 证据中大量性能声明明确标注为作者或厂商自述：Qwen2.5-VL-72B 匹敌 GPT-4o[^2]、InternVL3-78B 开源新 SOTA[^3]、Chameleon 在图像描述上达到 SOTA 且文本任务超过 Llama-2[^4]、LLaVA-1.5 在 11 个基准上达到 SOTA[^18]、DeepSeek-VL2 以更少激活参数达到 competitive 或 SOTA[^19][^24]、Qwen-VL-Max 与 Gemini Ultra 和 GPT-4V 表现相当[^21]、LLaVA-NeXT-34B 在某些基准上超越 Gemini Pro[^22]、InternVL3.5-241B-A28B 在开源 MLLM 中取得 SOTA[^6]、InternVL2_5-78B 是首个在 MMMU 上超过 70% 的开源 MLLM[^6]。这些主张未经独立验证。

**（四）闭源侧证据的时效性与二手性。** Google 的模型页与技术页内容随时间滚动更新，当前展示的 Gemini 3.1 Pro 与 Gemini 4 Argon 并非 2025 年时点的状态[^27][^28]。OpenAI 侧仅有 Wikipedia 二手来源[^9]。字节侧仅有 agent 工程仓库[^10]。这意味着「主要厂商在 2025-2026 年发布的多模态模型所宣称的架构取向」这一问题，在 OpenAI 与字节两个方向上**现有公开资料不足以确认**。

**（五）GitHub 检索的信号偏差。** 按 stars 排序检索多模态相关仓库，头部多为基准/评测类仓库（MMStar ★218、MMT-Bench ★119、MMIU ★99）与论文代码库，而非主流旗舰多模态模型仓库；检索 Qwen2.5-VL / InternVL / LLaVA 时头部为通用框架与工具仓库（huggingface/transformers ★167711、Mintplex-Labs/anything-llm ★67055、sgl-project/sglang ★37126）[^10]。这说明**stars 数不能直接作为架构路线流行度的代理指标**。此外，facebookresearch/chameleon 仓库的 stars 数与最近提交时间因 GitHub API 403 rate limit 未能获取[^23]。

**（六）统一路线的评测口径不统一。** Janus-Pro 的多模态理解综合分由 POPE、MME-Perception、GQA、MMMU 四项平均（MME-Perception 分数除以 20 归一化到 [0,100]），视觉生成在 GenEval 与 DPG-Bench 两个指令跟随基准上评估，图像分辨率为 384×384[^15]。这一自定义综合分口径与其他论文的单项基准报告方式不可直接比较。

---

## 参考来源

[^1]: A Survey on Multimodal Large Language Models (arXiv:2306.13549v4) — https://arxiv.org/abs/2306.13549v4 (arxiv, 2023-06-23)
[^2]: Qwen2.5-VL Technical Report — https://arxiv.org/abs/2502.13923 (arxiv, 2025-02-19)
[^3]: InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models — https://arxiv.org/abs/2504.10479 (arxiv, 2025-04-14)
[^4]: Chameleon: Mixed-Modal Early-Fusion Foundation Models — https://arxiv.org/abs/2405.09818 (arxiv, 2024-05-16)
[^5]: QwenLM/Qwen2.5-VL (README, 现为 Qwen3-VL 主页) — https://github.com/QwenLM/Qwen2.5-VL (github, 2025-11-27)
[^6]: OpenGVLab/InternVL (README) — https://github.com/OpenGVLab/InternVL (github, 2025-08-30)
[^7]: Gemini 2.5: Our most intelligent AI model (Google blog) — https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/ (web, 2025-03-25)
[^8]: The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation (Meta AI blog) — https://ai.meta.com/blog/llama-4-multimodal-intelligence/ (web, 2025-04-05)
[^9]: GPT-4o (Wikipedia) — https://en.wikipedia.org/wiki/GPT-4o (web)
[^10]: GitHub 仓库检索结果（stars 排序，元数据） — https://github.com/search?q=multimodal+large+language+model+vision+language+model&type=repositories (github)
[^11]: Emu3: Next-Token Prediction is All You Need — https://arxiv.org/abs/2409.18869 (arxiv, 2024-09-27)
[^12]: Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model — https://arxiv.org/abs/2408.11039 (arxiv, 2024-08-20)
[^13]: Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation — https://arxiv.org/abs/2410.13848 (arxiv, 2024-10-17)
[^14]: Show-o: One Single Transformer to Unify Multimodal Understanding and Generation — https://arxiv.org/abs/2408.12528 (arxiv, 2024-08-22 (v1); last revised 2025-09-08 (v7); ICLR 2025)
[^15]: Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling — https://arxiv.org/abs/2501.17811 (arxiv, 2025-01-29)
[^16]: xGen-MM (BLIP-3): A Family of Open Large Multimodal Models — https://arxiv.org/abs/2408.08872 (arxiv, 2024-08-16)
[^17]: MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training — https://arxiv.org/abs/2403.09611 (arxiv, 2024-03-14)
[^18]: Improved Baselines with Visual Instruction Tuning (LLaVA-1.5) — https://arxiv.org/abs/2310.03744 (arxiv, 2023-10-05)
[^19]: DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding — https://arxiv.org/abs/2412.10302 (arxiv, 2024-12-13)
[^20]: QwenLM/Qwen3-VL (official repo) — https://github.com/QwenLM/Qwen3-VL (github, 2026-01-30)
[^21]: QwenLM/Qwen-VL (official repo, first generation) — https://github.com/QwenLM/Qwen-VL (github, 2024-08-07)
[^22]: haotian-liu/LLaVA (official repo) — https://github.com/haotian-liu/LLaVA (github, 2024-08-12)
[^23]: facebookresearch/chameleon (official repo) — https://github.com/facebookresearch/chameleon (github)
[^24]: deepseek-ai/DeepSeek-VL2 (official repo) — https://github.com/deepseek-ai/DeepSeek-VL2 (github, 2025-02-26)
[^25]: deepseek-ai/DeepSeek-VL (official repo, first generation) — https://github.com/deepseek-ai/DeepSeek-VL (github, 2024-04-24)
[^26]: Introducing Claude 4 (Anthropic official announcement) — https://www.anthropic.com/news/claude-4 (web, 2025-05-22)
[^27]: Gemini Pro model page (Google DeepMind) — https://deepmind.google/models/gemini/pro/ (web)
[^28]: Gemini technologies page (Google DeepMind) — https://deepmind.google/technologies/gemini/ (web)