## 摘要

2025 年多模态大模型（MLLM）的架构路线可归纳为三条主线，而非两条：**(A) 模块化拼接（encoder–connector–LLM）**，以 Qwen2.5-VL/Qwen3-VL、InternVL3/3.5、LLaVA-OneVision 为代表，是开源侧事实上的主流工程范式；**(B) 原生统一（early-fusion / 单一 transformer 同时理解与生成）**，内部再分化为 B1 纯离散 token + 下一 token 预测（Chameleon、Emu3、Emu3.5）与 B2 连续表示 + 扩散/流目标（Transfusion、BLIP3-o、JanusFlow、Show-o、BAGEL）；**(C) 任意模态输入输出（any-to-any / omni）**，以 NExT-GPT 的「LLM + 适配器 + 多扩散解码器」和 Qwen2.5-Omni 的 Thinker–Talker 为代表，本质是模块化路线的模态扩展变体。

需要前置声明三点：(1) 本报告能核对的架构级数字仅限 Qwen 与 InternVL 两族；Gemini 2.5 只有厂商博客的「native multimodality」一句表述，GPT-4o/o 系列、Llama 4、Pixtral、Step-3 在给定证据中零覆盖，因此无法构建 2025 年代表性模型的横向参数对比表。(2) 「native / 原生多模态」一词在证据中至少指代两种不同事物，报告中将显式区分。(3) 各模型的 SOTA 主张几乎全部为作者/厂商自述，未获独立验证。

---

## 一、基础定义：MLLM 的架构抽象与三条路线的分野坐标

按 Yin 等的综述，MLLM 被形式化定义为「基于 LLM、能够接收、推理并输出多模态信息的模型」（LLM-based model with the ability to receive, reason, and output with multimodal information）[^1]。其典型架构被抽象为三个模块：**预训练模态编码器（modality encoder）、预训练 LLM、以及连接二者的模态接口（modality interface / connector）**；部分 MLLM 还附带生成器（generator）以输出文本之外的模态 [^1]。连接器被明确分为三类——projection-based、query-based、fusion-based，前两类采用 token 级融合（token-level fusion），把特征处理成 token 与文本一同送入 LLM [^1]。该综述把 MLLM 归入「生成式」范式（以 OFA 的 sequence-to-sequence 统一多模态任务为代表），区别于 CLIP 为代表的「判别式」范式，并指出 MLLM 相对传统多模态模型的两大特征：十亿级参数 LLM 基座、以及多模态指令微调等新训练范式 [^1]。

这个三模块抽象正是路线分野的坐标原点：**是否保留独立的模态编码器与连接器**，决定了模型落在模块化拼接一侧还是原生统一一侧。综述本身已指出编码器选择上存在「有编码器」与「无编码器（encoder-free）」两条路线，Fuyu-8B 直接将图像 patch 投影后送入 LLM，从而天然支持灵活分辨率 [^1]。

视觉-语言映射的根本困难在于模态表征差异：语言是离散的，而视觉存在于维度高得多的连续空间中，概念并不总能被轻易离散化 [^2]。这一表征差异是后文 B1（离散化视觉）与 B2（保留连续视觉）两条技术分支分歧的物理根源。

---

## 二、路线 A：模块化拼接（ViT–MLP–LLM）

### 2.1 范式定义与 2025 年的工程主导地位

该路线的算法结构可明确写出：**输入**为原始图像/视频/文本；**步骤**为 (i) 视觉编码器（ViT）把图像切 patch 并编码为视觉特征，(ii) 投影器/merger 把视觉特征压缩并对齐到 LLM 的 hidden 维度，(iii) 视觉 token 与文本 token 拼接后送入 LLM 自回归解码；**产出**为文本（或经额外生成器产出其他模态）。InternVL3 论文正文明确写明其架构与前代遵循同一总体框架，即「ViT-MLP-LLM」范式 [^29]；InternVL3.5 官方博客同样明确表示其架构「follow the ViT-MLP-LLM paradigm」[^37]。

**Qwen 族**：Qwen2.5-VL 采用三模块架构——视觉编码器 + 视觉-语言 Merger（投影器）+ LLM 解码器 [^10]。其视觉编码器为从零训练的原生动态分辨率 ViT：hidden size 1280、32 层、16 个注意力头、intermediate size 3456、patch size 14、window size 112，三个尺寸（3B/7B/72B）共用同一 ViT 配置；ViT 仅在 4 层使用全自注意力（full attention block indexes {7,15,23,31}），其余层使用最大 112×112（8×8 patch）窗口注意力，并采用 2D-RoPE、RMSNorm、SwiGLU [^10]。Vision-Language Merger 输入通道 1280，输出通道分别为 2048（3B）/3584（7B）/8192（72B），与对应 LLM hidden size 对齐 [^10]。LLM 骨干由 Qwen2.5 LLM 预训练权重初始化，并把 1D RoPE 改为对齐绝对时间的 MRoPE [^10]。图像输入高宽在训练与推理时都 resize 到 28 的倍数，ViT 以 stride 14 切 patch；视频按 3D patch 划分，连续两帧成组以减少送入 LLM 的 token 数 [^10]。

Qwen3-VL 沿用三模块架构（视觉编码器 + 基于 MLP 的视觉-语言 merger + LLM），但视觉编码器**改用 SigLIP-2 架构**（不再是 Qwen2.5-VL 的自研 ViT），从官方预训练 checkpoint 初始化后以动态分辨率继续训练，默认 SigLIP2-SO-400M，2B/4B 小模型用 SigLIP2-Large（300M）[^11]。投影器为两层 MLP，把 2×2 视觉特征压缩为单个视觉 token 并对齐 LLM hidden 维度 [^11]。模型族包含 dense 2B/4B/8B/32B 与 MoE 30B-A3B/235B-A22B，全部基于 Qwen3 骨干；旗舰 235B-A22B 总参 235B、每 token 激活 22B [^11]。原生支持最长 256K token 的交错多模态上下文 [^11]。架构三项升级为 interleaved-MRoPE、DeepStack（从 ViT 三个不同层级抽取视觉 token，经专用 merger 注入 LLM 前三个层的 hidden states）、以及用文本时间戳 token 替代 Qwen2.5-VL 的绝对时间位置编码 [^11]。GitHub 侧 README 列出的架构更新与之一致，并称原生 256K 上下文可扩展至 1M [^35]。

**InternVL 族**：InternVL3 视觉编码器提供 InternViT-300M 与 InternViT-6B 两种配置，LLM 骨干用 Qwen2.5 系列与 InternLM3-8B 的预训练 base 模型（不使用指令微调版本）初始化；MLP 投影器为两层网络、随机初始化；沿用 pixel unshuffle 把视觉 token 数压缩到四分之一，每个 448×448 图像 tile 表示为 256 个视觉 token [^12][^29]。InternVL3.5 视觉编码器仍为 InternViT-300M 或 InternViT-6B，语言模型改用 Qwen3 系列与 GPT-OSS 初始化，并保留 Dynamic High Resolution 策略 [^13]。其尺寸配置（视觉/语言/总参数）为：1B = InternViT-300M + Qwen3-0.6B（0.3B/0.8B/1.1B）；8B = InternViT-300M + Qwen3-8B（0.3B/8.2B/8.5B）；38B = InternViT-6B + Qwen3-32B（5.5B/32.8B/38.4B）[^13]。MoE 配置包括 20B-A4B（InternViT-300M + GPT-OSS-20B，激活 4B）、30B-A3B（Qwen3-30B-A3B，激活 3B）、241B-A28B（InternViT-6B + Qwen3-235B-A22B，激活 28B）[^13]。视觉 token 压缩上，每个图像 patch 先由编码器表示为 1024 个视觉 token，再经 pixel shuffle 压缩为 256 个送入 LLM [^13]。

**LLaVA-OneVision**：作为开放 LMM 家族，整合了 LLaVA-NeXT 系列关于数据、模型与视觉表征的经验，作者称其为首个能在单图、多图、视频三类场景同时推进开放 LMM 性能边界的单一模型，其设计支持跨模态/跨场景的强迁移学习并由此产生新的涌现能力 [^7]。该论文被定位为「视觉编码器-投影器-LLM」组合式路线的代表，与 Chameleon/Emu3 的 early-fusion 路线形成对照 [^7]。

### 2.2 该路线内部的两个工程改进轴

**轴一：输入分辨率的扩展方式**。综述将其归为两类——direct scaling（直接放大输入/换更高分辨率预训练编码器，如 CogAgent 的双编码器机制）与 patch-division（切块复用低分辨率编码器，如 Monkey、SPHINX）[^1]。2025 年的 Qwen3-VL 走的是「动态分辨率 + 2D-RoPE 插值绝对位置嵌入（沿用 CoMP 方法）」的路径 [^11]；InternVL 族走的是 Dynamic High Resolution + pixel unshuffle/pixel shuffle 压缩的路径 [^13]。

**轴二：视觉 token 预算的动态分配**。InternVL3.5-Flash 加入 Visual Resolution Router (ViR)：为每个 patch 增加一个更高压缩率的 pixel shuffle 模块，可把视觉 token 压缩到 64 个；patch router 依据语义丰富度选择压缩率（如 1/4 或 1/16），可减少 50% 视觉 token 而保持接近 100% 的性能 [^13]。官方博客对此的表述是动态调整视觉 token 分辨率以在不损失性能前提下优化效率 [^37]。

### 2.3 训练范式的分化：「native pre-training」的术语陷阱

InternVL3 提出 native multimodal pre-training 范式：不是在纯文本 LLM 上做后置适配，而是在单一预训练阶段同时从多模态数据与纯文本语料中联合获得多模态与语言能力；论文把 conventional post-hoc pipeline 描述为「先预训练纯文本 LLM，再通过多阶段流程/多模态对齐改造为支持视觉输入的 MLLM」，并指出这会带来模态对齐挑战 [^29][^12]。

**此处必须显式澄清一个术语冲突**：InternVL3 的「native」指**训练数据混合方式**，而非架构层面的统一建模——其架构仍是 ViT-MLP-LLM，且实际选择用预训练 ViT 与 LLM 权重初始化以降低算力成本，尽管该范式理论上支持从零训练 MLLM [^29]。而 Chameleon/Emu3/Transfusion 语境下的「原生多模态」指**架构层面**用单一 transformer 统一建模所有模态 [^4][^6][^5]。同一术语指代两种不同事物，混用会造成事实性误导。

Qwen3-VL 的训练流程为：预训练四阶段——先只更新 merger（视觉-语言投影）层的 warm-up 对齐阶段、其余参数冻结，随后全参数训练，上下文长度依次为 8K、32K、256K；后训练三阶段——长链式思维 SFT、从更强教师模型蒸馏、强化学习 [^11]。InternVL3.5 的训练流程为四阶段：Multimodal Continual Pre-Training (CPT)、SFT、Cascade RL（先离线 MPO，后在线 GSPO）[^37]，其论文侧表述为后训练提出 Cascade RL——离线 RL 阶段保证稳定收敛，在线 RL 阶段精调对齐 [^13]。

---

## 三、路线 B：原生统一（单一 transformer 同时理解与生成）

该路线的共同目标是消除「理解模块」与「生成模块」的架构割裂。按生成目标的表示形式，证据中清晰分化为两支。

### 3.1 B1：纯离散 token + 下一 token 预测

**Chameleon** 是 early-fusion、token-based 的 mixed-modal 模型家族，能在任意交错序列中理解并生成图像与文本；论文声称给出了从训练初期即稳定的训练方法、对齐配方，以及专为 early-fusion token-based mixed-modal 设定设计的架构参数化 [^4]。评估覆盖 VQA、图像描述、文本生成、图像生成与长形式混合模态生成 [^4]。论文将自身定位为「统一建模完整多模态文档（unified modeling of full multimodal documents）」的一步 [^4]。需注意：其「图像描述任务 SOTA」「文本任务超过 Llama-2、与 Mixtral 8x7B 和 Gemini-Pro 相当」「长形式混合模态生成上匹配或超过 Gemini Pro 与 GPT-4V」均为作者自述主张，本次未通过独立来源交叉验证 [^4]。

**Emu3** 是一组仅用 next-token prediction 训练的多模态模型：把图像、文本、视频 tokenize 到离散空间，从零训练单个 transformer 于混合多模态序列上 [^6]。作者称其在生成与感知任务上均超过若干任务专用模型，超过 SDXL 与 LLaVA-1.6，且不需要扩散或组合式架构；能通过在视频序列上预测下一个 token 生成高保真视频 [^6]。论文明确把自身路线表述为「收敛到 token 这一单一焦点」，以简化复杂多模态模型设计，并称 next-token prediction 是通向语言之外通用多模态智能的路径 [^6]。上述超越 SDXL / LLaVA-1.6 为作者自述主张，本次未取得结果表格与独立验证 [^6]。

**Emu3.5** 将该路线推进到 world model 定位：以统一的 next-token prediction 目标端到端预训练，原生预测视觉与语言的下一个状态；预训练语料为视觉-语言交错数据、超过 10 trillion tokens，主要来自互联网视频的连续帧与转录文本；模型接受交错视觉-语言输入并生成交错视觉-语言输出；后训练使用大规模强化学习增强多模态推理与生成能力 [^23]。其提出的 Discrete Diffusion Adaptation (DiDA) 把逐 token 解码转为双向并行预测，作者称单图推理加速约 20x 且不损失性能——该加速比与「与 Gemini 2.5 Flash Image (Nano Banana) 在图像生成/编辑上相当」均为摘要中的作者自述主张，未在结果表格中核对 [^23]。Emu3.5 已开源 [^23]。

### 3.2 B2：连续表示 + 扩散 / 流目标

**Transfusion** 是该分支的配方定义者：把语言建模损失（next token prediction）与扩散（diffusion）结合，用单个 transformer 训练混合模态序列；从零预训练了多个最高 7B 参数的模型，数据为文本与图像混合，并给出单模态与跨模态基准上的 scaling law [^5]。关键实证结论有两条：其一，Transfusion 的扩展性显著优于「把图像量化后用离散图像 token 训练语言模型」的做法 [^5]；其二，引入模态专属的编码与解码层可进一步提升性能，并能把每张图像压缩到仅 16 个 patch [^5]。将配方扩展到 7B 参数、2T 多模态 token 后，模型生成图像与文本的能力与同规模扩散模型和语言模型相当 [^5]。该工作代表「理解+生成统一」中与纯离散 token 路线（Chameleon/Emu3）相对的另一支：连续图像表示 + 扩散目标 [^5]。

**BLIP3-o** 走的是「扩散 transformer 生成语义丰富的 CLIP 图像特征、而非传统 VAE 表示」的变体，即生成端用连续语义特征 + 扩散目标而非离散视觉 token；作者称该设计同时带来更高的训练效率和更好的生成质量 [^25]。其训练策略为顺序式预训练：先在图像理解上训练，再训练图像生成，从而在获得强生成能力的同时保留理解能力 [^25]。配套构建了 BLIP3o-60k 生成指令微调数据集（用多样化 caption 提示 GPT-4o 生成），并完全开源代码、权重、训练脚本与数据集 [^25]。

**JanusFlow** 在单一模型中整合自回归语言模型与 rectified flow；其关键发现是 rectified flow 可以直接在 LLM 框架内训练，无需复杂架构改动；采用两项策略：(i) 解耦理解编码器与生成编码器，(ii) 在统一训练中对二者表示做对齐 [^26]。**Janus-Pro** 作为前作 Janus 的进阶版，三项改进为优化的训练策略、扩展的训练数据、扩展到更大模型规模，作者称这些改进同时提升了多模态理解与文生图指令跟随能力并增强了生成稳定性 [^27]。需注意：Janus-Pro 的 abs 页摘要未直接陈述视觉 tokenizer 与生成目标细节，且 arXiv 管理员标注其与 arXiv:2410.13848 (Janus) 存在文本重叠，架构细节需回溯前作 [^27]。

**Show-o** 是单一统一 transformer，统一了自回归建模与（离散）扩散建模以自适应处理各种混合模态输入输出；与完全自回归模型不同，它混合使用自回归与离散扩散两种目标；支持视觉问答、文生图、文本引导的 inpainting/extrapolation 以及混合模态生成 [^28]。该论文为 ICLR 2025 接收 [^28]。

**BAGEL** 是开源基础模型，原生支持多模态理解与生成，为统一 decoder-only 架构；预训练数据为万亿级交错的文本、图像、视频与网页数据；采用 Mixture-of-Transformer-Experts (MoT) 架构，对模态专属参数做选择性激活；模型规模为 7B 激活参数 / 14B 总参数 [^24]。其架构主张值得单独标出：与在生成/理解模块之间引入 bottleneck connector 的先前方法不同，BAGEL 通过共享 self-attention 实现理解与生成之间的长上下文交互（bottleneck-free 设计）[^24]。作者称其在标准多模态理解榜单上超过当时顶级开源 VLM，文生图质量与 SD3、FLUX.1-dev 等公开生成器相当，并展示自由形式图像编辑、多视角合成、世界导航等 world-modeling 能力 [^24]。

### 3.3 与纯生成式扩散路线的边界

需要区分「统一理解+生成」与「纯生成」：Imagen 3 是 latent diffusion 模型，从文本提示生成高质量图像，论文内容聚焦质量评估与责任（安全、表征）评估，不含统一理解+生成的单一 transformer 架构 [^9]。它可作为「生成侧仍以扩散为主」这一背景的对照证据，但不属于本报告的路线 A/B/C 任何一支 [^9]。

---

## 四、路线 C：任意模态输入输出（any-to-any / omni）

该路线的出发点是模块化路线的模态覆盖缺口。NExT-GPT 论文指出当时多数 MM-LLM 的局限是「仅具备输入侧的多模态理解能力，无法以多模态产出内容」[^3]。其解法是「LLM + 多模态适配器 + 多个扩散解码器」的端到端系统结构，可接受并生成文本、图像、视频、音频的任意组合输入输出；系统复用已有的高性能编码器与解码器，仅微调约 1% 的参数（certain projection layers），从而降低训练成本并便于扩展到更多模态——这是「模块化拼接」路线的典型工程特征 [^3]。配套提出 modality-switching instruction tuning (MosIT) 并人工构建高质量数据集 [^3]。该工作发表于 ICML 2024（Oral）[^3]，代码与模型已在 GitHub 发布 [^17]。综述亦以 NExT-GPT 为例说明除输入外还可支持不同模态的输出 [^1]。

Qwen2.5-Omni 的官方定位是「end-to-end multimodal model」，采用 Thinker-Talker 架构，端到端处理文本、图像、音频、视频输入并流式输出文本与语音；官方明确表述「the entire architecture operates as a cohesive single model, enabling end-to-end training and inference」[^36]。但同一博客也写明 Thinker 是 Transformer decoder，并「accompanied by encoders for audio and image」——即**仍保留独立的音频/图像编码器，属于编码器+统一主干而非完全无编码器的原生 token 化** [^36]。这使「end-to-end」的表述与架构层面的「无编码器」不是同一件事，引用时不可混同。技术上提出 TMRoPE（Time-aligned Multimodal RoPE）用于同步视频与音频时间戳 [^36]。官方路线图表述「we strive to integrate more modalities towards an omni-model」，指向继续向更多模态统一的方向演进 [^36]。其「OmniBench 等任务 SOTA」「音频能力超过同规模 Qwen2-Audio、与 Qwen2.5-VL-7B 相当」为厂商自述主张，需交叉验证 [^36]。

开源生态侧，mini-omni 自述为「open-source multimodal large language model that can hear, talk while thinking」，具备实时端到端语音输入与流式音频输出对话能力，仓库 3585 stars，last push 为 2024-11-05 [^18]。该仓库已超过一年无推送，作为 2025 年活跃度证据不成立，仅可作为该路线存在开源实现的佐证。

---

## 五、闭源侧的表述：Gemini 的「native multimodality」及其证据强度问题

Gemini 2.5 的官方博客以「native multimodality」描述其继承的模型特性，原文为「Gemini 2.5 builds on what makes Gemini models great — native multimodality and a long context window」[^21]。Gemini 2.5 Pro 发布时提供 100 万 token 上下文窗口，并称 200 万 token「coming soon」；可处理输入模态包括 text、audio、images、video 以及 entire code repositories；被定位为 thinking model，通过「significantly enhanced base model + improved post-training」实现性能提升 [^21]。**该博客未出现 vision encoder、early fusion 或 Mixture-of-Experts / MoE 的任何表述** [^21]。

Gemini 2.0 Flash 被描述为具备「multimodal reasoning across vast amounts of information with a context window of 1 million tokens」；2.0 系列发布时「All of these models will feature multimodal input with text output on release, with more modalities ready for general availability in the coming months」——即发布时为多模态输入、文本输出；2.0 Pro Experimental 提供该系列最大上下文窗口 200 万 token，2.0 Flash 与 2.0 Flash-Lite 均为 100 万 token [^22]。**该博客同样未出现 vision encoder、early fusion、native multimodal 或 MoE 的表述** [^22]。

结论必须明确写出：Gemini 2.5 的「native multimodality」**无法与本文第二节/第三节建立的架构分类对齐**——没有技术报告、没有参数量、没有编码器或融合方式细节，也没有第三方交叉验证 [^21][^22]。将其作为「原生多模态路线」的代表性证据过于单薄，本报告只能记录该表述的存在，不能据此判定 Gemini 2.5 属于路线 B。

---

## 六、矛盾与不确定性

**1. 覆盖度缺口（最严重）。** 本可核对的架构级数字仅限 Qwen2.5-VL/Qwen3-VL 与 InternVL3/3.5 两族 [^10][^11][^12][^13][^29]。Gemini 2.5 仅有两条官方博客且证据本身注明未出现编码器/融合方式/MoE 表述 [^21][^22]；GPT-4o/o 系列、Llama 4、Pixtral、Step-3 在给定证据中**零覆盖**。因此「2025 年代表性模型架构参数横向对比表」在本证据集下无法成立，现有公开资料不足以确认这些模型的架构细节。

**2. 「native / 原生多模态」术语冲突（未解决）。** InternVL3 的 native multimodal pre-training 指训练数据混合方式，其架构仍为 ViT-MLP-LLM 且用预训练 ViT/LLM 权重初始化 [^29][^12]；Chameleon/Emu3/Transfusion 语境下的原生多模态指架构层面用单一 transformer 统一建模 [^4][^6][^5]；Gemini 博客的 native multimodality 则无任何技术定义 [^21]。三者不可互换，任何将其合并的表述都是事实性误导。

**3. 「end-to-end」与「无编码器」不是同一件事。** Qwen2.5-Omni 官方同时声称「cohesive single model, end-to-end training and inference」与「Thinker 是 Transformer decoder，accompanied by encoders for audio and image」[^36]。这说明端到端训练/推理的宣称不蕴含架构上取消独立编码器。

**4. SOTA 主张普遍未验证。** Chameleon 的「图像描述 SOTA、长形式生成匹配或超过 GPT-4V」[^4]、Emu3 的「超过 SDXL 与 LLaVA-1.6」[^6]、Emu3.5 的「20x 加速、与 Nano Banana 相当」[^23]、BLIP3-o 的「多数基准 superior」[^25]、InternVL3-78B 的「MMMU 72.2 为开源 SOTA」[^12][^29]、InternVL3.5 的「+16.0% 与 4.05× 加速、241B-A28B 开源 SOTA」[^13][^37]、Qwen2.5-VL-72B 的「与 GPT-4o、Claude 3.5 Sonnet 相当」[^10]、Qwen2.5-Omni 的「OmniBench SOTA」[^36]、Molmo 的「超过 Claude 3.5 Sonnet、Gemini 1.5 Pro/Flash，仅次于 GPT-4o」[^8]、Imagen 3 的「优于其他 SOTA」[^9]，全部为作者/厂商自述，本次均未获独立来源或结果表格核对。

**5. 路线优劣的公开表态全部来自利益相关方。** 现有对模块化 vs 原生统一的优劣论述，仅见于 Qwen Blog 与 InternVL Blog 等厂商自家渠道 [^36][^37]，无独立媒体、第三方评测或学术对比来源。缺少「何时选模块化、何时选原生统一」的第三方工程实践讨论——例如 vLLM 与 SGLang 对两类架构的支持差异、部署成本对比。vLLM README 列出的关键技术包括 PagedAttention、continuous batching、chunked prefill、prefix caching、speculative decoding、disaggregated prefill/decode/encode [^32]，SGLang 自述为同时服务纯文本 LLM 与多模态模型的推理框架 [^14]，但两者均未在给定证据中说明对 early-fusion 统一架构的支持情况。InternVL3.5 的 DvD（把视觉编码器与语言模型分置不同 GPU 以平衡计算负载）是模块化路线在推理部署上的直接收益主张，但同样出自厂商博客 [^37][^13]。

**6. GitHub 活跃度证据的时效性问题。** haotian-liu/LLaVA 有 25078 stars，但 last push 为 2024-08-12，README 最新 Release 停在 2024/05/10 [^30]，已实质停更近两年，**不能用于论证 2025 年生态活跃度**，只能表述为历史影响力。相对地，OpenGVLab/InternVL last push 2025-09-22、README 最新 News 为 2025/08/30 开源 InternVL3_5-GPT-OSS-20B-A4B 训练代码与 CascadeRL [^31]；Qwen3-VL 仓库 20116 stars、last push 2026-01-30 [^15]；Qwen2.5-VL 仓库 URL 已 301 重定向至 Qwen3-VL 仓库，即原仓库被改名，本次未能取得该仓库的 stars/forks/last push 数值 [^35]。另需注意多条 GitHub 元数据的 last push 时间落在 2026 年（如 sglang 2026-10-25 [^14]、vllm 2026-10-25 [^32]、ms-swift 2026-10-23 [^33]），与本报告「2025 年」的时间范围不一致，引用时应以仓库内容而非时间戳作为路线证据。

**7. 数据开放性与架构路线是两个独立维度，不应混谈。** Molmo 的贡献在于配套发布 PixMo 数据集（预训练高细节图像描述数据、微调自由形式图像问答数据、2D 指向数据集），全部数据采集未使用外部 VLM，作者以此区别于依赖专有 VLM 合成数据蒸馏的开放权重模型，并指出当时最强开放权重模型严重依赖闭源 VLM 的合成数据 [^8]。该工作代表组合式路线中「数据与训练配方」而非架构创新的改进方向 [^8]。同理，BLIP3-o 的完全开源（代码、权重、训练脚本、数据集）[^25] 与 Emu3 的开源关键技术 [^6]、Emu3.5 的开源 [^23]、InternVL3 承诺公开训练数据与权重 [^12]、InternVL3.5 开源各训练阶段权重 [^37]，都属于开放性维度，不能作为架构优劣的论据。

**8. 评测与训练基础设施的路线中立性。** stanford-crfm/helm 自述为对基础模型（含 LLM 与多模态模型）进行整体、可复现、透明评测的开源 Python 框架 [^20]；ms-swift 自述为面向大模型与多模态大模型微调的轻量基础设施，topics 含 multimodal、qwen3-vl、qwen3-omni、grpo、lora、megatron、moe [^33]。两者均未在给定证据中体现对特定架构路线的偏向，故不能用于支撑任一路线的优劣判断。Open-Sora 2.0 (11B) 称在 VBench 与 Human Preference 上与 11B HunyuanVideo、30B Step-Video 表现相当、训练成本约 $200K [^34]，属视频生成侧证据，与本文三条 MLLM 架构路线不构成直接对照。

## 参考来源

[^1]: A Survey on Multimodal Large Language Models (Yin, Fu, Zhao et al.; arXiv:2306.13549v4; accepted by National Science Review) — https://arxiv.org/abs/2306.13549（arxiv, 2023-06-23 (v1); last revised 2024-11-29 (v4)）
[^2]: An Introduction to Vision-Language Modeling (Bordes, Pang, Ajay et al.; arXiv:2405.17247) — https://arxiv.org/abs/2405.17247（arxiv, 2024-05-27）
[^3]: NExT-GPT: Any-to-Any Multimodal LLM (Wu, Fei, Qu, Ji, Chua; arXiv:2309.05519; ICML 2024 Oral) — https://arxiv.org/abs/2309.05519（arxiv, 2023-09-11 (v1); v3 2024-06-25）
[^4]: Chameleon: Mixed-Modal Early-Fusion Foundation Models — https://arxiv.org/abs/2405.09818v2（arxiv, 2024-05-16）
[^5]: Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model — https://arxiv.org/abs/2408.11039v1（arxiv, 2024-08-20）
[^6]: Emu3: Next-Token Prediction is All You Need — https://arxiv.org/abs/2409.18869v1（arxiv, 2024-09-27）
[^7]: LLaVA-OneVision: Easy Visual Task Transfer — https://arxiv.org/abs/2408.03326v3（arxiv, 2024-08-06）
[^8]: Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models — https://arxiv.org/abs/2409.17146v2（arxiv, 2024-09-25）
[^9]: Imagen 3 — https://arxiv.org/abs/2408.07009v3（arxiv, 2024-08-13）
[^10]: Qwen2.5-VL Technical Report (arXiv:2502.13923v1) — https://arxiv.org/html/2502.13923v1（arxiv, 2025-02-19）
[^11]: Qwen3-VL Technical Report (arXiv:2511.21631v2) — https://arxiv.org/html/2511.21631v2（arxiv, 2025-11-26）
[^12]: InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models (arXiv:2504.10479v3) — https://arxiv.org/html/2504.10479v3（arxiv, 2025-04-14）
[^13]: InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency (arXiv:2508.18265v2) — https://arxiv.org/html/2508.18265v2（arxiv, 2025-08-25）
[^14]: sgl-project/sglang — high-performance serving framework for LLMs and multimodal models — https://github.com/sgl-project/sglang（github）
[^15]: QwenLM/Qwen3-VL — multimodal large language model series by Qwen team, Alibaba Cloud — https://github.com/QwenLM/Qwen3-VL（github）
[^17]: NExT-GPT/NExT-GPT — any-to-any multimodal LLM (ICML 2024) — https://github.com/NExT-GPT/NExT-GPT（github）
[^18]: gpt-omni/mini-omni — open-source omni multimodal LLM with real-time speech input/output — https://github.com/gpt-omni/mini-omni（github）
[^20]: stanford-crfm/helm — holistic evaluation framework incl. multimodal models — https://github.com/stanford-crfm/helm（github）
[^21]: Gemini 2.5: Our most intelligent AI model (Google 官方博客, Koray Kavukcuoglu) — https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/（web, 2025-03-25）
[^22]: Gemini 2.0 is now available to everyone (Google 官方博客, Koray Kavukcuoglu) — https://blog.google/technology/google-deepmind/gemini-model-updates-february-2025/（web, 2025-02-05）
[^23]: Emu3.5: Native Multimodal Models are World Learners — https://arxiv.org/abs/2510.26583（arxiv, 2025-10-30）
[^24]: Emerging Properties in Unified Multimodal Pretraining (BAGEL) — https://arxiv.org/abs/2505.14683v3（arxiv, 2025-05-20）
[^25]: BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset — https://arxiv.org/abs/2505.09568v1（arxiv, 2025-05-14）
[^26]: JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation — https://arxiv.org/abs/2411.07975v2（arxiv, 2024-11-12）
[^27]: Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling — https://arxiv.org/abs/2501.17811（arxiv, 2025-01-29）
[^28]: Show-o: One Single Transformer to Unify Multimodal Understanding and Generation — https://arxiv.org/abs/2408.12528（arxiv, 2024-08-22）
[^29]: InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models — https://arxiv.org/abs/2504.10479v3（arxiv, 2025-04-14）
[^30]: haotian-liu/LLaVA — https://github.com/haotian-liu/LLaVA（github, 2024-08-12）
[^31]: OpenGVLab/InternVL — https://github.com/OpenGVLab/InternVL（github, 2025-09-22）
[^32]: vllm-project/vllm — https://github.com/vllm-project/vllm（github, 2026-10-25）
[^33]: modelscope/ms-swift — https://github.com/modelscope/ms-swift（github, 2026-10-23）
[^34]: hpcaitech/Open-Sora — https://github.com/hpcaitech/Open-Sora（github, 2026-04-09）
[^35]: QwenLM/Qwen2.5-VL（现重定向为 QwenLM/Qwen3-VL） — https://github.com/QwenLM/Qwen2.5-VL（github）
[^36]: Qwen2.5-Omni 官方博客（Qwen Blog） — https://qwenlm.github.io/blog/qwen2.5-omni/（web）
[^37]: InternVL3.5 官方博客（InternVL Blog, 2025-08-26） — https://internvl.github.io/blog/2025-08-26-InternVL-3.5/（web, 2025-08-26）