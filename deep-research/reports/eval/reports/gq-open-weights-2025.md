# 2025年开源权重大模型格局：主要玩家与代表性模型

## 摘要

2025年开源权重模型生态呈现三个可验证的结构性变化。第一，**中国实验室成为开源权重模型的主要供给方**：基于 Hugging Face 全量周度下载数据（2020年6月–2025年8月，覆盖85.1万个模型、22亿次下载）的研究记录到美国开源权重产业主导地位大幅下降，让位于无隶属开发者、社区组织以及截至2025年的中国产业，DeepSeek 与 Qwen 可能预示市场权力的新一轮集中[^13]；另一份基于19个月每日下载序列的报告显示，Qwen 自有仓库占文本与视觉语言模型下载量的比例从2025年3月的9%升至2026年9月的30%（4月峰值50%），而 Meta 的份额从10%降至3%[^39]。第二，**架构层面 MoE 成为主流**：同一研究记录到 MoE 架构采用量增长7倍、平均模型规模增长17倍、量化增长5倍[^13]。第三，**"开放权重"与"真正开源"出现分化**：2025年"开放权重"模型首次超过"真正开源"模型，数据透明度出现令人担忧的下降[^13]——这与 OSI 对 Open Source AI 的严格定义（须同时提供 Data Information、Code、Parameters 三类 preferred form）[^1] 和 Meta 自称 Llama Materials 为"proprietary"的许可现实[^2] 形成直接对照。

主要玩家可归为五类：Meta（Llama 4）、中国实验室（DeepSeek、Qwen、Kimi/Moonshot）、Google（Gemma 3）、Mistral AI（Apache 2.0 阵营最彻底的一家）、以及微软 Phi 与 AI2 OLMo 2（分别代表"合成数据路线"与"完全开放路线"）。xAI 与 Cohere 在2025年开源权重维度的证据严重不足，无法支撑结论。

---

## 一、定义基线：什么算"开源权重"，什么算"开源"

讨论格局前必须先固定判据，否则"开源"一词会被营销话术稀释。

**Open Source AI 的 OSI 定义**：以授予四项自由的方式提供的 AI 系统——(1) 为任何目的使用而无需请求许可；(2) 研究其运作方式并检查其组件；(3) 为任何目的修改，包括改变其输出；(4) 为任何目的分享（含或不含修改）[^1]。OSI 明确这些要求对 system、model、weights and parameters 或其他结构元素一视同仁；行使自由的前提是能获得"修改该系统的首选形式（preferred form to make modifications）"[^1]。

对机器学习系统，**preferred form 必须包含三类要素**[^1]：
- **Data Information**：训练数据的足够详细信息，使有技能者可构建实质等同系统；须披露全部训练数据的完整描述（含不可共享数据）、数据来源与范围特征、获取与筛选方式、标注流程、数据处理与过滤方法，并列出所有公开可得的训练数据及其获取位置。
- **Code**：训练与运行系统的完整源代码，须采用 OSI 认可的许可证。
- **Parameters**：模型参数如权重或配置，须以 OSI 认可条款提供。

OSI 同时给出术语定义：**AI model 由 model architecture、model parameters（含 weights）与 inference code 三部分构成**；**AI weights 是叠加在模型架构之上、用于从给定输入产生输出的那组学习到的参数**[^1]。

以此对照，**Llama 3.1 的许可不是开源许可**：它采用 Community License Agreement，授予的是"non-exclusive, worldwide, non-transferable and royalty-free limited license"（有限、非独占、不可转让）[^2]；再分发须随附协议副本并显著展示"Built with Llama"[^2]；用其输出训练并分发的模型名称须以"Llama"开头[^2]；月活超7亿须向 Meta 申请且"Meta may grant to you in its sole discretion"（Meta 单方面裁量）[^2]；使用受 Acceptable Use Policy 约束（即存在使用目的限制）[^2]；协议文本将 Llama Materials 描述为"Meta's proprietary Llama 3.1 and Documentation"[^2]。

这一区分不是学理洁癖。上述下载研究明确指出：2025年"开放权重（open weights）"模型首次超过"真正开源（truly open source）"模型[^13]。换言之，2025年生态的主流产品形态是**开放权重但非开源**。

---

## 二、Meta：Llama 4 转向 MoE 与原生多模态

Llama 4 是 Meta 首批采用 MoE 架构的模型，也是其首批原生多模态开放权重模型[^22]。产品线三档：

| 型号 | 激活参数 | 专家数 | 上下文 | 部署 |
|---|---|---|---|---|
| Scout | 17B | 16 | 10M tokens | 单张 H100（Int4 量化）[^22] |
| Maverick | 17B | 128 | — | 单张 H100 主机[^22] |
| Behemoth | 288B | 16 | — | 教师模型，发布时仍在训练，未开放下载[^22] |

Scout 与 Maverick 于2025年4月5日发布，可在 llama.com 与 Hugging Face 下载[^22]。Maverick 的实验性 chat 版本在 LMArena 上取得 ELO 1417[^22]。

**须注意**：上述全部数据来自 Meta 官方博客单一来源，且博客自称 Scout 是"同级别中全球最好的多模态模型"、Maverick"优于 GPT-4o 和 Gemini 2.0 Flash"，属营销性主张[^22]。证据池中缺少 Llama 4 的许可类型（Llama 4 Community License 具体条款）、训练 token 量、以及独立第三方（LMArena 官方榜、Hugging Face 模型卡、技术报告 arXiv）的交叉验证。因此 Llama 4 的关键规格在本次取证范围内**不满足"每个子问题至少2个独立来源"的要求**。

---

## 三、中国阵营：DeepSeek、Qwen、Kimi

### 3.1 DeepSeek：MoE 效率路线与纯 RL 推理路线

**DeepSeek-V3** 是 MoE 语言模型，总参数 671B，每 token 激活 37B[^3][^29]。架构采用 Multi-head Latent Attention (MLA) 与 DeepSeekMoE（细粒度专家）以实现高效推理与低成本训练[^3][^29]。预训练使用 14.8 万亿 tokens[^3][^29]，训练全程仅需 2.788M H800 GPU 小时，过程稳定，未出现不可恢复的 loss spike 或回滚[^3][^29]。技术贡献包括首创无辅助损失的负载均衡策略（auxiliary-loss-free）与多 token 预测（multi-token prediction）训练目标[^3][^29]。权重发布于 github.com/deepseek-ai/DeepSeek-V3[^3][^29]。论文声称其性能超过其他开源模型并与领先闭源模型相当，属作者自述主张[^29]。

**DeepSeek-R1** 走另一条路线：通过纯强化学习激励 LLM 推理能力，无需人工标注的推理轨迹[^4]。RL 框架使模型涌现出自我反思、验证、动态策略调整等推理模式[^4]；在数学、编程竞赛、STEM 等可验证任务上，其表现超过用传统监督学习（人类示范）训练的同类模型[^4]；大规模模型涌现的推理模式可被系统性用于指导并增强小模型的推理能力[^4]。

### 3.2 Qwen：Apache 2.0 全量开放 + 长上下文

**Qwen3** 系列同时包含 dense 与 MoE 架构，参数规模从 0.6B 到 235B[^5]。核心设计是把 thinking 模式（复杂多步推理）与 non-thinking 模式（快速上下文响应）整合进统一框架，支持按查询或 chat template 动态切换，并引入 thinking budget 机制以在推理时自适应分配计算资源[^5]。多语言支持从 Qwen2.5 的29种扩展到119种语言和方言[^5]。**所有 Qwen3 模型以 Apache 2.0 许可公开发布**[^5]——这是与 Llama 系许可的关键差别。

长上下文方向由 **Qwen2.5-1M** 推进：上下文长度扩展到100万 tokens，相比此前128K版本显著增强[^6]。开源模型包括 Qwen2.5-7B-Instruct-1M 与 Qwen2.5-14B-Instruct-1M，另有 API 访问的 Qwen2.5-Turbo[^6]。采用长度外推方法，可在无需额外训练的情况下将上下文长度扩展至少4倍[^6]；在1M tokens 场景下实现3x到7x的 prefill 加速[^6]；Qwen2.5-14B-Instruct-1M 在长上下文任务上显著优于 GPT-4o-mini，并支持8倍长的上下文[^6]。

### 3.3 Kimi K2：万亿参数 MoE 与 agentic 后训练

Kimi K2 是 MoE 大模型，激活参数32B，总参数1万亿[^11]。训练侧提出 **MuonClip 优化器**，在 Muon 基础上引入 QK-clip 技术以解决训练不稳定问题，基于此在15.5万亿 token 上预训练，全程零 loss spike[^11]。后训练包含大规模 agentic 数据合成流水线与联合 RL 阶段，模型通过与真实及合成环境交互提升能力[^11]。

非思考设置下的成绩：Tau2-Bench 66.1、ACEBench(En) 76.5、SWE-Bench Verified 65.8、SWE-Bench Multilingual 47.3，论文称取得开源非思考模型 SOTA[^11]；编码/数学/推理成绩为 LiveCodeBench v6 53.7、AIME 2025 49.5、GPQA-Diamond 75.1、OJBench 27.1（均无扩展思考）[^11]。

### 3.4 中国阵营的生态位证据

下载数据提供了独立于模型卡的外部验证：Qwen 自有仓库占文本与视觉语言模型下载量的比例从2025年3月的9%升至2026年9月的30%（4月峰值50%），同期 Meta 从10%降至3%[^39]。同时，下载量最高的十个模型所占份额从2025年3月的34%降至22%，Hub 头部集中度下降[^39]。全部下载中43%流向衍生模型（此前为18%），仅量化一项就从9%升至28%[^39]。

---

## 四、Google Gemma 3：多模态 + 长上下文的 KV-cache 优化

Gemma 3 是 Gemma 系列的多模态新增版本，参数规模从1B到27B[^7]。引入视觉理解能力、更广的语言覆盖，以及至少128K tokens 的更长上下文[^7]。其长上下文策略有明确工程取向：通过提高局部注意力层相对全局注意力层的比例、并保持局部注意力跨度较短，来降低长上下文下易膨胀的 KV-cache 内存[^7]。训练层面，Gemma 3 使用蒸馏（distillation）训练[^7]。

---

## 五、Mistral AI：Apache 2.0 阵营中许可最彻底的一家

Mistral 在2025年发布了多条产品线，**全部以 Apache 2.0 许可发布**：

- **Mistral Small 3**（2025年1月）：24B 参数，同时发布预训练（base）与指令微调（instruct）两个 checkpoint，均为 Apache 2.0[^17]。官方称 MMLU 准确率超81%、延迟约150 tokens/s，且未使用 RL 或合成数据训练[^17]；性能与 Llama 3.3 70B instruct 相当，同硬件上快3倍以上[^17]。
- **Mistral Small 3.1**（2025年3月）：改进文本性能与多模态理解，上下文窗口扩展至最高128k tokens，Apache 2.0[^18]。官方称推理速度150 tokens/s，可在单张 RTX 4090 或32GB RAM 的 Mac 上运行，同时发布 base 与 instruct 两个 checkpoint[^18]；性能超过 Gemma 3 与 GPT-4o Mini 等同类模型[^18]。
- **Magistral**（2025年6月）：Mistral 首个推理模型，双版本——Magistral Small（24B 开源版）与 Magistral Medium（企业版）[^19]。AIME2024 得分：Medium 73.6%、majority voting @64 下90%；Small 分别为70.7%与83.3%[^19]。配套论文涵盖评估、训练基础设施、强化学习算法[^19]。chain-of-thought 支持多语言（英语、法语、西班牙语、德语、意大利语、阿拉伯语、俄语、简体中文）[^19]。后续 **Magistral Small 1.2**（仓库名 Magistral-Small-2509）基于 Mistral Small 3.2 (2506) 构建，加入推理能力，在 Magistral Medium 轨迹上做 SFT 并叠加 RL，safetensors 元数据总参数 24,011,361,280，许可标注 apache-2.0，量化后可在单张 RTX 4090 或32GB RAM MacBook 上本地部署，相比1.1版本新增视觉能力，base_model 为 mistralai/Mistral-Small-3.2-24B-Instruct-2506[^21]。
- **Mistral 3 系列**（2025年12月）：包含三个稠密小模型（14B、8B、3B）以及 **Mistral Large 3**[^20]。Mistral Large 3 是稀疏 MoE 模型，41B 激活参数、675B 总参数，从零训练，使用3000张 NVIDIA H200 GPU，同时发布 base 与 instruction fine-tuned 版本，均为 Apache 2.0[^20]。官方称其在 LMArena 排行榜 OSS 非推理模型类别中位列第2（OSS 模型整体第6）[^20]；将发布 NVFP4 格式 checkpoint，可在单台 8×A100 或 8×H100 节点上用 vLLM 运行[^20]。Ministral 3 系列提供3B、8B、14B 三种规模，每种规模均发布 base、instruct、reasoning 三个变体，均支持图像理解，均为 Apache 2.0[^20]。

Hugging Face 模型卡 **mistralai/Mistral-Large-3-675B-Instruct-2512** 与官方博客相互印证：命名中675B 与2512（2025年12月）对应参数规模与版本日期，base_model 字段指向 mistralai/Mistral-Large-3-675B-Base-2512（即存在对应 Base 版本），tags 含 mistral-common 与 compressed-tensors（量化/压缩张量格式）[^10]。模型卡默认系统提示声明该模型由 Mistral AI 创建，知识库最后更新于2023-10-01，具备图像读取能力（可读图，不能生成图，不能读音频/视频）[^10]。

---

## 六、微软 Phi 与 AI2 OLMo 2：两条非主流但重要的路线

### 6.1 Phi-4：合成数据驱动

phi-4 是14B 参数语言模型，训练配方以数据质量为核心[^8]。与多数以网页/代码等有机数据为主的模型不同，phi-4 在训练全程策略性地引入合成数据[^8]。关键结论是：phi-4 在 STEM 类问答能力上大幅超越其教师模型（GPT-4），表明其数据生成与后训练技术超越了单纯蒸馏[^8]；相对 phi-3 架构改动极小，性能提升主要来自改进的数据、训练课程与后训练方案创新[^8]。

产品线扩展：**microsoft/phi-4** 仓库创建于2024-12-11，许可标签为 MIT，关联论文 arXiv:2412.08905[^25]；**Phi-4-mini-instruct** 创建于2025-02-19，许可 MIT，关联论文 arXiv:2503.01743，标签含 multilingual 及23种语言代码，sizeRank 显示属 <6B 档[^26]；**Phi-4-reasoning** 创建于2025-04-09，系统提示描述其为 Phi 系列语言模型，采用 Thought/Solution 两段式推理格式[^27]。

### 6.2 OLMo 2：完全开放路线

OLMo 2 是新一代**完全开放（fully open）**语言模型，包含7B、13B、32B 三个 dense 自回归模型规模[^9]。完全释放的产物包括：模型权重、完整训练数据、训练代码与配方、训练日志以及数千个中间 checkpoint[^9]——这是本次证据池中唯一在"Data Information + Code + Parameters"三要素上接近 OSI preferred form 要求的开源实践[^1]。

技术要点：预训练数据混合引入新的专用数据配比 Dolmino Mix 1124，通过晚期课程训练（预训练退火阶段引入专用数据）显著提升下游基准表现[^9]；OLMo 2-Instruct 借鉴 Tülu 3 最佳实践，聚焦宽松许可（permissive）数据，并将最终阶段强化学习扩展为可验证奖励的 RL（RLVR）[^9]。性能定位上，OLMo 2 base 模型处于性能-训练算力帕累托前沿，常匹配或超越 Llama 3.1、Qwen 2.5、Gemma 2 等仅开放权重模型，且使用更少 FLOPs[^9]。

---

## 七、RL 训练系统的开源化：DAPO

DAPO 论文指出，SOTA 推理模型（如 OpenAI o1、DeepSeek R1）的关键技术细节被隐藏，社区难以复现其 RL 训练结果[^12]。作为回应，论文提出 **Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)** 算法，完全开源了一个大规模 RL 系统，使用 Qwen2.5-32B 基座模型在 AIME 2024 上达到50分[^12]；公开了四项关键技术细节，并开源训练代码（基于 verl 框架）与经整理处理的数据集，以增强可复现性[^12]。

---

## 八、生态与工具层：推理框架与量化中介

开源权重模型的价值实现依赖推理与微调工具链。以下仓库元数据可确认其规模，但**所有 last push 时间均为2026-10-14，是2026年的快照，无法支撑"2025年内工程生态活跃度"的时序主张**：

- **ggml-org/llama.cpp**：★131,152、forks 24,394、C++、描述"LLM inference in C/C++"[^34]。README 声明主要目标是在广泛硬件上以最小配置实现 LLM（及 VLM）推理并达到 SOTA 性能，为无依赖纯 C/C++ 实现[^34]；Apple Silicon 是一等公民（ARM NEON、Accelerate、Metal），支持 x86 的 AVX/AVX2/AVX512/AMX 与 RISC-V 的 RVV/ZVFH/ZFH/ZICBOP/ZIHINTPAUSE[^34]；量化位宽支持从1.5-bit 到5-bit（原文在5-bit 处被截断）[^34]；提供 llama cli 与 llama serve 命令，支持直接从 Hugging Face 下载并运行 GGUF 模型（示例为 ggml-org/Qwen3.5-0.8B-GGUF），内置 OpenAI 兼容 API server 与 Web UI[^34]。
- **vllm-project/vllm**：★93,774、forks 23,373、Python、描述"high-throughput and memory-efficient inference and serving engine for LLMs"[^32]。README 声明使用 PagedAttention 管理 attention key/value 内存[^32]；量化支持 FP8、MXFP8/MXFP4、NVFP4、INT8、INT4、GPTQ/AWQ、GGUF、compressed-tensors、ModelOpt、TorchAO[^32]；支持 continuous batching、chunked prefill、prefix caching、speculative decoding（n-gram、suffix、EAGLE、DFlash）、disaggregated prefill/decode/encode[^32]；无缝支持 Hugging Face 上200+模型架构，涵盖 decoder-only LLM、MoE LLM、混合注意力与状态空间模型、多模态模型[^32]；硬件覆盖 NVIDIA/AMD/Intel GPU 及 x86/ARM/PowerPC CPU，并有 Google TPU、Intel Gaudi、IBM Spyre、Huawei Ascend、Apple Silicon 等插件；最初由 UC Berkeley Sky Computing Lab 开发，现由2000+贡献者维护[^32]。
- **sgl-project/sglang**：★37,129、forks 9,554、Python[^33]。README 称其为面向大语言、视觉语言和扩散模型的开源推理框架，针对 agentic 工作负载、大规模 serving 和 RL rollouts 优化[^33]；内置 SGLang Diffusion 图像与视频生成引擎[^33]；硬件覆盖 NVIDIA（B200/B300/GB200/GB300、H100/H200、A100、RTX 30/40/50系列等）、AMD（MI300X/MI325X/MI350X/MI355X）、Google TPU（v6e/v7）、Intel GPU/CPU、Apple Silicon（Metal/MLX）、Huawei Ascend、Moore Threads，集成中平台包括 AWS Trainium、Alibaba T-Head PPU、Cambricon MLU、Qualcomm QAIC、MetaX、Hygon HCU/DCU、Iluvatar CoreX[^33]；生态项目含 SpecForge（投机解码 draft 模型训练）、SGLang Omni（音频 TTS/ASR）及 Miles/slime/AReaL/Tunix/verl 等 RL 训练框架集成[^33]。

**学术基础**：SGLang 论文提出由前端语言和运行时组成的系统，运行时通过 RadixAttention 实现 KV cache 复用、通过压缩有限状态机实现更快的结构化输出解码，实验显示相比当时最先进推理系统实现最高6.4x 吞吐提升，评测任务涵盖 agent control、逻辑推理、few-shot 学习基准、JSON 解码、RAG 流水线和多轮对话[^35]。vLLM 的 PagedAttention 论文提出受操作系统虚拟内存与分页技术启发的注意力算法，在 PagedAttention 之上构建 vLLM，实现 KV cache 内存近零浪费及请求内和跨请求的 KV cache 灵活共享，评测显示在相同延迟水平下将主流 LLM 吞吐提升2-4x（对比 FasterTransformer 和 Orca），且该提升在更长序列、更大模型和更复杂解码算法下更为显著[^36]。

**量化中介层**：上述下载研究记录到生态中涌现出一层新的开发者中介（developer intermediaries），专注于对基础模型进行量化与适配，服务于效率与艺术表达两类目的[^13]。Model Pulse 平台明确追踪每个模型的量化、微调、适配器与合并等衍生关系[^38]。这与下载数据中"仅量化一项就从9%升至28%"的观察一致[^39]。

---

## 九、需求侧结构：下载量分布揭示的真实使用形态

理解"格局"不能只看发布侧，还要看实际下载结构。

**头部集中度**：Hugging Face Hub 下载量最高的50个实体占 Hub 总下载量的80.22%[^37]。在这50个实体中，企业占下载量63.2%（50个实体中20个），大学占20.7%（10个），个人占12.1%（16个），非营利组织占3.8%（4个），混合实验室占0.3%（1个）[^37]。

**规模分布与"大模型叙事"的张力**：在可获知模型规模的模型中（占 top 50的96.94%、占整个 Hub 的77.76%），小模型下载量占绝对主导——92.48%的下载来自参数少于10亿的模型，86.33%来自少于5亿参数的模型，69.83%来自少于2亿参数的模型，40.17%来自少于1亿参数的模型[^37]。

**模态与架构分布**：按模态，NLP 占58.1%，CV 占21.2%，音频占15.1%，多模态占3.3%，时间序列占1.7%[^37]。文本编码器（base 模型及其任务微调）占总下载量45%以上（占 NLP 模态的77.5%），而解码器仅占9.5%（模态的16.5%），编码器-解码器占3%（模态的5%）[^37]。语言分布上，英语占使用语言标签模型下载量的79.46%以上（若仅计有语言标签的模型则为92.85%），法语以17.48%位居第二[^37]。

**平台规模**：Hugging Face Model Hub 目前托管200万+模型，累计独立下载量17亿次（论文自注其使用的下载度量比 Hub 页面原始计数更精确）[^14]；该 Hub 自2019年分发 BERT 与 GPT-2 的开放 PyTorch 格式起步，已演变为全球共享开放权重 AI 模型的首要平台[^14]。另一数据源称 Hub 每天服务约1亿次模型下载（2026年9月约1.03亿次/日），一年前约为6100万次/日[^39]；数据集下载量一年增长2.5倍，达到约800万次/日[^39]。

**计数方法学警告**：Hugging Face 官方文档说明，Hub 使用一组 query files 进行下载计数以避免对单个模型仓库的多文件下载重复计数；默认（未指定 library 时）使用 config.json 作为默认 query file，其他库可能检查 pytorch_model.bin 或 adapter_config.json 等文件；GGUF 文件因自包含且不绑定单一库，全部计入下载量，在整仓库克隆场景下可能造成重复计数[^40]。该文档仅说明计数方法学，未提供 Hub 模型总数或累计下载总量的汇总数字[^40]。这意味着上述下载份额结论受计数口径影响，跨来源比较需谨慎。

---

## 十、监管与许可环境：EU GPAI Code of Practice

2025年开源权重模型的发布环境受到 EU AI Act 框架的直接约束。

**The General-Purpose AI Code of Practice (GPAI CoP)** 是欧盟委员会于2025年7月10日发布的支持 EU AI Act 合规的工具[^41]。它为通用目的 AI 模型提供者提供操作指引，特别是针对2025年8月2日生效的 AI Act 第53条与第55条[^41]。准则分三章：Transparency、Copyright、Safety and Security[^41][^42]。

**约束力性质**：虽为非约束性，但遵循该准则意味着 EU 监管机构将假定提供者满足相应法律要求；遵循是自愿的，但 AI Act 合规不是自愿的[^41]。AI Act 第56条授权 EU AI Office 制定自愿性规则手册，规定模型提供者如何满足第53条与第55条义务[^41]。

**算力阈值**：根据第53条与第55条，训练算力超过 10^23 FLOPs 且投放 EU 市场的 GPAI 模型开发者必须满足透明度义务并制定 EU 版权法政策[^41]。训练算力超过 10^25 FLOPs 的模型被归类为具有系统性风险，须接受增强安全要求；委员会也可在该算力阈值以下指定某模型为系统性风险（Annex XIII 标准）[^41]。Safety and Security 章适用于系统性风险 GPAI 模型——即训练超过 10^25 FLOP 或由委员会基于覆盖范围与能力指定的模型[^43]。

**具体合规要求**：提供者须维护训练数据来源、计算资源与能源消耗的详细记录；能源报告要求以至少两位有效数字的兆瓦时记录测量值[^43]。安全要求包括防范内部威胁、使用基于硬件的可信执行环境实施机密计算，以及对访问模型权重的人员实施多因素认证[^43]。框架适用于在 EU 市场投放通用目的 AI 模型的提供者，无论提供者所在地[^43]。

**制定过程**：准则在2025年7月10日发布前经历三轮迭代（2024年11月、2024年12月、2025年3月），比原计划晚两个多月[^41]。由13名独立专家在四个专题工作组中起草，由 EU AI Office 协调，吸收了近1000个利益相关方的意见[^41]。准则官方仅以英文发布，站内其他语言版本为 AI 生成翻译[^42]。

**签署方格局**：美国公司签署方包括 Amazon、Anthropic、Google、IBM、Microsoft、OpenAI；xAI 仅签署了安全与安保章；欧洲签署方包括 Aleph Alpha 与 Mistral AI[^41]。截至2026年1月，Meta 是最显著拒绝签署该准则的公司；Alibaba、Baidu、DeepSeek 等主要中国 AI 公司也未签署[^41]。

**须注意**：证据 [15] 仅基于搜索结果元数据（标题/URL），未成功读取页面正文，故不作为实质结论依据[^15]。

---

## 十一、算力与出口管制背景

开源权重模型的训练能力受硬件可得性约束，2025年这一约束出现调整。

NVIDIA 于7月15日宣布华盛顿决定，H20 GPU 恢复对华销售；AMD 的 MI308 出口许可申请也在审查中等待最终批准[^45]。H20 是 NVIDIA H100 的刻意降规版本，为应对美国出口规则而设计，基于2022年的 Hopper 架构，算力仅为 H100 的一小部分[^45]。2023年 BIS 规则设定了严格的"性能密度（performance density, PD）"阈值，以算力与芯片面积之比为公式；H20 低于该限值，可绕过完整的商务部许可流程对华销售[^45]。NVIDIA 此前已有"中国特供"产品先例：2024年的 H800 与更早的 A800 都是 H100、A100 的定制降规版本[^45]。自2023年10月 BIS 规则以来，H20 曾是唯一被设计为可合法进入中国市场的 AI 加速器，随后在中美贸易摩擦加剧下受到出口禁令，直至2025年7月中旬解禁[^45]。2025年4月的半导体规则导致 H20 与 MI308 对华出口"瘫痪数月"[^45]。7月中旬的放松不限于 AI 加速器，还包括放宽对华出口电子设计自动化（EDA）工具的管制[^45]。文章认为华盛顿愿意以有限准入换取谈判筹码，同时维持刻意降规以放缓中国步伐[^45]。

学术侧，一篇来自 UC Berkeley Berkeley Risk and Security Lab 与 Berkeley AI Research Lab、发表于 AI Frontiers Initiative 的论文摘要称这是"首个公开的具体证据"，展示中国领先 AI 实验室如何规避和绕过美国出口管制[^44]。论文指出中国公司（尤以腾讯为例）不仅在使用受美国出口管制限制的芯片，还通过软件与建模技术最大化利用性能较弱的硬件来规避管制[^44]。论文具体主张：腾讯用非出口管制的 NVIDIA H20 训练其 Hunyuan-Large 模型，说明机器学习效率提升已侵蚀美国出口管制最初建立的护城河[^44]。论文正文提出两点核心论断：(1) 中国主要 AI 公司尽管面临出口管制，仍能获取高端受管制芯片；(2) 中国 AI 实验室利用 ML 训练工具进步，在较低质量、非出口管制的芯片（含 NVIDIA H20）上成功训练出 SOTA 模型[^44]。论文认为基于硬件阈值的出口策略可以通过更好的软件被克服，并强调跟踪机器学习与算力效率前沿对预测竞争对手能力增长的重要性[^44]。

---

## 十二、长上下文的能力边界

2025年多家厂商将长上下文作为核心卖点（Llama 4 Scout 10M[^22]、Qwen2.5-1M 1M[^6]、Gemma 3 至少128K[^7]、Mistral Small 3.1 最高128k[^18]），但独立评测给出警示。

**NoLiMa** 是一个扩展 NIAH（needle-in-a-haystack）的长上下文评测基准，其 needle 集合设计使问题与 needle 之间词面重叠极小，要求模型推断潜在关联来定位 needle[^31]。论文评测了13个声称支持至少128K token 上下文的流行 LLM：这些模型在短上下文（<1K）表现良好，但随着上下文长度增加性能显著下降；在32K 时，13个模型中有11个跌破其短长度强基线的50%[^31]。GPT-4o 从近乎完美的99.3%基线下降到69.7%[^31]。分析认为性能下降源于注意力机制在缺少字面匹配的长上下文中更难检索相关信息；即便加入推理能力或 CoT 提示，模型也难以在长上下文中维持性能[^31]。数据集与评测代码开源于 github.com/adobe-research/NoLiMa[^31]。

---

## 矛盾与不确定性

**1. Llama 4 的规格与性能主张缺乏独立验证。** Scout 为17B 激活参数/16专家、10M 上下文；Maverick 为17B 激活/128专家、LMArena ELO 1417；Behemoth 288B 激活——全部来自 Meta 官方博客单一来源，属营销性主张（博客自称 Scout 是"同级别全球最好"、Maverick"优于 GPT-4o"），无独立第三方（如 LMArena 官方榜、Hugging Face 模型卡、技术报告 arXiv）交叉验证[^22]。且缺少 Llama 4 的许可类型（Llama 4 Community License 具体条款）、训练 token 量、发布日期以外的版本细节。本报告第二节因此不满足"每个子问题至少2个独立来源"的要求。

**2. Cohere 阵营实际无有效证据。** 证据 [28] 自述"抓取到的页面内容仅包含 chat_template 与 config 字段，未渲染出模型卡正文"，参数量、发布日期、许可类型等核心事实全部缺失[^28]。该来源不能支撑任何关于 Cohere 的结论，等于 Cohere 在本次取证中实际缺席。

**3. xAI 在2025年是否发布开源权重模型未被证实或证伪。** 仅有 Grok 2 权重仓库（2024年训练），无参数计数（模型卡仅以约500GB 权重体积与 TP=8 描述，需8张 >40GB 显存 GPU 部署），权重以 Grok 2 Community License Agreement 授权[^23]。组织页显示 xAI 仅发布 grok-1（Updated Mar 28, 2024）与 grok-2（Updated Nov 5, 2025）两个权重仓库，未列出 Grok 2.5 或 Grok 3 的权重仓库[^24]。因此 xAI 无法被列为2025年开源权重的代表性玩家。

**4. 工具层 GitHub 元数据与课题时间边界冲突。** unsloth（★77,994）[^16]、vLLM（★93,774）[^32]、SGLang（★37,129）[^33]、llama.cpp（★131,152）[^34] 的 last push 均为2026-10-14，是2026年的快照，无法支撑"2025年内工程生态活跃度"的时序主张。且 unsloth 描述中出现 Qwen3.8、DeepSeek-V4、MiniMax-H3、Gemma 4 等疑似未来版本或工具噪声，未加甄别[^16]；该条仅为搜索工具返回的元数据，未读取 README/模型卡正文，按取证纪律不得作为结论依据[^16]。本报告第八节因此仅陈述仓库自述能力，不主张2025年内的
活跃度。

**5. 下载量数据的口径不一致，跨来源比较需谨慎。** 证据 [13] 使用 Hugging Face Model Hub 完整周度下载历史数据集（2020年6月–2025年8月），覆盖85.1万个模型、每模型200+聚合属性、22亿次下载[^13]；证据 [14] 称 Hub 托管200万+模型、累计独立下载量17亿次，并自注其使用的下载度量比 Hub 页面原始计数更精确[^14]；证据 [39] 称 Hub 每天约1亿次模型下载（2026年9月约1.03亿次/日），基于160万模型、19个月每日下载序列[^39]；证据 [38] 称 Model Pulse 追踪1,640,282个模型[^38]。四组数字（85.1万 / 200万+ / 160万 / 164万）与下载总量口径（22亿 / 17亿 / 每日1亿）互不一致，且 Hugging Face 官方文档明确 GGUF 文件在整仓库克隆场景下可能重复计数[^40]。因此本报告引用的份额类结论（如 Qwen 从9%升至30%、Meta 从10%降至3%[^39]）应视为特定口径下的观测，而非可直接横向对比的绝对量。

**6. 时间边界问题。** 证据 [39] 的份额数据点延伸至2026年9月[^39]，证据 [41] 的签署方状态截至2026年1月[^41]，证据 [16][^32][^33][^34] 的 GitHub 快照为2026年10月。本课题限定为"2025年"，上述超出2025年底的数据点仅可作为趋势佐证，不能作为2025年状态的事实陈述。

**7. 未纳入的玩家。** 证据池中未包含任何关于以下主体的开源权重发布证据：Anthropic、OpenAI（其开源权重模型情况）、NVIDIA（Nemotron 系列）、Apple、字节跳动（Seed/豆包）、腾讯（Hunyuan，仅在出口管制论文中被提及用于说明算力规避，非作为开源权重模型发布方[^44]）、01.AI、InternLM、MiniMax（仅在 unsloth 描述中被提及，且该条不得作为依据[^16]）。因此本报告无法就这些主体在2025年开源权重格局中的位置作出任何判断。

**8. "开源"标签的内部张力。** 本报告同时呈现了两类互相冲突的事实：一方面 OSI 定义要求 Data Information、Code、Parameters 三要素齐备才算 Open Source AI[^1]；另一方面2025年生态中"开放权重"模型首次超过"真正开源"模型[^13]，且头部玩家（Meta）在许可文本中自称其材料为"proprietary"[^2]、设置月活7亿门槛与单方面裁量条款[^2]。这意味着"2025年开源权重大模型格局"这一提法本身包含一个未解决的分类问题：若严格采用 OSI 判据，则本报告讨论的多数模型（Llama 4、Gemma 3、phi-4 系列、Mistral 多数产品线、DeepSeek、Qwen、Kimi）均无法确认满足开源标准，因为证据池中没有任何一条提供了这些模型完整的训练数据描述（Data Information 要素）；若采用宽松的"权重可下载"判据，则上述模型均可纳入。本报告采用后者作为组织框架，但读者应知悉这一选择会系统性高估生态的"开放程度"。

**9. 性能主张的来源性质。** 本报告中大量性能数字来自模型方自述：DeepSeek-V3 论文声称"性能超过其他开源模型并与领先闭源模型相当"[^29]；Mistral Small 3 官方称"性能与 Llama 3.3 70B instruct 相当"[^17]；Mistral Small 3.1 官方称"性能超过 Gemma 3 与 GPT-4o Mini"[^18]；Mistral Large 3 官方称 LMArena OSS 非推理模型类别第2[^20]；Kimi K2 论文称"开源非思考模型 SOTA"[^11]。这些均为发布方自述，证据池中缺少统一的第三方评测框架下的横向对比。唯一具备独立评测性质的是 NoLiMa 对13个长上下文模型的评测，其结论（32K 时11/13跌破短上下文基线50%）与厂商的长上下文宣传形成直接张力[^31]。

**10. 中国阵营的算力路径存在两种叙事。** 一方面出口管制论文主张中国实验室通过软件与建模技术最大化利用性能较弱硬件（如 H20）来规避管制，并称腾讯用 H20 训练 Hunyuan-Large[^44]；另一方面 ECIPE 文章记录 H20 在2025年4月规则下对华出口"瘫痪数月"，直至7月中旬解禁[^45]。两者并不矛盾（时间窗口不同），但证据池中没有任何一条直接说明 DeepSeek、Qwen、Kimi 这三个本报告重点讨论的模型具体使用了何种硬件训练——DeepSeek-V3 仅披露"2.788M H800 GPU 小时"[^3][^29]，H800 本身即为中国特供降规型号[^45]。因此"中国开源权重模型在多大程度上依赖受管制或降规硬件"这一问题，现有公开资料不足以确认。

**11. 开源权重与监管合规的交叉点未被证据覆盖。** EU GPAI CoP 对训练算力超 10^23 FLOPs 的模型设定透明度义务、超 10^25 FLOPs 设定系统性风险义务[^41][^43]，而本报告讨论的多个模型（DeepSeek-V3 671B 总参数/14.8T tokens[^3]、Kimi K2 1T 总参数/15.5T tokens[^11]、Mistral Large 3 675B 总参数[^20]）的对应训练算力未被证据披露，因此无法判断哪些模型落入哪一档义务。同时，签署方名单显示主要中国 AI 公司（Alibaba、Baidu、DeepSeek）与 Meta 均未签署该准则[^41]，但证据池未说明未签署对开源权重分发是否产生实际影响。

## 参考来源

[^1]: The Open Source AI Definition – version 1.0 (Open Source Initiative) — https://opensource.org/ai/open-source-ai-definition（web, 2024-10-28）
[^2]: LLAMA 3.1 COMMUNITY LICENSE AGREEMENT (meta-llama/llama-models) — https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE（github, 2024-07-23）
[^3]: DeepSeek-V3 Technical Report — https://arxiv.org/abs/2412.19437v2（arxiv, 2024-12-27）
[^4]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning — https://arxiv.org/abs/2501.12948v2（arxiv, 2025-01-22）
[^5]: Qwen3 Technical Report — https://arxiv.org/abs/2505.09388v1（arxiv, 2025-05-14）
[^6]: Qwen2.5-1M Technical Report — https://arxiv.org/abs/2501.15383v1（arxiv, 2025-01-26）
[^7]: Gemma 3 Technical Report — https://arxiv.org/abs/2503.19786v1（arxiv, 2025-03-25）
[^8]: Phi-4 Technical Report — https://arxiv.org/abs/2412.08905v1（arxiv, 2024-12-12）
[^9]: 2 OLMo 2 Furious — https://arxiv.org/abs/2501.00656v3（arxiv, 2024-12-31）
[^10]: mistralai/Mistral-Large-3-675B-Instruct-2512 (Hugging Face model card) — https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512（github）
[^11]: Kimi K2: Open Agentic Intelligence — https://arxiv.org/abs/2507.20534v2（arxiv, 2025-07-28）
[^12]: DAPO: An Open-Source LLM Reinforcement Learning System at Scale — https://arxiv.org/abs/2503.14476v2（arxiv, 2025-03-18）
[^13]: Economies of Open Intelligence: Tracing Power & Participation in the Model Ecosystem (arXiv:2512.03073v1) — https://arxiv.org/abs/2512.03073v1（arxiv, 2025-11-27）
[^14]: Economies of Open Intelligence (HTML full text, Section 1 Introduction) — https://arxiv.org/html/2512.03073v1（arxiv, 2025-11-27）
[^15]: The General-Purpose AI Code of Practice — European Commission (digital-strategy.ec.europa.eu) — https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai（web）
[^16]: unslothai/unsloth (GitHub repository metadata) — https://github.com/unslothai/unsloth（github）
[^17]: Mistral Small 3 (official blog) — https://mistral.ai/news/mistral-small-3（web, 2025-01-30）
[^18]: Mistral Small 3.1 (official blog) — https://mistral.ai/news/mistral-small-3-1（web, 2025-03-17）
[^19]: Magistral (official blog) — https://mistral.ai/news/magistral（web, 2025-06-10）
[^20]: Introducing Mistral 3 (official blog, incl. Mistral Large 3) — https://mistral.ai/news/mistral-3（web, 2025-12-02）
[^21]: mistralai/Magistral-Small-2509 (HuggingFace model card, Magistral Small 1.2) — https://huggingface.co/mistralai/Magistral-Small-2509（github, 2025-09-12）
[^22]: The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation (Meta 官方博客) — https://ai.meta.com/blog/llama-4-multimodal-intelligence/（web, 2025-04-05）
[^23]: xai-org/grok-2 (Hugging Face model card) — https://huggingface.co/xai-org/grok-2（web）
[^24]: xai-org organization page (Hugging Face) — https://huggingface.co/xai-org（web）
[^25]: microsoft/phi-4 (Hugging Face model card) — https://huggingface.co/microsoft/phi-4（web, 2024-12-11）
[^26]: microsoft/Phi-4-mini-instruct (Hugging Face model card) — https://huggingface.co/microsoft/Phi-4-mini-instruct（web, 2025-02-19）
[^27]: microsoft/Phi-4-reasoning (Hugging Face model card) — https://huggingface.co/microsoft/Phi-4-reasoning（web, 2025-04-09）
[^28]: CohereLabs/c4ai-command-a-03-2025 (Hugging Face model card) — https://huggingface.co/CohereLabs/c4ai-command-a-03-2025（web）
[^29]: DeepSeek-V3 Technical Report — https://arxiv.org/abs/2412.19437（arxiv, 2024-12-27）
[^31]: NoLiMa: Long-Context Evaluation Beyond Literal Matching — https://arxiv.org/abs/2502.05167v3（arxiv, 2025-02-07）
[^32]: vllm-project/vllm (GitHub repository) — https://github.com/vllm-project/vllm（github）
[^33]: sgl-project/sglang (GitHub repository) — https://github.com/sgl-project/sglang（github）
[^34]: ggml-org/llama.cpp (GitHub repository) — https://github.com/ggml-org/llama.cpp（github）
[^35]: SGLang: Efficient Execution of Structured Language Model Programs — https://arxiv.org/abs/2312.07104v2（arxiv, 2023-12-12）
[^36]: Efficient Memory Management for Large Language Model Serving with PagedAttention — https://arxiv.org/abs/2309.06180v1（arxiv, 2023-09-12）
[^37]: Model statistics of the 50 most downloaded entities on Hugging Face (HF Community Blog) — https://huggingface.co/blog/lbourdois/huggingface-models-stats（web, 2025-10-13）
[^38]: Model Pulse — download history for every Hugging Face model — https://modelpulse.ifsp.dev/（web）
[^39]: What 19 months of daily downloads say about the Hugging Face Hub (Model Pulse report) — https://modelpulse.ifsp.dev/report（web）
[^40]: Hub documentation — Models Download Stats (Hugging Face) — https://huggingface.co/docs/hub/models-download-stats（web）
[^41]: General-Purpose AI Code of Practice - Wikipedia — https://en.wikipedia.org/wiki/General-Purpose_AI_Code_of_Practice（web）
[^42]: GPAI Code of Practice · Final Version · 10 July 2025 (Code of Practice Explorer, Future of Life Institute) — https://artificialintelligenceact.eu/en/cop-home/（web, 2025-07-10）
[^43]: EU publishes final General-Purpose AI Code of Practice (PPC Land) — https://ppc.land/eu-publishes-final-general-purpose-ai-code-of-practice/（web）
[^44]: Whack-a-Chip: The Futility of Hardware-Centric Export Controls — https://arxiv.org/abs/2411.14425（arxiv, 2024-11-21）
[^45]: H20 Back in the Chinese Market: Washington's Calculated Adjustment (ECIPE) — https://ecipe.org/insights/nvidia-in-chinese-market/（web, 2025-08）