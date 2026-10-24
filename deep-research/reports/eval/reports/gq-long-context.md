## 摘要

把上下文窗口推到 100 万 token 级别，主流做法不是单一技术，而是四条技术线的组合：(1) **位置编码外推/插值**——Position Interpolation、YaRN、LongRoPE 通过改造 RoPE 位置索引，用极少微调步数把窗口从 4K/8K 拉到 32K–2048K[^2][^3][^5]；(2) **注意力与架构效率**——FlashAttention 的 IO 感知精确注意力[^6]、Ring Attention 的序列并行[^7]、NSA 与 DSA 的原生可训练稀疏注意力[^9][^14][^39]、Mamba 的选择性状态空间与 MiniMax-01 的 lightning attention 线性注意力[^8][^12]；(3) **长数据与训练配方**——长数据合成、渐进式预训练、多阶段 SFT[^13]；(4) **推理系统**——稀疏注意力 + chunked prefill、kernel 优化、prefix caching 等[^13][^32]。开源侧已有 1M 上下文模型落地（Qwen2.5-1M 系列[^13]、MiniMax-Text-01 训练 1M/外推 4M[^12]），闭源侧 Gemini 3.1 Pro 官方声明输入上限 1M token[^20]。

关于「长上下文是否让 RAG 过时」：证据支持的是**分工而非替代**。长上下文确实在知识库小于约 200,000 token 时可以直接替代 RAG[^38]，且检索更多文档配合长上下文能提升 RAG 质量[^37]；但「有效上下文长度」远短于宣称窗口[^1]、context rot 在所有模型上出现[^18][^19]、Lost-in-the-Middle 位置偏置[^29]、NoLiMa 显示 32K 时 13 个模型中 11 个跌破短上下文基线 50%[^30]，这些共同构成长上下文不可单独承担检索职责的证据。RAG 框架侧仍在高频维护（LlamaIndex 52,639 stars，最近 push 2026-10-23）[^17]，且 RAG 自身在向「上下文工程」演进（contextual retrieval 降低检索失败率 49%–67%）[^38]。

---

## 一、位置编码层：把窗口「撑开」的最低成本手段

**Position Interpolation（PI）** 的做法是线性缩小（down-scale）输入位置索引，使其匹配原始上下文窗口大小，而不是外推到训练长度之外[^2]。其动机是外推会产生「灾难性偏高的注意力分数」，彻底破坏自注意力机制；理论分析表明插值的上界至少比外推的上界小约 600 倍[^2]。工程代价极低：把 LLaMA 7B–65B 扩展到最长 32768 只需 1000 步以内微调，且保留原架构、可复用既有优化与基础设施[^2]。代价是长上下文扩展与短上下文性能之间存在权衡[^2]。

**YaRN** 是计算效率更高的 RoPE 扩展方法，相比此前方法只需 1/10 的 token 与 1/2.5 的训练步数，并表现出可外推到微调数据有限上下文之外的能力[^3]。其官方仓库发布了 LLaMA 2 的 32K/64K/128K、Mistral 7B 的 64K/128K、SOLAR 10.7B 的 32K/64K 微调模型，训练用 DeepSpeed ZeRO-3、评测用 lm-evaluation-harness，仓库约 1.8k stars、MIT 许可[^3][^4]。需要注意「超过此前 SOTA」是论文自述[^3]。

**LongRoPE** 把这条路推到 2048k token：在 256k 训练长度下仅需不超过 1k 步微调，同时保持原始短上下文窗口性能[^5]。它的三项创新是：通过高效搜索识别位置插值中的两种**非均匀性**（为微调提供更好初始化，并在免微调场景实现 8 倍扩展）；**渐进式扩展**（先微调 256k，再对已扩展模型做第二次位置插值到 2048k）；在 8k 长度上**重新调整**以恢复短上下文性能[^5]。这体现了位置插值从「均匀」走向「非均匀」的演进，也明示了扩展长上下文会损害短上下文、需要专门回退调整[^5]。

Anthropic 从机制侧给出同一现象的注脚：位置编码插值可让模型处理更长序列，但会带来 token 位置理解上的退化[^19]。

## 二、注意力与架构层：从「省显存」到「改复杂度」

**精确注意力的 IO 优化。** 自注意力的时间与显存复杂度随序列长度二次增长，这是 Transformer 长序列慢且耗内存的根源[^6]。FlashAttention 是 IO 感知的**精确**注意力算法，用 tiling 减少 GPU HBM 与片上 SRAM 之间的读写次数；它不改变注意力的数学结果，收益来自显存层级 IO 优化而非降低渐近复杂度[^6]。实测 BERT-large（512）端到端墙钟加速 15%、GPT-2（1K）3 倍、long-range arena（1K–4K）2.4 倍，并在 Path-X（16K，61.4%）与 Path-256（64K，63.1%）上首次优于随机[^6]。扩展到 block-sparse 后成为近似算法[^6]。

**序列并行突破单设备显存墙。** Ring Attention 采用自注意力与前馈的 blockwise 计算，把长序列分布到多设备，并将 KV 块通信与 blockwise 注意力计算完全重叠；可支持序列长度达此前显存高效 Transformer 的「设备数量倍」，不使用近似、不引入额外通信与计算开销，实验显示可支持百万级 token 上下文[^7]。它解决的是单设备显存墙（分摊 KV/激活），而非降低单设备内的注意力计算复杂度[^7]。

**原生可训练稀疏注意力。** NSA 采用动态分层稀疏策略：粗粒度 token 压缩 + 细粒度 token 选择，兼顾全局上下文感知与局部精度；两项关键创新是算术强度平衡的算法设计与硬件实现优化、以及支持端到端训练[^9]。用 NSA 预训练的模型在通用基准、长上下文任务与指令推理上维持或超过 Full Attention 模型，在 64k 序列上解码、前向、反向各阶段均有显著加速[^9]。DeepSeek 的 DSA 走同一路线：官方称其首次实现 fine-grained sparse attention，在保持输出质量几乎不变的同时显著提升长上下文训练与推理效率，V3.2-Exp 的训练配置刻意与 V3.1-Terminus 对齐以验证「on par」[^39][^40]；随模型发布 API 价格下调 50% 以上[^40]。DSA 被列为 DeepSeek-V3.2 的首要技术突破[^14]。

**线性注意力 / SSM 路线。** Mamba 的关键判断是：线性注意力、门控卷积、SSM 等次二次架构此前在语言模态上不如注意力，弱点在于缺乏基于内容推理的能力；让 SSM 参数成为输入的函数（选择性 SSM）可让模型沿序列维度按当前 token 选择性地传播或遗忘信息[^8]。代价是高效卷积不可用，因此作者设计了 recurrent 模式下的硬件感知并行算法[^8]。Mamba 是不含注意力甚至不含 MLP 块的端到端架构，推理吞吐比 Transformer 高 5 倍、随序列长度线性扩展，可处理达百万长度的序列，Mamba-3B 超过同尺寸 Transformer 并与两倍尺寸相当[^8]。MiniMax-01 则把线性注意力与 softmax 注意力混合：MiniMax-Text-01 共 80 层，同时整合 lightning attention 与 softmax attention[^42]；模型为 32 专家 MoE、总参数 456B、每 token 激活 45.9B，训练上下文 1M、推理可外推到 4M[^12]。其工程约束很具体：需在单机最多 8 张 GPU、640GB 内存、8-bit 量化下处理超过 1M token，为此实现了定制 CUDA kernel，在 Nvidia H20 上端到端 MFU 超过 75%[^12][^42]。论文明确指出二次复杂度使进一步扩展困难，这是采用线性注意力的动机[^12][^42]。

**KV cache 侧优化。** ThinK 的观察是 KV cache 在**通道维度**上存在大量冗余（依据是注意力权重幅度分布不均与低秩结构），据此做查询相关的通道剪枝，相比原始 KV cache 驱逐与量化方法内存开销降低超过 20%；与 KIVI 结合可实现 2.8 倍峰值内存降低且质量几乎不变，单 GPU 批大小提升至 5 倍[^10]。

## 三、训练配方与推理系统：1M 落地的两个必要条件

**训练侧。** Qwen2.5-1M 把上下文从 128K 扩到 1M，采用长数据合成、渐进式预训练、多阶段监督微调来提升长上下文能力并降低训练成本[^13]。其开源模型为 Qwen2.5-7B-Instruct-1M 与 14B-Instruct-1M，上下文/生成长度均为 1M/8K、Apache 2.0，架构沿用 Qwen2.5 并用 GQA 提升 KV cache 利用效率（7B：28 层、Q/KV 头 28/4；14B：48 层、40/8）[^41][^13]。Qwen2.5-14B-Instruct-1M 在长上下文任务上显著优于 GPT-4o-mini 且上下文长 8 倍[^13]。

**推理侧。** Qwen2.5-1M 开源了推理框架，其中长度外推方法可在无需额外训练的情况下把上下文扩展至少 4 倍甚至更多；为降低推理成本实现了稀疏注意力配合 chunked prefill 优化与 sparsity refinement，加上 kernel 优化、流水线并行与调度优化，在 1M 场景下实现 3x–7x 的 prefill 加速[^13]。通用推理引擎侧，vLLM（95,274 stars、Apache-2.0、截至 2026-10-24 仍每日级活跃）[^15] 与 SGLang（37,685 stars，同期高频提交）[^16] 构成主要承载层；vLLM 的 Automatic Prefix Caching 缓存已有查询的 KV cache，新查询若共享相同前缀可直接复用、跳过共享部分计算，官方明确列出「长文档反复提问」与「多轮对话」两类受益负载，但也说明 APC 只减少 prefill 耗时、不减少 decode，答案很长或无共享前缀时无收益[^32]。vLLM 提供 -O0 到 -O3 四个优化级别（默认 -O2），并在 KV cache 不足时抢占请求、空间恢复后重算[^33]。

**闭源侧规格。** Gemini 3.1 Pro（Preview）官方 Model information 声明输入上限 1M tokens、输出上限 64k tokens[^20]。Gemini 1.5 报告称长上下文检索达到近乎完美的跨模态召回，next-token prediction 与检索性能在至少 10M token 范围内持续改善、检索准确率 >99%[^11]。

## 四、长上下文是否让 RAG 过时

**支持「部分替代」的证据。** Anthropic 给出了明确边界：知识库小于 200,000 token（约 500 页材料）时，可以直接把整个知识库放进 prompt，无需 RAG 或类似方法；其 prompt caching 使这种整库塞入显著更快更省钱——延迟降低 2 倍以上、成本最多降低 90%[^38]。Databricks 的实验（2,000+ 次实验、13 个模型）也显示，检索更多文档会提高正确信息被传入 LLM 的概率，具备长上下文的现代模型可以利用这一点提升整体 RAG 表现[^37]。

**支持「不可替代」的证据。** 同一篇 Anthropic 文章立刻划出上限：该做法不可扩展，知识库增长到超出上下文窗口时 RAG 仍是典型解决方案[^38]。更关键的是「宣称窗口 ≠ 有效窗口」：RULER 评测 17 个模型、13 类任务，几乎所有模型在 NIAH 上接近满分，但随上下文变长大幅下降；这些模型均宣称 32K 或更长，其中只有一半能在 32K 维持令人满意性能；宣称 200K 的 Yi-34B 随输入长度与任务复杂度上升仍有很大改进空间[^1]。RULER 明确把「宣称的上下文窗口」与「实际可用（有效）上下文长度」区分开，并以基准通过长度作为后者的操作性度量[^1]。

**性能随长度衰减是普遍现象，不是个别模型缺陷。** Chroma 的 Context Rot 报告覆盖 18 个 LLM（闭源与开放权重），结论是随输入长度增加性能呈**非均匀**下降；实验保持任务复杂度不变、只改输入长度以隔离单一变量[^18]。Anthropic 称该衰减「在所有模型上都会出现」，只是部分模型衰减更平缓，并将其归因于 transformer 架构约束（n 个 token 产生 n² 个成对关系）、训练数据中短序列更常见、以及注意力预算（attention budget）随每个新 token 被消耗；其工程原则是找到「最小可能的高信号 token 集合」[^19]。LangChain 把长上下文的问题归为超出窗口、成本/延迟膨胀、agent 性能退化三类，并引用 context poisoning / distraction / confusion / clash 四种具体失效形式[^34]。

**位置与词法匹配两个具体失效轴。** Lost in the Middle 显示：改变相关信息在上下文中的位置会导致性能显著下降，信息位于开头或结尾时最高、必须访问中段时显著退化，且该现象在明确的长上下文模型上同样存在[^29]。NoLiMa 进一步去掉字面匹配：其 needle 与问题词汇重叠极少，要求模型推断潜在关联；评测 13 个宣称至少 128K 的模型，短上下文（<1K）表现良好但随长度显著退化，32K 时 13 个中有 11 个跌破短上下文强基线的 50%，表现最好的 GPT-4o 也从 99.3% 降到 69.7%；即使有推理增强或 CoT prompting 也难以维持[^30]。Databricks 的实测给出模型级拐点：Llama-3.1-405b 在 32k token 之后开始下降、GPT-4-0125-preview 在 64k 之后开始下降，只有少数模型能在所有数据集上维持稳定长上下文 RAG 性能，且失败方式各异（因版权顾虑拒答、总是对上下文做摘要等），许多行为表明长上下文后训练不足[^37]。

**RAG 侧在同步进化，而非静止等待被替代。** LlamaIndex 官方生产级 RAG 指南列出的关键技术包括解耦检索用 chunk 与合成用 chunk、面向大文档集的结构化检索、按任务动态检索、rerank、优化 context embedding；文档明确指出「embedding giant chunks leads to lost in the middle problems」，即靠放大 chunk 塞更多上下文会触发 lost-in-the-middle，因此建议 embed 句子再链接到周围窗口[^35]。标准 top-k + 基础切分在文档数量扩展时表现不佳（如 100 个不同 PDF 的场景），需引入元数据过滤 + 自动检索[^35]。Anthropic 的 Contextual Retrieval 直指传统 RAG 的上下文破坏问题（chunk 切小后可能缺少公司名、时间范围等定位关键上下文），Contextual Embeddings + Contextual BM25 可将检索失败率降低 49%，与 reranking 结合降低 67%；同时指出 embedding 擅长语义但可能漏掉精确匹配，BM25 的词法匹配在「Error code TS-999」这类唯一标识符查询上更有效[^38]。框架侧活跃度也支持「RAG 未退场」：LlamaIndex 52,639 stars、MIT、最近 push 2026-10-23，topics 中 rag 与 agents 并列[^17]；node postprocessor 作为检索后、response synthesis 前的标准变换/过滤环节仍是文档化模块[^36]。

**综合判断。** 现有证据支持的是分层结论：在**小知识库、单次或少次查询、且任务对中段信息与语义推断要求不高**的场景，1M 级窗口可以直接吞下语料，RAG 的检索环节可被省略[^38]；在**知识库持续增长、需要精确词法匹配、需要多跳/干扰项消歧、或成本敏感**的场景，长上下文不能替代检索——因为有效窗口短于宣称窗口[^1]、性能随长度非均匀衰减[^18][^19][^30]、中段信息利用差[^29]，而检索侧的 contextual retrieval 与 rerank 仍在带来两位数百分比的失败率下降[^38][^35]。Databricks 把「长上下文模型是否会最终吞并 RAG 工作流」明确列为业界争论而非已决结论[^37]。

---

## 矛盾与不确定性

**1. 三个主流模型的百万级窗口主张在本轮取证中零一手支撑（blocking）。** Claude 的 200K–1M 上下文（含 Sonnet 4.5 的 1M beta）：docs.claude.com 返回地区封锁页、无法取得任何窗口规格[^22]，Sonnet 4.5 官方公告正文未提及 1M beta、长上下文溢价倍率或 200K 窗口规格[^21]，GitHub 文档镜像按推测路径读取均 404、未提供可用事实[^23]。GPT-4.1 的 1M 窗口：Azure 模型页仅把 GPT-4.1 系列列在 Model highlights 型号列表，未给出 context window / max output / 定价[^24]；openai-cookbook 只取到仓库元数据，notebook 路径 404[^25]。Llama 4 的 10M 宣称窗口：Meta 官方博客连接失败[^26]、HF 模型卡连接失败[^27]、meta-llama/llama-models 返回 403[^28]，三条来源均无正文。因此本报告中**不把这三项数字作为事实陈述**；现有公开资料不足以确认其规格。

**2. Gemini 的「10M 上 >99% 检索」与「1M 上 MRCR 26.3%」表面冲突，口径不同但并列引用会误导。** Gemini 1.5 报告称 next-token prediction 与检索在至少 10M token 范围内持续改善、检索准确率 >99%[^11]；而 Gemini 3.1 Pro 官方 Performance 表显示 MRCR v2 1M (pointwise) 档 Gemini 3.1 Pro Thinking (High) 与 Gemini 3 Pro Thinking (High) 均为 26.3%，同表标注 Sonnet 4.6、Opus 4.6、GPT-5.2 在该档 "Not supported"[^20]。二者任务不同（NIAH 式检索 vs 8-needle MRCR），且前者为 2024 年数据、相对取证时点 2026-10-24 已偏旧；后者为 Google 自报数据[^20]。同一页面 128k (average) 档为 84.9% / 77.0%，与 1M 档的 26.3% 落差本身即说明「窗口长度」与「多针共指消解能力」不是同一维度[^20]。此外该页 Model information 只声明输入 1M/输出 64k，模型卡正文本轮未取到[^20]。

**3. 基准效度本身存在争议。** NIAH 只反映表层理解能力[^1]，本质是简单检索、只测直接词法匹配[^18]；NoLiMa 中 72.4% 的 needle-question 对需要外部世界知识，因此更接近「非词法匹配 + 世界知识」的联合测试[^18]。这意味着不同报告之间的「长上下文分数」不可直接横向比较。

**4. 量化维度整体空缺：KV cache 显存与每百万 token 成本无一手数值。** Qwen2.5-1M 正文（架构与预训练章节）未出现任何 KV cache 显存 GB 数值或每百万 token 成本数值[^41]；MiniMax-01 已读正文段落同样未出现，相关对比可能位于被工具裁剪掉的后续章节[^42]。DeepSeek 侧只有 DSA 的定性描述与「API 价格下调 50% 以上」，无绝对价格[^40]。因此本报告无法给出百万级上下文的成本量级对比。可用的只有间接线索：Qwen2.5-1M 在 1M 场景下 3x–7x prefill 加速[^13]、ThinK 的 2.8 倍峰值内存降低[^10]、APC 的「只省 prefill 不省 decode」边界[^32]。

**5. 部分仓库证据仅为元数据，不构成技术结论。** Lost-In-The-Middle-experiments 仓库自述含 7 个实验、用 Position Bias Index 量化 U 形曲线，但 star 2、fork 0，且未读取 README 正文，按取证纪律不得作为结论依据[^31]。vLLM 优化文档的导航栏虽列有 Context Parallel Deployment、KV Offloading、Paged Attention 等条目，但抓取正文未包含 chunked prefill、context parallel 或 KV cache 量化的具体说明[^33]。

**6. 厂商自述与第三方评测的强度差异需保留。** YaRN「超过此前 SOTA」为论文自述[^3]；Gemini 3.1 Pro 的 MRCR 数据为 Google 自报[^20]；MiniMax-01「与 GPT-4o、Claude-3.5-Sonnet 相当且上下文长 20–32 倍」为论文自述[^12]；DeepSeek 的「质量几乎不变 / on par」为官方自述[^39][^40]。这些均未经本轮独立复现。

## 参考来源

[^1]: RULER: What's the Real Context Size of Your Long-Context Language Models? — https://arxiv.org/abs/2404.06654v3（arxiv, 2024-04-09）
[^2]: Extending Context Window of Large Language Models via Positional Interpolation — https://arxiv.org/abs/2306.15595v2（arxiv, 2023-06-27）
[^3]: YaRN: Efficient Context Window Extension of Large Language Models — https://arxiv.org/abs/2309.00071v3（arxiv, 2023-08-31）
[^4]: jquesnelle/yarn — YaRN 官方代码与模型仓库 — https://github.com/jquesnelle/yarn（github）
[^5]: LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens — https://arxiv.org/abs/2402.13753v1（arxiv, 2024-02-21）
[^6]: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — https://arxiv.org/abs/2205.14135v2（arxiv, 2022-05-27）
[^7]: Ring Attention with Blockwise Transformers for Near-Infinite Context — https://arxiv.org/abs/2310.01889v4（arxiv, 2023-10-03）
[^8]: Mamba: Linear-Time Sequence Modeling with Selective State Spaces — https://arxiv.org/abs/2312.00752v2（arxiv, 2023-12-01）
[^9]: Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention — https://arxiv.org/abs/2502.11089v2（arxiv, 2025-02-16）
[^10]: ThinK: Thinner Key Cache by Query-Driven Pruning — https://arxiv.org/abs/2407.21018v3（arxiv, 2024-07-30）
[^11]: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context — https://arxiv.org/abs/2403.05530v5（arxiv, 2024-03-08）
[^12]: MiniMax-01: Scaling Foundation Models with Lightning Attention — https://arxiv.org/abs/2501.08313v1（arxiv, 2025-01-14）
[^13]: Qwen2.5-1M Technical Report — https://arxiv.org/abs/2501.15383v1（arxiv, 2025-01-26）
[^14]: DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models — https://arxiv.org/abs/2512.02556v1（arxiv, 2025-12-02）
[^15]: vLLM — Easy, fast, and cheap LLM serving for everyone — https://github.com/vllm-project/vllm（github, 2026-10-24）
[^16]: SGLang — Fast inference for LLMs and multimodal models — https://github.com/sgl-project/sglang（github, 2026-10-24）
[^17]: LlamaIndex Framework — https://github.com/run-llama/llama_index（github, 2026-10-23）
[^18]: Context Rot: How Increasing Input Tokens Impacts LLM Performance (Chroma 技术报告) — https://research.trychroma.com/context-rot（web, 2025-07）
[^19]: Effective context engineering for AI agents (Anthropic Engineering) — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents（web）
[^20]: Gemini 3.1 Pro — Google DeepMind 官方模型页（Model information / Performance 表） — https://deepmind.google/models/gemini/pro/（web）
[^21]: Introducing Claude Sonnet 4.5 (Anthropic 官方发布公告) — https://www.anthropic.com/news/claude-sonnet-4-5（web, 2025-09-29）
[^22]: Anthropic 官方文档站 context-windows 页面（访问受阻，仅取得地区限制提示） — https://docs.claude.com/en/docs/build-with-claude/context-windows（web）
[^23]: thevibeworks/claude-code-docs — Anthropic 文档 markdown 归档镜像（仅取得仓库元数据） — https://github.com/thevibeworks/claude-code-docs（github, 2025-06-24）
[^24]: Azure OpenAI / Microsoft Foundry Models — model capabilities table (GPT-4.1 series listed in model highlights only) — https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/models（web）
[^25]: openai/openai-cookbook — repository metadata (README/notebook 正文未取得) — https://github.com/openai/openai-cookbook（github）
[^26]: Meta AI 官方博客：Llama 4 multimodal intelligence（抓取失败，未取得正文） — https://ai.meta.com/blog/llama-4-multimodal-intelligence/（web）
[^27]: Hugging Face 模型卡：meta-llama/Llama-4-Scout-17B-16E-Instruct（抓取失败，未取得正文） — https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct（web）
[^28]: Meta 官方仓库 meta-llama/llama-models（详情接口 403，未取得 README / MODEL_CARD 正文） — https://github.com/meta-llama/llama-models（github）
[^29]: Lost in the Middle: How Language Models Use Long Contexts (arXiv:2307.03172v3) — https://arxiv.org/abs/2307.03172v3（arxiv, 2023-07-06）
[^30]: NoLiMa: Long-Context Evaluation Beyond Literal Matching (arXiv:2502.05167v3) — https://arxiv.org/abs/2502.05167v3（arxiv, 2025-02-07）
[^31]: abhishek-pandey7/Lost-In-The-Middle-experiments — https://github.com/abhishek-pandey7/Lost-In-The-Middle-experiments（github）
[^32]: Automatic Prefix Caching — vLLM documentation — https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html（web, 2026-10-13）
[^33]: Optimization and Tuning — vLLM documentation — https://docs.vllm.ai/en/latest/configuration/optimization.html（web）
[^34]: Context Engineering for Agents (LangChain 官方博客) — https://blog.langchain.com/context-engineering-for-agents/（web, 2025-07-02）
[^35]: Building Performant RAG Applications for Production (LlamaIndex 官方文档) — https://docs.llamaindex.ai/en/stable/optimizing/production_rag/（web）
[^36]: Querying Node Postprocessors (LlamaIndex 官方文档) — https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/（web）
[^37]: Long Context RAG Performance of LLMs (Databricks Blog) — https://www.databricks.com/blog/long-context-rag-performance-llms（web）
[^38]: Introducing Contextual Retrieval (Anthropic) — https://www.anthropic.com/news/contextual-retrieval（web）
[^39]: deepseek-ai/DeepSeek-V3.2-Exp (official repository README) — https://github.com/deepseek-ai/DeepSeek-V3.2-Exp（github, 2025-09-29）
[^40]: DeepSeek API Docs — Introducing DeepSeek-V3.2-Exp (official news, 2025-09-29) — https://api-docs.deepseek.com/news/news250929（web, 2025-09-29）
[^41]: Qwen2.5-1M Technical Report（正文 §2 Architecture，Table 1） — https://arxiv.org/html/2501.15383v1（arxiv, 2025-01-26）
[^42]: MiniMax-01: Scaling Foundation Models with Lightning Attention（正文 §2 架构与线性注意力章节） — https://arxiv.org/html/2501.08313v1（arxiv, 2025-01-14）