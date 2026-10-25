# 百万级上下文窗口的技术路径与 RAG 的存废之争

## 摘要

把上下文窗口推到 100 万 token 级别，公开文献中可辨识的技术手段分布在四个层面：（1）**位置编码重标定**——以 RoPE 插值为核心的 PI → NTK-aware → NTK-by-parts → YaRN 谱系，以及 LongRoPE 系列的非均匀插值搜索与渐进式扩展[^1][^2][^3][^7]；（2）**注意力与架构替换**——滑动窗口/稀疏注意力、线性注意力与 SSM 等次二次结构，以及混合注意力（如 MiniMax-01 的 7:1 lightning/softmax 配比）[^36][^37][^38][^6][^8]；（3）**训练数据与课程**——长数据打包、超评估长度训练、短指令 SFT，ProLong 用 40B token 把 Llama-3-8B 推到 512K[^26][^27]；（4）**推理侧显存治理**——KV cache 驱逐、量化、低秩压缩与 IO-aware kernel，这是 1M 上下文能否实际服务的关键约束[^15][^16][^17][^20][^22][^29][^5]。

关于 RAG 是否过时：证据不支持「过时」这一判断。多项系统评测在资源充足时显示长上下文平均优于 RAG[^32][^34]，但 RAG 在成本、对话类与通用问句查询上仍有优势[^32][^33][^34]；Anthropic 给出的工程阈值是知识库小于约 200,000 token 时可直接全量入 prompt[^35]。更准确的表述是：长上下文压缩了 RAG 的适用区间，并在评测层面暴露出「宣称长度 ≠ 有效长度」的问题[^4][^25]，二者正在走向路由式混合。

---

## 一、位置编码重标定：把训练长度「拉伸」而非「外推」

**问题定义。** RoPE 类模型无法泛化到训练序列长度之外[^2]。外推失败的机制被明确表述为：超出训练上下文长度会产生灾难性高的注意力分数，从而破坏自注意力机制[^1]。

**Position Interpolation（PI）的完整算法。** 输入：一个基于 RoPE 预训练的 LLM、目标上下文长度、少量微调数据。步骤：将输入 token 的位置索引**线性下采样**到原始上下文窗口大小之内，而不是外推到训练长度之外；随后做极少步微调（1000 步以内）。产出：保留原架构、上下文扩展后的模型。理论依据是插值的上界至少比外推的上界小约 600 倍，因此更稳定[^1]。LLaMA 7B–65B 由此扩展到最多 32768，且在原始窗口内的任务（passkey retrieval、语言建模、长文档摘要）质量保持相对良好[^1]。

**YaRN 及其谱系。** YaRN 论文把 RoPE 插值方法梳理为一条演进链：PI（位置插值）→ NTK-aware（考虑高频损失）→ 两个改进分支：Dynamic NTK（免微调）与 NTK-by-parts（少量长上下文数据微调时表现最好）→ YaRN[^2]。YaRN 本身只需此前方法 1/10 的 token 与 1/2.5 的训练步数，在少于原始预训练数据约 0.1% 的数据上微调即达到当时 SOTA；配合推理时 Dynamic Scaling，Dynamic-YaRN 可在完全不微调的情况下实现 2 倍以上扩展[^2]。工程采用情况：Code Llama 使用 NTK-aware，Qwen 7B 使用 Dynamic NTK[^2]。适用范围限于 RoPE 训练的模型家族（LLaMA、GPT-NeoX、PaLM）[^2]。

**LongRoPE：非均匀性与渐进式扩展。** LongRoPE 首次把预训练 LLM 扩展到 2048k token，微调步数最多 1k，训练长度在 256k 以内[^3]。其关键创新是识别并利用位置插值中的**两种非均匀性**，通过高效搜索得到更好的微调初始化，并在免微调场景下实现 8 倍扩展；策略上是渐进式的：先微调出 256k 模型，再对扩展后模型做第二次位置插值达到 2048k，并在 8k 长度上重新调整以恢复短上下文性能[^3]。论文把此前扩展多止步于约 128k 归因于三点：高微调成本、长文本稀缺、新 token 位置引入的灾难性数值[^3]。

**LongRoPE2：把 OOD 归因到高维 RoPE 维度训练不足。** 其假设是「高维 RoPE 维度训练不足」导致现有方法持续出现分布外问题；方法上用 needle-driven perplexity 引导的进化搜索求 rescaling 参数，并提出混合上下文窗口训练——长序列用重缩放 RoPE、短序列用原始 RoPE 以保住短上下文性能[^7]。结果：LLaMA3-8B 扩展到 128K 有效上下文，保留超过 98.5% 的短上下文性能，仅用 10B token，比 Meta 的方法少 80 倍，而后者未达到目标有效上下文长度[^7]。

**代价。** Anthropic 指出位置编码插值等技术可让模型处理更长序列，但会带来 token 位置理解上的退化[^12]。

---

## 二、注意力机制与架构层：从二次复杂度里脱身

**复杂度根因。** 自注意力的时间与内存复杂度对序列长度是二次的，这是长序列上 Transformer 慢且吃内存的根因[^5]。

**精确注意力的 IO 优化（不改变数学）。** FlashAttention 是 IO 感知的**精确**注意力算法，用 tiling 减少 GPU HBM 与片上 SRAM 之间的读写次数，不牺牲模型质量；其 HBM 访问次数少于标准注意力，并在一系列 SRAM 尺寸下达到最优[^5]。实测加速：BERT-large（512）端到端 15%，GPT-2（1K）3 倍，Long Range Arena（1K–4K）2.4 倍；并使 Path-X（16K）61.4%、Path-256（64K）63.1% 首次超过随机水平[^5]。论文明确批评此前的近似注意力方法以牺牲模型质量为代价、且往往拿不到实际墙钟加速[^5]。工程侧，FlashAttention 已是主流推理栈的标配 kernel 之一[^9]，其独立仓库有 25,211 stars[^11]。

**局部/稀疏注意力。** Longformer 的滑动窗口在固定窗口 w 下每 token 只 attend 左右各 1/2·w 个 token，复杂度 O(n×w) 随输入长度线性增长（对比全注意力 O(n²)）；ℓ 层堆叠后顶层感受野为 ℓ×w，膨胀窗口（间隔 d）则为 ℓ×d×w，即使 d 很小也能达到数万 token[^36]。但 Longformer 明确指出纯局部窗口不足，需引入任务驱动的 global attention，且论文本身未给出相对全注意力的信息损失上界或形式化近似误差界[^36]。BigBird 把稀疏注意力形式化为有向图 D 上的广义注意力——边集即被计算的内积集合，邻接矩阵 A(i,j)=1 表示 query i attend key j，全注意力对应完全有向图，于是「降二次复杂度」变成图稀疏化问题；其构造为 g 个全局 token + w 个局部邻居 + r 个随机 token，复杂度对 token 数线性，并证明它是序列函数的通用近似器且图灵完备，O(1) 个全局 token 带来可证明收益[^37]。BigBird 摘要称该稀疏注意力可处理长度达此前同硬件下 8 倍的序列[^37]。Mistral 7B 采用 SWA：每层信息最多前移 W 个 token，k 层后最多 k×W；配置 window_size=4096、n_layers=32、context_len=8192，最后一层理论注意力跨度约 131K token；SWA 使 KV cache 可用固定大小 rolling buffer（位置 i 存于 i mod W），序列超过 W 后缓存不再增长，32k 序列上缓存内存降低 8 倍[^38]。

**次二次架构与混合注意力。** Mamba 让 SSM 参数成为输入的函数（selective），使模型能沿序列长度维度选择性传播或遗忘信息，从而补上此前次二次架构「缺乏基于内容推理能力」的关键弱点；推理吞吐比 Transformer 高 5 倍，对序列长度线性缩放，性能随序列长度提升直至百万级，且是不含注意力甚至不含 MLP 块的端到端架构[^6]。MiniMax-01 走混合路线：每 7 个使用 lightning attention（linear attention 的 I/O-aware 实现）的 transnormer block 后接 1 个 softmax attention transformer block，共 80 层；训练阶段上下文窗口达 100 万 token，推理阶段外推到 400 万 token[^8]。其 RoPE 施加于一半的注意力 head 维度，base frequency 设为 10,000[^8]。论文把上下文扩展归因于更强 GPU 与更好的 I/O-aware softmax attention 实现，并指出进一步扩展受限于 transformer 的二次计算复杂度[^8]。

---

## 三、训练侧：数据配比、长度课程与 SFT 策略

ProLong 系统研究了 continued training 与 SFT 两个阶段，三条可操作结论：（1）code repositories 与 books 是优质长数据来源，但**必须与高质量短上下文数据结合**；（2）用超过评估长度的序列长度训练可提升长上下文性能；（3）SFT 阶段仅用短指令数据集即可在长上下文任务上取得强性能[^26]。

工程细节可核验：ProLong 从 Llama-3-8B 继续训练并 SFT，最大窗口 512K；两阶段课程为先在 64K 数据上训 20B token、再在 512K 数据上训 20B token；SFT 用 UltraChat（约 1B token）[^27]。数据构造用自研 datatools 按长度过滤并打包，示例命令 `pack --pack_length 65536 --min_length 65536`，即只保留长度不低于目标上下文长度的文档再拼接；长数据来源域清单包括 Code repos 689GB、Books 180GB、FineWeb 864GB、ArXiv 210GB 等[^27]。训练代码基于 Hugging Face Transformers，改动包括 mosaicml-streaming 数据格式、FlashAttention-2 变长注意力（document masking）与 in-batch 长度排序 dataloader[^27]。结果：ProLong-8B 训练 40B token，在 128K 上于同规模模型中报告 SOTA（作者自述），多数长上下文任务优于 Llama-3.1-8B-Instruct 而长上下文训练 token 仅为其 5%，可有效处理最长 512K[^26]。

Gemini 1.5 侧的证据表明规模路线的收益不止于检索：仅将 500 页参考语法、词典与约 400 条平行句放入上下文，模型即可学习英语→Kalamang 翻译，水平接近用同样材料学习的人类[^28]。

---

## 四、推理侧显存治理：1M 上下文能否真正服务

**为什么这是瓶颈。** KV cache 主导显存：batch size 128、序列长度 2048 时，OPT-175B 的 KV cache 约 1152GB 而模型权重仅 325GB；LLaMA-65B 为 640GB vs 130GB；BLOOM 为 950GB vs 352GB[^18]。MHA 的标准计算式是每 token 缓存 2·n_h·d_h·l 个元素（n_h 头数、d_h 每头维度、l 层数）[^30]。

**驱逐（保留子集）。** StreamingLLM 发现 attention sink 现象——Softmax 需归一化，模型需要一个「倾倒」多余注意力的位置，因此即使初始 token 语义不重要也获得大量注意力分数；方法上只保留 4 个 sink token 的 KV 加滑动窗口内最近 token 的 KV，无需微调即可让有限窗口训练的 LLM 泛化到无限序列，cache 大小与序列长度解耦、显存恒定；可使 Llama-2-[7,13,70]B、MPT、Falcon、Pythia 稳定建模 400 万 token 甚至更长，流式设置下最高 22.2x 加速[^15]。H2O 观察到注意力矩阵稀疏度超过 95%（仅 5% KV 即可解码出相同输出 token，理论支持最高 20x 缩减），且累积注意力分数服从幂律、存在一小组 heavy hitter，完全移除会导致显著退化；方法上动态保留「近期 token + H2 token」，主推 20% heavy hitter 预算，在 OPT-6.7B/30B 上相比三个基线分别最高提升 29x、29x、3x 吞吐[^16]。Scissorhands 基于「重要性持久性假设」——只有曾在某步产生显著影响的 pivotal token 才会在未来继续显著影响，实证上句子靠后 token 主要只关注早期 token 已重点关注过的 token，多数层重叠率超 90%；无需微调即可把 KV cache 推理显存降低最多 5x，与 4-bit 量化结合可达 20x[^18]。SnapKV 的关键补充是：既有方法（StreamLLM、H2O、FastGen、Scissorhands）主要压缩解码阶段追加的 KV，而**忽略了 prompt 阶段 KV 的压缩**，后者才是长上下文场景的内存瓶颈；其方法利用 prompt 末尾 observation window 在生成前判断各头的重要 KV，16K 输入下生成速度提升 3.6x、内存效率提升 8.2x，单张 A100-80GB 可处理最多 380K 上下文 token[^17][^19]。

**量化。** KIVI 是无需微调的 2-bit KV 量化，核心发现是 key cache 应按 per-channel、value cache 应按 per-token 量化；Llama/Falcon/Mistral 上 2bit 维持几乎相同质量，峰值显存降低 2.6×，batch size 最大提升 4×，吞吐提升 2.35×~3.47×[^20]。其 Table 1 给出配置敏感性的量化证据：Llama-2-13B 上 16bit 基线 CoQA 66.37 / TruthfulQA 29.53；2bit（K per-token, V per-token）降至 52.93 / 24.98；2bit（K per-channel, V per-token）为 63.53 / 28.60，是 2bit 下最准配置；而 value 按 per-channel 量化会崩溃（CoQA 2.88 / 0.74 与 2.80 / 0.26）[^21]。误差分析解释了原因：K 按 per-token 的 attention score 相对误差 47.00、per-channel 为 9.60；V 按 per-token 的注意力输出相对误差 3.55、per-channel 为 49.89[^21]。KVQuant 用四项技术（Per-Channel Key Quantization、Pre-RoPE Key Quantization、非均匀量化、Per-Vector Dense-and-Sparse）做到 3-bit 下 perplexity 退化 < 0.1，带来 4.8× 显存压缩与 4.8× 更长上下文，使 LLaMA-7B 可在单张 A100-80GB 上服务最长 1M 上下文、8-GPU 系统上最长 10M；仅移除 1% outlier 即可在 3-bit 下达到 <0.1 退化[^22]。QServe 选 KV4 而非 KV8 的理由是解码阶段 attention 是计算强度 1 MAC/element 的 batched GEMV、访存由 KV cache 主导，KV4 相对 KV8 提供 2× attention 峰值性能；并用 SmoothAttention 把激活量化难点从 keys 转移到不被量化的 queries[^23]。

**低秩压缩（架构级）。** DeepSeek-V2 的 MLA 把 KV cache 压缩成一个 latent vector：对 key 与 value 做低秩联合压缩 c_t^KV = W^DKV · h_t，再由 W^UK / W^UV 上投影得到 k_t^C 与 v_t^C，因此只需缓存压缩后的 latent 而非完整 K/V[^30]。相比 DeepSeek 67B，KV cache 减少 93.3%，最大生成吞吐提升至 5.76 倍，训练成本节省 42.5%[^29]。论文明确 MQA/GQA 虽降低 KV cache 但性能不及 MHA，MLA 声称性能优于 MHA 同时 KV cache 显著更小[^29][^30]。

**服务栈。** vLLM 通过 PagedAttention 管理 KV 显存，支持 continuous batching、chunked prefill、prefix caching，并提供 tensor/pipeline/data/expert/context 五种并行（context parallelism 与长上下文切分直接相关）；attention kernel 覆盖 FlashAttention、FlashInfer、TRTLLM-GEN、FlashMLA、Triton；声称无缝支持 Hugging Face 上 200+ 架构，含混合注意力与状态空间模型（Mamba、Qwen3.5）[^9]。仓库 95,404 stars、23,762 次提交[^9]；SGLang 为 38,111 stars[^10]。

---

## 五、宣称长度 ≠ 有效长度：评测侧的校正

RULER 区分「宣称的上下文长度」与「实际有效性能长度」：几乎所有模型在 vanilla NIAH 上接近满分，但随上下文变长出现大幅性能下降；17 个宣称支持 32K 或更长的模型中只有一半能在 32K 上维持令人满意性能；NIAH 只反映表层检索能力，RULER 因此增加多跳追踪与聚合等 13 类任务；支持 200K 的 Yi-34B 在输入变长与任务变复杂时仍有很大改进空间[^4]。HELMET 进一步指出 NIAH 这类合成任务不能可靠预测下游性能，各任务类别趋势各异且相关性低；多数模型 NIAH 满分，但当任务需要全上下文推理或遵循复杂指令时，开源模型显著落后闭源且差距随长度扩大[^25]。LongBench v2 用 503 道高难度多选题（8k–2M words，六大任务类）显示：15 分钟时限下人类专家仅 53.7%，直接作答最佳模型仅 50.1%，o1-preview 57.7%[^24]。更早的 Lost in the Middle 已证明性能在相关信息位于开头或结尾时最高、必须访问中间信息时显著退化，即使对显式支持长上下文的模型也如此[^31]。

---

## 六、长上下文是否让 RAG 过时

**支持「长上下文更优」的证据。** Self-Route 论文在多个公开数据集与三个 LLM 上系统对比，结论是资源充足时长上下文在平均性能上稳定优于 RAG，但 RAG 显著更低的成本仍是明确优势[^34]。Long Context vs. RAG 在过滤掉「无需外部上下文即可回答」的问题、识别最有效检索方法并扩展数据集后重新评测，结论是问答基准上 LC 总体优于 RAG，尤其在 Wikipedia 类问题上；摘要式检索与 LC 相当，分块式检索落后于 LC[^32]。Gemini 1.5 论文的表述更激进：在更真实的多模态长上下文基准上，Gemini 1.5 Pro 在所有模态上优于「即使被外部检索方法增强」的竞争模型[^28]。

**支持 RAG 仍不可替代的证据。** 同一篇 Long Context vs. RAG 明确指出 RAG 在对话类与通用问句查询上具有优势，并强调既有研究忽视了「上下文相关性」的重要性[^32]。其 Table 1 汇总的既有研究呈现明显分歧：ChatQA2 发现在不超过 32K 的序列上 RAG 优于 LC，且 3K→24K 更大窗口对 RAG 更有利；LongBench 发现检索对 4K 模型有帮助但对 16K/32K 模型没有帮助；Ret-LC LLM 发现 RAG 在 70B/43B 模型上对所有上下文长度都有提升；OP-RAG 发现高效检索可优于暴力全上下文输入；Leng et al. 2024 发现多数开源模型的 RAG 在 16k–32k 达峰后性能下降；LongBench v2 发现 Qwen2.5 与 GLM-4-Plus 在 RAG 上下文超 32k 后性能下降[^33]。关于「LC 与 RAG 结合是否有利」也存在分歧：部分研究认为结合有效，另一些认为可能无益[^33]。

**成本阈值与混合路由。** Anthropic 给出明确工程阈值：知识库小于 200,000 token（约 500 页）时可直接把整个知识库放进 prompt 无需 RAG，配合 prompt caching 可降低延迟 >2x、成本最多降低 90%；超过则需 Contextual Retrieval，其 Contextual Embeddings + Contextual BM25 在嵌入与建索引前为每个 chunk 前置 50–100 token 的 chunk 专属解释性上下文，把失败检索数量降低 49%、结合 rerank 后降低 67%[^35]。Self-Route 的做法是基于模型自我反思把查询路由到 RAG 或长上下文，在保持与纯长上下文可比性能的同时显著降低计算成本[^34]。Anthropic 的 context engineering 立场与此一致：上下文是有限资源且边际收益递减，随窗口 token 数增加模型准确召回能力下降（context rot，所有模型都有、幅度不同），归因于 transformer 的 n² 成对关系摊薄了「attention budget」；工程原则是找到能最大化目标结果概率的**最小高信号 token 集合**[^12]。其 agent 构建指南也建议先找最简单方案，对许多应用而言用检索与 in-context examples 优化单次调用通常已足够[^13]。

**结论性判断。** 证据支持「长上下文压缩了 RAG 的适用区间、并在平均性能上取得优势」，不支持「RAG 已过时」：RAG 的成本优势[^34]、对话类/通用问句优势[^32]、以及开源模型在 RAG 上下文超 32k 后性能反降的现象[^33]，都指向二者互补。同时，长上下文自身的有效性受「宣称 ≠ 有效」约束[^4][^25]与中间信息退化约束[^31]，这进一步削弱了「全量塞入即可替代检索」的论断。ERASE 从另一角度提示 RAG 的问题不在「是否检索」而在「知识更新」——它通过增量删除或改写知识库条目而非改进检索或推理，在两个新基准上相对常规 RAG 绝对提升 7–13%（Mixtral-8x7B）与 6–10%（Llama-3-8B），但该证据仅取到摘要且对比对象是常规 RAG 而非长上下文模型，与本节主线关系较弱[^14]。

---

## 矛盾与不确定性

**1. 1M 级模型的现状清单严重不完整。** 本报告只能覆盖 MiniMax-01（训练 1M / 推理外推 4M）[^8]与 Gemini 1.5（宣称至少 10M，NIAH >99.7% 的数据来自 2024 年 2 月版本）[^28]。2025–2026 年当前主流 1M 级模型（Gemini 2.x/3 系列、Llama 4、Qwen 长上下文变体、Kimi、GLM 等）的窗口大小与扩展策略在给定证据中完全缺失。因此第一节至第四节归纳的「技术手段」应理解为**公开文献中可核验的方法谱系**，而非对当前工业界主流配比的判断——现有公开资料不足以确认 2026 年主流模型实际采用哪几种手段的组合。

**2. KV cache 每 token 字节数无法给出。** 证据只提供压缩倍数（KIVI 2.6×[^20]、KVQuant 4.8×[^22]、MLA 减少 93.3%[^29]、Scissorhands 5x/20x[^18]）与 MHA 的 2·n_h·d_h·l 元素计数式[^30]，没有任何一条给出具体模型在具体精度下的 bytes/token 数值。若需该量级，只能由公式自行推算，而推算所需的模型配置（层数、头数、头维度、精度）在证据中并不齐备——现有公开资料不足以确认。

**3. 吞吐/延迟随上下文长度的退化曲线缺失。** MiniMax-01 只描述了 Figure 2 的二次函数拟合与测试条件（H800、TP=8、W8A16，其他模型经官方 API），未给出可引用数值[^8]；H2O、SnapKV、QServe 的加速比均相对特定基线而非随长度变化的曲线[^16][^17][^23]。从 1K→128K→1M 的 prefill/decode 延迟实测数据点在证据中不存在。

**4. 「长上下文优于 RAG」的结论对评测设计高度敏感。** 同一主题下不同论文结论相反[^32][^33]，且 Long Context vs. RAG 自身承认既有研究忽视上下文相关性[^32]。RULER 与 HELMET 又表明 NIAH 类指标不能预测下游性能[^4][^25]，这意味着任何以 NIAH 为依据的「长上下文够用」论断都不可靠。

**5. 稀疏/局部注意力的理论保证不统一。** BigBird 给出通用近似与图灵完备性证明[^37]，而 Longformer 论文本身未给出相对全注意力的信息损失上界或形式化近似误差界[^36]；两者对「此前稀疏方法是否缺乏理论保证」的表述也不完全一致（BigBird 称多数此前方法为启发式、Longformer 是其主要例外）[^37]。此外，FlashAttention 明确区分自己（精确）与 block-sparse 变体（近似）[^5]，引用时不应混为一谈。

**6. 部分来源为二手或受限。** SGLang 与 flash-attention 仅有仓库元数据、README 正文未读取[^10][^11]；ERASE 仅取到摘要[^14]；Anthropic agent 一文顶部注明所述工具生态自 2024 年 12 月起已变化[^13]；Gemini 1.5 的 Figure 1 性能数据明确来自 2024 年 2 月版本而非更新版[^28]。这些来源支撑的论断应视为方向性而非精确值。

## 参考来源

[^1]: Extending Context Window of Large Language Models via Positional Interpolation (Position Interpolation, PI) — https://arxiv.org/abs/2306.15595v2（arxiv, 2023-06-27）
[^2]: YaRN: Efficient Context Window Extension of Large Language Models — https://arxiv.org/abs/2309.00071（arxiv, 2023-08-31）
[^3]: LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens — https://arxiv.org/abs/2402.13753（arxiv, 2024-02-21）
[^4]: RULER: What's the Real Context Size of Your Long-Context Language Models? — https://arxiv.org/abs/2404.06654（arxiv, 2024-04-09）
[^5]: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — https://arxiv.org/abs/2205.14135（arxiv, 2022-05-27）
[^6]: Mamba: Linear-Time Sequence Modeling with Selective State Spaces — https://arxiv.org/abs/2312.00752（arxiv, 2023-12-01）
[^7]: LongRoPE2: Near-Lossless LLM Context Window Scaling — https://arxiv.org/abs/2502.20082（arxiv, 2025-02-27）
[^8]: MiniMax-01: Scaling Foundation Models with Lightning Attention (arXiv:2501.08313v1) — https://arxiv.org/abs/2501.08313（arxiv, 2025-01-14）
[^9]: vllm-project/vllm — README (high-throughput LLM inference and serving engine) — https://github.com/vllm-project/vllm（github, 2026-10-25）
[^10]: sgl-project/sglang — 仓库元数据（SGLang serving framework） — https://github.com/sgl-project/sglang（github, 2026-10-25）
[^11]: Dao-AILab/flash-attention — 仓库元数据（Fast and memory-efficient exact attention） — https://github.com/Dao-AILab/flash-attention（github, 2026-10-24）
[^12]: Effective context engineering for AI agents (Anthropic Engineering) — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents（web）
[^13]: Building effective agents (Anthropic Engineering) — https://www.anthropic.com/engineering/building-effective-agents（web, 2024-12-19）
[^14]: Language Modeling with Editable External Knowledge (ERASE), arXiv:2406.11830 — https://arxiv.org/abs/2406.11830（arxiv, 2024-06-17）
[^15]: Efficient Streaming Language Models with Attention Sinks (StreamingLLM) — https://arxiv.org/abs/2309.17453（arxiv, 2023-09-29 (v1), last revised 2024-04-07 (v4); ICLR 2024）
[^16]: H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models — https://arxiv.org/abs/2306.14048（arxiv, 2023-06-24 (v1), latest v3）
[^17]: SnapKV: LLM Knows What You are Looking for Before Generation — https://arxiv.org/abs/2404.14469（arxiv, 2024-04-22 (v1), last revised 2024-06-17 (v2)）
[^18]: Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time — https://arxiv.org/abs/2305.17118（arxiv, 2023-05-26 (v1), latest v2）
[^19]: SnapKV (related-work 章节对各方法的横向定位) — https://arxiv.org/html/2404.14469v2（arxiv, 2024-06-17 (v2)）
[^20]: KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (ICML 2024) — https://arxiv.org/abs/2402.02750v2（arxiv, 2024-02-05）
[^21]: KIVI 正文 Table 1：Llama-2-13B 上不同 KV 量化配置的模拟结果（group size 32） — https://arxiv.org/html/2402.02750v2（arxiv, 2024-02-05）
[^22]: KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization — https://arxiv.org/abs/2401.18079v6（arxiv, 2024-01-31）
[^23]: QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving (MLSys 2025) — https://arxiv.org/abs/2405.04532v3（arxiv, 2024-05-07）
[^24]: LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks — https://arxiv.org/abs/2412.15204v2（arxiv, 2024-12-19）
[^25]: HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly — https://arxiv.org/abs/2410.02694v3（arxiv, 2024-10-03）
[^26]: How to Train Long-Context Language Models (Effectively) (ProLong, arXiv:2410.02660v4, ACL 2025) — https://arxiv.org/abs/2410.02660（arxiv, 2024-10-03 (v1); v4 2025-12-03）
[^27]: princeton-nlp/ProLong — Homepage for ProLong and paper "How to Train Long-Context Language Models (Effectively)" — https://raw.githubusercontent.com/princeton-nlp/ProLong/main/README.md（github, last push 2025-09-12）
[^28]: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context (v5, Gemini Team, Google) — https://arxiv.org/html/2403.05530v5（arxiv, 2024-03-08 (v1); v5 revised 2024-12-16）
[^29]: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model — https://arxiv.org/abs/2405.04434v5（arxiv, 2024-05-07）
[^30]: DeepSeek-V2 (HTML 全文) — Section 2.1 Multi-Head Latent Attention: Boosting Inference Efficiency — https://arxiv.org/html/2405.04434v5（arxiv, 2024-05-07）
[^31]: Lost in the Middle: How Language Models Use Long Contexts (arXiv:2307.03172v3) — https://arxiv.org/abs/2307.03172v3（arxiv, 2023-07-06）
[^32]: Long Context vs. RAG for LLMs: An Evaluation and Revisits (arXiv:2501.01880v1) — https://arxiv.org/abs/2501.01880v1（arxiv, 2024-12-27）
[^33]: Long Context vs. RAG for LLMs: An Evaluation and Revisits — Table 1 汇总的既有对比研究结论 — https://arxiv.org/html/2501.01880v1（arxiv, 2024-12-27）
[^34]: Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach (Self-Route) — https://arxiv.org/abs/2407.16833（arxiv, 2024-07-23 (v1); v2 2024-10-17）
[^35]: Introducing Contextual Retrieval (Anthropic) — https://www.anthropic.com/news/contextual-retrieval（web）
[^36]: Longformer: The Long-Document Transformer (arXiv:2004.05150v2) — https://ar5iv.labs.arxiv.org/html/2004.05150（arxiv, 2020-04-10）
[^37]: Big Bird: Transformers for Longer Sequences (arXiv:2007.14062v2, NeurIPS 2020) — https://ar5iv.labs.arxiv.org/html/2007.14062（arxiv, 2020-07-28）
[^38]: Mistral 7B (arXiv:2310.06825) — https://ar5iv.labs.arxiv.org/html/2310.06825（arxiv, 2023-10-10）