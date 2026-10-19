# 百万 token 上下文的主流技术路径，以及长上下文对 RAG 的冲击边界

## 摘要

把上下文窗口推到 100 万 token 级别，目前不存在单一"银弹"，而是四类技术叠加的结果：**位置编码外推/插值**（PI、NTK 系列、YaRN、LongRoPE）、**注意力与 KV cache 的结构性压缩**（MLA、GQA、StreamingLLM 的 attention sink）、**IO 感知的注意力算子与解码并行**（FlashAttention 1/2/3、Flash-Decoding）、以及**推理侧的 KV cache 复用与稀疏化**（prefix caching、RadixAttention、稀疏注意力 + chunked prefill）。位置编码决定"能不能把位置索引喂进去"，注意力算子与 KV 压缩决定"喂进去之后显存和算力扛不扛得住"，推理框架决定"跑起来贵不贵"——三者缺一，1M 窗口就只是纸面参数。

长上下文并未让 RAG 过时，但确实改变了 RAG 的分工。系统性对比显示，资源充足时 long-context 在平均性能上一致优于 RAG[^44]，但 RAG 的成本优势依然明确[^44]；LongRAG 这类工作则直接把长上下文 LLM 当作 reader，把检索单元从 100 词段落放大到 4K token，主张的是**融合**而非替代[^15]。真正被削弱的是"短单元 + 重检索器 + 轻阅读器"这一特定架构，而不是检索本身。

需要提前声明：本报告证据链在若干关键点上存在硬缺口——Gemini 1.5 的具体窗口数值、Llama 3 的长上下文配置、OpenAI 全系模型的上下文数据、以及任何 1M 级模型的逐基准实测分数，均未在本次取证中读到，下文相应位置会显式标注。

---

## 一、位置编码层：把"没见过的位置"变成"见过的位置"

### 1.1 问题的根源

Transformer 的自注意力让序列中任意两位置直接交互，这是长上下文复杂度的理论根源[^1]。位置信息如何注入，决定了模型能否处理超出预训练长度的序列。RoPE 用旋转矩阵编码绝对位置，同时在自注意力公式中融入显式相对位置依赖，并具备序列长度灵活性[^2]，因此成为后续几乎所有长度外推方法的基础[^2]。

### 1.2 Position Interpolation：插值而非外推

Position Interpolation (PI) 的核心操作是**线性下缩放输入位置索引**，使其匹配原始上下文窗口，而不是外推到训练长度之外[^3][^36]。论文明确指出外推会导致灾难性高注意力分数、破坏自注意力机制[^3]；理论分析表明插值的上界至少比外推小约 600 倍，这是其稳定性的依据[^3][^36]。

工程代价极低：PI 可将 RoPE 预训练 LLM（如 LLaMA）的窗口扩展至最多 32768，仅需 1000 步以内的微调[^3][^36]，且扩展后模型保留原架构，可复用大部分既有优化与基础设施[^3][^36]。LongRoPE 论文把 PI 描述为"按扩展比例线性插值 RoPE 的旋转角度"，以将新位置索引缩放到预训练范围内[^28]。

### 1.3 NTK 系列与 YaRN

YaRN 论文完整梳理了三种此前未正式发表的 NTK 类插值方法[^26]：

- **NTK-aware**（bloc97, 2023b）：核心思想是"考虑高频信息的损失"[^26]；
- **Dynamic NTK**（emozilla, 2023）：面向无需微调的预训练模型[^26]；
- **NTK-by-parts**（bloc97, 2023a）：在少量长上下文数据微调时表现最好[^26]。

其中 NTK-aware 已被 Code Llama 采用，Dynamic NTK 已被 Qwen 7B 采用[^26]。YaRN 本身是计算高效的 RoPE 窗口扩展方法，摘要声称相比此前方法需要 10 倍更少的 token 和 2.5 倍更少的训练步数，使 LLaMA 能有效利用并外推到远超原始预训练长度的上下文，并展示了外推到超出微调数据集有限上下文之外的能力[^25]。作者来自 Nous Research、EleutherAI、University of Geneva，代码已开源[^25]。

### 1.4 LongRoPE：2M token 与"非均匀性"

LongRoPE 首次将预训练 LLM 的上下文窗口扩展到 2048k（2M）token[^27]。其关键洞察是位置插值中存在**两种非均匀性**，通过高效搜索识别并利用它们，可为微调提供更好的初始化，并在非微调场景下实现 8 倍扩展[^27]。

训练策略是**渐进式扩展**：先微调一个 256k 长度的 LLM，再对已扩展模型做第二次位置插值以达到 2048k[^27]。在 256k 训练长度内，仅需最多 1k 微调步数[^27]。此外还需在 8k 长度上重新调整以恢复原始短上下文性能[^27]。实验覆盖 LLaMA2 与 Mistral，扩展后模型保留原始架构，仅对位置嵌入做微小修改[^27]。

LongRoPE 论文明确列出扩展窗口的三大障碍：未训练的新位置索引引入灾难性数值、长文本稀缺且训练昂贵、注意力分散导致原始短上下文性能下降[^28]。从 4k 扩展到 >1000k 会引入超过 90% 的新位置，这是微调难以收敛的核心挑战之一[^28]。论文同时指出，当时近期工作可将预训练 LLM 窗口通过长文本微调扩展到约 128k[^28]。

### 1.5 工业界的落地形态

Qwen2.5-1M 的训练采用渐进式扩展：预训练阶段上下文从 4K 逐步增至 256K，RoPE base 从 10,000 提升到 10,000,000（Adjusted Base Frequency）[^42]。其开源推理框架包含长度外推方法，可在无需额外训练的情况下将上下文长度扩展至少 4 倍甚至更多[^12]。

---

## 二、注意力算子与解码并行：让长序列在硬件上跑得动

### 2.1 FlashAttention 系列

FlashAttention 是 IO 感知的精确注意力算法，用 tiling 减少 GPU HBM 与片上 SRAM 之间的内存读写次数[^29]。论文分析了其 IO 复杂度，证明 HBM 访问次数少于标准注意力，并在一定 SRAM 尺寸范围内达到最优[^29]。实测端到端加速：BERT-large（seq 512）相比 MLPerf 1.1 训练速度记录有 15% 墙钟加速，GPT-2（seq 1K）3× 加速，long-range arena（seq 1K–4K）2.4× 加速[^29]。在长序列能力上，Path-X（seq 16K）达 61.4% 准确率、Path-256（seq 64K）达 63.1%，为首次优于随机水平的 Transformer[^29]。

FlashAttention-2 指出前代仅达到理论最大 FLOPs/s 的 25–40%，瓶颈在于线程块与 warp 之间的工作划分不佳[^30]。三项改进为：调整算法减少非 matmul FLOPs；即使对单个 head 也跨线程块并行化注意力计算以提高占用率；在线程块内将工作分配到各 warp 以减少共享内存通信[^30]。结果相比 FlashAttention 约 2× 加速，在 A100 上达到理论最大 FLOPs/s 的 50–73%[^30]；端到端训练 GPT 类模型时单张 A100 最高 225 TFLOPs/s（72% 模型 FLOPs 利用率）[^30]。

FlashAttention-3 面向 H100：前代在 H100 上仅达到 35% 利用率[^31]。三项技术为：利用 Tensor Core 与 TMA 的异步性，通过 warp-specialization 重叠计算与数据搬运；交错执行分块 matmul 与 softmax；块量化与 incoherent processing 以利用 FP8 低精度硬件支持[^31]。在 H100 上实现 1.5–2.0× 加速，FP16 达最高 740 TFLOPs/s（75% 利用率），FP8 接近 1.2 PFLOPs/s[^31]；FP8 版本相比基线 FP8 注意力实现 2.6× 更低的数值误差[^31]。

**证据边界**：FlashAttention 1/2/3 三篇摘要中报告的最长序列分别为 64K、未给出、未给出，**均未涉及 1M token 的实测数据**[^29][^30][^31]。官方实现仓库 README 摘录仅覆盖到介绍与 Usage 开头，未读取到 benchmark 章节中的 1M token 实测数字[^33]。该仓库 25,137 stars、3,157 forks，最近 push 为 2026-10-14，BSD-3-Clause 许可[^33]。

### 2.2 Flash-Decoding：解码阶段的并行维度

长上下文推理的瓶颈在解码阶段与训练阶段不同。Flash-Decoding 基于 FlashAttention 新增一个并行维度——keys/values 的序列长度，先并行加载 KV，再分别 rescale 并合并结果以保持正确的注意力输出[^32]。

动机很具体：解码时 query length 通常为 1，FlashAttention 只在 batch size 与 query 块上并行；若 batch size 小于 GPU 的 SM 数量（A100 为 108），则只用到 GPU 的一小部分——batch size 为 1 时 FlashAttention 使用不到 1% 的 GPU[^32]。而长上下文场景下恰恰需要更小 batch size 才能装入显存，进一步加剧低占用率[^32]。该技术对非常长的序列带来最高 8× 更快的生成速度[^32]。

**证据边界**：该博客正文仅抓取到前半部分，未获取到具体序列长度与显存节省的量化数字[^32]。

---

## 三、KV cache 与注意力结构：1M 窗口真正的成本中心

### 3.1 MLA：把 KV cache 压成 latent vector

DeepSeek-V2 采用 Multi-head Latent Attention (MLA)，通过将 Key-Value (KV) cache 压缩为 latent vector 来实现高效推理[^8]。效果量级：相比 DeepSeek 67B，DeepSeek-V2 将 KV cache 减少 93.3%，最大生成吞吐提升至 5.76 倍，训练成本节省 42.5%[^8]。模型总参数 236B，每 token 激活 21B 参数，预训练语料 8.1T tokens，支持 128K token 上下文[^8]。

### 3.2 GQA / MQA：减少 KV head 数量

MQA (multi-query attention) 只使用单个 key-value head，可大幅加速 decoder 推理，但可能导致质量下降，且为更快推理单独训练一个模型并不理想[^34]。GQA 论文提出将现有多头 checkpoint 通过 uptraining 转为 MQA 模型的方案，仅使用原预训练算力的 5%[^34]；并进一步提出 grouped-query attention (GQA)，是 multi-query attention 的泛化，使用介于 1 与 query head 数之间的中间数量 key-value heads[^34]。论文声称 uptrained GQA 达到接近 multi-head attention 的质量，同时具备与 MQA 相当的速度[^34]。

### 3.3 StreamingLLM 与 attention sink

StreamingLLM 观察到 attention sink 现象：即使初始 token 语义上不重要，注意力分数仍强烈指向它们；保留初始 token 的 KV 可大幅恢复 window attention 的性能[^9]。基于此提出的框架，使以有限长度注意力窗口训练的 LLM 无需微调即可泛化到无限序列长度[^9]，可让 Llama-2、MPT、Falcon、Pythia 在高达 400 万 token 及以上长度下进行稳定高效的语言建模[^9]。在流式设置下，相比滑动窗口重计算基线最高实现 22.2 倍加速[^9]。论文还发现预训练时加入一个占位 token 作为专用 attention sink 可进一步改善流式部署[^9]。

### 3.4 稀疏注意力与 chunked prefill

Qwen2.5-1M 为降低推理成本，实现了稀疏注意力（sparse attention）与 chunked prefill 优化，以及提升精度的 sparsity refinement 方法[^12]。借助该推理框架，Qwen2.5-1M 在 100 万 token 上下文场景下实现 3x 到 7x 的 prefill 加速[^12]。

### 3.5 证据缺口

计划中列出的 **KV 量化**与 **token 驱逐类方法（H2O、SnapKV、Quest 等）**在本次 48 条证据中完全缺失。因此本报告无法就这两类方法的机制与效果做任何事实性陈述。同样，**1M 上下文场景下的绝对 KV cache 显存占用（GB 级）与每百万 token 成本量级**在证据中不存在——仅有 DeepSeek-V2 的 93.3% 压缩比例[^8]与 Qwen 的 3–7x prefill 加速[^12]这两个相对数字。

---

## 四、推理框架层：KV cache 复用与上下文工程

### 4.1 Prefix caching

vLLM 的 automatic prefix caching (APC) 采用 hash-based 方案：对每个 KV-cache block 用「块内 token + 块前 prefix token」共同计算哈希来唯一标识[^45]。哈希组件包含三部分：parent hash value（父块哈希）、block tokens（本块 token 元组，用于降低哈希碰撞）、extra hashes（LoRA ID、多模态输入哈希、多租户隔离用的 cache salt）[^45]。vLLM 只缓存完整的 block[^45]。自 v0.11 起默认哈希算法为 sha256，以解决此前版本哈希键不保证无碰撞的问题，可通过 `--prefix-caching-hash-algo` 控制[^45]。文档将 prefix caching 描述为"almost a free lunch"且不改变模型输出，并称其已被 OpenAI、Anthropic 等公开端点及多数开源推理框架（如 SGLang）广泛采用[^45]。

### 4.2 RadixAttention

SGLang 运行时通过 RadixAttention 实现 KV cache 复用，是其核心优化之一[^46]。论文摘要称 SGLang 在多种大语言与多模态模型、多类任务上相比当时 SOTA 推理系统实现最高 6.4x 吞吐提升[^46]。受益任务明确包含 retrieval-augmented generation (RAG) pipelines、agent control、logical reasoning、few-shot learning、JSON decoding、multi-turn chat[^46]。另一核心优化为 compressed finite state machines，用于加速结构化输出解码[^46]。

两个框架的体量：vLLM 93,854 stars、23,416 forks，Apache-2.0 许可，最近 push 2026-10-15[^47]；SGLang 37,173 stars、9,585 forks，Apache-2.0 许可，最近 push 2026-10-15[^48]。

### 4.3 上下文工程作为独立工具类别

GitHub 上已出现围绕 prompt caching 的独立工具：ostk-cache 是面向 Anthropic 与 OpenAI 的上下文感知缓存代理，目标是通过把长生命周期上下文锚定到稳定边界来最大化 prompt-cache 命中率，自述可作为 Claude Code、Codex、Cursor 及自定义 harness 的 drop-in 组件[^23]。该仓库 stars 10、forks 1，Rust 语言，最后 push 为 2026-06-12[^23]——体量很小，但其存在表明"围绕 prompt caching 做工程优化"已形成独立工具类别[^23]。

更宏观地，context-engineering-tools 收录 160+ 工具，覆盖 LLM 记忆、提示、上下文管理、压缩、缓存、MCP、检索与评测，并将"缓存""检索""上下文压缩"并列为上下文工程的并列子领域[^24]。

另一个工程信号是"1M 上下文 + 高 prompt cache 命中率"被作为组合卖点出现：Whale 是面向 DeepSeek 的终端优先 AI 编码 agent，自述约 98% prompt cache 命中率、1M 上下文、支持 MCP 工具[^22]。

---

## 五、1M 级模型的实际供给

### 5.1 有明确证据的模型

| 模型 | 上下文窗口 | 关键机制 | 来源 |
|---|---|---|---|
| Qwen2.5-7B/14B-Instruct-1M | 1M token | 渐进式预训练（4K→256K）、RoPE base 10,000→10,000,000、稀疏注意力 + chunked prefill | [^12][^42] |
| Qwen2.5-Turbo | 1M token（API） | 同上 | [^12] |
| MiniMax-Text-01 | 训练 1M，推理可外推至 4M | lightning attention + MoE（32 专家、4560 亿总参数、每 token 激活 459 亿） | [^13] |
| Gemini 2.0 Flash | 100 万 token | — | [^41] |
| Gemini 2.0 Pro（实验版） | 200 万 token | Google 迄今最大上下文窗口 | [^41] |
| Claude（部分模型） | "Up to 1M varies by model" | — | [^43] |

Qwen2.5-1M 系列包含开源模型 Qwen2.5-7B-Instruct-1M 与 Qwen2.5-14B-Instruct-1M，以及 API 访问的 Qwen2.5-Turbo[^12]。评测显示 Qwen2.5-14B-Instruct-1M 在长上下文任务上显著优于 GPT-4o-mini，并支持 8 倍长的上下文[^12]；官方结论称该系列在多数长上下文任务上显著优于其 128K 版本，尤其在超过 64K 的序列上[^42]。

MiniMax-01 系列包含 MiniMax-Text-01 与 MiniMax-VL-01，核心是 lightning attention 及其高效扩展，并与 MoE 结合[^13]。MiniMax-VL-01 通过 5120 亿视觉-语言 token 的继续训练构建[^13]。实验显示模型性能匹配 GPT-4o 与 Claude-3.5-Sonnet 等 SOTA 模型，同时提供 20–32 倍更长的上下文窗口[^13]。MiniMax-01 已公开发布[^13]。

### 5.2 证据不足或存在矛盾的部分

**Gemini 1.5**：技术报告标题即宣称可在数百万 token 上下文上实现多模态理解[^6][^11]，由 Gemini Team Google 发布，v1 提交于 2024-03-08，最新版本 v5 修订于 2024-12-16[^11]。但本次取证**仅取到 arXiv 摘要页元数据，正文中的具体上下文窗口数值与 NIAH 基准分数未读到**[^6]，因此无法支撑任何具体参数主张。

**Claude 系列存在直接矛盾**：Anthropic 定价页在 'Models and usage' 表中标注各计划 'Context window: Up to 1M varies by model'[^43]；但 Claude 3.5 Sonnet 官方公告标 200K token[^38]，Claude 3.7 Sonnet 官方公告未给出窗口数值但给出输出上限 128K token[^39]，Claude 4 公告给出的是定价而非窗口[^40]。两来源直接冲突且未解决，也未说明 1M 是否需 beta header 或仅限特定模型。

**Llama 3**：官方技术报告涵盖模型族训练与长上下文能力（含长上下文预训练阶段）[^7]，但同样**仅取到摘要页元数据，长上下文阶段的具体配置与基准分数未读到**[^7]。**Llama 4 完全无证据**。

**OpenAI 全系**：48 条证据中**无任何 OpenAI 官方来源**，GPT-4.1 / GPT-4o 的上下文长度与长上下文实测表现完全空白。唯一相关的是第三方论文中的对比结论——Qwen2.5-14B-Instruct-1M 显著优于 GPT-4o-mini[^12]，以及 MiniMax-01 性能匹配 GPT-4o[^13]。

**Anthropic 定价细节**：Claude 3.5 Sonnet 定价为每百万输入 token $3、每百万输出 token $15[^38]；Claude 3.7 Sonnet 在标准模式与扩展思考模式下定价相同（含思考 token）[^39]；Claude Opus 4 为 $15 输入 / $75 输出，Claude Sonnet 4 为 $3 输入 / $15 输出[^40]。但**无 Gemini / GPT / Qwen 定价，也无任何 prompt caching 折扣（如 Anthropic cache write/read 折扣、OpenAI cached input 折扣）的官方数据**。

---

## 六、名义窗口 ≠ 有效窗口

这是长上下文领域最重要的实证结论，也是判断"1M 是否真的可用"的前提。

**Lost in the Middle** 指出：语言模型虽能接收长上下文输入，但对其利用能力有限[^4]。在多文档问答与键值检索任务中，改变相关信息位置会显著影响性能，说明模型不能稳健利用长输入上下文[^4]。性能通常在相关信息位于输入开头或结尾时最高，当相关信息位于长上下文中部时显著下降，即使对显式长上下文模型也如此[^4]。该现象即"lost in the middle"，是有效上下文长度低于名义上下文窗口的经典证据[^4]。

**RULER** 进一步量化了这一点。NIAH（needle-in-a-haystack）测试被广泛用于评测长上下文 LM，但仅反映表层检索能力[^5]。RULER 是可控序列长度与任务复杂度的合成基准，扩展 NIAH 并新增 multi-hop tracing 与 aggregation 任务类别[^5][^10]。评测 17 个长上下文 LM、13 类任务的结果是：几乎所有模型在 vanilla NIAH 上近乎满分，但随上下文长度增加出现大幅性能下降[^5][^10]。这些模型均声称支持 32K token 或更大上下文，但**仅一半能在 32K 长度维持满意性能**[^5][^10]。Yi-34B 声称支持 200K 上下文，随输入长度与任务复杂度增加仍有很大改进空间[^5][^10]。RULER 代码已开源（COLM 2024）[^10]。

**训练侧的方法论回应**：ProLong 论文明确主张不使用困惑度或简单 NIAH 测试，而使用广泛的长上下文下游任务，并在 SFT 之后评测模型，因为这样更能揭示长上下文能力[^35]。其数据配比结论是：代码仓库和书籍是优秀的长数据来源，但关键是要将它们与高质量短上下文数据结合[^35]；训练策略结论是使用超过评测长度的序列长度进行训练可提升长上下文性能[^35]；SFT 结论是仅使用短指令数据集即可在长上下文任务上取得强性能[^35]。最终模型 ProLong-8B 从 Llama-3 初始化，在 40B tokens 上训练，在 128K 长度下于同规模模型中达到 SOTA 长上下文性能，在多数长上下文任务上超过 Llama-3.1-8B-Instruct，尽管其长上下文训练所用 token 数仅为后者的 5%[^35]。ProLong 可有效处理最多 512K tokens[^35]。论文被 ACL 2025 接收，代码、数据与模型已开源[^35]。

**证据边界**：本次取证**缺少逐模型、逐基准的分数对照**。仅有 RULER 基准论文本身的统计结论[^5][^10]与 Qwen 官方博客的定性结论[^42]，无法支撑"某模型在 1M 长度下 RULER 得分 X"这类具体主张。

---

## 七、长上下文是否让 RAG 过时？

### 7.1 系统性对比的结论

《Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach》对 RAG 与 long-context (LC) LLM 在多个公开数据集上、使用三个最新 LLM 进行了性能与效率两个维度的系统性对比[^44]。核心结论：

- **资源充足时，LC 在平均性能上一致地优于 RAG**[^44]；
- **RAG 的显著更低成本仍是其明确优势**（LLM API 按输入 token 计费，RAG 大幅缩短输入长度）[^44]；
- LC 与 RAG 的预测在**超过 60% 的查询上完全相同**[^44]；
- 该文与先前工作 Xu et al. (2023) 的发现不同，强调近期 LLM 在长上下文理解上的进步[^44]。

论文提出的 Self-Route 方法基于模型自我反思将查询路由到 RAG 或 LC，在保持与 LC 相当性能的同时显著降低计算成本[^44]。该文被 EMNLP 2024 industry track 接收，作者来自 Google DeepMind 与 University of Michigan[^44]。

### 7.2 融合而非替代：LongRAG

LongRAG 的诊断是：传统 RAG 中短检索单元（如 100 词 Wikipedia 段落）造成"重检索器、轻阅读器"的不平衡，导致次优性能[^15]。其框架由"long retriever"与"long reader"组成[^15]。

具体做法与结果：在 NQ 与 HotpotQA 两个 Wikipedia 数据集上，LongRAG 将整个语料处理为 4K-token 单元，大幅减少单元总数，仅需少于 8 个 top 单元即可获得强检索性能[^15]。无需任何训练，在 NQ 上达到 EM 62.7%、HotpotQA 上 EM 64.3%，与（完全训练的）SoTA 模型相当[^15]。在非 Wikipedia 数据集 Qasper 与 MultiFieldQA-en 上，将每篇文档作为单个长单元处理，取得 Qasper F1 25.9%、MultiFieldQA-en F1 57.5%[^15]。

论文的自我定位是关键：**"结合 RAG 与长上下文 LLM 的未来路线图"**，即长上下文与 RAG 融合而非替代[^15]。该论文为 Technical Report，v1 提交于 2024-06-21，v3 修订于 2024-09-01[^15]。

### 7.3 工程侧的信号

LongRAG 官方仓库对应论文 arXiv:2406.15319，MIT 许可，Python 语言，stars 249、forks 18、open issues 4[^16]。但创建于 2024-06-18，最后 push 为 2024-08-25，此后无更新（截至检索时已停滞约两年），README 自述"We are still in the process to polish our repo"[^16]。**论文层面的影响力与工程层面的维护状态出现明显背离**。

与之对照，memvid 定位为"AI agent 的单文件记忆层"，主张用 serverless、单文件记忆层替代复杂 RAG 流水线，为 agent 提供即时检索与长期记忆[^17]。仓库 stars 16,599、forks 1,432、open issues 34，Rust 语言，Apache-2.0 许可，创建于 2025-05-27，最后 push 为 2026-07-14，属活跃维护状态[^17]。README 强调"Persistent, versioned, and portable memory, without databases"[^17]，topics 包含 rag、retrieval-augmented-generation、vector-database、semantic-search、knowledge-graph 等[^17]。**注意其 topics 仍包含 rag 与 vector-database**——即便主张"替代复杂 RAG 流水线"，其定位坐标依然在检索生态内。

另一个信号是 memoket-kite，自述为面向 AI agent 的开源、无向量（vector-free）长期记忆引擎，声称在 LoCoMo 与 LongMemEval 上以显著更少的上下文达到 SOTA[^21]。stars 144、forks 12，最后 push 为 2026-10-12[^21]。**"SOTA"为仓库自述营销性主张，未在本次取证中找到独立权威来源验证**[^21]。

学术社区的整理工作也显示两个方向被并列而非对立：Awesome-Agent-Memory 汇集 LLM/MLLM 记忆相关的系统、基准与论文，覆盖 long-term context、retrieval 与 reasoning，其主题将"长期上下文"与"检索"并列为同一记忆体系下的组成部分[^19]。Awesome-LLM-Long-Context-Modeling 作为长上下文建模的论文与博客精选列表，stars 2,183、forks 106，最后 push 为 2026-08-17，说明该方向持续有文献产出与社区维护[^18]。

### 7.4 本节结论

综合来看，**长上下文削弱的是"短检索单元 + 重检索器 + 轻阅读器"这一特定 RAG 架构，而不是检索本身**。证据支持三点：

1. 资源充足时 LC 平均性能优于 RAG，但 60% 以上查询两者预测完全相同[^44]——意味着性能差异集中在少数查询上，而成本差异是全局的；
2. RAG 的成本优势在按输入 token 计费的 API 模式下是结构性的[^44]；
3. 长上下文与检索的融合路线（LongRAG 的 4K 长单元、Self-Route 的路由）在性能与成本之间取得了比任一单独方案更好的平衡[^15][^44]。

**证据边界**：本次取证**无任何向量数据库厂商（Pinecone/Weaviate/Qdrant 等）或重排器（Cohere Rerank 等）的官方动态、营收或采用数据**，因此无法就"企业检索栈是否被长上下文替代"这一产业侧问题给出有证据支撑的判断。

---

## 八、矛盾与不确定性

### 8.1 已识别的直接矛盾

**Claude 上下文窗口**：Anthropic 定价页标注 "Context window: Up to 1M varies by model"[^43]，而 Claude 3.5 Sonnet 官方公告标 200K token[^38]。两来源直接冲突，未解决。定价页也未说明 1M 是否需 beta header 或仅限特定模型[^43]。

**LongRAG 的影响力 vs 维护状态**：论文提出的是被广泛引用的融合路线图[^15]，但官方仓库自 2024-08-25 后无更新，README 自述仍在打磨中[^16]。论文结论的可靠性与其参考实现的可用性之间存在落差。

### 8.2 证据链硬缺口（影响结论强度）

1. **Gemini 1.5 的 1M 上下文**：仅取到 arXiv 摘要页元数据，正文中的具体上下文窗口数值与 NIAH 基准分数未读到[^6]。本报告中 Gemini 1.5 的所有主张仅限于"标题宣称数百万 token"这一层面[^6][^11]。

2. **Llama 3 / Llama 4**：Llama 3 报告仅取到摘要元数据，长上下文阶段配置与基准分数缺失[^7]；Llama 4 完全无证据。

3. **OpenAI 全系空白**：GPT-4.1 / GPT-4o 的上下文长度与长上下文实测表现无任何官方来源。仅有第三方论文的对比性陈述[^12][^13]。

4. **逐基准实测分数缺失**：无任何 1M 级模型在 RULER / NIAH / LongBench / ∞Bench 上的逐项分数。仅有 RULER 基准论文的统计结论（17 个模型中仅一半能在 32K 维持满意性能）[^5][^10]与 Qwen 官方博客的定性结论[^42]。

5. **KV cache 绝对开销缺失**：仅有 DeepSeek-V2 的 93.3% 相对压缩比例[^8]与 Qwen 的 3–7x prefill 加速[^12]，缺少 1M 场景下 GB 级显存占用与每百万 token 成本量级。

6. **KV 量化与 token 驱逐方法完全缺失**：H2O、SnapKV、Quest 等在本报告证据集中不存在，因此第三节无法覆盖这两类主流技术。

7. **产业侧检索栈数据缺失**：无向量数据库厂商或重排器的官方动态、营收或采用数据，第七节的产业判断因此受限。

8. **定价数据不完整**：仅有 Claude 3.5/3.7/4 的输入输出单价[^38][^39][^40]，无 Gemini / GPT / Qwen 定价，无任何 prompt caching 折扣的官方数据。

### 8.3 需要谨慎对待的营销性主张

- memoket-kite 的 "SOTA" 自述未获独立验证[^21]；
- memvid 的"替代复杂 RAG 流水线"是仓库定位主张，其 topics 仍包含 rag 与 vector-database[^17]；
- Whale 的"约 98% prompt cache 命中率"为仓库自述，无第三方验证[^22]；
- FlashAttention 系列论文摘要中的加速数字均为特定硬件与特定序列长度下的实测值，**不可外推至 1M token 场景**[^29][^30][^31]。

## 参考来源

[^1]: Attention Is All You Need (Transformer 原论文) — https://arxiv.org/abs/1706.03762v7（arxiv, 2017-06-12）
[^2]: RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE 原论文) — https://arxiv.org/abs/2104.09864v5（arxiv, 2021-04-20）
[^3]: Extending Context Window of Large Language Models via Positional Interpolation — https://arxiv.org/abs/2306.15595v2（arxiv, 2023-06-27）
[^4]: Lost in the Middle: How Language Models Use Long Contexts — https://arxiv.org/abs/2307.03172v3（arxiv, 2023-07-06）
[^5]: RULER: What's the Real Context Size of Your Long-Context Language Models? — https://arxiv.org/abs/2404.06654v3（arxiv, 2024-04-09）
[^6]: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context — https://arxiv.org/abs/2403.05530v5（arxiv, 2024-03-08）
[^7]: The Llama 3 Herd of Models — https://arxiv.org/abs/2407.21783v3（arxiv, 2024-07-31）
[^8]: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model — https://arxiv.org/abs/2405.04434v5（arxiv, 2024-05-07）
[^9]: Efficient Streaming Language Models with Attention Sinks (StreamingLLM) — https://arxiv.org/abs/2309.17453v4（arxiv, 2023-09-29）
[^10]: RULER: What's the Real Context Size of Your Long-Context Language Models? — https://arxiv.org/abs/2404.06654（arxiv, 2024-04-09）
[^11]: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context — https://arxiv.org/abs/2403.05530（arxiv, 2024-03-08）
[^12]: Qwen2.5-1M Technical Report — https://arxiv.org/abs/2501.15383（arxiv, 2025-01-26）
[^13]: MiniMax-01: Scaling Foundation Models with Lightning Attention — https://arxiv.org/abs/2501.08313（arxiv, 2025-01-14）
[^15]: LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs — https://arxiv.org/abs/2406.15319（arxiv, 2024-06-21）
[^16]: TIGER-AI-Lab/LongRAG (official repo) — https://github.com/TIGER-AI-Lab/LongRAG（github, 2024-06-18）
[^17]: memvid/memvid — https://github.com/memvid/memvid（github, 2025-05-27）
[^18]: Xnhyacinth/Awesome-LLM-Long-Context-Modeling — https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling（github）
[^19]: TeleAI-UAGI/Awesome-Agent-Memory — https://github.com/TeleAI-UAGI/Awesome-Agent-Memory（github）
[^21]: memoket/memoket-kite — https://github.com/memoket/memoket-kite（github）
[^22]: usewhale/Whale — https://github.com/usewhale/Whale（github）
[^23]: os-tack/ostk-cache — https://github.com/os-tack/ostk-cache（github）
[^24]: aglio-lab/context-engineering-tools — https://github.com/aglio-lab/context-engineering-tools（github）
[^25]: YaRN: Efficient Context Window Extension of Large Language Models — https://arxiv.org/abs/2309.00071（arxiv, 2023-08-31）
[^26]: YaRN: Efficient Context Window Extension of Large Language Models (full text, intro) — https://arxiv.org/html/2309.00071v3（arxiv, 2023-08-31）
[^27]: LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens — https://arxiv.org/abs/2402.13753（arxiv, 2024-02-21）
[^28]: LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens (full text, intro) — https://arxiv.org/html/2402.13753v1（arxiv, 2024-02-21）
[^29]: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — https://arxiv.org/abs/2205.14135v2（arxiv, 2022-05-27）
[^30]: FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning — https://arxiv.org/abs/2307.08691v1（arxiv, 2023-07-17）
[^31]: FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision — https://arxiv.org/abs/2407.08608v2（arxiv, 2024-07-11）
[^32]: Flash-Decoding for long-context inference (PyTorch Blog) — https://pytorch.org/blog/flash-decoding/（web）
[^33]: Dao-AILab/flash-attention (official implementation repo) — https://github.com/Dao-AILab/flash-attention（github, 2022-05-19）
[^34]: GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints — https://arxiv.org/abs/2305.13245v3（arxiv, 2023-05-22）
[^35]: How to Train Long-Context Language Models (Effectively) (ProLong) — https://arxiv.org/abs/2410.02660（arxiv, 2024-10-03）
[^36]: Extending Context Window of Large Language Models via Positional Interpolation — https://arxiv.org/abs/2306.15595（arxiv, 2023-06-27）
[^38]: Claude 3.5 Sonnet (Anthropic official announcement) — https://www.anthropic.com/news/claude-3-5-sonnet（web, 2024-06-21）
[^39]: Claude 3.7 Sonnet and Claude Code (Anthropic official announcement) — https://www.anthropic.com/news/claude-3-7-sonnet（web, 2025-02-24）
[^40]: Introducing Claude 4 (Anthropic official announcement) — https://www.anthropic.com/news/claude-4（web, 2025-05-22）
[^41]: Gemini 2.0 is now available to everyone (Google official blog) — https://blog.google/technology/google-deepmind/gemini-model-updates-february-2025/（web, 2025-02-05）
[^42]: Qwen2.5-1M: Deploy Your 1M Context Length Solutions (Qwen official blog) — https://qwenlm.github.io/blog/qwen2.5-1m/（web, 2025-01-27）
[^43]: Anthropic Pricing (consumer plans page) — https://www.anthropic.com/pricing（web）
[^44]: Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach — https://arxiv.org/abs/2407.16833（arxiv, 2024-07-23）
[^45]: vLLM Design Document — Automatic Prefix Caching — https://docs.vllm.ai/en/latest/design/prefix_caching.html（web）
[^46]: SGLang: Efficient Execution of Structured Language Model Programs (arXiv:2312.07104v2) — https://arxiv.org/abs/2312.07104v2（arxiv, 2023-12-12）
[^47]: vllm-project/vllm (GitHub repository) — https://github.com/vllm-project/vllm（github, 2023-02-09）
[^48]: sgl-project/sglang (GitHub repository) — https://github.com/sgl-project/sglang（github, 2024-01-08）