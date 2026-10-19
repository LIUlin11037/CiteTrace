# 大模型 tokenizer：BPE 类方法原理，及其对模型能力与成本的影响

## 摘要

分词器（tokenizer）把原始文本映射为模型可处理的离散 token 序列，是 LLM 输入输出的第一道也是最末一道工序：Gemini 官方文档把 token 定义为「模型处理输入输出的粒度」，全部 token 的集合称为 vocabulary，切分过程称为 tokenization[^18]；DeepSeek 官方文档同样把 token 定义为「模型表示自然语言的基本单位，也是计费单位」[^20]。BPE（Byte Pair Encoding）类方法的基本原理是：先用预分词把文本切成词/片段并统计频次，以所有字符构成基础词表，然后反复合并频次最高的相邻符号对，直到达到目标词表规模，因此 **词表规模 = 基础词表规模 + 合并次数**（GPT 系列为 40,478 = 478 个基础 token + 40,000 次合并）[^4]。BPE 的原始动机是开放词表机器翻译：把罕见词与未登录词编码为子词单元序列，替代回退到词典的做法，在 WMT 15 上相对回退词典基线提升 1.1 BLEU（En-De）与 1.3 BLEU（En-Ru）[^1]。BPE 只产生唯一子词序列，而基于 unigram 语言模型的切分存在多种候选，Kudo 据此提出 subword regularization，在训练时按概率采样多种切分以提升鲁棒性[^11][^12]。

对成本的影响链条清晰且有官方定价佐证：计费按 token 数进行[^15][^18][^19]，而同一段文本在不同语言下的 token 长度差异最高可达 15 倍，字符级与字节级模型在某些语言对上仍有超过 4 倍的编码长度差异[^13]。Anthropic 明确记录了一次分词口径变更：Claude 4.7 及之后模型使用新 tokenizer，对同一文本产生约多 30% 的 token[^16]。对能力的影响方面，现有证据只能支持一个较窄的结论：词表规模本身是 scaling law 的一个变量，最优词表大小随算力预算增长，更大的模型需要更大的词表[^5]。至于字节回退、special token 设计、算术/拼写/代码等具体能力维度，现有公开资料不足以确认。

---

## 一、BPE 类方法的基本原理

### 1.1 BPE：从压缩算法到子词切分

BPE 用于分词的做法源自 Sennrich、Haddow、Birch 的论文，其目标是把罕见词和未登录词编码为子词单元序列，从而实现开放词表翻译，取代回退到词典的方案；切分基于 byte pair encoding 压缩算法，论文同时与简单字符 n-gram 模型做了对比[^1]。

Hugging Face 文档给出的 BPE 标准流程定义如下[^4]：

1. **预分词（pre-tokenization）**：把文本切分为「唯一词 + 频次」的形式；
2. **构建基础词表**：由所有出现的字符构成 base vocabulary；
3. **迭代合并**：反复合并频次最高的相邻符号对（most frequent adjacent pair），直到达到目标词表规模。

由此得到的关键恒等式是：**BPE 词表规模 = 基础词表规模 + 合并次数**，文档给出的实例是 GPT 使用 40,478 = 478 个基础 token + 40,000 次合并[^4]。文档同时指出，任何不在基础词表中的字符会映射为 `<unk>` 之类的未知 token，但实践中基础词表覆盖训练中见过的全部字符[^4]。

### 1.2 BPE 的性质：可逆、无损、可压缩

tiktoken 的 README 对 BPE 的性质给出了明确描述：BPE 是可逆/无损的，可作用于任意文本（包括未见过的文本），能压缩文本（平均每个 token 约 4 字节），并让模型看到常见子词（例如 `ing`）[^3]。tiktoken 自我定位为「用于 OpenAI 模型的高速 BPE 分词器」，提供 `o200k_base`、`cl100k_base` 等 encoding，以及 `encoding_for_model()` 按模型取分词器（如 gpt-4o）[^3]。README 还称 tiktoken 在 1GB 文本上以 GPT-2 分词器测量时，比同类开源分词器快 3–6 倍，并附带 `tiktoken._educational` 教育子模块，内含训练与可视化 BPE 过程的代码[^3]。

### 1.3 Unigram、subword regularization 与 BPE 的对照

Kudo（ACL 2018）提出基于 unigram 语言模型的子词切分算法，用于更好的子词采样，并提出 subword regularization：训练时按概率对多种子词切分进行采样，以提升 NMT 鲁棒性[^11]。该论文明确指出同一句子在同一词表下可存在多种子词切分（segmentation ambiguity），而 BPE 只产生唯一子词序列[^11]。

论文正文以 `Hello World` 为例，展示同一句子在同一词表下可对应 `_Hell/o/_world`、`_H/ello/_world` 等多种子词序列，说明切分存在虚假歧义（spurious ambiguity）；论文主张训练时使用多种切分候选可让模型对噪声与切分错误更鲁棒，并间接学习词的组合性（如 books = book + s）[^12]。论文把 Schuster and Nakajima (2012)、Chitnis & DeNero (2015)、Sennrich et al. (2016)、Wu et al. (2016) 并列为把罕见词拆分为子词单元的早期工作，并称 BPE（Sennrich et al. 2016）是事实标准的子词切分算法[^12]。

### 1.4 工程实现：SentencePiece 与 tokenizers

SentencePiece 被描述为一个快速、轻量的无监督文本分词/反分词器，用于神经文本生成，其词表规模在训练前即固定；它实现了包括 BPE（Sennrich et al.）与 unigram 语言模型（Kudo）在内的子词单元[^2]。其关键设计是直接从原始句子训练，把输入视为 Unicode 字符的原始序列，从而消除语言相关的前后处理；它用元符号 ▁（U+2581）转义空白并将其纳入分词，使反分词成为与语言无关的无损字符串拼接操作；它还支持 subword regularization 与 BPE-dropout[^2]。

huggingface/tokenizers 自我定位为「The SOTA tokenization library on all languages, all models, all hardwares」，目标是成为「the industry's standard tokenization engine」；当前处于 v1.0.0.rc.0 候选发布阶段，正从 0.23 迁移到 v1.0.0，迁移期库未包含全部功能；Rust 为核心参考实现，Python 与 Node.js 绑定已完成，C / C++ / Java / Go 等绑定标注为 planned[^6]。v1 路线图包含恢复训练能力、批量编码零拷贝视图、dlpack 互操作、offset 输出、decode_stream、窗口/stride 编码等；Rust 核心待办包含 bitnorm 归一化优化、训练/解码性能改进，以及修复 ModernBERT / OLMo 加载失败（64-bit hash collision）；1.0.0 之后计划重写 bitcannon crate、增加 Java/Go/Swift 绑定，以及探索性的 tk-devices（在 GPU 上做编码与批量解码，词表上传一次、并行计算输出位置）[^6]。README 示例显示可直接 `from_pretrained("meta-llama/Llama-3.1-8B")` 加载 Llama 3.1 的 tokenizer[^6]。

### 1.5 具体 encoding 的配置细节

tiktoken 的 encoding 定义文件显示：`cl100k_base` 通过 `load_tiktoken_bpe` 从 `cl100k_base.tiktoken` 加载 mergeable_ranks，未设置 `explicit_n_vocab` 字段；其 special_tokens 为 `{ENDOFTEXT: 100257, FIM_PREFIX: 100258, FIM_MIDDLE: 100259, FIM_SUFFIX: 100260, ENDOFPROMPT: 100276}`；其正则 `pat_str` 为 `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s'`[^21]。`o200k_base` 同样由 `load_tiktoken_bpe` 加载 mergeable_ranks 且未设 `explicit_n_vocab`，special_tokens 为 `{ENDOFTEXT: 199999, ENDOFPROMPT: 200018}`，其分词正则更复杂，按 Unicode 大小写/标记类别（`\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}` 等）区分，并含 `\p{N}{1,3}` 数字切分[^21]。作为对照，`gpt2()` 与 `r50k_base()` 均设置 `explicit_n_vocab: 50257`，`p50k_base()` 设置 `explicit_n_vocab: 50281`[^21]。源码注释还指出 o200k_base 的 token 分配在跨语言上仍可更高效（"I think you can allocate tokens more efficiently across languages"）[^21]。

---

## 二、tokenizer 如何影响成本

### 2.1 计费口径：token 数即账单

OpenAI Cookbook 明确 OpenAI 使用开源 tokenizer tiktoken，不同模型使用不同 encoding，并给出对应关系：`o200k_base` → gpt-4o、gpt-4o-mini；`cl100k_base` → gpt-4-turbo、gpt-4、gpt-3.5-turbo、text-embedding-ada-002/3-small/3-large；`p50k_base` → Codex 系列；`r50k_base`(gpt2) → GPT-3 系列；官方明确 API 调用成本由 token 数决定（usage is priced by token）[^15]。该页面为 2022-12-16 归档内容，可能引用过时模型或 API[^15]。

Gemini 官方文档给出换算直觉：1 token ≈ 4 个字符，100 tokens ≈ 60–80 个英文单词；所有输入与输出（含文本、图像文件及其他非文本模态）均被 tokenize；启用计费后，调用成本部分由输入与输出 token 数决定[^18]。DeepSeek 官方文档给出中英文换算比：1 个英文字符 ≈ 0.3 token，1 个中文字符 ≈ 0.6 token，并说明不同模型 tokenization 方法不同、换算比会变化，实际 token 数以模型返回的 usage 为准；官方提供离线 tokenizer 工具包（`deepseek_tokenizer.zip`）用于计算输入/输出 token 用量[^20]。DeepSeek 还给出计费公式：`expense = number of tokens × price`，优先扣减 granted balance[^19]。

### 2.2 价格结构：输入、输出、缓存与档位

三家主要厂商的定价结构高度一致地围绕 token 数展开，但倍数关系不同。

**OpenAI**：定价单位为每 1M tokens，区分 Input / Cached input / Cache writes / Output 四类计费项；Standard 档短上下文下，gpt-6-astra 为 Input $10.00 / Cached input $1.00 / Cache writes $12.50 / Output $50.00，gpt-6.1-sol 为 $2.00 / $0.10 / $2.50 / $10.00，gpt-6-luna 为 $0.10 / $0.01 / $0.125 / $0.50[^14]。缓存折扣约为标准 input 的 1/10，output 约为 input 的 5 倍；长上下文档 input/output 翻倍（astra 长上下文 Input $20.00 / Output $75.00）；存在 Batch / Flex / Fast / Ultrafast 多档服务层级，Batch 与 Flex 约为 Standard 的 5 折，Fast 为 2 倍，Ultrafast 为 6 倍；区域处理（data residency）端点与 FedRAMP 端点对 2026-03-05 及之后发布的模型加收 10% uplift；Priority processing 于 2026-07-30 更名为 Fast mode；GPT-5.6 Sol 促销价至少持续至 2026-11-21[^14]。

**Anthropic**：定价单位为 $/MTok，区分 Input / Output / 5m cache writes / 1h cache writes / Hits and refreshes 五类；Claude Fable 5.1 为 Input $10 / Output $50 / 5m writes $12.50 / 1h writes $20 / Hits $0.25；Opus 5.5 为 $4 / $20 / $5 / $8 / $0.20；Sonnet 5.5 为 $2 / $10 / $2.50 / $4 / $0.10；Haiku 5.5 在 ≤100k tokens 时为 Input $0.10、Output $0.50，>100k tokens 时为 Input $0.50、Output $2.50，5m writes $0.125、1h writes $0.20、Hits $0.01[^16]。cache hits 约为标准 input 的 1/10，缓存写入高于标准 input（5m writes ≈ 1.25×，1h writes ≈ 2×），output 为 input 的 5 倍；区域/多区域端点（Bedrock、Google Cloud）相对 global 端点有 10% 溢价[^16]。

**Gemini**：定价单位为 per 1M tokens in USD，区分 Input / Output（含 thinking tokens）/ Context caching；Gemini 4 Argon（Standard 档）Input $2.00（至 2026-11-30），2026-12-01 起 $4.00；Output（含 thinking tokens）$10.00（至 2026-11-30），2026-12-01 起 $20.00；Context caching $0.10（至 2026-11-30），2026-12-01 起 $0.20，且仅支持隐式缓存（implicit caching），不支持显式缓存[^17]。缓存折扣约为标准 input 的 1/20，output 为 input 的 5 倍；Batch 与 Flex 档为 Standard 的 5 折（Input $1.00 / Output $5.00），Priority 档为 1.8 倍（Input $3.60 / Output $18.00）；官方称 Batch API 提供 50% cost reduction；Gemini 4 Argon 不提供 Free tier，也不在 Google AI Pro 计划中；Grounding with Google Search 每月 5,000 次免费（Gemini 3 及更新模型共享），之后 $14 / 1,000 次请求[^17]。

**DeepSeek**：定价单位为 per 1M tokens，区分 Cache Hit / Cache Miss / Output，并区分 Peak / Off-peak 时段；deepseek-flash 为 Cache hit off-peak $0.003 / peak $0.006，Cache miss off-peak $0.15 / peak $0.30，Output off-peak $0.6 / peak $1.2；deepseek-v4-pro 为 Cache hit off-peak $0.022 / peak $0.044，Cache miss off-peak $0.66 / peak $1.32，Output off-peak $1.98 / peak $3.96[^19]。cache hit 约为 cache miss 的 1/50，output 约为 cache-miss input 的 4 倍；off-peak 价格为 peak 的 5 折，peak 时段为 UTC 周一至周五 01:00–04:00 与 06:00–10:00（不含中国法定节假日），其余时段（含周末及中国法定节假日）均为 off-peak；上下文长度 1M，最大输出 384K，支持 thinking 与非 thinking 模式（默认 thinking）；官方声明产品价格可能变动并保留调价权[^19]。

### 2.3 分词口径变更直接改变账单

Anthropic 文档记录了一次分词口径的重大变化：Claude 4.7 及之后模型（含 Mythos Preview）使用新 tokenizer，对同一文本产生约多 30% 的 token；Claude Sonnet 4.6 及更早使用旧 tokenizer[^16]。这意味着在单价不变的前提下，同一段输入文本的计费 token 数会因分词器换代而上升约 30%。

### 2.4 跨语言不公平性：token 数差异可达 15 倍

NeurIPS 2023 的论文指出，同一段文本翻译成不同语言后，tokenization 长度差异巨大，某些情况下可达 15 倍；即使是为多语言支持而专门训练的 tokenizer，这种 token 长度差异依然存在；字符级（character-level）和字节级（byte-level）模型在某些语言对上仍表现出超过 4 倍的编码长度差异[^13]。该差异导致部分语言社区在商业语言服务成本、处理时间与延迟、以及可作为上下文提供给模型的内容量方面受到不公平对待——即 token 数直接影响可提供的上下文内容量；作者主张未来语言模型应使用多语言公平的子词 tokenizer[^13]。

---

## 三、tokenizer 如何影响模型能力

### 3.1 有直接实证支持的结论：词表规模是 scaling law 的变量

arXiv 2407.13623 研究词表大小如何影响 LLM 的 scaling law，训练了 33M 到 3B 参数、最多 500B 字符的模型，并配置不同词表大小；论文提出三种互补方法预测 compute-optimal 词表大小：IsoFLOPs 分析、导数估计、损失函数的参数化拟合；结论是最优词表大小取决于算力预算，更大的模型需要更大的词表[^5]。论文指出多数 LLM 使用的词表偏小，预测 Llama2-70B 的最优词表至少应为 216K，是其实际 32K 词表的 7 倍；论文用 3B 参数模型在不同 FLOPs 预算下做了实证验证：将词表从常规 32K 增至 43K，在相同 2.3e21 FLOPs 下 ARC-Challenge 分数从 29.1 提升到 32.0[^5]。该论文发表于 NeurIPS 2024，代码与 demo 位于 github.com/sail-sg/scaling-with-vocab 与 huggingface space sail/scaling-with-vocab-demo[^5]。

这是现有证据中唯一同时给出「词表规模变化 → 具体能力指标变化」的量化结果：词表从 32K 增至 43K，ARC-Challenge 从 29.1 → 32.0[^5]。

### 3.2 切分歧义与鲁棒性

Kudo 的 subword regularization 主张，训练时使用多种切分候选可让模型对噪声与切分错误更鲁棒，并间接学习词的组合性（如 books = book + s）[^12]。这是关于「分词方式影响模型泛化行为」的机制性论述，但该来源未提供针对 LLM 下游能力基准的量化结果[^11][^12]。

### 3.3 字节级替代路线：BLT

Byte Latent Transformer（BLT）是一种字节级 LLM 架构，作者称其首次在规模上匹配基于分词的 LLM 性能，并在推理效率与鲁棒性上有显著改进；BLT 将字节编码为动态大小的 patch，patch 作为主要计算单元，分段依据下一个字节的熵（entropy of the next byte），在数据复杂度高的地方分配更多计算与模型容量[^26]。论文给出了首个字节级模型的 FLOP 受控扩展研究，规模达 8B 参数、4T 训练字节；结论主张在固定推理成本下，BLT 的扩展性显著优于基于分词的模型，可同时增大 patch 与模型规模；论文报告在推理与长尾泛化（long tail generalization）上有定性改进[^26]。

官方代码库 facebookresearch/blt 称 BLT 包含新的注意力机制以最大化字节与 patch 隐表示之间的信息流，以及一种新的字节序列记忆（byte-sequence memory）；README 称其为首个可从字节端到端训练、无需分词或其他预处理的规模化模型（scaling study 至 8B 参数、8T 训练字节）；已发布 BLT 1B 与 BLT 7B 的 HuggingFace 权重，作者称正与 HF 合作将 BLT 集成进 Transformers；仓库处于活跃开发状态（65 commits），作者称正在持续更新代码以便复现结果[^27]。

### 3.4 能力维度的证据缺口

现有 29 条证据中，没有任何一条直接测量 tokenizer 设计对算术、拼写、代码生成或长尾字符处理能力的影响。字节回退（byte fallback）与 special token 设计的实证影响同样完全无来源。因此本报告不对这些维度作任何结论性论述。

---

## 四、代表性模型的词表规模与分词方案

### 4.1 可确认的词表规模

| 模型 | 词表规模 | 来源性质 |
|---|---|---|
| Qwen3-8B | 151936 | 官方 config.json[^7] |
| DeepSeek-V3 | 129280 | 官方 config.json[^8] |
| Gemma 3 27B IT | 262208（text_config.vocab_size） | 第三方 unsloth 镜像[^9] |
| Llama 3 | 128K tokens | 官方 MODEL_CARD[^22] |

Qwen3-8B 的 config.json 给出 vocab_size 为 151936，bos_token_id = 151643，eos_token_id = 151645，架构为 Qwen3ForCausalLM，hidden_size 4096，36 层，32 个注意力头 / 8 个 KV 头，max_position_embeddings 40960；该配置文件未声明分词算法类型（BPE/Unigram），仅给出词表规模与特殊 token id[^7]。

DeepSeek-V3 的 config.json 给出 vocab_size 为 129280，bos_token_id = 0，eos_token_id = 1，架构为 DeepseekV3ForCausalLM，MoE 结构为 256 个路由专家 + 1 个共享专家、每 token 激活 8 个专家、61 层，max_position_embeddings 为 163840，采用 YaRN rope scaling（factor 40，original_max_position_embeddings 4096）；该配置文件同样未声明分词算法类型[^8]。

Gemma 3 27B 的 text_config.vocab_size 为 262208；多模态特殊 token 索引为 image_token_index = 262144、boi_token_index = 255999、eoi_token_index = 256000，每张图 256 个 token（mm_tokens_per_image = 256）；bos_token_id = 2，eos_token_id = 106，pad_token_id = 0；文本塔 62 层、hidden_size 5376，max_position_embeddings 131072，滑动窗口 1024（sliding_window_pattern 6）[^9]。**需注意该 config 来自第三方镜像仓库 unsloth/gemma-3-27b-it（含 unsloth_fixed 标记），非 Google 官方仓库**[^9]。

Llama 3 使用词表规模为 128K tokens 的 tokenizer（原文："Llama 3 uses a tokenizer with a vocabulary of 128K tokens"），在 8,192 token 的序列上训练，包含 8B 与 70B 两个规模、预训练与指令微调两个变体，发布日期为 2024 年 4 月 18 日[^22]。

### 4.2 分词方案的普遍缺失

Llama 3.1 的 model card 列出 8B、70B、405B 三个规模，上下文长度 128k，均使用 GQA，预训练语料为 15T+ tokens，知识截止 2023 年 12 月，支持语言为英语、德语、法语、意大利语、葡萄牙语、印地语、西班牙语、泰语；**该 model card 未提及 tokenizer 词表规模或分词方案**[^23]。The Llama 3 Herd of Models 论文摘要称最大模型为 405B 参数的 dense Transformer，上下文窗口最高 128K tokens，Llama 3 在约 15T 多语言 tokens 的语料上预训练，v1 提交于 2024-07-31、v3 最后修订于 2024-11-23[^24]。

Llama 4 包含 Scout（17B 激活/109B 总参数，16 experts）与 Maverick（17B 激活/400B 总参数，128 experts）两个 MoE 模型，采用 mixture-of-experts 架构并支持原生多模态（early fusion），Scout 上下文长度 10M、Maverick 为 1M，训练 token 数分别约 40T 与 22T，发布日期为 2025 年 4 月 5 日；**该 model card 未提及 tokenizer 词表规模、BPE 变体或 tiktoken 兼容性**[^25]。

Gemma 3 model card 说明其为多模态模型，支持文本与图像输入、文本输出，含预训练与指令微调变体；27B 模型训练使用 14 万亿 tokens，知识截止日期为 2024 年 8 月；图像被归一化到 896x896 分辨率并编码为每张 256 个 token；4B/12B/27B 尺寸总输入上下文为 128K tokens，1B 与 270M 为 32K tokens；**模型卡正文未给出 vocab_size 数值，也未描述分词算法（BPE/SentencePiece）**[^28]。Gemma 3 技术报告由 Gemma Team (Google DeepMind) 于 2025-03-25 提交（arXiv 2503.19786v1），摘要称 Gemma 3 为多模态模型、规模 1B 至 27B 参数、支持至少 128K tokens 上下文，并通过增加局部/全局注意力层比例来降低 KV-cache 内存；**抓取仅获得摘要与作者列表，未读到正文 tokenizer 章节，故无法从该来源直接引用 vocab_size 或分词算法**[^29]。

关于 o200k 的实际词表规模：tiktoken 源码对 `o200k_base` 与 `cl100k_base` 均未设置 `explicit_n_vocab` 字段，因此无权威数字[^21]。

---

## 五、工程生态现状

GitHub 检索（tokenizer 相关，按 stars 排序）返回的结果为：NLPOptimize/flash-tokenizer（★458，forks 11，C++，描述为 "EFFICIENT AND OPTIMIZED TOKENIZER ENGINE FOR LLM INFERENCE SERVING"，最近 push 2026-02-02）；Systemcluster/kitoken（★62，forks 2，Rust，描述称兼容 SentencePiece、Tokenizers、Tiktoken，支持 BPE、Unigram、WordPiece，可在 JavaScript、Python、Rust 中使用，最近 push 2026-08-10）；actual-computer/toks（★98，forks 6，C，描述为 "exact tokenizers ids, as fast as the hardware allows"，最近 push 2026-10-09）；SunayHegde2006/OmniToken（★1，Rust，声称统一 BPE/WordPiece/Unigram 为单一 trie-walker 自动机并带 AVX2 SIMD 加速，声称 3.02+ GB/s 多核吞吐）[^10]。**该检索未命中 tiktoken、sentencepiece、huggingface/tokenizers 等主流仓库的 stars 数据，返回结果以低星/教学类仓库为主**[^10]。OmniToken 的性能主张来自低星未验证仓库[^10]。

---

## 六、矛盾与不确定性

**1. 能力影响维度的实证证据严重不足。** 29 条证据中，仅 [^13]（多语言 tokenization 不公平性，NeurIPS 2023）与 [^5]（词表规模 → loss / ARC-Challenge）直接相关，且均未覆盖算术、拼写、代码、长尾字符等具体能力维度；字节回退（byte fallback）与 special token 设计的实证影响完全无来源。该子问题目前无法支撑任何结论性论述。

**2. 分词方案（算法类型）普遍缺失。** Qwen3-8B[^7]、DeepSeek-V3[^8]、Gemma 3[^9] 的 config.json 均未声明 BPE/Unigram；Llama 3 的 128K 词表仅来自 MODEL_CARD 一句话[^22]；Llama 4 model card 完全未提 tokenizer[^25]；Gemma 3 的 vocab_size=262208 仅来自第三方 unsloth 镜像而非 Google 官方来源[^9]，且官方 model card[^28] 与技术报告抓取结果[^29] 均未提供该数值。o200k 的实际词表规模也无权威数字，因 tiktoken 源码未设 `explicit_n_vocab`[^21]。

**3. 工程生态热度无法判断。** GitHub 检索未命中主流仓库的 stars 与最近提交数据，返回结果以低星/教学类仓库为主，无法据此判断工程生态热度[^10]。

**4. 定价证据的时间戳异常。** 部分定价来源包含 2026 年及之后的日期与模型名（如 gpt-6-astra、Claude Fable 5.1、Gemini 4 Argon、DeepSeek-V4.1-Flash），这些内容来自抓取时的官方页面，本报告仅如实转述，不对其时间线一致性作判断[^14][^16][^17][^19]。

**5. 归档内容的时效性。** OpenAI Cookbook 的 tiktoken 计数页面为 2022-12-16 归档内容，可能引用过时模型或 API[^15]。

**6. 跨语言差异的量化边界。** 「最高 15 倍」与「字符级/字节级模型超过 4 倍」来自同一篇论文[^13]，本报告未获得其他独立来源的交叉验证。

---

## 参考来源

[^1]: Neural Machine Translation of Rare Words with Subword Units (Sennrich, Haddow, Birch) — https://arxiv.org/abs/1508.07909v5 （arxiv, 2015-08-31）
[^2]: google/sentencepiece — README — https://github.com/google/sentencepiece （github）
[^3]: openai/tiktoken — README — https://github.com/openai/tiktoken （github）
[^4]: Hugging Face Transformers Docs — Tokenization algorithms (Summary of the tokenizers) — https://huggingface.co/docs/transformers/en/tokenizer_summary （web）
[^5]: Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies — https://arxiv.org/abs/2407.13623 （arxiv, 2024-07-18）
[^6]: huggingface/tokenizers — README (v1.0.0.rc.0) — https://github.com/huggingface/tokenizers （github）
[^7]: Qwen3-8B config.json (HuggingFace) — https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json （web）
[^8]: DeepSeek-V3 config.json (HuggingFace) — https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json （web）
[^9]: Gemma 3 27B IT config.json (unsloth 镜像) — https://huggingface.co/unsloth/gemma-3-27b-it/raw/main/config.json （web）
[^10]: GitHub 仓库检索结果（tokenizer 相关，按 stars 排序） — https://github.com/NLPOptimize/flash-tokenizer （github）
[^11]: Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (Kudo, ACL 2018) — https://arxiv.org/abs/1804.10959 （arxiv, 2018-04-29）
[^12]: Subword Regularization (Kudo 2018) — HTML full text, Introduction — https://arxiv.org/html/1804.10959v1 （arxiv, 2018-04-29）
[^13]: Language Model Tokenizers Introduce Unfairness Between Languages — https://arxiv.org/abs/2305.15425 （arxiv, 2023-05-17）
[^14]: OpenAI API Pricing (official pricing page) — https://platform.openai.com/docs/pricing （web）
[^15]: How to count tokens with Tiktoken (OpenAI Cookbook) — https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken （web, 2022-12-16）
[^16]: Anthropic Claude API Pricing (official docs) — https://docs.anthropic.com/en/docs/about-claude/pricing （web）
[^17]: Gemini Developer API Pricing (official docs) — https://ai.google.dev/gemini-api/docs/pricing （web）
[^18]: Understand and count tokens (Gemini API official docs) — https://ai.google.dev/gemini-api/docs/tokens （web）
[^19]: DeepSeek Models & Pricing (official API docs) — https://api-docs.deepseek.com/quick_start/pricing （web）
[^20]: DeepSeek Token & Token Usage (official API docs) — https://api-docs.deepseek.com/quick_start/token_usage （web）
[^21]: openai/tiktoken — tiktoken_ext/openai_public.py (encoding definitions) — https://raw.githubusercontent.com/openai/tiktoken/main/tiktoken_ext/openai_public.py （github）
[^22]: meta-llama/llama-models — Llama 3 MODEL_CARD.md — https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3/MODEL_CARD.md （github, 2024-04-18）
[^23]: meta-llama/llama-models — Llama 3.1 MODEL_CARD.md — https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/MODEL_CARD.md （github, 2024-07-23）
[^24]: The Llama 3 Herd of Models (arXiv:2407.21783v3) — https://arxiv.org/abs/2407.21783v3 （arxiv, 2024-07-31）
[^25]: meta-llama/llama-models — Llama 4 MODEL_CARD.md — https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama4/MODEL_CARD.md （github, 2025-04-05）
[^26]: Byte Latent Transformer: Patches Scale Better Than Tokens — https://arxiv.org/abs/2412.09871 （arxiv, 2024-12-13）
[^27]: facebookresearch/blt (Byte Latent Transformer) — https://github.com/facebookresearch/blt （github）
[^28]: Gemma 3 model card (Google AI for Developers) — https://ai.google.dev/gemma/docs/core/model_card_3 （web）
[^29]: Gemma 3 Technical Report (arXiv:2503.197