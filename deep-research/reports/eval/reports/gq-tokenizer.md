## 摘要

BPE（Byte-Pair Encoding）是一种从数据压缩算法改造而来的子词切分方法：Sennrich 等把 Gage (1994) 的字节对替换算法改造为词切分，合并对象从字节对改为字符或字符序列[^1]。其标准流程是：符号词表以字符词表初始化，每个词表示为字符序列并附加词尾符号 `·`；随后反复统计相邻符号对频次、取频次最大的 pair 合并、重复 num_merges 次[^1]。合并准则是纯频次驱动的贪心确定性替换[^1]，这既是它简单高效的原因，也是它无法给出带概率的多切分、难以直接支撑子词正则化的原因[^2]。WordPiece 被 Kudo 视为 BPE 的变体，区别在于以似然而非频次作为合并判据[^2]。

tokenizer 对能力与成本的影响是真实存在但难以用单一指标预测的。训练侧，tokenizer 选择可显著影响下游性能与训练成本，把英语中心 tokenizer 用于多语言训练会造成严重下游退化，并因分词效率低下带来最高 68% 的额外训练成本[^23]；覆盖五种最高频欧洲语言的多语言 tokenizer 相比英语需要约 3 倍词表规模扩张[^23]。词表规模本身存在随算力预算增长的最优值，最优词表参数与非词表参数呈幂律关系 N_v^opt ∝ N_nv^γ（γ≈0.83）[^25]。能力侧，数字被切碎会显著妨碍精确算术推理[^7]，数字 tokenization 方向（从左到右 vs 从右到左）可造成大幅算术性能差异[^30]。但预测侧的证据是负面的：对约 200 个模型、九种语言的系统检验显示，没有任何单一 tokenizer 内在指标能在模型家族、任务和语言之间一致预测下游性能[^6]；fertility 与 parity 也不总能预测下游性能[^23]。

## 一、BPE 类方法的基本原理

### 1.1 BPE 的算法定义

BPE 的原始出处是 Gage (1994) 的数据压缩算法：迭代地把序列中最频繁的字节对替换为一个未使用的单字节[^1]。Sennrich 等的改造包含三处关键变化：合并对象从字节对改为字符或字符序列；符号词表以字符词表初始化；每个词附加特殊词尾符号 `·`（end-of-word），使翻译后可以还原原始分词[^1]。

算法（Learn BPE operations）的输入、步骤与产出可明确写出[^1]：

- **输入**：从文本抽取的词频字典（每个词以其频次加权），以及合并次数 num_merges。
- **步骤**：反复执行 get_stats 统计相邻符号对频次 → 取频次最大的 pair → merge_vocab 合并 → 重复 num_merges 次（论文示例中 num_merges=10）。
- **产出**：一组有序的合并操作，应用于任意文本即得到切分。

为效率起见不考虑跨词边界的符号对，因此算法可直接在词频字典上运行[^1]。论文给出的示例：从字典 {'low','lowest','newer','wider'} 学到的合并操作为 `r·→r·`、`l o→lo`、`lo w→low`、`e r·→er·`；未见词 'lower' 会被切分为 'low er·'[^1]。论文评估了两种应用方式：源端与目标端各自学习独立编码，或在两端词表并集上学习编码（joint BPE）[^1]。论文主张 BPE 能用固定大小的变长字符序列词表表示开放词表，比大词表加回退词典更简单有效，在 WMT15 英德、英俄任务上相对回退词典基线分别提升 1.1 和 1.3 BLEU[^1][^4]。

### 1.2 BPE 的固有局限与 Unigram 的对照

Kudo 明确指出 BPE 的局限：基于贪心且确定性的符号替换，无法给出带概率的多种切分，因此难以直接用于依赖切分概率 P(x|X) 的子词正则化[^2]。作为对照，Unigram 语言模型假设各子词独立出现，子词序列 x=(x1..xM) 的概率为各子词出现概率之积 P(x)=∏p(xi)，且 Σ_{x∈V} p(x)=1，V 为预先确定的词表；最可能切分为 x* = argmax_{x∈S(X)} P(x)，其中 S(X) 是由输入句构建的切分候选集合，用 Viterbi 算法求得[^2]。给定词表时，子词概率通过 EM 估计以最大化边缘似然，把切分视为隐变量；词表本身未知，故采用迭代算法：先从训练语料启发式构造较大的种子词表，再重复「固定词表下用 EM 估计概率 → 计算删除每个子词后的总似然损失 → 按损失排序只保留前 η%（如 η=80）的子词」直到 |V| 达到目标大小，并始终保留单字符子词以避免 OOV[^2]。

Kudo 对 WordPiece 与 BPE 关系的判断值得单独记录：严格说 WordPiece 模型不同于 BPE，但因其同样采用增量式词表生成、只是损失函数不同，作者将其视为 BPE 的变体；WordPiece 使用似然而非频次作为合并判据[^2]。Kudo 还给出 BPE 的典型词表规模经验：通常词表仅需 16k–32k 即可不显著增加编码句长[^2]。

### 1.3 工程实现中的 BPE

tiktoken 是用于 OpenAI 模型的快速 BPE 分词器，示例编码包括 o200k_base、cl100k_base，并可通过 `encoding_for_model("gpt-4o")` 取得对应模型的编码[^3]。其 README 明确列出 BPE 的四条性质：可逆且无损，token 能还原为原始文本；可作用于任意文本，包括不在分词器训练数据中的文本；压缩文本，token 序列短于原文对应字节，实践中平均每个 token 约对应 4 个字节；让模型看到常见子词，例如 "ing" 使 "encoding" 常被切成 "encod"+"ing"，有助于模型泛化与理解语法[^3]。tiktoken 的 Encoding 对象由 pat_str（预分词正则）、mergeable_ranks（可合并的 rank 表）、special_tokens 等参数构造，说明其实现以「预分词 + 合并 rank 表」为核心数据结构[^3]。README 提供 `_educational` 子模块用于教学与可视化 BPE 过程[^3]。其「比同类开源分词器快 3–6 倍」的说法是项目自述的性能主张（在 1GB 文本上以 GPT-2 分词器测量，对比 tokenizers==0.13.2 / transformers==4.24.0 / tiktoken==0.2.0），非独立第三方评测[^3]。

huggingface/tokenizers 明确列出四种模型实现：BPE、Unigram、WordPiece、WordLevel，并与 Normalizer、PreTokenizer、Model、PostProcessor、Decoder 共同构成完整 pipeline[^31]。v0.21.0 版 README 正文只列举三种模型（BPE、WordPiece、Unigram），未在正文提及 WordLevel[^32]——这是同一项目不同版本间的一处表述不一致。该库自述定位为 "The SOTA tokenization library on all languages, all models, all hardwares"，属仓库自我营销表述，无第三方权威背书[^31]；v0.21.0 给出的唯一吞吐数字是「在服务器 CPU 上分词 1GB 文本不到 20 秒」，该数字是分词吞吐而非训练速度[^32]。当前 main 分支处于 v1.0.0.rc.x 发布候选阶段，README 称 "the current library does not ship all features"，训练与从 Hub 下载改为默认关闭的 opt-in feature[^31]。其预分词组件 bitcannon 声称覆盖 gpt2/ByteLevel、cl100k、o200k、tekken、deepseek 和 kimi-k2 的 byte-exact 语法[^31]。

## 二、tokenizer 如何影响模型能力

### 2.1 数字与算术推理：最直接的能力证据

基于文本的 tokenization 经常把数字切碎，显著妨碍精确的算术推理[^7]。ANI 提出混合框架，用上下文感知的门控机制选择性地把数值 embedding（具体为 FoNE）注入隐空间，在多种 LLM 上相对官方参考模型在 MATH 上提升 9.5 个点，同时在通用语言 benchmark 上保持稳健表现[^7]。

更细的机制证据来自数字 tokenization 方案的差异：LLaMA 与 PaLM 选择单数字（single-digit）tokenization，而 GPT-3.5 与 GPT-4 对每个 1 位、2 位、3 位数字都有独立 token[^30]。对 GPT-3.5/GPT-4 而言，从右到左的 tokenization（推理时用逗号分隔数字强制实现）相比标准从左到右 tokenization 大幅提升性能；使用标准从左到右 tokenization 时，模型错误呈现刻板的错误模式，说明模型计算是系统性的而非近似性的[^30]。模型能够在不同 tokenization 之间转换，因此可用 chain-of-thought 式方法恢复从左到右 tokenized 输入上的性能；tokenization 方向造成的性能差距随模型规模增大而缩小，可能表明更大模型更能覆盖这种 tokenization 依赖的归纳偏置[^30]。

### 2.2 多语言与跨语言能力

tokenizer 选择可显著影响模型下游性能与训练成本[^23]。作者训练了 24 个 2.6B 参数规模的单语与多语言 LLM 做消融，报告：覆盖五种最高频欧洲语言的多语言 tokenizer 相比英语需要约 3 倍词表规模扩张；将英语中心 tokenizer 用于多语言 LLM 训练会导致严重的下游性能退化，并因词表 tokenization 效率低下带来最高 68% 的额外训练成本[^23]。

词表剪枝方向的证据从反面印证了词表配置与语言覆盖的耦合：对 Qwen3、Gemma-3、Llama-3、Aya 等模型删除与目标应用无关语言的 token 与 embedding 参数后，token pruning 通过消除语言混淆显著提升生成稳定性，在机器翻译上常常提升韩语特定任务表现；指令遵循能力呈现与架构相关的差异，作者将其与潜在跨语言表征联系起来[^9]。英阿语言对的词表剪枝研究把词表规模从 12.8 万以上降到约 1 万 token，实现 60% 内存节省且性能无损；剪枝并微调后的 M2M100 取得 BLEU 42.04（对比 OPUS-MTen-ar 双语模型的 44.59），但在 COMET 上显著更优（0.8730 vs 0.7911）[^8]。

### 2.3 为什么「单一指标预测能力」这条路走不通

对约 200 个语言模型、覆盖九种语言、同时评测模型与其 tokenizer 的系统研究检验了各类 tokenizer 内在指标与多 benchmark 表现之间的关系，核心结论是：尽管已有大量且多样的 tokenizer 内在指标，没有任何一个指标能在模型家族、任务和语言之间一致地预测下游性能[^6]。作者据此主张需要新的 tokenizer 评测指标与方法，并提出一个基于语言学的新指标，称其能捕捉现有指标遗漏的 tokenizer 质量维度，但预测力仍可能随模型家族、语言和任务而变化[^6]。该文被 EMNLP 2026 Findings 接收，10 页[^6]——需注意 ACL Anthology 的 EMNLP 2026 会议页未直接显示该论文条目，因此不能据此确认其 Anthology 书目信息[^34]。

与之呼应，Tokenizer Choice 一文的作者明确指出常用的 tokenizer 评估指标 fertility 与 parity 并不总能预测模型下游性能，因此认为这些指标作为下游性能代理指标值得质疑[^23]。

## 三、tokenizer 如何影响成本

### 3.1 词表规模与参数/算力分配

语言模型用 embedding 矩阵表示 token，其参数量随词表规模线性增长[^10]。Scaling Laws with Vocabulary 把总参数量分解为非词表参数 N_nv 与词表参数 N_v，N = N_nv + N_v，并用 N_v = V·d 表示词表参数（V 为词表大小，d 为 embedding 维度）；作者用 V·d 而非 2V·d，理由是实测主要计算负担（以 FLOPs 计）来自输出层而非词嵌入层[^25]。实验训练 33M 到 3B 参数模型、最多 500B 字符，核心结论是最优词表大小随算力预算增长，最优词表参数与非词表参数呈幂律关系 N_v^opt ∝ N_nv^γ，γ ≈ 0.83 < 1，即词表参数应比非词表参数增长得更慢[^25]。论文预测 Llama2-70B 的最优词表大小至少应为 216K，是其实际 32K 词表的 7 倍；实证上在相同 2.3e21 FLOPs 预算下把词表从 32K 增大到 43K，ARC-Challenge 从 29.1 提升到 32.0[^25]。论文还提出用 f(V) = a·log²(V) + b·log(V) + c 从词表大小估计压缩比（token 数 D 与字符数 H 之比 D/H），拟合 1K 到 1024K 词表的多个 tokenizer 得 a=0.0064、b=−0.1581、c=1.2047[^25]。收益来自更好的 tokenization fertility 与覆盖更广概念，风险是稀有 token 欠拟合，尤其在数据受限情形下[^25]。论文指出当前多数 LLM 词表参数低于预测最优值（Llama2-7B 用 32K、Gemma-7B 用 256K），StarCoder2-3B、OLMo-7B、InternLM2-20B、Gemma2-27B 最接近其规模的最优配置[^25]。官方代码仓库 sail-sg/scaling-with-vocab 标注 [NeurIPS-2024]，提供三个脚本分别复现 IsoFLOPs 幂律拟合、导数法、参数化拟合三种预测方法，以及 optimal_Nv_predict.py 直接预测最优 N_v 与 V[^33]。

### 3.2 压缩率、生成速度与有效上下文

tokenizer 的 size、pre-tokenization 正则表达式与训练数据会显著影响模型的生成速度、有效上下文大小、内存占用与下游性能[^26]。作者在代码生成任务（HumanEval、MBPP）上训练专用 BPE 代码 tokenizer 并做广泛消融，发现当微调超过 50B token 时，可以为预训练 LLM 专门化 tokenizer，从而在生成速度与有效上下文大小上获得大幅收益[^26]。tiktoken README 给出的经验值是平均每个 token 约对应 4 个字节[^3]，这是「token 数 → 字节数 → 计费与上下文占用」换算的常用锚点，但它是项目自述的实践观察，非受控测量。

### 3.3 降低词表参数成本的替代路线

MultiHashFormer 针对「embedding 参数量随词表线性增长」提出替代方案：每个 token 表示为多个独立哈希函数生成的离散哈希 ID 短序列（hash signature），由 Hash Encoder 压缩为单一隐向量，再由 Hash Decoder 生成下一 token 的哈希签名并映射回文本；在 100M、1B、3B 三个规模上一致优于标准 Transformer LM，并能以恒定参数开销处理多语言词表扩展而无需任何修改[^10]。词表剪枝路线则直接削减既有词表：英阿案例实现 60% 内存节省且性能无损[^8]；韩语中心模型的实验验证词表规模显著缩减是内存受限、领域特定部署的有效优化策略，但推理延迟收益有限[^9]。

## 四、主流模型的 tokenizer 参数：证据严重不足

本节只能给出极有限的确认事实，其余必须显式标注空缺。

**可确认**：Qwen2.5 使用 Qwen 的 tokenizer，实现的是 byte-level byte-pair encoding (BBPE)，词表规模 151,643 个 regular tokens；相比此前 Qwen 版本，控制 token 从 3 个扩展到 22 个，其中新增 2 个用于工具功能的 token，其余分配给其他模型能力；这一扩展在所有 Qwen2.5 模型上建立了统一词表，以提升一致性并减少潜在兼容性问题[^28]。Qwen2.5 开源权重覆盖 0.5B/1.5B/3B/7B/14B/32B/72B 七种尺寸，另有 MoE 的 Qwen2.5-Turbo 与 Qwen2.5-Plus 两个 API 模型[^28]。

**无法确认**：Llama 3 的证据只覆盖参数量（8B/70B/405B 稠密 Transformer）、上下文窗口（最高 128K tokens）、预训练语料规模（约 15T 多语言 tokens，旗舰 405B 在 15.6T 文本 tokens 上预训练，8K 上下文后扩展至 128K）与数据过滤流程（行级去重、脏词计数过滤、基于 token 分布 KL 散度的离群文档过滤），未包含 tokenizer 章节的任何参数[^12]；其元数据记录同样不含 tokenizer 参数，不能作为结论依据[^27]。DeepSeek-V3 的证据明确称除显式说明外其他细节沿用 DeepSeek-V2 的设置[^13]，而 DeepSeek-V2 抓取到的正文（摘要、引言、2.1 节 MLA）未包含 tokenizer 词表大小与分词算法类型的原文表述，tokenizer 内容位于未抓取到的 2.2 节[^29]。o200k_base 仅在 tiktoken README 中被提及名称，无词表大小与压缩率[^3]。Gemini、Mistral、GPT-4o 的 tokenizer 一手参数：现有公开资料不足以确认。

## 五、工具生态与量化能力：证据不足以支撑结论

现有 GitHub 证据几乎全部是搜索接口返回的元数据，README 正文均未成功读取，因此无法支撑任何关于工具能力或「tokenizer 与成本量化关系」的实质结论。

可记录的元数据事实：openai/tiktoken 约 19,521 stars、1,661 forks，主语言 Python，最近 push 2026-08-17，简介自述为「a fast BPE tokeniser for use with OpenAI's models」[^14]。围绕 tiktoken 形成了多语言移植生态：pkoukk/tiktoken-go（约 956 stars，Go，最近 push 2026-05-12）、tiktoken-go/tokenizer（约 459 stars，Go，最近 push 2026-10-21）、zurawiki/tiktoken-rs（约 408 stars，Rust，最近 push 2026-09-24）、dmitry-brazhenko/SharpToken（约 263 stars，C#，基于 tiktoken Python 库）、daulet/tokenizers（约 228 stars，提供 HF Tokenizers 与 Tiktoken 的 Go/Wasm 绑定）[^17]；niieani/gpt-tokenizer 约 856 stars，TypeScript，自称 tiktoken 的 JavaScript 移植并声称支持 gpt-5、gpt-o*、gpt-4o 等模型（「fastest」为营销性表述，未获第三方验证）[^16]；M4THYOU/TokenDagger 约 474 stars，C++，自述为 tiktoken 的高性能实现[^18]；NLPOptimize/flash-tokenizer 约 457 stars，C++，自述为面向 LLM 推理服务的优化分词引擎[^19]；dqbd/tiktokenizer 约 1,714 stars，TypeScript，定位为 OpenAI 分词器的在线 playground[^15]。与「每 token 计算量/显存成本」量化相关的 RahulSChand/gpu_poor 约 1,403 stars，简介称可计算任意 LLM 的 token/s 与 GPU 显存需求[^20]。zero-noise-bench 针对字符串反转、字符计数、位置抽取等任务以暴露 tokenization 盲点，但仅 0 stars、0 forks，社区热度极低，不能作为热度证据[^21]。

检索层面存在明确的方法论问题：以 'tokenizer benchmark comparison token count' 检索仅返回 4 个仓库（stars 4/1/0/0）且均非 tokenizer 对比 benchmark；以 'tokenizer comparison fertility multilingual benchmark' 检索返回 0 个仓库；以 'huggingface tokenizers' 检索未返回 huggingface/tokenizers 官方仓库本身[^22]。这说明检索式本身失效，而非该生态不存在——huggingface/tokenizers 的 README 是通过 raw 路径间接获得的[^31][^32]。

## 矛盾与不确定性

**一、tokenizer 指标与下游性能的关系存在直接冲突。** 一方面，Tokenizer Choice 报告 tokenizer 选择可显著影响下游性能与训练成本，并给出 68% 额外训练成本这样的量级[^23]；另一方面，同一作者又指出 fertility 与 parity 不总能预测下游性能[^23]，而 200 模型 × 九语言的系统检验给出更强的否定结论：没有任何单一内在指标能跨模型家族、任务、语言一致预测下游性能[^6]。这两者并不逻辑矛盾（「tokenizer 重要」与「现有代理指标不好用」可以同时成立），但意味着实践中不能靠 fertility 类指标选 tokenizer，而必须做端到端消融。新提出的语言学指标其预测力也被作者自己限定为可能随模型家族、语言、任务变化[^6]。

**二、词表规模的「最优」结论依赖建模假设。** Scaling Laws with Vocabulary 用 N_v = V·d 而非 2V·d，理由是 FLOPs 主要来自输出层[^25]；这一简化直接影响最优词表大小的预测值（如 Llama2-70B 应为 216K、为实际的 7 倍[^25]）。若参数计数口径改变，结论的数值需重新校准。此外该结论建立在 33M–3B 参数、最多 500B 字符的实验规模上[^25]，向 70B 级外推属于论文自身的预测而非实测。

**三、词表扩张与词表剪枝两条路线并存且都自称有效。** 前者主张当前 LLM 词表参数普遍低于最优、应扩大[^25][^33]；后者在具体部署场景下把词表从 12.8 万降到约 1 万仍称性能无损、省 60% 内存[^8]，或删除无关语言 token 提升生成稳定性[^9]。二者的适用条件不同（前者是给定算力预算下的训练期最优分配，后者是内存受限的领域部署），但现有证据不足以给出统一的判据边界。

**四、主流模型 tokenizer 参数大面积空缺。** 仅 Qwen2.5 一条满足词表大小 + 算法类型 + 特殊 token 的完整要求[^28]。Llama 3 的 tokenizer 章节未取到[^12][^27]，DeepSeek-V3 的 tokenizer 细节指向未抓取的 V2 2.2 节[^13][^29]，o200k_base 只有名称[^3]。因此本报告无法就「主流模型 tokenizer 参数对比」给出任何跨模型结论，现有公开资料不足以确认。

**五、工具生态的量化能力无实质证据。** 所有 GitHub 条目均为元数据，README 正文未读取[^14]–[^21]，且检索式本身失效[^22]。关于「词表大小与 embedding/输出层参数占比、压缩率对计费与有效上下文的影响」的量化关系，本报告未获得任何一手证据，现有公开资料不足以确认。

**六、若干单点证据的强度需打折。** tiktoken「快 3–6 倍」是项目自述、非独立评测[^3]；huggingface/tokenizers 的 "SOTA"/"industry's standard" 为自我营销表述[^31]，bitcannon「每个预分词正则都最快」亦为仓库自述、无第三方验证[^31]；gpt-tokenizer 的「fastest」同为营销性表述[^16]。此外同一项目 v0.21.0 与 main 分支 README 对模型实现的列举不一致（三种 vs 四种）[^31][^32]。EMNLP 2026 Findings 的会议页存在且规模庞大（2479 papers）[^34]，但未列出本报告引用的目标论文条目，故其书目信息不能据此确认[^6]。

## 参考来源

[^1]: Neural Machine Translation of Rare Words with Subword Units (Sennrich, Haddow, Birch; ACL 2016) — https://arxiv.org/html/1508.07909v5（arxiv, 2015-08-31 (v1); v5 2016-06-10）
[^2]: Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (Taku Kudo; ACL 2018) — https://arxiv.org/html/1804.10959v1（arxiv, 2018-04-29）
[^3]: openai/tiktoken — README (fast BPE tokeniser for OpenAI models) — https://raw.githubusercontent.com/openai/tiktoken/main/README.md（github）
[^4]: Neural Machine Translation of Rare Words with Subword Units — arXiv 摘要页（元数据核对） — https://arxiv.org/abs/1508.07909（arxiv, 2015-08-31 (v1); v5 2016-06-10）
[^6]: No Single Tokenizer Feature Reliably Predicts Downstream Language Model Performance — https://arxiv.org/abs/2610.21679（arxiv, 2026-10-20）
[^7]: ANI: Adaptive Numerical Injection for Unifying Semantic and Arithmetic Representations in Numerical Reasoning — https://arxiv.org/abs/2609.39294（arxiv, 2026-09-30）
[^8]: Efficient Multilingual Neural Machine Translation via Corpus-Driven Vocabulary Pruning: An English-Arabic Case Study — https://arxiv.org/abs/2608.03480（arxiv, 2026-08-04）
[^9]: Optimizing Korean-Centric LLMs via Token Pruning — https://arxiv.org/abs/2604.16235（arxiv, 2026-04-17）
[^10]: MultiHashFormer: Hash-based Generative Language Models — https://arxiv.org/abs/2606.28057（arxiv, 2026-08-28）
[^12]: The Llama 3 Herd of Models (arXiv:2407.21783v3) — https://arxiv.org/html/2407.21783v3（arxiv, 2024-07-31 (v1); v3 revised 2024-11-23）
[^13]: DeepSeek-V3 Technical Report (arXiv:2412.19437v2) — https://arxiv.org/html/2412.19437v2（arxiv, 2024-12 (v2)）
[^14]: openai/tiktoken — fast BPE tokeniser for use with OpenAI's models — https://github.com/openai/tiktoken（github）
[^15]: dqbd/tiktokenizer — Online playground for OpenAI tokenizers — https://github.com/dqbd/tiktokenizer（github）
[^16]: niieani/gpt-tokenizer — JS BPE tokenizer for OpenAI GPT models — https://github.com/niieani/gpt-tokenizer（github）
[^17]: pkoukk/tiktoken-go / tiktoken-go/tokenizer / zurawiki/tiktoken-rs — 多语言 tiktoken 移植生态 — https://github.com/pkoukk/tiktoken-go（github）
[^18]: M4THYOU/TokenDagger — High-Performance Implementation of OpenAI's TikToken — https://github.com/M4THYOU/TokenDagger（github）
[^19]: NLPOptimize/flash-tokenizer — optimized tokenizer engine for LLM inference serving — https://github.com/NLPOptimize/flash-tokenizer（github）
[^20]: RahulSChand/gpu_poor — token/s 与 GPU 显存需求估算器 — https://github.com/RahulSChand/gpu_poor（github）
[^21]: shivnathtathe/zero-noise-bench — 暴露 tokenization 盲点的基准套件 — https://github.com/shivnathtathe/zero-noise-bench（github）
[^22]: 检索结果：未找到 tokenizer 对比 benchmark 类仓库（检索式 'tokenizer benchmark comparison token count'） — https://github.com/search?q=tokenizer+benchmark+comparison+token+count（github）
[^23]: Tokenizer Choice For LLM Training: Negligible or Crucial? — https://arxiv.org/abs/2310.08754（arxiv, 2023-10-12 (v1); v4 2024-03-17）
[^25]: Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies — https://arxiv.org/abs/2407.13623（arxiv, 2024-07-18 (v1), v3 2024-11-01, NeurIPS 2024）
[^26]: Getting the most out of your tokenizer for pre-training and domain adaptation — https://arxiv.org/abs/2402.01035（arxiv, 2024-02-01 (v1), v2 2024-02-07）
[^27]: The Llama 3 Herd of Models — metadata record (arXiv:2407.21783) — https://arxiv.org/abs/2407.21783（arxiv, 2024-07-31 (v1); last revised 2024-11-23 (v3)）
[^28]: Qwen2.5 Technical Report (arXiv:2412.15115v2), Section 2 "Architecture & Tokenizer" — https://arxiv.org/html/2412.15115v2（arxiv, 2024-12-19 (v1); v2 2025-01-03）
[^29]: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (arXiv:2405.04434v5) — https://arxiv.org/html/2405.04434v5（arxiv, 2024-05-07 (v1); v5 后续修订）
[^30]: Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs — https://arxiv.org/abs/2402.14903（arxiv, 2024-02-22）
[^31]: huggingface/tokenizers README (main branch, v1.0.0.rc.x) — https://raw.githubusercontent.com/huggingface/tokenizers/main/README.md（github）
[^32]: huggingface/tokenizers README (v0.21.0 tag) — https://raw.githubusercontent.com/huggingface/tokenizers/v0.21.0/README.md（github）
[^33]: sail-sg/scaling-with-vocab (official code for NeurIPS-2024 vocabulary scaling laws) — https://raw.githubusercontent.com/sail-sg/scaling-with-vocab/main/README.md（github）
[^34]: ACL Anthology — The 2026 Conference on Empirical Methods in Natural Language Processing (event page) — https://aclanthology.org/events/emnlp-2026/（web）