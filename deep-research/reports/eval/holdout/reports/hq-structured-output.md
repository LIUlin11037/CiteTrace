# 让 LLM 稳定输出符合 schema 的 JSON：约束解码 vs 提示词约束

## 摘要

约束解码（constrained decoding）与提示词约束（prompt-based enforcement）解决的是同一个问题——让自回归 LLM 的下一 token 分布只落在「格式合法」的支撑集上——但机制层级完全不同：前者在**采样环节**改写 logits，后者在**上下文环节**施加软性影响。约束解码的标准形式化是 token 级布尔掩码：给定已生成序列 $s$，构造掩码 $m(s) \in \{0,1\}^N$（$N$ 为词表大小），把非法 token 的 logits 置零后从归一化分布采样 [^14]；掩码由当前有限状态机状态决定，状态在采样过程中被增量跟踪 [^14]。提示词约束则没有任何机制性保证，其可靠性完全取决于模型对指令的服从程度。

证据给出的核心结论有三条。第一，**机制层面**约束解码确实提供 token 级文法合法性保证，但「100% 合法」的自述边界模糊：llama.cpp 明确只支持 JSON Schema 子集且不支持的特性被静默跳过 [^29]，第三方实测显示厂商 structured-output 通道会在模型运行前删除 `minimum`/`pattern`/`additionalProperties` 等约束 [^13]。第二，**可靠性层面**提示词约束在弱模型上近乎失效——SLM 几乎完全无视 prompt-based 约束 [^5]；但强模型 + 宽松提取的实测 schema 通过率可达 0.972 [^13]，说明提示词约束的失败主要是包装问题而非正确性问题。第三，**代价层面**约束解码并非免费：它可能削弱推理能力 [^1]，且各框架对真实 schema 的覆盖率差异显著，最佳框架支持的 schema 数量是最差框架的两倍 [^20]。

---

## 一、约束解码的机制定义

### 1.1 标准形式化：token 级掩码

Willard & Louf 把引导式生成重构为「有限状态机状态之间的转移」，并给出 token-level masking 的形式化：定义布尔掩码 $m: \mathcal{P}(V) \to \{0,1\}^N$ 作用于 logits $\alpha$，得到未归一化条件分布；掩码限制下一 token 随机变量 $s_{t+1}$ 的支撑集（support）[^14]。带掩码的采样算法（Algorithm 2）为：$\alpha \leftarrow \text{LLM}(s,\theta)$；构造掩码 $m(s)$；$\tilde{\alpha} \leftarrow m \odot \alpha$；从 $\text{Categorical}(\tilde{\alpha})$ 采样 $\tilde{s}$；若 $\tilde{s} = \text{EOS}$ 则停止 [^14]。论文指出常见且充分的朴素做法是对整个词表重复评估以判定哪些 token 合法、把非法 token 概率置零，代价为每生成一个 token 固定 $O(N)$ [^14]。

其状态机基础采用 Sipser《Introduction to Theory of Computation》Definition 1.5 的五元组有限自动机定义：$(Q, \Sigma, \delta, q_0, F)$，$Q$ 为有限状态集，$\Sigma$ 为有限字母表，$\delta: Q \times \Sigma \to Q$ 为转移函数，$q_0$ 为起始状态，$F$ 为接受状态集 [^14]。FSM 状态可在采样过程中被跟踪，从而无需每次从已生成序列开头重新读取即可继续状态机；掩码由当前 FSM 状态决定 [^14]。论文给出具体掩码示例：正则表达式 `([0-9]*)?\.?[0-9]*`（浮点数），词表仅含 `"A"`、`"."`、`"42"`、`".2"`、`"1"`；初始状态 0 下 `"A"` 被掩码，只能采样 `"."`、`"42"`、`".2"`、`"1"`；采样 `".2"` 后进入状态 3，此时仅 `"42"` 与 `"1"` 为合法续接 [^14]。

索引构造是让掩码代价从 $O(N)$ 降到平均 $O(1)$ 的关键：先对词表中每个字符串 $v$ 求出 FSM 接受它的所有状态子序列（Algorithm 3 `find_sub_sequences`），再构建映射 $\sigma: Q \to \mathcal{P}(V)$，把 FSM 状态与在该状态下会被接受的词表子集关联（Algorithm 4）[^14]。索引的关键设计是「考虑从每一个可行的 FSM 状态出发」，因为词表字符串可能匹配正则表达式的任意片段，而这些片段隐含地就是 FSM 状态 [^14]。

### 1.2 文法约束版本：增量解析器

Geng et al. 把 grammar-constrained decoding（GCD）的机制定义为：在解码过程中进行干预，剪枝（pruning）概率分布，使其只包含形式文法所允许的 token 子集 [^15]。允许的 token 子集由一个增量解析器（incremental parser）返回，其输入是部分已生成序列与形式文法，输出是下一步允许的 token 集合 [^15]。解析器的角色被抽象为「补全引擎」（completion engine，引 Poesia et al. 2022），从部分序列与文法 $G$ 推导出补全候选；该文具体使用 Grammatical Framework 的增量解析器（Angelov, 2009）[^15]。形式文法定义为四元组 $G = (V, \Sigma, P, S)$：$V$ 为非终结符有限集，$\Sigma$ 为终结符有限集，$P$ 为产生式规则有限集，$S \in V$ 为起始符号；为可处理性，框架聚焦于上下文无关文法 [^15]。该约束过程与任何解码算法兼容，包括贪心解码、beam search、top-k 采样 [^15]。

PICARD 是同一机制在 text-to-SQL 上的早期实例：输入是当前假设的 token id 与模型语言建模头对每个词表 token 预测的 log-softmax 分数，对被拒绝 token 的分数置为 $-\infty$、被接受 token 保留原分数 [^16]。它在每个生成步先只保留 top-k 最高概率 token，再对被判定失败的 token 赋 $-\infty$，因此与贪心与 beam search 均可直接集成 [^16]。PICARD 提供四档检查完备度：off、lexing（仅词法）、parsing without guards（语法解析）、parsing with guards（最高档），通过更高档检查的预测必然通过更低档 [^16]。

### 1.3 工程实现：预检划分与栈加速

XGrammar 的问题动机是：用上下文无关文法做约束解码时，需要在运行时对词表中所有 token 遍历多个栈状态，带来不可忽略的开销 [^8]。其核心机制是把词表划分为 context-independent tokens（可预检查）与 context-dependent tokens（需运行时解释），并做文法变换以扩展上下文、减少 context-independent token 数量；工程上用高效 persistent stack 加速 context-dependent token 检查，并与推理引擎协同设计，把文法计算与 GPU 执行重叠（overlap）[^8]。

另一条工程路线来自对自动机结构的利用：在「模型被要求输出符合 schema 的 JSON 或带类型参数的工具调用、由自动机通过禁止破坏格式的 token 来强制格式」这一场景下，该自动机具有特殊性质——从任一状态出发，每个 token 只沿唯一一条路径前进 [^2]。据此构造的推理引擎让格式强制要求的文本不运行模型直接写出，mask 在 GPU 上重算而无需任何查表，递归格式使用小栈，独立字段并行解码并校验 [^2]。在单台 16 GB Apple M2 Pro 上，用 Qwen3.5-2B 与 4B 对比 MLX + llguidance：schema 约束抽取完成时间快 1.2–1.3 倍且答案相同；一个文法占 3 MB 而非最多 1.5 GB；一台服务器可容纳 16 个文法而查表方案会内存耗尽；16 个工具调用 agent 完成时间快 2.5 倍 [^2]。

### 1.4 生态与接入面

vLLM 的 structured outputs 支持 xgrammar 或 guidance 作为后端，支持的结构化参数为 choice（枚举之一）、regex、json（JSON Schema）、grammar（上下文无关 EBNF 文法）、structural_tag（在指定标签内遵循 JSON Schema）；后端通过 `--structured-outputs-config.backend` 配置，默认 `auto` 会依据请求细节自动选择 [^12]。v0.12.0 起移除旧字段 `guided_json`/`guided_regex`/`guided_choice`/`guided_grammar`/`guided_whitespace_pattern`/`structural_tag`/`guided_decoding_backend`，统一改为 `structured_outputs` 参数 [^12]。不同后端的正则语法不同：xgrammar、guidance、outlines 使用 Rust 风格正则，lm-format-enforcer 使用 Python `re` 模块 [^12]。

XGrammar README 自述是多数 LLM 推理引擎的默认结构化生成后端，包括 vLLM、SGLang、TensorRT-LLM、MLC-LLM，并列出集成时间线：2024/11 SGLang 与 MLC-LLM；2024/12 vLLM；2025/01 TensorRT-LLM；2025/02 Modular MAX；2025/09 OpenVINO GenAI；2025/12 Mirai [^7]。仓库元数据为 stars 1973、forks 246、open issues 136、Apache-2.0、创建于 2024-06-28、最近 push 2026-10-23 [^7]。Outlines 的 API 形态为 `model(prompt, output_type)`，直接传入 Python 类型（Literal、int、Pydantic 模型、Enum）作为输出约束，README 称同一份代码可跨 OpenAI、Ollama、vLLM 等后端运行 [^9]；仓库 stars 15973、最近 push 2026-09-21 [^9]。Guidance 自述为「steering language models」的高效编程范式，stars 21807、MIT 许可、最近 push 2026-05-21 [^10]。LMQL 定位为面向大语言模型的查询/编程语言，stars 4224、最近 push 2025-05-22，活跃度明显低于上述三者 [^11]。

llama.cpp 走的是自有文法格式 GBNF（GGML BNF），是 BNF 的扩展，主要增加少量现代类正则特性；终结符支持完整 Unicode，重复与可选语法包括 `*`、`+`、`?`、`{m}`、`{m,}`、`{m,n}`、`{0,n}`，`root` 规则定义整个输出必须匹配的起点 [^29]。GBNF 支持按 token 而非字符匹配（token ID 或 token 字符串形式），并可用 `!` 前缀取反；token 字符串形式要求该字符串在词表中恰好切分为单个 token，否则文法解析失败 [^29]。

---

## 二、提示词约束的机制与失效条件

提示词约束没有统一的机制定义，证据中出现的形态是「把 schema 粘贴进提示词」[^13] 与「基于提示词的强制（prompt-based enforcement）」[^5]。它的作用路径是上下文条件化，不触碰采样分布，因此不提供任何 token 级保证。

失效条件在证据中有两处直接观测。其一是弱模型：PTC-Decoder 论文报告现有 plan-solve 范式依赖 prompt-based enforcement，实验显示 SLM 几乎完全无视这种约束——弱模型无法调用 plan [^5]。其二是包装问题：在 216 次 API 调用的实测中，plain 模式（schema 粘贴进提示词）72/72 返回 HTTP 200，52/72 能被严格解析为 JSON，70/72 在宽松提取后通过 schema 校验 [^13]。20 个严格解析失败中 18 个来自同一模型 groq/qwen3.6-27b（严格解析 0/18），但该模型 18/18 的响应在从代码围栏（11 次）或按 span 扫描（7 次）提取 JSON 后均通过 schema 校验；宽松提取器把 0.722 的严格解析率提升到 0.972 的 schema 通过率 [^13]。plain 模式按模型的 schema 通过率为：gemini-3.1-flash-lite 18/18、gpt-oss-120b 18/18、gpt-oss-20b 16/18、qwen3.6-27b 18/18 [^13]。

值得注意的是，提示词约束与约束解码并非互斥。vLLM 文档明确建议即使有 schema 约束，也应在 prompt 中说明 JSON schema 与字段填充方式，可显著提升效果 [^12]。llama.cpp 文档则从反面确认了这一分工：JSON schema 仅用于约束模型输出，不会被注入 prompt；模型看不到 schema，若希望模型理解期望结构需在 prompt 中显式描述（工具调用场景例外，schema 会注入 prompt）[^29]。

---

## 三、两者的关键差别

### 3.1 保证强度：token 级文法合法性 vs 无保证

约束解码的保证是机制性的：掩码在每一步把非法 token 的支撑集排除，因此输出必然属于文法所定义的语言 [^14][^15]。XGrammar README 把这一点表述为「100% structural correctness」[^7][^17]，OpenAI 把 structured outputs 表述为「保证模型总是生成符合所提供 JSON Schema 的响应」[^24]。但「保证」的边界需要拆开看：

- **保证的是 token 级文法合法性**，不保证语义正确。PTC-Decoder 论文明确指出最终答案准确率仍是未解决问题 [^5]。
- **不保证 schema 全特性被强制执行**。llama.cpp 文档承认只支持把 JSON Schema 的一个子集转换为 GBNF，不支持的特性会被静默跳过，建议用命令行 Python 转换器查看警告并用 `llama-gbnf-validator` 校验 [^29]。其列出的已知限制包括：`additionalProperties` 默认为 false；同一类型中不能混用 `properties` 与 `anyOf`/`oneOf`（issue #7703）；`prefixItems` 损坏；`minimum`/`exclusiveMinimum`/`maximum`/`exclusiveMaximum` 目前仅支持 integer 类型、不支持 number；嵌套 `$ref` 损坏（issue #8073）；`pattern` 必须以 `^` 开头并以 `$` 结尾；C++ 版本不支持远程 `$ref`；`string format` 缺少 `uri`、`email`；不支持 `patternProperties` [^29]。文档另列出「不太可能实现」的不支持特性：`uniqueItems`、`contains`/`minContains`、`$anchor`、`not`、条件语句 `if/then/else`/`dependentSchemas` [^29]。
- **厂商 structured-output 通道可能在模型运行前就删除约束**。第三方实测中，native 模式（schema 原样交给厂商 structured-output 字段）仅 44/72 返回 HTTP 200，44/72 通过 schema 校验；28/72 的调用在模型运行之前就被厂商以 HTTP 400 拒绝，拒绝的是 schema 本身而非模型输出 [^13]。gemini-3.1-flash-lite 端点 18/18 全部返回 400，错误信息为 `Invalid JSON payload received. Unknown name "additionalProperties"`，即 draft 2020-12 的 `additionalProperties: false` 无法原样提交给该端点 [^13]。三个 Groq 端点（openai-gpt-oss-120b、openai-gpt-oss-20b、qwen3.6-27b）在 OpenAI 风格 strict 模式下拒绝含可选属性的 schema，错误信息为 `` `required` … must include every key in properties ``，`s5_optional_and_null` 这个 schema 被全部三个端点拒绝 [^13]。native_sanitised 模式达到 71/72 通过（唯一失败是一次 HTTP 503，与 schema 无关），但该清洗过程是减法式的：72 个 schema 中有 54 个被改动，删除了 `minimum`（36 行）、`maximum`（24 行）、`pattern`（24 行）、`additionalProperties`（18 行），并把 `required` 扩宽为包含全部属性（9 行）、把 `type: [x, "null"]` 改写为 nullable（3 行）[^13]。作者据此指出：0.986 的 schema 接受率是真实的，但其背后的约束执行并不真实——bounds、正则、封闭对象在到达厂商之前就已被删除；若依赖 structured-output 模式强制执行 `minimum`/`pattern`，必须自行实测而不能假设 [^13]。

这一实测与 OpenAI 官方示例的设计取向一致：示例 schema 中每个 object 层级都同时出现 `"required": [...]` 与 `"additionalProperties": false`，并与 `"strict": True` 一起放在 `json_schema` 定义内 [^24]；Azure OpenAI 的官方示例同样在嵌套 object 与顶层 object 上均显式写出 `"additionalProperties": false`，且 `required` 数组列出该层全部属性 [^25]。换言之，strict 模式要求的是「全属性必填 + 封闭对象」这一受限子集，而非任意 JSON Schema。

提示词约束一侧则没有任何机制性保证，其可靠性完全由模型行为决定，且已被观测到在弱模型上近乎失效 [^5]。

### 3.2 对推理能力的影响：约束可能伤害推理

CRANE 论文指出此前工作已实证观察到：严格强制形式约束常常削弱 LLM 的推理能力 [^1]。论文给出理论解释：把 LLM 输出约束到只允许语法合法最终答案的极强限制性文法，会降低模型的推理能力 [^1]。其主张是通过在输出文法中增补精心设计的额外规则，总是可以在保证输出语法与语义正确的同时保留 LLM 的推理能力，并据此提出 reasoning-augmented constrained decoding 算法 CRANE，在约束生成的正确性与无约束生成的灵活性之间做平衡 [^1]。在多个开源 LLM 与基准上，CRANE 同时优于 SOTA 约束解码策略与标准无约束解码，在 GSM-symbolic 与 FOLIO 等困难符号推理基准上相对基线最高提升 10 个百分点准确率 [^1]。

另一条相关证据来自格式修复的归因分析：在 Qwen3.5 (0.8B–9B)、Gemma-4-12B 及两个 API 前沿模型上，用 29 个主实验单元加一个前沿实验组，把 always-revise 的准确率变化分解为 content margin（两个答案都可解析）与 format-recovery/loss margin（可解析性变化）；在 12 个存在显著不可解析答案率的单元中，格式效应超过内容效应（Wilcoxon p=1.7e-3）[^4]。为做因果检验，论文强制已生成的推理过程经过 grammar-constrained decoding，使每个答案按构造都可解析：在 14 个单元中，这弥合了朴素总效应与 content-margin 估计之间差距的中位数 71% [^4]。聚类模型显示 floor-scale（0.8B/2B）模型发生内容层面改变与伤害的几率远高于 capable-scale 模型（p<1e-7）[^4]。前沿模型检查显示格式主导性随规模增强：5 个单元中 content margin 精确为零，尽管总效应最高达 +0.275（该实验组统计功效较低）[^4]。

这两条证据共同指向一个非对称结论：约束解码在**格式指标**上几乎必然改善，但在**内容指标**上可能造成退化，且退化幅度与模型规模、约束强度相关。提示词约束一侧缺少同条件的对照数据——现有证据只有 grammar-constrained decoding 一侧的推理退化实测 [^1][^4]，以及一个单日单次、机器生成、无人类核查的第三方仓库 [^13]。

### 3.3 性能代价：不是免费的

JSONSchemaBench 从三个维度评估约束解码方法：生成符合约束输出的效率、对多样约束类型的覆盖率、以及生成输出的质量 [^19]。基准包含 10K 个真实世界 JSON Schema，与官方 JSON Schema Test Suite 配对使用 [^19]。效率指标定义为 GCT（语法编译时间）、TTFT（首 token 时间）、TPOT（每输出 token 时间），表中数值均为样本中位数 [^20]。实验配置为 Llama-3.1-8B-Instruct 配合 Llamacpp 推理引擎作为 Outlines、Guidance、Llamacpp 的后端，因 XGrammar 不支持 Llamacpp 后端而额外增加 Hugging Face Transformers 后端实验；硬件为单张 NVIDIA A100-SXM4-80GB GPU、AMD EPYC 7543（12 核）CPU，batch size 设为 1 [^20]。

LlamaCpp 后端下的实测（GCT/TTFT/TPOT）：GlaiveAI 上 LM only 为 NA/0.10s/15.40ms，Guidance 0.00/0.24/6.37，Llamacpp 0.05/0.20/29.98，Outlines 3.48/3.65/30.33；GitHub Easy 上 Guidance 0.00/0.34/7.44，Llamacpp 0.05/0.18/27.22，Outlines 3.71/3.97/39.78；Snowplow 上 Guidance 0.00/0.28/6.55，Llamacpp 0.05/0.20/28.90，Outlines 3.91/4.14/42.66；GitHub Medium 上 Guidance 0.01/0.54/7.57，Llamacpp 0.06/0.30/29.08，Outlines 8.05/8.38/46.57；Kubernetes 上 Guidance 0.01/0.45/9.47，Llamacpp 0.05/0.28/28.04，Outlines 5.29/5.55/46.10 [^20]。HF Transformers 后端下：GlaiveAI 上 Guidance 0.01/0.36/36.92 对比 XGrammar 0.12/0.30/66.78；GitHub Easy 上 Guidance 0.01/0.37/42.03 对比 XGrammar 0.11/0.33/65.57；GitHub Medium 上 Guidance 0.01/0.55/44.21 对比 XGrammar 0.20/0.48/65.51；GitHub Hard 上 Guidance 0.01/0.73/35.88 对比 XGrammar 0.30/0.65/65.20 [^20]。论文指出 Outlines 与 Llamacpp 的吞吐显著低于 LM-only 方案，而 Guidance 通过 guidance acceleration 对部分生成步骤做 fast-forwarding，效率甚至更高；在 HF Transformers 后端下 Guidance 的 TPOT 显著优于 XGrammar [^20]。

论文给出的优化方向是：mask 计算可与 LM 前向传播并行、语法编译可与 prefill 计算并发，以及语法缓存与基于约束的投机解码 [^20]。三条主要发现为：约束解码相比无约束解码可将生成过程加速 50%；各框架对真实世界 JSON Schema 的实际支持差异显著，最佳框架支持的 schema 数量是最差框架的两倍；约束解码持续提升下游任务表现最多 4%，即使在 GSM8k 这类结构极少的任务上也是如此 [^20]。第 5 节 Coverage 开篇指出：每个约束解码框架在把 JSON Schema 翻译为能可靠保证 LM 输出有效性的约束集合时都存在局限，为系统评估，论文定义了三种 coverage 概念 [^20]。

XGrammar 论文报告相对已有方案最高 100x 加速，与推理引擎结合后，端到端低并发 LLM serving 下可实现 near-zero overhead 的结构化生成 [^8]。但该摘要层面未指明被比较的具体引擎名称与测试条件 [^27]。

### 3.4 适用边界：API 不可用与后端差异

GCD 可应用于任何自回归语言模型，前提是能访问每一步解码时对词表的分布；由于 OpenAI 等 API 服务不提供词表分布访问，因此无法与 GCD 配合使用 [^15]。这是约束解码与提示词约束在部署形态上的硬边界：闭源 API 只能走厂商提供的 structured-output 通道或纯提示词约束，前者受厂商 schema 子集限制 [^13][^29]，后者无机制保证 [^5]。

后端选择本身也影响结果。llama.cpp 的 server 特性列表同时包含「Schema-constrained JSON response format」与「Speculative decoding」，二者被并列为独立特性，但该 README 未说明 grammar/JSON schema 约束与 speculative decoding 同时启用时的行为或限制 [^30]；speculative decoding 实现中定义 `SPEC_VOCAB_MAX_SIZE_DIFFERENCE` 为 128，并要求 draft 模型词表与目标模型词表足够接近，否则拒绝使用投机解码，该文件正文未出现 grammar/GBNF 相关逻辑，未直接陈述文法约束与投机解码的兼容性 [^31]。

### 3.5 实践中的组合策略

证据支持的做法是把两者叠加而非二选一：

1. **以约束解码兜底格式**，同时在 prompt 中描述 schema 语义。vLLM 文档建议即使有 schema 约束也应在 prompt 中说明 JSON schema 与字段填充方式 [^12]；llama.cpp 文档指出 schema 不注入 prompt、模型看不到 schema，需在 prompt 中显式描述期望结构 [^29]。
2. **对需要语义正确性的任务，考虑 reasoning-augmented 约束**。CRANE 的出发点正是严格形式约束削弱推理，其方案是在输出文法中增补额外规则以保留推理能力 [^1]。
3. **对弱模型不要依赖提示词约束**。PTC-Decoder 的消融显示，移除 TC-Decoder（对工具名施加 token 级硬约束的确定性有限自动机）后所有质量指标显著下降且计算成本未降低，确认 TC-Decoder 是主要驱动因素 [^5]；该框架在 7 个 SLM、200 个真实遥感卫星任务上平均总分提升 +1.21（p<0.01，95% CI [+1.13, +1.29]）[^5]。
4. **对 schema 特性做实测而非假设**。第三方 conformance 实测的结论是：若依赖 structured-output 模式强制执行 `minimum`/`pattern`，必须自行实测而不能假设 [^13]；llama.cpp 文档建议用 `test-gbnf-validator` 工具针对字符串测试文法，示例调用为 `./llama-cli -m <model> --grammar-file grammars/some-grammar.gbnf -p 'Some prompt'` [^29]。
5. **注意 schema 的 strict 子集约束**。OpenAI 与 Azure 的官方示例均采用「每层 `additionalProperties: false` + `required` 列全属性」的写法 [^24][^25]，这与第三方实测中 Groq 端点拒绝含可选属性 schema 的行为一致 [^13]。

---

## 四、矛盾与不确定性

**（1）「100% 合法」的自述与实测冲突。** XGrammar README 的「100% structural correctness」[^7][^17] 与 OpenAI 的「保证模型总是生成符合所提供 JSON Schema 的响应」[^24] 均为厂商自述，且与两类实测证据冲突：llama.cpp 文档承认只支持 JSON Schema 子集、不支持的特性被静默跳过 [^29]；第三方 conformance 实测显示 native structured-output 模式下 `minimum`/`pattern`/`additionalProperties` 在到达厂商前就被删除 [^13]。现有证据中缺少一个权威来源正面界定「保证」的边界——即保证的是 token 级文法合法性，不保证语义正确、不保证 schema 全特性被强制执行。该边界只能从上述反面证据推断，而非由任一来源直接陈述。

**（2）tokenization 边界与 schema 属性顺序敏感性缺少权威表述。** 这是约束解码的已知理论局限，但 31 条证据中无一条正面覆盖。Geng et al. 2023 只提到 BPE 等分词方法允许同一字符串有多种切分（例如 `"[[["` 可切为 `"[[ ["`、`"[[["` 或 `"[ [ ["`），造成 token 级文法与字符级文法之间的一对多映射；作者用加空格并强制单括号为 token 的方式规避，但明确称该做法「不 principled、依赖特定 tokenizer」，并把 token 级文法的原则化定义列为未来工作 [^15]。Willard & Louf 只给出 FSM 索引构造 [^14]。XGrammar README 明确无 limitations 章节，未提及 tokenization 边界导致的不可达字符串、JSON Schema 属性顺序敏感性、或 UTF-8 多字节字符与正则约束冲突 [^17]。Guidance 文档目录中存在「Token healing」教程页与「Prompt Boundaries and Token Healing」文章，说明该库将 token 边界问题作为独立主题处理，但抓取到的首页仅为目录/安装页，未给出 token healing 的具体机制说明或示例正文，不能作为实质证据 [^18]。**该缺口未被填补，现有公开资料不足以确认这两个局限的权威表述。**

**（3）Outlines 性能结论方向相反。** Outlines 官方博客声称结构化生成「comes at virtually no cost during inference」、「Generating structured text is free」，并提出 coalescence 机制声称带来 5x 加速 [^28]；JSONSchemaBench 实测 Outlines 的 GCT 为 3.48–8.05s、TPOT 为 30–46ms，且显著慢于 LM-only 方案 [^20]。两个来源对同一实现的性能结论方向相反，且未在任何证据中被解释或调和。可能源于 coalescence 版本差异、后端差异或 schema 复杂度差异，但证据中无说明。**这是 blocking 级一致性冲突，报告若直接引用任一方都会误导。** 需注意博客本身属厂商自述、不是独立第三方评测 [^28]，而 JSONSchemaBench 的测量条件是 Llama-3.1-8B-Instruct + Llamacpp 后端、A100、batch size 1 [^20]，两者条件不可直接对齐。

**（4）XGrammar 加速主张缺少可对齐的第三方数据。** XGrammar 论文摘要给出「up to 100x speedup over existing solutions」，但摘要层面未指明被比较的具体引擎名称与测试条件 [^27]；仅取得摘要页元数据，未读取正文结果表格，因此无法确认其对比基线是否包含 Outlines / Guidance / llama.cpp GBNF，也无法确认是否报告首 token 开销 [^27]。JSONSchemaBench 的 HF Transformers 后端实测中 XGrammar 的 TPOT（65–67ms）反而显著差于 Guidance（36–44ms）[^20]，与「最快引擎」自述方向不一致。XGrammar 论文作者为引擎作者团队，属厂商/作者自述而非独立第三方评测 [^27]。**缺少可对齐的第三方横向数据。**

**（5）约束解码对下游任务的影响方向不一致。** JSONSchemaBench 报告约束解码持续提升下游任务表现最多 4%，即使在 GSM8k 这类结构极少的任务上也是如此 [^20]；CRANE 报告严格强制形式约束常常削弱 LLM 的推理能力 [^1]。两条结论表面矛盾，证据中未给出调和说明。可能因任务类型、约束强度、模型规模不同，但证据中无直接解释。报告若并列引用而不区分条件会自相矛盾。相关地，Calibration Floor 的分解显示格式效应在中低规模模型上可超过内容效应 [^4]，这为「约束解码改善格式指标但可能伤害内容指标」提供了机制性线索，但仍不足以调和上述两条结论。

**（6）提示词约束一侧缺少同条件对照数据。** 现有证据中，「强约束导致准确率下降/推理退化」的实测只有 grammar-constrained decoding 一侧 [^1][^4]，以及一个单日单次、机器生成、无人类核查的第三方仓库 [^13]。该仓库作者自陈的方法学局限为：单日单次运行、4 个端点 2 家厂商、6 个手写 schema、每格 3 次试验（足以发现硬性 400，远不足以估计稀有失败率），厂商行为随时变化，所有数字应视为有日期的一次测量而非当前状态；并声明这不是排行榜、不是对模型质量的断言；该仓库自述由机器运行的机构产出，无人类撰写或核查这些数字 [^13]。**缺少厂商官方文档对 JSON mode 无语法约束的遵循率说明，也缺少 prompt-only vs constrained 的同条件对照。**

**（7）扩散语言模型的约束建模不适用于自回归场景。** 有论文主张扩散语言模型（dLLM）按任意顺序预测被 mask 的位置，其精确约束解码器仍把约束编码为序列语言，对关系型约束会指数级膨胀（对 same-order copy，任何有限自动机都需要 $4^k$ 个状态，任何上下文无关文法的大小为 $2^{\Omega(k)}$），因此约束应建模为图（factor graph）而非链 [^3]。该结论针对 dLLM 的任意顺序预测特性，**不能直接外推到自回归 LLM 的约束解码**；本报告讨论的掩码机制 [^14][^15] 均建立在自回归逐 token 生成的前提上。

**（8）功能调用场景的评测口径差异。** BFCL V4 的 Overall Accuracy 定义为所有子类别的未加权平均，榜单区分 FC（原生支持 function/tool calling）与 Prompt（用模型普通文本生成能力绕过 function calling）两种接入方式，Format sensitivity 测试用例仅对 prompt（非 FC）模型提供支持 [^21]。BFCL v1 博客指出在 simple function calling 上开源与闭源模型表现相当，但在 multiple 与 parallel 调用上 GPT 系列明显优于开源对手 [^23]；该博客设有「Common Mistakes」小节讨论常见失败模式，但该小节正文在本次抓取中被截断，未取得逐类错误占比数字 [^23]。**因此「约束解码 vs 提示词约束」在功能调用任务上的量化差异，现有证据不足以给出。**

## 参考来源

[^1]: CRANE: Reasoning with constrained LLM generation — https://arxiv.org/abs/2502.09061（arxiv, 2025-02-13）
[^2]: Breaking the Space Barrier and its Application to Language Model Inference — https://arxiv.org/abs/2610.09139（arxiv, 2026-10-06）
[^3]: Constraints Are Graphs, Not Chains: Exact Decoding for Diffusion Language Models — https://arxiv.org/abs/2609.32900（arxiv, 2026-09）
[^4]: The Calibration Floor: Format Repair Can Masquerade as Self-Correction at Small-to-Mid Scale — https://arxiv.org/abs/2608.04355（arxiv, 2026-08-04）
[^5]: PTC-Decoder: Towards Intelligent SLMs on Offline Resource-Constrained Edge Devices — https://arxiv.org/abs/2609.30836（arxiv, 2026-09-25）
[^7]: mlc-ai/xgrammar — Efficient, Flexible and Portable Structured Generation (README) — https://github.com/mlc-ai/xgrammar（github, 2024-06-28）
[^8]: XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models (arXiv:2411.15100v3) — https://arxiv.org/abs/2411.15100v3（arxiv, 2024-11-22）
[^9]: dottxt-ai/outlines — Structured outputs for LLMs (README) — https://github.com/dottxt-ai/outlines（github, 2023-03-17）
[^10]: guidance-ai/guidance — efficient programming paradigm for steering language models — https://github.com/guidance-ai/guidance（github, 2022-11-10）
[^11]: eth-sri/lmql — A programming language for large language models — https://github.com/eth-sri/lmql（github, 2022-11-24）
[^12]: vLLM 官方文档 — Structured Outputs — https://docs.vllm.ai/en/latest/features/structured_outputs.html（web）
[^13]: LLM structured-output conformance — 216 measured API calls, 2026-08-20 (YtinuMoc/llm-structured-output-conformance) — https://github.com/YtinuMoc/llm-structured-output-conformance（github, 2026-08-27）
[^14]: Efficient Guided Generation for Large Language Models (Willard & Louf, 2023) — https://arxiv.org/html/2307.09702v4（arxiv, 2023-07-19）
[^15]: Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning (Geng et al., EMNLP 2023) — https://arxiv.org/html/2305.13971v6（arxiv, 2023-05-23）
[^16]: PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models (Scholak et al., 2021) — https://arxiv.org/html/2109.05093v1（arxiv, 2021-09-10）
[^17]: mlc-ai/xgrammar — README (Efficient, Flexible and Portable Structured Generation) — https://raw.githubusercontent.com/mlc-ai/xgrammar/main/README.md（github）
[^18]: Guidance 官方文档首页（Install — Guidance latest documentation） — https://guidance.readthedocs.io/en/latest/（web）
[^19]: JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models (arXiv:2501.10868v3) — https://arxiv.org/abs/2501.10868（arxiv, 2025-01-18）
[^20]: JSONSchemaBench (arXiv:2501.10868v3) — 正文：数据收集、数据集构成与效率实验设置 — https://arxiv.org/html/2501.10868v3（arxiv, 2025-02-27）
[^21]: Berkeley Function Calling Leaderboard (BFCL) V4 — 官方排行榜页面 — https://gorilla.cs.berkeley.edu/leaderboard.html（web, 2026-04-12）
[^23]: Blog 8: Berkeley Function-Calling Leaderboard（BFCL v1 方法与失败模式） — https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html（web, 2024-08-19）
[^24]: openai/openai-cookbook — Introduction to Structured Outputs (notebook) — https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/Structured_Outputs_Intro.ipynb（github）
[^25]: Microsoft Learn — Azure OpenAI: How to use structured outputs — https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs（web）
[^27]: XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models (arXiv:2411.15100, MLSys '25) — https://arxiv.org/abs/2411.15100（arxiv, 2024-11-22）
[^28]: Coalescence: making LLM inference 5x faster — .txt (Outlines) 官方博客 — https://blog.dottxt.co/coalescence.html（web）
[^29]: llama.cpp grammars/README.md — GBNF Guide（官方文法文档） — https://raw.githubusercontent.com/ggml-org/llama.cpp/master/grammars/README.md（github）
[^30]: llama.cpp tools/server/README.md — LLaMA.cpp HTTP Server 特性列表 — https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md（github）
[^31]: llama.cpp common/speculative.cpp — speculative decoding 实现（draft 词表约束） — https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/speculative.cpp（github）