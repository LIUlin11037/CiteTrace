# 上下文工程（Context Engineering）：定义、边界与同提示词工程的关系

## 摘要

上下文工程指在 LLM 推理期间，对进入有限上下文窗口的 token 集合进行策展与维护，以在模型固有约束下最大化达成期望结果的概率[^1]。Anthropic 将其定义为「从不断演化的可能信息宇宙中，策展哪些内容进入有限上下文窗口的艺术与科学」[^1]；LangChain 将其定义为「在 agent 轨迹的每一步，用恰到好处的信息填充上下文窗口的艺术与科学」[^2]；中科院等机构的 166 页综述则把它上升为「一门正式学科」，涵盖对 LLM 信息载荷的系统性优化，明确「超越简单的 prompt design」[^3]。

三者共同点在于：**对象是「上下文窗口内的全部 token」，而非仅指令文本**。Anthropic 给出的组成清单包括 system instructions、tools、MCP、external data、message history 等「整个上下文状态」[^1]；LangChain 用 umbrella 框架覆盖 Instructions、Knowledge、Tools 三类上下文[^2]。

与提示词工程的关系，一手来源的表述高度一致且**不支持「取代论」**：Anthropic 称其为 prompt engineering 的「自然演进」（natural progression），并明确区分二者——prompt engineering 是撰写与组织 LLM 指令（尤其 system prompt）的方法，context engineering 是推理期间策展与维护最优 token 集合的策略，包含 prompt 之外所有可能进入上下文的信息[^1]；Cognition 称其为 prompt engineering 的「下一个层次」，重点是在动态系统中自动完成这件事[^26]。全部 40 条证据中，无任何一手来源出现「提示词工程已死」或「已被取代」的表述[^1][^26][^27]。

---

## 一、定义：上下文工程究竟在工程什么

### 1.1 对象定义：context = 采样时被包含的那组 token

Anthropic 的定义最为操作化：context 是「采样 LLM 时被包含的那组 token」，因此工程问题是「在 LLM 固有约束下优化这些 token 的效用，以稳定达成期望结果」[^1]。这个定义的关键在于它把边界画在**采样时刻的完整输入**上，而不是画在「人写的那段话」上。

由此推出的组成要素清单是：system instructions、tools、Model Context Protocol (MCP)、external data、message history 等，即「整个上下文状态」[^1]。LangChain 的分类与之呼应但更偏工程分层，把上下文分为 Instructions（prompts、memories、few-shot examples、tool descriptions）、Knowledge（facts、memories）、Tools（tool calls 的反馈）三类[^2]。

### 1.2 目标函数：最小高信号 token 集合

Anthropic 给出的指导原则是：好的 context engineering 是找到「最小的高信号 token 集合」以最大化期望结果的概率[^1]。LangChain 的表述是「用恰到好处的信息填充上下文窗口」[^2]。二者指向同一目标函数——不是「塞更多」，而是「在有限预算下最大化单位 token 的信息增益」。

### 1.3 学科化定义：超越 prompt design 的系统性优化

综述（arXiv:2507.13334，Mei 等，中科院计算所等，v1 2025-07-17、v2 2025-07-21，166 页，引用 1411 篇文献）把 Context Engineering 界定为「一门正式学科（a formal discipline），超越简单的 prompt design，涵盖对 LLM 信息载荷（information payloads）的系统性优化」[^3]。其两层分类法为：

**基础组件（三项）**：(1) Context Retrieval and Generation，含 prompt-based generation 与 external knowledge acquisition；(2) Context Processing，含长序列处理、self-refinement、结构化信息整合；(3) Context Management，含记忆层级、压缩、优化[^3]。

**系统实现（四项）**：(1) Retrieval-Augmented Generation，含模块化、agentic、图增强架构；(2) Memory Systems，支持持久交互；(3) Tool-Integrated Reasoning，函数调用与环境交互；(4) Multi-Agent Systems，通信与编排协调[^3]。

综述明确指出该领域研究高度碎片化、各子方向被孤立研究，因此需要统一框架来组织技术、澄清原理并说明相互依赖[^3]。其配套仓库自述定位为「from prompt engineering to production-grade AI systems」，约 3330 stars、292 forks，最近推送 2026-05-28[^4]。

### 1.4 为什么需要它：三条约束来源

**(a) 注意力预算与 n² 关系。** Anthropic 用「attention budget」解释约束来源：transformer 全注意力带来 n² 的成对关系，上下文变长时捕获能力被摊薄；且训练数据中短序列更常见，模型对长程依赖经验更少[^1]。

**(b) context rot。** Anthropic 提出该概念：随上下文窗口 token 数增加，模型准确召回信息的能力下降；该特性在所有模型上都出现，只是退化坡度不同[^1]。

**(c) 位置偏置的实证。** Lost in the Middle 通过受控实验系统性改变上下文长度与相关信息位置，观察到特征性的 U 形性能曲线：信息位于开头（首因偏置）或末尾（近因偏置）时性能最高，位于中间时显著退化[^19]。具体而言，当相关信息置于输入上下文中间时，GPT-3.5-Turbo 在多文档问答上的表现低于不给任何文档的闭卷设置（后者 56.1%）[^19]。该研究覆盖 MPT-30B-Instruct、LongChat-13B (16K)、GPT-3.5-Turbo、Claude-1.3[^19]，并发现扩展上下文模型常与其非扩展版本表现相同，说明扩展上下文模型不一定更善于利用其输入上下文[^19]。论文已被 TACL 2023 接收[^18]。

**(d) 长上下文不是充分替代。** MemGPT 论证：直接扩展 transformer 上下文长度会因自注意力带来计算时间与内存的二次增长，且已有研究表明长上下文模型难以有效利用额外上下文[^22]。Mem0 进一步给出两条理由：真实人机关系跨越数周数月必然超出任何上下文上限；真实对话很少保持主题连续性，全上下文方案必须在上千 token 的无关内容中推理，关键偏好被淹没[^24]。

---

## 二、与提示词工程的关系：演进而非替代

### 2.1 三个一手来源的一致表述

| 来源 | 对 prompt engineering 的界定 | 对二者关系的表述 |
|---|---|---|
| Anthropic[^1] | 撰写与组织 LLM 指令（尤其 system prompt）的方法 | context engineering 是其「自然演进」（natural progression），包含 prompt 之外所有可能进入上下文的信息 |
| Cognition[^26] | 「为让任务以理想格式呈现给 LLM 聊天机器人而付出努力」所创造的术语 | context engineering 是「the next level of this」，重点是在动态系统中自动完成这件事 |
| LangChain[^2] | 未做直接对立表述 | 把 context engineering 当作 umbrella，Instructions 类目下仍包含 prompts、few-shot examples、tool descriptions |

Anthropic 在《Building effective agents》正文中亦未出现「prompt engineering is dead」或「已被 context engineering 取代」的表述[^27]。因此，若以「厂商宣称提示词工程已死」为框架组织论述，将与已收集证据直接冲突。

### 2.2 差异的四个可操作维度

**(1) 范围：单条指令 vs 整个 token 状态。** prompt engineering 的作用域是 system prompt 与用户指令的组织；context engineering 的作用域是 system instructions、tools、MCP、external data、message history 的全体[^1]。LangChain 的 umbrella 框架把 prompts 降为 Instructions 类目下的一个子项[^2]。

**(2) 时间性：离散任务 vs 迭代过程。** Anthropic 明确对比：与写 prompt 这一离散任务相对，context engineering 是迭代性的，每次决定传给模型什么时都要重新做策展[^1]。Cognition 的对应表述是「在动态系统中自动完成这件事」[^26]。

**(3) 约束来源：表达质量 vs 注意力预算。** prompt engineering 的失败模式主要是表达歧义与格式不当——Mind Your Format 在 21 个模型（770M–70B）、4 个分类数据集上发现，模板选择不当可使最强模型与最强推理方法的性能下降到随机猜测水平，且最优模板无法跨设置迁移、甚至无法在同一模型家族的不同模型间迁移[^38]。context engineering 的失败模式则来自注意力预算、context rot 与位置偏置[^1][^19]。

**(4) 责任归属：写作环节 vs 系统架构。** Cognition 称 context engineering「需要更多细微考量，并且实际上是构建 AI agent 的工程师的第一号工作」[^26]；LangChain 转引同一说法为「#1 job」[^2]。这一定位把责任从「写提示词的人」转移到「设计 agent 上下文流的人」。

### 2.3 提示词工程并未失效：两类反证

**其一，提示策略的收益仍可测量且可被自动化超越。** PE2 在 meta-prompt 中注入详细描述、上下文规格说明、逐步推理模板三个组件，自动搜索到的提示词在 MultiArith 上比「let's think step by step」高 6.3%、GSM8K 上高 3.1%、反事实任务上比竞争基线高 6.9%[^11]。这说明手工通用启发式提示并非最优，但优化对象仍是提示词本身。

**其二，提示策略无法弥补模型能力差距。** TheoremQA 用 800 道覆盖 350 个定理的专家标注题目评估 16 个模型在不同提示策略（含 CoT 与 Program-of-Thoughts）下的表现：GPT-4 配合 Program-of-Thoughts 达 51% 为当时最高，而所有开源模型均低于 15%，仅勉强超过随机猜测基线[^10]。提示工程有明确上限。

**其三，提示干预的效果不稳定。** SimpleSafetyTests 用 100 条覆盖 5 类危害领域的测试提示测试 11 个开源/开放获取模型与 4 个闭源模型，多数模型对超过 20% 的提示给出不安全回复，极端情况超过 50%；在提示前添加强调安全的 system prompt 可显著降低但不能完全阻止不安全回复——同一 system prompt 干预在不同模型上的效果不一致、不彻底[^39]。

### 2.4 谱系位置：提示词工程是上下文工程的一个真子集

综合上述，二者不是替代关系而是包含关系：prompt engineering 处理「指令文本如何组织」，是 context engineering 的一个真子集；context engineering 额外处理工具定义、外部数据、消息历史、记忆与压缩策略。综述的定位与此一致——「超越简单的 prompt design」，但并未宣称 prompt design 不再是其中一环[^3]。另有仓库将 Context Engineering 与 Prompt Engineering、Ontology Engineering、Graph Engineering、Loop Engineering、Harness Engineering 并列为同一综述谱系中的并列概念[^40]，但仅获得元数据与简介、未读取正文，无法据此提供术语起源或时间线信息。

---

## 三、工程实践：四类策略与代表性机制

LangChain 把常见策略归为四类：**write**（写出到上下文窗口之外，如 scratchpad、跨会话 memory）、**select**（把相关内容拉入上下文窗口）、**compress**（压缩）、**isolate**（隔离）[^2]。以下按此框架组织证据。

### 3.1 Write：把状态移出窗口

**MemGPT 的虚拟上下文管理。** 论文明确提出「把上下文窗口当作一种受约束的记忆资源」，借鉴操作系统虚拟内存分页机制，在主内存（上下文窗口）与磁盘（外部存储）之间搬运数据，在固定上下文模型上制造「无限上下文」的假象[^22]。主上下文被划分为三段：只读 system instructions、固定大小可读写的 working context（存关键事实与用户偏好）、存放滚动消息历史的 FIFO queue[^22]。外部上下文对应 archival storage（任意长度文本对象）与 recall storage（消息数据库）[^22]。

溢出控制由队列管理器实现：prompt tokens 超过 warning token count（如窗口 70%）时插入系统消息发出 memory pressure 警告，让 LLM 有机会用函数把重要信息写入 working context 或 archival storage；超过 flush token count（如 100%）时驱逐约 50% 窗口的消息并生成新的递归摘要[^22]。被驱逐消息永久保存在 recall storage，可通过函数调用读回并重新追加到队列尾部[^22]。记忆编辑与检索完全由 LLM 自主发起（self-directed）[^22]。

**A-MEM 的 Zettelkasten 结构化。** 论文把现有记忆系统缺陷定位为「需要开发者预定义存储结构、指定工作流中的存储点、设定检索时机」，固定操作与结构限制跨任务适应性[^23]。A-MEM 以「原子笔记 + 灵活链接」构建互联知识网络，每条笔记为结构化元组 m_i = {c_i, t_i, K_i, G_i, X_i, e_i, L_i}，即原始内容、时间戳、LLM 生成的关键词、标签、上下文描述、稠密向量嵌入、链接集合[^23]。链接生成基于嵌入余弦相似度检索 top-k 历史笔记，无需预定义规则[^23]；memory evolution 使新记忆写入时可触发对既有记忆的上下文表示与属性更新[^23]。该论文将 MemGPT 归类为「利用类缓存架构优先保留近期信息」，并批评此类方法受预定义结构与固定工作流约束、在新环境中泛化差[^23]。

**Mem0 的提取+更新流水线。** 以消息对 (m_{t-1}, m_t) 为输入，结合数据库中的对话摘要 S 与最近 m 条消息的近期窗口做记忆提取；更新阶段把新提取记忆与相似既有记忆比对，通过 Tool Call 施加增/改/删/合并操作[^24]。Mem0^g 变体把记忆存为有向标记图（实体为节点、关系为边）以支持跨多条记忆的关系路径推理[^24]。报告结果：LLM-as-a-Judge 指标相对 OpenAI 提升 26%；图记忆版本整体分数比基础版高约 2%；相比全上下文方案 p95 延迟降低 91%、token 成本节省超过 90%[^24]。

**Generative Agents 的记忆流。** 论文把「管理不断增长的记忆」设为架构层面核心问题[^25]。memory stream 以自然语言记录 agent 完整体验清单；检索模型把 relevance、recency、importance 三者结合，在记忆量远超上下文容量时做选择性检索而非全量注入[^25]。reflection 把记忆随时间综合为更高层推断，planning 把推断转为行动计划并递归细化，反思与计划又写回 memory stream[^25]。消融实验证明 observation、planning、reflection 各组件对行为可信度均有关键贡献[^25]。

**工程侧的 memory 选择惯例。** LangChain 举例：Claude Code 使用 CLAUDE.md，Cursor 与 Windsurf 使用 rules 文件保存「程序性」记忆；ChatGPT 则从大量用户记忆集合中检索，常用 embedding 和/或知识图谱做记忆索引[^2]。

### 3.2 Select：按需拉入

**ratel 的渐进式披露。** README 以「The context engineering layer for AI agents」自我定位，核心主张是只选择与每一轮对话相关的工具与技能，以挽回因工具过载损失的准确率并降低每次调用成本[^30]。问题陈述为：每个工具 schema、技能与系统提示中的指令都是每次调用都要付费的 token，上下文增长会使模型变差、选错选项并偏离任务[^30]。机制是把工具与技能索引为 catalog，由 agent 渐进式披露，按轮次检索并只注入匹配能力；常驻事实（如店铺地址、品牌语气）注册为 facts 推入上下文，仅在转录中不再新鲜时重新注入[^30]。默认检索算法为 BM25，作用于工具名称/描述/参数名与技能名称/描述/标签，语义与混合排序可按 catalog 或按调用启用[^30]。README 宣称「No vector DB, no infra.」[^30]，并指向 benchmark.ratel.sh 作为完整结果出处，但本次抓取被截断，未读到具体基准分数[^30]。

**Anthropic 对工具契约的告诫。** Anthropic 指出工具定义了 agent 与信息/行动空间之间的契约，常见失败模式是工具集臃肿、功能重叠导致选择歧义[^1]。这与 ratel 的问题陈述同源。

**检索的边际收益递减。** Lost in the Middle 的开放域案例研究显示：在 NaturalQuestions-Open 上从 Wikipedia 检索时，模型性能远早于检索器召回率饱和；把检索文档从 20 篇增至 50 篇仅带来约 1.5%（GPT-3.5-Turbo）与约 1%（Claude-1.3）的边际提升[^19]。这为「select 不是越多越好」提供了定量支撑。

### 3.3 Compress：压缩

**LLMLingua。** coarse-to-fine 提示压缩方法，由三部分组成：budget controller（为 instruction、demonstrations、question 等不同提示成分动态分配不同压缩率）、token-level iterative compression algorithm（建模被压缩内容之间的相互依赖）、基于 instruction tuning 的分布对齐方法（对齐目标 LLM 与用于压缩的小语言模型）[^20]。作者报告在 GSM8K、BBH、ShareGPT、Arxiv-March23 四个数据集上取得 state-of-the-art，支持最高 20x 压缩且性能损失很小[^20]。该方法不通过 LLM 反向传播梯度，因此支持仅通过 API 访问的 LLM 场景[^20]。论文指出基于 prompt tuning 学习特殊 token 的压缩方法通常针对特定任务定制、部分需微调整个模型，限制了适用场景；利用 LLM 做摘要以形成记忆/知识的方法需多次调用 LLM、成本较高[^20]。

**LongLLMLingua。** 针对长上下文场景的三个挑战（计算成本更高、性能下降、位置偏差）提出提示压缩，目标是提升 LLM 对关键信息的感知[^21]。在 NaturalQuestions 上于 GPT-3.5-Turbo 以约 4x 更少 token 带来最高 21.4% 性能提升；在 LooGLE 上实现 94.0% 成本降低；对约 10k token 提示在 2x–6x 压缩率下端到端延迟加速 1.4x–2.6x[^21]。该论文被 ACL 2024 接收[^21]。

**Cognition 的压缩实践。** 提到为压缩动作与对话历史，引入专门 LLM 做压缩，并称已在内部微调小模型做这件事[^26]。

### 3.4 Isolate：隔离与分区

**Anthropic 的 right altitude 与分区建议。** 对 system prompt 的建议是「right altitude」——介于硬编码脆弱 if-else 逻辑与过度笼统/假设共享上下文两个失败模式之间，并建议用 XML 标签或 Markdown 标题分区（如 `<background_information>`、`<instructions>`、`## Tool guidance`、`## Output description`）[^1]。

**Cognition 的隔离原则。** 提出两条原则：共享完整上下文与完整 agent 轨迹（而非单条消息）；动作携带隐含决策，冲突的决策带来坏结果[^26]。主张默认排除不遵守上述原则的 agent 架构，并称最简单的遵循方式是单线程线性 agent[^26]。批评 OpenAI 的 swarm 与微软 autogen 所推动的多 agent 架构是「错误的构建 agent 方式」[^26]。以 Claude Code 子 agent 为例：截至 2025 年 6 月，其子任务 agent 不与主 agent 并行工作，通常只负责回答问题而非写代码[^26]。

**Context Engineering Kit 的按插件加载。** 定位为手工打造的先进上下文工程技术与模式集合，强调最小 token 占用，目标是提升 agent 结果质量与可预测性[^32]。Token 高效的具体做法是优先面向命令的技能配合子 agent，而非泛化信息技能，以减少无关上下文填充[^32]。安装插件不会把 agent/skill 载入上下文，只有安装的插件才加载其特定 agents、commands、skills[^32]。工作流包含 `/reflect`（反思并改进）与 `/memorize`（把解决策略写入 project memory）等命令[^14]。元数据：★1786、forks 163、TypeScript、last push 2026-08-26[^32][^36]。

**OpenCE 的闭环五支柱。** 自我定位为构建闭环 Context Engineering 系统的可插拔元框架[^31]。核心主张：经典 RAG 栈是开环的（取一次上下文即回答），OpenCE 补上两个缺失支柱——Evaluation（用领域评估器自动给每次 LLM 响应打分，如 ACE Reflector、RAGAS）与 Evolution（把评估信号反馈进长期记忆/策略模块，如 ACE Curator、自适应 RAG 策略）[^31]。标准化五个接口：Acquisition (IAcquirer)、Processing (IProcessor)、Construction (IConstructor)、Evaluation (IEvaluator)、Evolution (IEvolver)[^31]。元数据：★333、forks 47、Python、last push 2025-11-14[^31][^37]。

### 3.5 失效模式清单

LangChain 归纳长上下文导致性能问题的四类失效模式（引自 Drew Breunig）：context poisoning（幻觉进入上下文）、context distraction（上下文压过训练）、context confusion（多余上下文影响回答）、context clash（上下文各部分相互矛盾）[^2]。并指出 agent 交错进行 LLM 调用与工具调用，长任务与累积的工具反馈会消耗大量 token，可能导致超出上下文窗口、成本/延迟膨胀或 agent 性能退化[^2]。

---

## 四、经济性：缓存与成本结构如何改变策展决策

**Anthropic prompt caching（2024-12-17 GA）。** 允许在 API 调用之间缓存频繁使用的上下文，官方称对长提示词可降低成本最高 90%、降低延迟最高 85%[^17]。定价机制：cache write 比该模型基础 input token 价格贵 25%；cache read 仅为基础 input 价格的 10%[^17]。以 Claude 3.5 Sonnet 为例：基础 input $3/MTok，cache write $3.75/MTok，cache read $0.30/MTok，output $15/MTok，上下文窗口 200K[^17]。实测用例：10 万 token 缓存提示的「与书对话」场景，首 token 延迟从 11.5s 降至 2.4s（-79%），成本降低 90%；many-shot prompting（1 万 token）延迟从 1.6s 降至 1.1s（-31%），成本降低 86%；长 system prompt 的 10 轮对话延迟从约 10s 降至约 2.5s（-75%），成本降低 53%[^17]。官方明确表示借助缓存可在提示词中放入「数十个」多样化高质量示例（而非仅几个），并可将完整知识库嵌入提示词[^17]。

**Bedrock 的实现约束。** 支持 Implicit Prompt Caching（服务端自动复用可缓存前缀）与 Explicit Prompt Caching（调用方用 cache checkpoint 显式标记）[^28]。隐式缓存的工程要求是把静态内容放在 prompt 开头、动态内容放在末尾以提高精确前缀匹配概率，且为 best-effort，重复相同 prompt 不保证命中[^28]。显式缓存的前缀必须在请求间保持静态，前缀一旦改变即 cache miss[^28]。cache checkpoint 有最小 token 门槛，当前各模型为 512 至 4,096 tokens，按每个断点之前的整个前缀累计计算（含 tools、system、messages 字段）[^28]。缓存有 TTL，每次成功命中会重置，许多模型支持 5 分钟、部分支持 1 小时[^28]。prompt caching 仅支持 on-demand 推理端点，不支持 batch inference API[^28]。

**对策展策略的含义。** 缓存机制把「上下文布局」变成一个显式的成本变量：静态前缀（system prompt、工具定义、few-shot 示例）与动态后缀（当轮消息）的排序直接决定命中率[^28]，而命中与否决定该部分 token 按 10% 还是 100% 计价[^17]。这与「最小高信号 token 集合」的原则[^1]并不矛盾，但增加了第二个优化轴——**同一组 token 的排列顺序也有价格**。

**时效性警告。** 唯一有具体数字的定价来源是 Anthropic 2024-12-17 的 GA 公告，距 2026-10-25 已近两年，且未覆盖 2025–2026 的模型代际与价格调整[^17]。Bedrock 定价页的具体价格数值以 JS 模板占位符形式返回（如 `{priceOf!bedrock/...}`），未渲染出实际数字，因此不能用于确认任何具体单价[^29]；其页面中 Nova 系列表格脚注写明 cache read input tokens 比 on-demand input token 价格低 75%（即约为 on-demand 的 25%）[^29]，与 Anthropic「cache read 为基础 input 的 10%」口径不一致且证据未解释差异来源。Bedrock 另提供 Standard、Flex、Priority、Reserved 四种服务层级，同一模型在不同层级价格不同；部分基础模型 batch inference 比 on-demand 低 50%[^29]。因此「2025–2026 年缓存与长上下文定价如何改变实践重心」这一问题，**现有公开资料不足以确认**。

---

## 五、生态：术语在工程社区的落地形态

### 5.1 以 context engineering 为产品定位的仓库

| 仓库 | stars / last push | 核心主张（已读 README） |
|---|---|---|
| ratel-ai/ratel[^30] | ★529 / 2026-10-23[^16] | 按轮次选择工具与技能，BM25 默认检索，宣称无向量库、约 80% token 减少[^30] |
| NeoLabHQ/context-engineering-kit[^32] | ★1786 / 2026-08-26[^36] | 面向 Claude Code 等的最小 token 占用技能市场，按插件加载[^32] |
| sci-m-wang/OpenCE[^31] | ★333 / 2025-11-14[^37] | 闭环 CE 元框架，五支柱接口，补 Evaluation 与 Evolution[^31] |
| volcengine/MineContext[^33] | ★5542 / 2026-05-07[^35] | 主动式上下文感知 AI 伙伴，含 Context Engineering Architecture 全生命周期[^33] |

MineContext 的中文版 README 与英文版一致地以「上下文工程」自我定位，明确写出「基于底层的上下文工程框架，主动推送洞察、日/周总结、待办、活动记录等高质量信息」；「上下文工程架构」条目描述为支持多模态、多源数据从捕获、处理、存储到管理、检索、消费的完整生命周期，支持生成六种类型的智能上下文[^34]。中文版标注「智能浮现」功能状态为「实现中」，英文版对应条目未标注该状态[^34]。隐私设计上默认所有数据本地存储，支持基于 OpenAI API 协议的自定义模型服务与全本地模型[^33]。关联项目 OpenViking 定位为面向 AI Agents 的开源上下文数据库，以「文件系统范式」统一管理记忆、资源与技能[^33]。

### 5.2 未采用该术语但功能重叠的框架

**LangGraph** 自我定位为「Low-level orchestration framework for building stateful agents」，README 正文未使用「context engineering」一词；核心能力为 durable execution、human-in-the-loop、comprehensive memory（短期工作记忆 + 跨会话长期持久记忆）、LangSmith 调试、生产级部署[^12]。灵感来自 Pregel 与 Apache Beam，公共接口受 NetworkX 启发，由 LangChain Inc 构建但可脱离 LangChain 独立使用[^12]。页面显示 43.9k stars、7.5k forks、7,211 commits、MIT[^12]。README 推荐上层包 Deep Agents 作为快速构建 agent 的入口[^12]。

**Letta（f.k.a. MemGPT）** 自我定位为「Build stateful agents with memory that can learn and improve over time」，README 正文未出现「context engineering」一词，定位围绕 stateful agents 与 memory 展开[^13]。当前源码已迁移至 letta-ai/letta-code 仓库，原仓库 archive 分支保留已退役的 Letta V1 API server[^13]。25.3k stars、2.7k forks、7,476 commits、Apache-2.0[^13]。

**DSPy** 自我定位为「Programming —not prompting—Foundation Models」，未使用「context engineering」一词；提供用于优化 prompt 与 weights 的算法，适用于简单分类器、复杂 RAG 流水线或 agent 循环[^15]。全称 Declarative Self-improving Python，主张以可组合的 Python 代码替代脆弱的 prompt[^15]。相关论文包括 GEPA（Jul'25）、Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs（Jun'24）、DSPy（ICLR 2024）[^15]。38.8k stars、3.5k forks、4,789 commits、MIT[^15]。

这一分布本身是一个值得注意的信号：**术语的采用与功能的实现并不同步**——LangGraph、Letta、DSPy 在功能上分别覆盖了 orchestration、memory、prompt 自动优化，但均未在 README 中自称 context engineering[^12][^13][^15]；而 ratel、OpenCE、Context Engineering Kit、MineContext 则把该术语写进了产品定位[^30][^31][^32][^33]。

### 5.3 术语扩散的其他迹象

GitHub 检索「context engineering」按 stars 排序的头部结果包括：dair-ai/Prompt-Engineering-Guide（★79502，简介同时涵盖 prompt engineering、context engineering、RAG 与 AI Agents）、gsd-build/get-shit-done（★64224）、muratcankoylan/Agent-Skills-for-Context-Engineering（★18210）、coleam00/context-engineering-intro（★13887，简介称 "Context engineering is the new vibe coding"）、jasontang-ai/Context-Engineering（★9263，简介引用 Karpathy 表述并称目标是「超越提示词工程，进入上下文设计、编排与优化的更广学科」）、Meirtz/Awesome-Context-Engineering（★3330）[^35]。另有检索结果显示 Windy3f3f3f3f/how-claude-code-works（★3778）、wquguru/harness-books（★3208）、EgoAlpha/prompt-in-context-learning（★2248）等[^16]。

**必须声明的覆盖缺口**：上述 stars 最高的几个仓库（coleam00/context-engineering-intro、muratcankoylan/Agent-Skills-for-Context-Engineering、jasontang-ai/Context-Engineering、dair-ai/Prompt-Engineering-Guide）仅有检索元数据与简介，未读 README 正文，其核心主张无法核验，而它们恰是社区影响力最大的样本[^35]。同样，ratel-ai/ratel、qf-studio/navigator、itallstartedwithaidea/ContextOS 等在检索结果中仅记录元数据，未逐一读取 README，其主张仅作候选线索[^16]。

---

## 六、尚未解决的技术问题

**理解与生成的不对称。** 综述识别出一个关键研究缺口：模型能力存在根本性不对称——在先进 context engineering 加持下模型理解复杂上下文的能力很强，但生成长篇复杂输出的能力明显受限[^3]。这意味着当前所有策展策略优化的都是「输入侧」，输出侧的质量瓶颈未被同等解决。

**提示策略的规模门槛。** CoT prompting 的推理能力被主张是在「足够大的语言模型」中自然涌现的，存在规模门槛[^6]；涌现能力无法通过外推小模型性能来简单预测[^9]。因此依赖规模门槛的技术，其收益不能线性外推到小模型上[^9]。这与 TheoremQA 的观察一致：所有开源模型在该基准上均低于 15%[^10]。

**模板不可迁移。** 最优模板无法在不同实验设置之间迁移，甚至无法在同一模型家族的不同模型之间迁移[^38]。这直接削弱了「把最佳实践固化成模板」的可行性，也是 context engineering 强调「每次重新策展」[^1]的实证基础之一。

**记忆系统的结构自适应。** A-MEM 批评 MemGPT 类方法受预定义结构与固定工作流约束、在新环境中泛化差，并批评 Mem0 等基于图数据库的方案依赖预定义 schema 与关系、无法随知识演化形成新的组织模式[^23]。但 A-MEM 自身仍需定义七元组结构与 top-k 链接生成规则[^23]，「完全自适应」尚未达成。

---

## 矛盾与不确定性

**1. 「提示词工程已死/被取代」的说法缺乏一手证据。** 全部 40 条证据中无任何一手来源出现该表述。Anthropic 明确称 context engineering 是 prompt engineering 的「natural progression」（非替代）[^1]，Cognition 称其为「the next level of this」[^26]，Anthropic《Building effective agents》正文亦无此说法[^27]。若报告沿用「厂商宣称已死」的框架，将与已收集证据直接冲突。本报告因此采用「包含关系」而非「替代关系」的表述。

**2. 缓存定价口径不一致且时效性不足。** 唯一有具体数字的来源是 Anthropic 2024-12-17 的 GA 公告（cache write +25%、cache read 10%），距 2026-10-25 已近两年，未覆盖 2025–2026 的模型代际与价格调整[^17]。Bedrock 定价页价格以 JS 占位符返回、无法确认单价[^29]，其「cache read 比 on-demand 低 75%」与 Anthropic「cache read 为 base input 的 10%」口径不一致且未解释差异来源[^17][^29]。因此「prompt caching / 长上下文定价如何改变 2025–2026 年的实践重心」这一问题，**现有公开资料不足以确认**。

**3. 仓库活跃度存在时间不一致与覆盖缺口。** (a) 时间不一致：OpenCE last push 为 2025-11-14[^37]，距审校日近一年，本报告已在其条目中单独标注该日期，不与 2026-08 仍活跃的仓库混同陈述。(b) 覆盖缺口：stars 最高的几个相关仓库（coleam00/context-engineering-intro ★13887、muratcankoylan/Agent-Skills-for-Context-Engineering ★18210、jasontang-ai/Context-Engineering ★9263、dair-ai/Prompt-Engineering-Guide ★79502）仅有检索元数据与简介，未读 README，其核心主张无法核验[^35]。(c) ratel 的 README 指向 benchmark.ratel.sh 作为完整结果出处，但本次抓取被截断，未读到具体基准分数[^30]，其「约 80% fewer tokens」的主张在本次证据中**无法核验**。

**4. 术语起源与时间线无法确认。** 有仓库将 Context Engineering 与 Prompt Engineering、Ontology Engineering、Graph Engineering、Loop Engineering、Harness Engineering 并列为同一综述谱系中的并列概念[^40]，但仅获得元数据与简介、未读取正文，无法提供术语起源或时间线信息。Karpathy 的类比（LLM 像新操作系统，LLM 是 CPU、上下文窗口是 RAM/工作记忆）与表述系经 LangChain 转引[^2]及仓库简介转引[^35]，未获得其原始出处，故本报告不将其作为一手定义使用。

**5. 「上下文工程」的学科化程度存在张力。** 综述称其为「一门正式学科」[^3
**5. 「上下文工程」的学科化程度存在张力。** 综述称其为「一门正式学科（a formal discipline）」[^3]，但同一篇论文同时指出该领域研究高度碎片化、各子方向被孤立研究，因此才需要统一框架来组织技术、澄清原理并说明相互依赖[^3]。换言之，「学科」这一自我定位更像是一个**待完成的目标**而非既成事实——若它已是成熟学科，就不必以 166 页篇幅论证统一框架的必要性。工业界的用法则更为松散：Anthropic 与 Cognition 用的是「艺术与科学」（art and science）这类修辞[^1][^26]，LangChain 用的是「umbrella」[^2]，三者都未采用「学科」的强主张。本报告因此把「正式学科」标注为综述作者的主张，而非领域共识。

**6. 定义之间的细微分歧未被任何来源调和。** 三个主定义各有侧重：Anthropic 强调「策展」（curation）与「有限上下文窗口」[^1]，LangChain 强调「agent 轨迹的每一步」[^2]，综述强调「信息载荷的系统性优化」[^3]。三者在「优化进入窗口的 token」上一致，但作用对象不完全重合——Anthropic 的清单包含 tools 与 MCP[^1]，LangChain 把 Tools 限定为「tool calls 的反馈」[^2]，综述则把 Tool-Integrated Reasoning 放在「系统实现」层而非「基础组件」层[^3]。这些差异是否会导致工程实践上的不同取舍，现有证据未提供对照实验，**不足以判断**。

**7. 多 agent 架构上的直接对立。** Cognition 主张默认排除不遵守「共享完整上下文与完整轨迹」原则的 agent 架构，称最简单的遵循方式是单线程线性 agent，并批评 OpenAI swarm 与微软 autogen 推动的多 agent 架构是「错误的构建 agent 方式」[^26]。而综述把 Multi-Agent Systems（通信与编排协调）列为 context engineering 的四大系统实现之一[^3]，LangGraph 亦以 orchestration 为核心能力并推荐支持 subagents 的 Deep Agents[^12]。这不是术语之争而是架构路线之争，本报告不裁定孰是，仅指出：**同一术语下存在互不兼容的架构主张**，且双方均未在给定证据中提供对照基准。

**8. 「最小高信号 token 集合」与「缓存前缀稳定性」的潜在冲突。** Anthropic 的原则是找到最小的高信号 token 集合[^1]；Bedrock 的工程要求是把静态内容放在 prompt 开头、动态内容放在末尾以提高精确前缀匹配概率，且前缀一旦改变即 cache miss[^28]。当「最小集合」的构成随轮次变化时，前缀稳定性与最小化可能指向不同的排列方案。现有证据中没有任何来源量化这一权衡，**无法确认**实践中的优先级。

**9. 压缩的收益边界未被统一测量。** LLMLingua 报告最高 20x 压缩且性能损失很小[^20]，LongLLMLingua 报告约 4x 更少 token 带来最高 21.4% 性能提升、LooGLE 上 94.0% 成本降低[^21]，Mem0 报告相比全上下文方案 token 成本节省超过 90%[^24]，ratel 宣称约 80% fewer tokens[^16][^30]。四组数字的基准、任务与计量口径均不同，且 ratel 的基准页面在本次抓取中被截断[^30]。这些数字**不可横向比较**，也不构成「压缩率越高越好」的证据。

**10. 长上下文模型的能力边界仍在移动。** Lost in the Middle 的结论建立在 MPT-30B-Instruct、LongChat-13B (16K)、GPT-3.5-Turbo、Claude-1.3 之上[^19]，并发现扩展上下文模型常与非扩展版本表现相同[^19]。Anthropic 的 context rot 则称该退化特性在所有模型上都出现，只是坡度不同[^1]。两者方向一致，但证据分别来自 2023 年与更晚的工程观察，**2025–2026 年新模型代际上 context rot 的具体坡度是否改变，现有公开资料不足以确认**。这一点直接影响「压缩与检索」相对「直接放长上下文」的性价比判断。

**11. 安全类提示干预的普适性存疑。** SimpleSafetyTests 显示强调安全的 system prompt 可显著降低但不能完全阻止不安全回复，且同一干预在不同模型上效果不一致、不彻底[^39]。在 context engineering 框架下，system prompt 只是上下文的一个组成要素[^1]，那么把安全约束分散到工具描述、外部数据与消息历史中是否更稳健，**证据未涉及**。

**12. 术语扩散与功能实现的错配。** 一方面，功能上覆盖 orchestration、memory、prompt 自动优化的三个高星框架（LangGraph 43.9k[^12]、Letta 25.3k[^13]、DSPy 38.8k[^15]）均未在 README 中自称 context engineering；另一方面，把该术语写进产品定位的仓库中，stars 最高的 MineContext（★5542）[^35]其「智能浮现」功能在中文 README 中标注为「实现中」[^34]。这说明术语的传播速度可能快于对应能力的交付速度。但需注意：README 未使用某术语**不等于**该框架不实践相应方法，本报告不据此推断任何框架的能力缺失。

**13. 未核验的证据边界汇总。** 以下主张在本次证据中仅有元数据或简介支撑，未读正文，故仅作线索记录、不作论断：coleam00/context-engineering-intro 的「Context engineering is the new vibe coding」[^35]；jasontang-ai/Context-Engineering 对 Karpathy 表述的引用及其「超越提示词工程」的自我定位[^35]；qf-studio/navigator、itallstartedwithaidea/ContextOS、jihoo-kim/awesome-context-engineering 等仓库的定位描述[^16]；DEEP-JLU/Awesome-Graph-Engineering 的谱系并列说法[^40]。此外，Meirtz/Awesome-Context-Engineering 的「数百篇论文、框架与实现指南」自述[^4]与 Context Engineering Kit 的「Scientifically proven」自述[^32]均为仓库自我宣称，未在给定证据中被第三方验证。

## 参考来源

[^1]: Effective context engineering for AI agents (Anthropic Engineering) — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents（web）
[^2]: Context Engineering for Agents (LangChain Blog) — https://blog.langchain.com/context-engineering-for-agents/（web, 2025-07-02）
[^3]: A Survey of Context Engineering for Large Language Models (arXiv:2507.13334, v2) — https://arxiv.org/abs/2507.13334（arxiv, 2025-07-21）
[^4]: Meirtz/Awesome-Context-Engineering — 综述配套仓库 — https://github.com/Meirtz/Awesome-Context-Engineering（github）
[^6]: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models — https://arxiv.org/abs/2201.11903v6（arxiv, 2022-01-28）
[^9]: Emergent Abilities of Large Language Models — https://arxiv.org/abs/2206.07682v2（arxiv, 2022-06-15）
[^10]: TheoremQA: A Theorem-driven Question Answering dataset — https://arxiv.org/abs/2305.12524v3（arxiv, 2023-05-21）
[^11]: Prompt Engineering a Prompt Engineer (PE2) — https://arxiv.org/abs/2311.05661（arxiv, 2023-11-09）
[^12]: langchain-ai/langgraph — README（Low-level orchestration framework for building stateful agents） — https://github.com/langchain-ai/langgraph（github）
[^13]: letta-ai/letta（f.k.a. MemGPT）— README — https://github.com/letta-ai/letta（github）
[^14]: NeoLabHQ/context-engineering-kit — README — https://github.com/NeoLabHQ/context-engineering-kit（github）
[^15]: stanfordnlp/dspy — README — https://github.com/stanfordnlp/dspy（github）
[^16]: GitHub 仓库检索结果：以 context engineering 为名的仓库及其 stars/活跃度元数据 — https://github.com/search?q=context+engineering+LLM&type=repositories（github）
[^17]: Prompt caching with Claude (Anthropic News, GA Dec 17 2024) — https://www.anthropic.com/news/prompt-caching（web, 2024-12-17）
[^18]: Lost in the Middle: How Language Models Use Long Contexts (arXiv:2307.03172v3) — https://arxiv.org/abs/2307.03172v3（arxiv, 2023-07-06）
[^19]: Lost in the Middle: How Language Models Use Long Contexts — 全文（HTML 版，方法与实验章节） — https://arxiv.org/html/2307.03172v3（arxiv, 2023-11-20）
[^20]: LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models — https://arxiv.org/abs/2310.05736v2（arxiv, 2023-10-09）
[^21]: LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression — https://arxiv.org/abs/2310.06839（arxiv, 2023-10-10）
[^22]: MemGPT: Towards LLMs as Operating Systems (arXiv:2310.08560v2) — https://arxiv.org/abs/2310.08560v2（arxiv, 2023-10-12）
[^23]: A-MEM: Agentic Memory for LLM Agents (arXiv:2502.12110v11) — https://arxiv.org/abs/2502.12110v11（arxiv, 2025-02-17）
[^24]: Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory (arXiv:2504.19413v1) — https://arxiv.org/abs/2504.19413v1（arxiv, 2025-04-28）
[^25]: Generative Agents: Interactive Simulacra of Human Behavior (arXiv:2304.03442v2, UIST '23) — https://arxiv.org/abs/2304.03442v2（arxiv, 2023-04-07）
[^26]: Don't Build Multi-Agents / Principles of Context Engineering (Cognition) — https://cognition.ai/blog/dont-build-multi-agents（web）
[^27]: Building effective agents (Anthropic Engineering) — https://www.anthropic.com/engineering/building-effective-agents（web）
[^28]: Prompt caching for faster model inference - Amazon Bedrock User Guide — https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html（web）
[^29]: Amazon Bedrock pricing — https://aws.amazon.com/bedrock/pricing/（web）
[^30]: ratel-ai/ratel — README（The context engineering layer for AI agents） — https://raw.githubusercontent.com/ratel-ai/ratel/main/README.md（github）
[^31]: sci-m-wang/OpenCE — README（Closed-Loop Context Engineering Toolkit） — https://raw.githubusercontent.com/sci-m-wang/OpenCE/main/README.md（github）
[^32]: NeoLabHQ/context-engineering-kit — README（Advanced context engineering techniques and patterns） — https://raw.githubusercontent.com/NeoLabHQ/context-engineering-kit/master/README.md（github）
[^33]: volcengine/MineContext — README（proactive context-aware AI partner，含 Context Engineering Architecture） — https://raw.githubusercontent.com/volcengine/MineContext/main/README.md（github）
[^34]: volcengine/MineContext — README_zh（中文版：上下文工程架构） — https://raw.githubusercontent.com/volcengine/MineContext/main/README_zh.md（github）
[^35]: GitHub 仓库检索结果：'context engineering' 按 stars 排序（含 MineContext 元数据） — https://github.com/search?q=context+engineering&type=repositories（github）
[^36]: GitHub 仓库检索结果：NeoLabHQ/context-engineering-kit 元数据 — https://github.com/search?q=NeoLabHQ%2Fcontext-engineering-kit&type=repositories（github）
[^37]: GitHub 仓库检索结果：sci-m-wang/OpenCE 元数据 — https://github.com/search?q=OpenCE+context+engineering&type=repositories（github）
[^38]: Mind Your Format: Towards Consistent Evaluation of In-Context Learning Improvements — https://arxiv.org/abs/2401.06766（arxiv, 2024-01-12 (v1); v3 2024-06-06; Accepted to Findings of ACL 2024）
[^39]: SimpleSafetyTests: a Test Suite for Identifying Critical Safety Risks in Large Language Models — https://arxiv.org/abs/2311.08370（arxiv, 2023-11-14 (v1); v2 2024-02-16）
[^40]: DEEP-JLU/Awesome-Graph-Engineering — https://github.com/DEEP-JLU/Awesome-Graph-Engineering（github）