# 上下文工程（Context Engineering）：定义、与提示词工程的关系与差异

## 摘要

上下文工程（Context Engineering）是 2025 年前后由头部模型厂商与 agent 框架社区共同确立的一个工程概念，指在 LLM 推理过程中**策展与维护最优 token 集合**的策略集合，其范围覆盖提示词之外所有可能进入上下文的信息[^1]。它与提示词工程（Prompt Engineering）不是替代关系，而是后者的「自然演进」：提示词工程关注如何撰写与组织指令（尤其是系统提示），上下文工程则关注在多轮、长时域运行的 agent 中管理**整个上下文状态**——system instructions、tools、MCP、外部数据与消息历史[^1]。学界同期将其形式化为一门「超越简单提示词设计」的学科，并给出两层分类法（基础组件 + 系统实现）[^10][^11]。工程侧的共识是：agent 不可靠的主因往往不是底层模型能力不足，而是「正确的」上下文没有被传给模型[^3]，因此上下文工程被 Cognition 称为构建 AI agent 工程师的「头号工作」[^2][^13]。

---

## 一、什么是上下文工程

### 1.1 权威定义

Anthropic 给出的定义分两层。首先定义 **context**：采样 LLM 时所包含的 token 集合（the set of tokens included when sampling from a large-language model）[^1]。在此基础上定义 **context engineering**：在 LLM 推理过程中，策展与维护最优 token（信息）集合的策略集合，**包括提示词之外所有可能进入上下文的信息**[^1]。

LangChain 的定义更强调时序与「恰到好处」：在 agent 轨迹的**每一步**，用恰到好处的信息填充上下文窗口的艺术与科学[^2]。其官方文档给出更工程化的版本：以**正确格式**提供**正确的信息与工具**，使 LLM 能完成任务，并称其为 AI 工程师的头号工作[^3]。

学术综述的定位则更宏观：LLM 的性能**根本上由推理时提供的上下文信息决定**；该综述提出 Context Engineering 作为一门超越简单提示词设计的正式学科，涵盖对 LLM 信息载荷（information payload）的系统性优化[^10]。

### 1.2 一个可操作的定义

综合上述来源，上下文工程可被完整表述为：

- **输入**：agent 在某一时刻的全部可用信息源——系统指令、消息历史、工具定义与工具返回、外部检索数据、长期记忆、运行时配置（如 user ID、权限）[^1][^3]。
- **操作**：在 agent 轨迹的每一步，对这些信息进行写入（write）、选择（select）、压缩（compress）、隔离（isolate）[^2]，并决定以何种格式传入模型[^3]。
- **产出**：一个恰好填满上下文窗口、且信息量最优的 token 集合，供本次 LLM 采样使用[^1][^2]。
- **性质**：迭代式的（iterative）——策展阶段在每次决定向模型传入什么时都会发生，区别于「写提示词」这一离散任务[^1]。

### 1.3 为什么需要它：上下文是有限资源

Anthropic 明确将 context 视为**有限资源**，存在边际收益递减，并引用 needle-in-a-haystack 基准中的 **context rot** 现象：上下文 token 数增加时，模型准确召回信息的能力下降[^1]。其解释框架是「**attention budget（注意力预算）**」——类比人类有限的工作记忆，每引入一个新 token 都会消耗该预算[^1]。

这一现象有独立的学术证据支撑。《Lost in the Middle》发现：当相关信息在上下文中的位置改变时，模型性能显著下降；性能通常在相关信息位于上下文**开头或结尾**时最高，而当模型必须访问长上下文**中间**的信息时性能显著退化——**即使对显式支持长上下文的模型也是如此**[^7]。

工程侧观察到四类具体失效模式[^12]：

| 失效模式 | 定义 |
|---|---|
| Context Poisoning | 当幻觉或其他错误进入上下文并被反复引用时发生[^12] |
| Context Distraction | 当上下文变得过长，模型过度关注上下文而忽略训练中学到的内容[^12] |
| Context Confusion | 当多余上下文影响回答[^12] |
| Context Clash | 当上下文各部分相互矛盾[^12] |

例证方面：DeepMind 在 Gemini 2.5 技术报告中记录，玩 Pokémon 的 Gemini agent 偶尔产生幻觉并污染其上下文，导致 goals/summary 等部分被错误信息污染，模型会执着于不可能或不相关的目标[^12]。Databricks 研究发现 Llama 3.1 405b 的模型正确率在约 32k 上下文时开始下降，更小的模型下降更早[^12]。Gemini 2.5 Pro 虽支持 1M+ token 上下文，但在 agentic 设置中上下文显著超过 100k token 后，agent 倾向于重复其庞大历史中的动作，而非综合出新的计划[^12]。

---

## 二、上下文工程与提示词工程的关系

### 2.1 官方定位：演进，而非替代

Anthropic 明确把 context engineering 定位为 prompt engineering 的「**自然演进**（natural progression）」，而非替代[^1]。其界定的 prompt engineering 是「为获得最优结果而撰写与组织 LLM 指令的方法」，重点是写好提示词、尤其是系统提示[^1]。

Cognition 的表述是「**next level**（更高一层）」：prompt engineering 是在静态提示上做优化，而 context engineering 是「在动态系统中自动完成这件事」，需要更多细微判断，并且「实际上是构建 AI agent 工程师的头号工作」[^13]。

LangChain 则把 context engineering 定位为覆盖多种上下文类型的「**伞形（umbrella）**」概念，并列出三类常见上下文[^2]：

- **Instructions**：提示词、记忆、few-shot 示例、工具描述等
- **Knowledge**：事实、记忆等
- **Tools**：工具调用的反馈

在这个框架下，提示词只是 Instructions 这一类中的一个子集。

### 2.2 提示词工程的学术边界

提示词工程本身已有系统化的学术梳理。《The Prompt Report》由 31 位作者撰写（v1 提交于 2024-06-06，v6 最后修订于 2025-02-26），提出了一套结构化分类法：包含 **33 个词汇术语、58 种 LLM prompting 技术、以及 40 种面向其他模态的技术**[^34]。该论文的动机正是提示词工程领域存在**术语冲突**与**本体理解碎片化**[^34]。论文指出，prompt 可以是文本（如 "Write a poem about trees."），也可以是图像、音频、视频或其组合[^35]。

这一边界与上下文工程形成对照：提示词工程的对象是「指令文本（及多模态提示）」，上下文工程的对象是「推理时提供给 LLM 的完整信息载荷」[^8]。

### 2.3 一条清晰的历史脉络

提示词工程范式的奠基来自 GPT-3 论文：GPT-3（1750 亿参数）在推理时**不做任何梯度更新或微调**，任务与少量示例完全通过文本交互（prompt）指定；few-shot 性能随模型规模放大而显著提升，有时可与当时的微调 SOTA 相竞争[^4]。这把「通过提示文本指定任务」确立为一种无需参数更新的任务适配方式[^4]。

随后 CoT prompting 通过在提示中提供少量包含中间推理步骤的示例，在算术、常识与符号推理任务上提升表现——用仅 8 个 CoT 示例提示 540B 参数模型，在 GSM8K 上达到当时 SOTA，超过带 verifier 的微调 GPT-3[^5]。ReAct 进一步让 LLM 以交错方式同时生成推理轨迹与任务特定动作，使模型能对接外部来源获取额外信息，在 HotpotQA 与 Fever 上通过与简单 Wikipedia API 交互克服了 CoT 中普遍存在的幻觉与错误传播问题[^6]。

这条脉络的终点正是上下文工程要处理的问题：一旦模型开始与外部环境交互、多轮运行，需要管理的信息就不再只是「一段提示词」，而是**整个上下文状态**[^1]。

---

## 三、差异：三个可操作的切面

### 3.1 范围：单一提示 vs 完整信息载荷

Anthropic 明确列举了 agent 上下文的组成：**system instructions、tools、Model Context Protocol (MCP)、external data、message history**[^1]。LangChain 官方文档把 Model Context 的组成细化为五项：System Prompt（开发者给 LLM 的基础指令）、Messages（发送给 LLM 的完整消息/对话历史）、Tools（agent 可用于采取行动的工具）、Model（被调用的实际模型及配置）、Response Format（模型最终响应的 schema 规范）[^3]。文档并指出，上述各类 model context 均可取自 state（短期记忆）、store（长期记忆）或 runtime context（静态配置）[^3]。

社区资源库的表述最直接：『Unlike traditional prompt engineering, context engineering encompasses the complete information payload provided to LLMs at inference time, including all structured informational components necessary for plausible task completion.』[^8]

### 3.2 时间性：离散任务 vs 迭代过程

Anthropic 强调 context engineering 是**迭代式**的，策展阶段在每次决定向模型传入什么时都会发生，区别于写提示词这一**离散任务**[^1]。LangChain 的表述与此一致：在 agent 轨迹的**每一步**填充上下文窗口[^2]。

这一差异在 agent 循环中有具体形态。LangChain 文档把 agent 循环拆为两步：Model call（带提示词与可用工具调用 LLM，返回响应或工具执行请求）与 Tool execution（执行工具并返回结果）[^3]。文档将上下文按可控性分为三类[^3]：

- **Model Context**：进入模型调用的内容（instructions、message history、tools、response format）
- **Tool Context**：工具可读写的内容（state、store、runtime context）
- **Life-cycle Context**：模型与工具调用之间发生的事（summarization、guardrails、logging）

文档还区分了 **transient context**（单次调用 LLM 所见，可修改 messages/tools/prompts 而不改变 state）与 **persistent context**（跨轮次保存在 state 中，由生命周期钩子与工具写入永久修改）[^3]。

### 3.3 目标：写好指令 vs 管理资源预算

提示词工程的目标是「为获得最优结果而撰写与组织 LLM 指令」[^1]。上下文工程的目标则带有资源管理色彩：在 attention budget 约束下，让每一步的 token 集合达到最优[^1]。

LangChain 把 agent 上下文工程的常见策略归为四类：**write（写入）、select（选择）、compress（压缩）、isolate（隔离）**[^2]。其文档指出，agent 交错进行 LLM 调用与工具调用，长任务与工具反馈累积会消耗大量 token，可能导致超出上下文窗口、成本/延迟膨胀或性能下降[^2]。

---

## 四、支撑上下文工程的基础设施

### 4.1 记忆分层

LangChain 官方按召回范围把记忆分为两类：**短期记忆**（thread-scoped，维护会话内消息历史）与**长期记忆**（跨 session、跨 thread 共享）[^18]。短期记忆由 LangGraph 作为 agent state 的一部分管理，通过 thread 级 checkpoint 持久化；state 通常包含对话历史以及其他有状态数据，如上传文件、检索到的文档、生成的产物[^18]。长期记忆保存在自定义 namespace 中，而非仅限单个 thread ID，由 stores 保存与召回[^18]。

官方给出的动机直接对应上下文工程的核心约束：完整历史可能放不进 LLM 上下文窗口，导致 context loss 或报错；即使模型支持完整上下文长度，多数 LLM 在长上下文上表现仍差，会被 stale 或 off-topic 内容「分心」，同时响应更慢、成本更高[^14][^18]。因此官方建议，许多应用可从移除或「遗忘」陈旧信息的技术中获益[^14]。

LangChain 文档列出 agent 访问的数据源三类[^3]：

- **Runtime Context**：静态配置，会话级，如 user ID、API keys、权限
- **State**：短期记忆，会话级，如当前消息、上传文件、工具结果
- **Store**：长期记忆，跨会话，如用户偏好、提取的洞察、历史数据

在消息层面，LangChain 定义 Messages 为模型上下文的**基本单位**，承载内容与元数据以表示对话状态；Message 对象包含 Role、Content、Metadata 三部分[^15]。官方列出的消息类型包括 SystemMessage、HumanMessage、AIMessage（含文本、tool calls 与元数据）、ToolMessage[^15]。

### 4.2 虚拟上下文与状态化 agent

MemGPT 提出 **virtual context management（虚拟上下文管理）**，借鉴传统操作系统的分层内存系统，通过在快/慢内存之间移动数据，营造出比实际更大的内存资源；它在 LLM 有限的上下文窗口内智能管理不同的内存层级，并使用 interrupts 管理其与用户之间的控制流[^9]。论文在两个领域评估该设计：文档分析（可分析远超底层 LLM 上下文窗口的大文档）与多轮会话聊天（可创建在长期交互中记忆、反思、演化的对话代理）[^9]。其代码与数据发布于 memgpt.ai，即后续 Letta 项目的前身[^9]。

Letta 文档描述了该思路的工程化形态：agent 的全部状态（memories、user messages、reasoning、tool calls）持久化到数据库，**即使被逐出上下文窗口也不会丢失**[^20]。其记忆被组织为 **memory blocks**（字符串形式的上下文片段），可由 agent 通过 memory tools 编辑，也可由开发者通过 API 直接编辑；被 attach 的 block 处于 in-context 状态（pinned 到 system prompt），同一 block 可同时 attach 到多个 agent（shared blocks）[^20]。由于 API 存储全部 messages，在 compaction / eviction 之后，旧消息仍可通过 API（面向开发者）和 retrieval tools（面向 agent）取回[^20]。一个 stateful agent 由 system prompt、memory blocks、messages（in-context 与 out-of-context）和 tools 构成[^20]。

LlamaIndex 侧的记忆抽象是 **Memory 类**，可同时具备短期记忆（消息的 FIFO 队列）和可选长期记忆（随时间抽取信息）；通过 `Memory.from_defaults(session_id=..., token_limit=...)` 创建并传入 `agent.run()`[^21]。其 `token_limit` 默认 30000，表示短期与长期记忆合计可存储的最大 token 数；`chat_history_token_ratio` 默认 0.7，表示短期聊天历史 token 占总 token 上限的比例，若聊天历史超过该比例，最旧的消息会被 flush 到长期记忆[^21]。原有的 ChatMemoryBuffer 已被标记为 deprecated[^21]。

### 4.3 协议与工具层

Anthropic 在列举上下文组成时明确包含 Model Context Protocol (MCP)[^1]。MCP 参考实现仓库 modelcontextprotocol/servers 是 MCP 参考实现的集合，同时收录社区构建的服务器与附加资源；README 声明该仓库仅托管由 MCP steering group 维护的少量参考服务器，完整列表应查阅 MCP Registry，并警告仓库内服务器为演示 MCP 特性与 SDK 用法的参考实现，定位为教学示例，**并非生产就绪方案**[^30]。仓库列出的参考服务器包括 Everything、Fetch、Filesystem、Git、Memory、Sequential Thinking、Time，并支持 C#、Go、Java、Kotlin、PHP、Python、Ruby、Rust、Swift、TypeScript 等多种语言的 SDK[^30]。

### 4.4 框架层的定位分化

LangChain 官方对框架分工的表述是：**Agent = Model + Harness**；`create_agent` 是最小、高度可配置的 harness，harness 指模型循环之外的一切——prompt、tools 以及任何塑造行为的 middleware[^19]。Deep Agents 是「batteries-included」的 agent，具备 automatic context compression、虚拟文件系统、subagent-spawning 等特性；LangGraph 是低层编排框架，用于结合确定性工作流与 agentic 工作流的进阶需求；LangSmith 用于对上述任一框架构建的 agent 进行 trace、debug 与 evaluate[^19]。

DSPy 走的是另一条路径：自我定位为「programming—rather than prompting—language models」的框架，提供用于优化 prompts 与 weights 的算法（optimizers），适用于分类器、RAG 流水线、Agent 循环等[^22]。其全称为 Declarative Self-improving Python，主张以可组合的 Python 代码替代脆弱的 prompt[^22]。

---

## 五、学术形式化：两层分类法

综述 arXiv:2507.13334《A Survey of Context Engineering for Large Language Models》提出了一个两层分类法，将 Context Engineering 分解为 **foundational Components（基础组件）** 与 **System Implementations（系统实现）**[^11]。

**基础组件共 3 类**[^11]：

1. **Context Retrieval and Generation（上下文检索与生成）**：涵盖 prompt-based generation 与 external knowledge acquisition
2. **Context Processing（上下文处理）**：涵盖 long sequence processing、self-refinement、structured information integration
3. **Context Management（上下文管理）**：涵盖 memory hierarchies、compression、optimization

需要特别注意：原文将 retrieval 与 generation 合并为同一组件（Context Retrieval and Generation），因此基础组件为 **3 类**，而非 retrieval/generation/processing/management 四类并列[^11]。

**系统实现共 4 类**[^11]：

1. **Retrieval-Augmented Generation (RAG)**：含 modular、agentic、graph-enhanced 架构
2. **Memory Systems**：支持持久化交互
3. **Tool-Integrated Reasoning**：用于函数调用与环境交互
4. **Multi-Agent Systems**：协调通信与编排

该论文的作者共 15 位（Lingrui Mei、Jiayu Yao、Yuyao Ge、Yiwei Wang、Baolong Bi、Yujun Cai、Jiazhi Liu、Mingyu Li、Zhong-Zhi Li、Duzhen Zhang、Chenlin Zhou、Jiayi Mao、Tianze Xia、Jiafeng Guo、Shenghua Liu），分类为 cs.CL，v1 于 2025-07-17 提交，v2 于 2025-07-21 修订，Comments 字段标注为 ongoing work; 166 pages, 1411 citations[^10]。作者机构包括中科院计算所（ICT, CAS）、中国科学院大学、UC Merced、昆士兰大学、北京大学、清华大学等[^11]。

论文摘要指出该综述系统分析了超过 1400 篇研究论文，并揭示一个关键研究缺口：模型能力存在根本性不对称——**模型在理解复杂上下文方面表现优异，但在生成同等复杂的长文本输出方面存在明显局限**[^10]。

---

## 六、社区生态与热度

GitHub 上，`Meirtz/Awesome-Context-Engineering` 自我定位为「Context Engineering 的综合综述与资源集合」，主题为「从静态提示（static prompting）到动态、上下文感知的 AI 系统的演进」，并进一步延伸到 agent runtime、记忆系统、协议、编码 agent 与可观测性栈[^8]。该仓库声明其配套综述论文 arXiv 编号为 2507.13334，标题为 'A Survey of Context Engineering for Large Language Models'[^8]。仓库元数据显示 stars 3326、forks 293、最近 push 时间 2026-05-28[^8]。

README 记录了 **2026 年 3 月的立场更新**：『As of March 2026, context engineering remains a useful and necessary concept, but it is no longer the whole story.』重心已从「如何打包最好的提示词」转向 agent 系统如何管理运行时状态、记忆、工具、协议、审批与长时程执行[^8]。该仓库保留的原始综述结构涵盖 long context、RAG、memory、agent communication、tool use、evaluation、applications，并新增 agent harness、生产环境上下文管理（compaction、caching、artifact-backed context、scoped instruction loading）、记忆工件与可移植性、开放协议（MCP、A2A、AG-UI、ACP）等主题[^8]。

相关框架仓库的规模数据（抓取时点）[^25][^26][^27][^28][^29]：

| 仓库 | Stars | Forks | 最近 push | 描述 |
|---|---|---|---|---|
| langchain-ai/langchain | 147843 | 24795 | 2026-10-14 | The agent engineering platform |
| langchain-ai/langgraph | 43254 | 7354 | 2026-10-14 | Build resilient agents |
| run-llama/llama_index | 52492 | 8336 | 2026-10-14 | LlamaIndex is the document processing platform for AI |
| stanfordnlp/dspy | 38633 | 3423 | 2026-10-13 | DSPy: The framework for programming—not prompting—language models |
| letta-ai/letta | 25132 | 2644 | 2026-09-10 | Platform for stateful agents: AI with advanced memory that can learn and self-improve over time |

---

## 七、矛盾与不确定性

**1. 「演进」与「对立」两种叙事并存。** Anthropic 明确把 context engineering 定位为 prompt engineering 的「自然演进」而非替代[^1]，Cognition 称其为「next level」[^13]。但社区资源库 README 使用了「Unlike traditional prompt engineering」的对立表述[^8]。这两种叙事在证据中同时存在，本报告不裁定孰是孰非。

**2. 「头号工作」的归属对象不同。** LangChain 引用 Cognition 的观点称「context engineering 实际上是构建 AI agent 工程师的头号工作」[^2]，Cognition 原文的表述是「effectively the #1 job of engineers building AI agents」[^13]。LangChain 官方文档则直接称其为「AI 工程师的头号工作」[^3]，范围比原文更宽。引用时需注意这一范围差异。

**3. 部分来源不覆盖该概念。** OpenAI 的 GPT-5 prompting guide 通篇以 'prompting' 为框架，未出现对 'context engineering' 与 'prompt engineering' 关系的定义或对比[^31]。Chip Huyen 的博客索引中不存在标题含 'context engineering' 的文章[^32]，其 Agents 一文（改编自《AI Engineering》2025）也未出现 'context engineering' 一词，未界定其与 prompt engineering 的边界[^33]。这说明该术语的采用在业界**并不统一**。

**4. 第三方来源的证据强度有限。** 关于 LlamaIndex MemoryBlocks 的描述来自一个 0 stars 的第三方演示仓库，非官方来源，仅可作为概念旁证[^23]。关于 Letta REST API 暴露 239 个端点、覆盖 36 个公开资源类别的说法来自第三方 API 画像（API Evangelist），具体数字**未经官方文档核实**[^24]。

**5. 关键仓库元数据缺失。** modelcontextprotocol/servers 抓取到的页面正文未包含 star 计数与最近提交日期[^30]，因此无法评估其工程热度。

**6. 术语边界仍在移动。** Awesome-Context-Engineering README 的 2026 年 3 月立场更新明确表示 context engineering「不再是故事的全部」，重心已转向 agent 系统的运行时状态、记忆、工具、协议、审批与长时程执行[^8]。这意味着本报告所描述的「上下文工程 vs 提示词工程」二分，可能正在被更宽的「agent 系统工程」框架吸收。现有公开资料不足以确认这一转向的最终形态。

**7. 综述的时效性。** arXiv:2507.13334 的 v2 修订于 2025-07-21[^10]，其 Comments 标注为 ongoing work[^10]。该综述对 2026 年后的协议演进（如 A2A、AG-UI、ACP）的覆盖情况，现有证据未予说明。

---

## 参考来源

[^1]: Effective context engineering for AI agents (Anthropic Engineering) — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents （web, 2025-09-29）
[^2]: Context Engineering (LangChain Blog) — https://blog.langchain.com/context-engineering-for-agents/ （web, 2025-07-02）
[^3]: Context engineering (LangChain Docs) — https://docs.langchain.com/oss/python/langchain/context-engineering （web）
[^4]: Language Models are Few-Shot Learners (GPT-3) — https://arxiv.org/abs/2005.14165v4 （arxiv, 2020-05-28）
[^5]: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models — https://arxiv.org/abs/2201.11903v6 （arxiv, 2022-01-28）
[^6]: ReAct: Synergizing Reasoning and Acting in Language Models — https://arxiv.org/abs/2210.03629v3 （arxiv, 2022-10-06）
[^7]: Lost in the Middle: How Language Models Use Long Contexts — https://arxiv.org/abs/2307.03172v3 （arxiv, 2023-07-06）
[^8]: Meirtz/Awesome-Context-Engineering — README（上下文工程综述资源库） — https://github.com/Meirtz/Awesome-Context-Engineering （github）
[^9]: MemGPT: Towards LLMs as Operating Systems — https://arxiv.org/abs/2310.08560v2 （arxiv, 2023-10-12）
[^10]: A Survey of Context Engineering for Large Language Models (arXiv:2507.13334, abs page) — https://arxiv.org/abs/2507.13334 （arxiv, 2025-07-17）
[^11]: A Survey of Context Engineering for Large Language Models (arXiv:2507.13334v2, HTML full text) — https://arxiv.org/html/2507.13334v2 （arxiv, 2025-07-21）
[^12]: How Long Contexts Fail — https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html （web, 2025-06-22）
[^13]: Don't Build Multi-Agents (Cognition 官方博客) — https://cognition.ai/blog/dont-build-multi-agents （web）
[^14]: LangChain Docs — Short-term memory — https://docs.langchain.com/oss/python/langchain/short-term-memory （web）
[^15]: LangChain Docs — Messages — https://docs.langchain.com/oss/python/langchain/messages （web）
[^16]: LangChain Docs — Long-term memory — https://docs.langchain.com/oss/python/langchain/long-term-memory （web）
[^17]: LangGraph Docs — Add memory — https://docs.langchain.com/oss/python/langgraph/add-memory （web）
[^18]: LangGraph Docs — Memory (conceptual guide) — https://docs.langchain.com/oss/python/langgraph/memory （web）
[^19]: LangChain Docs — create_agent / harness 概览（trim_messages 页面重定向后内容） — https://python.langchain.com/docs/how_to/trim_messages （web）
[^20]: Letta Docs — Introduction to Stateful Agents (Memory / Core API concepts) — https://docs.letta.com/guides/agents/memory （web）
[^21]: LlamaIndex Docs — Agents / Memory (Memory class, short-term & long-term) — https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/memory/ （web）
[^22]: stanfordnlp/dspy — README (official repository) — https://raw.githubusercontent.com/stanfordnlp/dspy/main/README.md （github）
[^23]: nadavlev/LlamaIndex-MemoryBlocks-Context-Management-Demo — README (third-party demo) — https://github.com/nadavlev/LlamaIndex-MemoryBlocks-Context-Management-Demo （github）
[^24]: api-evangelist/letta — third-party API profile (memory blocks / archival memory) — https://github.com/api-evangelist/letta （github）
[^25]: langchain-ai/langchain — https://github.com/langchain-ai/langchain （github, 2026-10-14）
[^26]: langchain-ai/langgraph — https://github.com/langchain-ai/langgraph （github, 2026-10-14）
[^27]: run-llama/llama_index — https://github.com/run-llama/llama_index （github, 2026-10-14）
[^28]: stanfordnlp/dspy — https://github.com/stanfordnlp/dspy （github, 2026-10-13）
[^29]: letta-ai/letta — https://github.com/letta-ai/letta （github, 2026-09-10）
[^30]: modelcontextprotocol/servers (README 正文，未含 star 数) — https://github.com/modelcontextprotocol/servers （github）
[^31]: GPT-5 prompting guide (OpenAI Cookbook) — https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide （web, 2025-08-07）
[^32]: Chip Huyen — Blog index — https://huyenchip.com/blog/ （web）
[^33]: Chip Huyen — Agents (blog post, adapted from AI Engineering 2025) — https://huyenchip.com/2025/01/07/agents.html （web, 2025-01-07）
[^34]: The Prompt Report: A Systematic Survey of Prompt Engineering Techniques — https://arxiv.org/abs/2406.06608v6 （arxiv, 2024-06-06）
[^35]: The Prompt Report: A Systematic Survey of Prompt Engineering Techniques (HTML full text) — https://arxiv.org/html/2406.06608v6 （arxiv, 2024-06-06）