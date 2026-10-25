## 摘要

2025 年以来 LLM Agent 记忆机制的主流方案可归为四条技术路线：**结构化笔记网络**（A-Mem 的 Zettelkasten 式互连笔记与 memory evolution）[^3][^19]、**抽取—整合—检索管线及其图增强变体**（Mem0 与 Mem0^g）[^4][^22]、**时序知识图谱记忆层**（Zep / Graphiti，事实带有效期、失效而非删除）[^5][^8]、**文件系统式可编辑记忆**（Letta 的 MemFS 与 Dreaming 后台整合）[^10]。此外还有面向"经验/技能沉淀"的自演化记忆分支，但现有证据中该分支只有低星仓库与 README 级主张，不足以确认其研究地位[^27][^28][^31]。

需要强调三点不确定性。其一，**本课题最关键的量化对比无法完成**：A-Mem 与 Mem0 的一手论文与官方评测仓库均只给出定性结论或相对提升，未提供可核验的 LoCoMo / LongMemEval 绝对分数[^18][^20][^21]；唯一的具体数字（LoCoMo 92.5、LongMemEval 94.4）出自 Mem0 README 标注的 2026 年 4 月新算法，且作者自承包含开源 SDK 不具备的专有优化[^23]。其二，厂商侧证据严重不全，只有 Anthropic 一家的官方发布页[^15]，OpenAI 与 Google 的记忆能力无任何来源，三家对比不成立。其三，MCP 规范正文中不存在 memory 原语，记忆只能借 Resources/Tools 间接承载[^16]。

---

## 一、概念基线：什么算"Agent 记忆"

按该领域首篇系统综述的定义框架，agent 的执行被形式化为：第 t 步的动作为 a_t、环境观测为 o_t，长度 T 的 trial 记为 ξ_T = {a_1, o_1, a_2, o_2, …, o_T, a_T}；step、trial、task 构成三层结构，每个 trial 内 agent 执行多步形成潜在解，每个 task 内 agent 可探索多个 trial[^2]。综述明确从"狭义与广义两个视角"定义 agent 记忆，但抓取内容在进入定义正文前被截断，**该定义的具体表述现有公开资料不足以确认**[^2]。可确认的是其功能定位：记忆是使 agent 区别于原始 LLM 的关键模块，决定 agent 如何积累知识、处理历史经验、检索有用知识以支撑行动[^1]；综述从认知心理学、自我进化、agent 应用三个视角论证记忆的必要性，并梳理记忆模块的信息来源、存储形式与操作机制[^1]。该综述 v1 提交于 2024-04-21，配套仓库最近 push 为 2025-07-28，仍在持续跟踪[^1][^17]——这意味着它是本课题的框架来源，而非 2025 年新方案本身。

---

## 二、路线一：结构化笔记网络（A-Mem）

**机制定义。** A-Mem 的目标是让记忆组织不依赖预定义存储结构与固定工作流[^3]。其数据结构直接借自 Zettelkasten 卡片盒笔记法：每条记忆表示为 m_i = {c_i, t_i, K_i, G_i, X_i, e_i, L_i}，依次为原始内容、时间戳、LLM 生成的关键词、标签、上下文描述、稠密向量、链接集合[^19]。写入流程分两步：先为新记忆生成上下文描述、关键词、标签等结构化属性与嵌入向量[^3]；再分析历史记忆库，基于语义相似度与共享属性建立链接——具体做法是用嵌入余弦相似度 s_{n,j} = e_n·e_j/(|e_n||e_j|) 取 top-k 候选，再由 LLM 判定是否真正连边[^19]。第三步是 **memory evolution**：新记忆被整合时触发对既有历史记忆的上下文表示与属性更新，使记忆网络随经验累积持续演化[^3][^19]。

**定位与对比。** 作者把 Mem0 归为"遵循 RAG 原则、用图数据库做存储与检索"的方案，并批评图数据库路线依赖预定义 schema 与关系，限制跨任务适应性[^3][^19]。

**发表与工程状态。** 论文被 NeurIPS 2025 接收，arXiv v1 为 2025-02-17，最新 v11 修订于 2025-10-08[^18]。代码分散在多个仓库：评测复现仓库使用 LoCoMo 的 locomo10 子集，检索超参 `--retrieve_k` 默认 10 并需按模型调整才能达到论文报告性能[^20]；面向构建的 memory system 包提供 add_note / read / search_agentic / update / delete 接口，用 ChromaDB 做向量存储[^36][^37]；另有生产版仓库[^3]。值得注意的是，`WujiangXu/AgenticMemory` 的 API 请求返回 301 重定向、页面抓取失败，按名称检索返回无结果，**该仓库当前的 stars 与最近提交时间无法取得**[^38]。

**证据强度评估。** A-Mem 是四条路线中机制描述最完整的一手证据，但**结果层面只有定性表述**：v11 摘要仅称"在六个基础模型上相对现有 SOTA 基线有显著提升"，未出现任何 LoCoMo 或 LongMemEval 分数[^18]；官方评测仓库 README 的 Results 小节同样只有一句定性表述，无数值表格、无与 Mem0/Zep/全上下文基线的分数对比[^20]。

---

## 三、路线二：抽取—整合—检索管线（Mem0 / Mem0^g）

**机制定义。** Mem0 针对 LLM 固定上下文窗口无法维持长多轮对话一致性的问题，核心是从持续进行的对话中动态抽取（extract）、整合归并（consolidate）、检索（retrieve）显著信息[^4]。论文给出两种架构：Mem0 是抽取—更新两阶段的增量式管线，抽取阶段以"对话摘要 S + 最近 m 条消息 + 当前消息对"构成 prompt 由 LLM 抽取候选记忆集合，更新阶段通过 Tool Call 机制对相似记忆执行操作；Mem0^g 则是以实体为节点、关系为边的有向标签图，更新阶段含冲突检测与消解，检索支持语义三元组匹配并按相似度阈值过滤排序[^22]。实验配置为 m=10 条历史消息、s=10 条相似记忆，所有 LLM 操作使用 GPT-4o-mini，向量库用稠密嵌入做相似度检索[^21]。

**量化结果（仅相对值可核验）。** 论文摘要给出的是相对提升：LLM-as-a-Judge 指标相对 OpenAI 提升 26%；带图记忆的 Mem0^g 总体分数比基础配置高约 2%；相比全上下文方法 p95 延迟降低 91%、token 成本节省超过 90%[^4][^21]。**绝对分数不可得**——HTML 抓取在 §2.2 处截断，未读到实验章节的结果表[^22]。

**一个必须拆开的版本混淆。** Mem0 仓库 README 给出的 LoCoMo 92.5、LongMemEval 94.4、BEAM(1M) 64.1、BEAM(10M) 48.6、token 6.7K–7.0K、p50 延迟 0.88–1.09s，全部标注在"New Memory Algorithm (April 2026)"标题下，并给出 Old → New 对照（LoCoMo 71.4 → 92.5，LongMemEval 67.8 → 94.4）[^7][^23]。该新算法的变更点包括：单次 ADD-only 抽取（一次 LLM 调用，无 UPDATE/DELETE，记忆只累积不覆盖）、agent 生成事实作为一等公民、实体链接、多信号检索（semantic + BM25 + entity 并行打分融合）、时间感知检索[^7][^23]。README 明确声明这些分数跑在"production-representative model stack"上、采用单次检索与 top_200 检索预算，且**反映 Mem0 托管平台、包含开源 SDK 中不具备的专有优化，开源用户只能期待方向相似而非相同数字**[^23]。因此这组数字不能回填为 2025 年 4 月论文所述系统的成绩。

工程侧另有可确认的事实：仓库 67,773 stars、Apache-2.0、Python，提供 Library / Self-Hosted Server / Cloud 三种部署形态，支持 User、Session、Agent 三层记忆，默认 LLM 为 gpt-5-mini[^7]。

---

## 四、路线三：时序知识图谱记忆层（Zep / Graphiti）

**机制定义。** Zep 是面向 AI Agent 的记忆层服务，核心组件 Graphiti 是一个时序感知的知识图谱引擎，动态综合非结构化对话数据与结构化业务数据，并在整合过程中保留历史关系[^5]。Graphiti 的 context graph 由四类组件构成：Entities（节点，带随时间演化的摘要）、Facts/Relationships（边，三元组带 temporal validity window）、Episodes（原始数据，作为 provenance 的 ground truth）、Custom Types（通过 Pydantic 模型定义的实体与边类型）[^8]。其 temporal fact management 的关键设计是：事实有有效期窗口，信息变化时**旧事实被 invalidated 而非删除**，可查询"当前为真"或"任一历史时点为真"的事实；支持增量更新与精确历史查询而无需完整重算图；检索为 semantic + keyword + graph traversal 的混合检索[^8]。

**量化结果。** Zep 报告在 Deep Memory Retrieval (DMR) 上 94.8% 对 MemGPT 的 93.4%；在 LongMemEval 上准确率最高提升 18.5%，同时相比基线降低 90% 响应延迟[^5]。这是本课题中少数在论文摘要层面给出具体基准数字的工作，但需注意其 LongMemEval 结果是"最高提升 18.5%"的相对表述，绝对准确率未在证据中给出[^5]。

**分工与边界。** Graphiti README 明确划界：Graphiti 是开源 Python 框架，需自备图数据库（Neo4j、FalkorDB 或 Amazon Neptune）、LLM 与 embedder；Zep 是面向企业的统一 context layer，使用自研图数据库 Konig，Graphiti 是 Zep 内部的图构建框架[^8]。README 甚至直接声明"两者都不是 agent memory 产品：Graphiti 是建图的框架，Zep 统一、治理并服务企业的 context"[^8]。工程热度：Graphiti 32,191 stars、Apache-2.0、Python，最近 push 2026-10-25[^8]。

---

## 五、路线四：文件系统式可编辑记忆（Letta / MemFS）

**机制定义。** Letta（前身 MemGPT）的 agent 使用 MemFS——一个 git-backed memory filesystem，agent 可以检视并编辑它；记忆在 agent 多个会话之间共享，并随 agent 学到关于用户与工作的持久信息而改进[^9][^10]。围绕它有一组显式操作：`/init` 引导或刷新当前项目的记忆；`/doctor` 审计记忆层级是否漂移或过大（含重复与 system-prompt token 用量）；`/remember` 是显式教学机制，由 agent 决定该经验归属何处并把更新 commit 到 MemFS，可用 `/memory` 或直接检视 `$MEMORY_DIR` 查看[^10]。**Dreaming** 是后台整合机制：用后台 subagent 回顾近期对话、整合有用经验并更新记忆，不打断当前工作，触发时机为完成一定数量 agent step 后或 context window 被压缩时；记忆重组流程会先备份当前仓库，再拆分大文件、合并重复项或重构层级[^10]。云记忆配额为 Free 每 repository 5 MiB、Pro 50 MiB，按各分支或 tag 最新快照的文件计，不含 Git 历史[^10]。

**工程状态。** 主仓库 25,250 stars、Apache-2.0，README 已重定向：源码迁至 letta-ai/letta-code（3,707 stars、TypeScript、创建于 2025-10-25），archive 分支保留已退役的 Letta V1 API server[^9]。letta-code 自述为"a stateful agent harness for creating agents that are more like people than tools"，agent 通过重写自身 memory、skills、prompts 乃至 harness（通过 mods）来长期演化[^9]。**该路线在现有证据中没有任何基准量化结果**，其主张均为产品文档层面的机制描述[^10]。

---

## 六、相邻但证据不足的分支

**上下文预算分配。** MemPrice 把活跃上下文视为共享资源，将累积的记忆压力转化为对上下文的自适应价格，只有当某条细节的预估保留价值足以抵消其额外占用时才保留细节；作者声称可证明不可逆压缩在跟随自身记忆轨迹时会导致线性最坏情况 regret，并在 LongMemEval、LoCoMo、AppWorld、WebShop 四个基准、三种基础模型上评估，覆盖 query-blind 记忆构建与目标导向的 agent 上下文管理两种场景[^24]。**摘要中未给出任何具体基准分数**[^24]。该文 arXiv 编号为 2610.24955、日期 2026-10-17，已超出"2025 年以来"的典型窗口[^24]。

**经验/技能沉淀（self-evolving memory）。** 这条线在概念上最贴近"自我进化"，但证据质量最弱。SkeMex 自述为医疗 agent 的"Self-Evolving Skill Memory"，徽章标注"Adaptation: No weight updates"，即通过外部记忆/技能沉淀而非权重更新实现适应，仓库仅 5 stars、0 forks[^27]。FactorMiner 记录公式的 lineage、metrics 与 admission decisions，并以"shared evaluation and memory contracts"抽象记忆[^28]。以 'self-evolving agent memory experience' 检索 GitHub 命中结果多为低星个人项目（2–123 stars），且均未读到方法与实验正文[^31]。一个经验生命周期综述配套仓库列出 acquisition、representation、retrieval、consolidation、abstraction、internalization、revision、distribution、governance 九个阶段，但该仓库仅 2 stars，只能作为文献索引线索[^29]。同期 GitHub 上 'skill library agent' 的高星结果（如 54,493 stars 的科学 Agent Skills 库）属于 2026 年出现的"Agent Skills 开放标准"工程生态，而非 agent 自主沉淀经验的程序性记忆研究方案[^30]。**结论：该分支目前不足以确认为 2025 年以来的主流方案之一。**

**其他记忆平台。** cognee 自述把文档、代码与对话转成可自托管知识图谱，可完全本地免费运行、依赖 CPU 小模型（本地 GLiNER + 本地 embedding），提供 remember / recall / improve / forget 四个核心操作，32,729 stars[^11]。LangMem 提供可与任意存储配合的 Core memory API、hot-path 内的记忆管理工具、以及后台自动抽取整合的 memory manager，1,722 stars、MIT[^12]。Memobase 官方文档定位为面向 AI 应用的长期用户记忆后端，内置 English/Chinese/Japanese 多语言 prompt，但首页仅设"Architecture Overview"入口而未展开架构内容[^35]；其 GitHub 仓库 README 正文抓取失败（main 与 master 分支均 404），**架构主张未获正文取证**[^14]。

---

## 七、评测基准：LoCoMo 与 LongMemEval 的设计差异

**LoCoMo**（arXiv:2402.17753，v1 2024-02-27）由 machine-human pipeline 生成：用 LLM-based agent 架构、以 persona 与 temporal event graph 为 grounding 生成对话，再由人工标注者校验长程一致性与事件图 grounding[^32]。规模为 50 段超长期对话，平均 304.9 turns、19.3 sessions、9,209.2 tokens，跨最多 35 个 session，时间跨度数月，支持多模态[^32][^33]。基准含 question answering、event summarization、multi-modal dialogue generation 三类任务[^32]。核心发现：long-context LLM 与 RAG 在 QA 上带来 22–66% 提升，但仍显著落后人类（低 56%），时序推理上落后更多（低 73%）；long-context LLM 在对抗性问题上比 base model 低 83%，尤其容易把对话或事件错误归属到错误的说话人[^33]。论文认为 RAG 是较平衡的折中，把对话转成关于每个说话人生活与 persona 的 assertion 数据库时表现尤佳[^33]。

**LongMemEval**（ICLR 2025）定义五项核心能力：信息抽取、多会话推理、时间推理、知识更新、拒答，含 500 道人工精编问题[^6]。实例定义为四元组 (S, q, t_q, a)，S 为按时间排序的 N 个历史 session 序列，每个 session 是带时间戳 t_i 的 user-assistant 多轮交互；受 needle-in-a-haystack 启发为每题编译长度可配置的连贯历史，两个标准设置为 LongMemEval_S（每题约 115k tokens）与 LongMemEval_M（500 sessions，约 1.5 million tokens）[^34]。它对既有基准提出两点批评：许多只做人-人对话、交互历史长度不可配置且通常只有几千 token；问题覆盖的记忆能力有限——论文明确指出"包括较新的 LoCoMo 在内的所有长期记忆基准"都未能评估对 assistant 所提供信息的回忆，以及对已更新用户信息的推理[^34]。Table 1 对比显示 LoCoMo 的 KU（知识更新）标为 ✗，而 LongMemEval 五项能力全覆盖[^34]。

**可直接用于设计的三条实证结论**（来自 LongMemEval 全文）：以 round 而非 session 作为存储与使用历史的粒度更优；进一步压缩为单条 user fact 会因信息损失损害整体表现，但能提升 multi-session reasoning 准确率；用抽取的用户事实扩展 key 可提升 recall@k 9.4%、下游 QA 准确率 5.4%；时间感知的索引与查询扩展在强 LLM 做查询扩展时把时序推理的记忆召回提升 6.8%–11.3%；即使召回完美，应用 Chain-of-Note 与结构化数据格式仍可提升最多 10 个绝对点的 QA 准确率[^34]。这些结论与上文 Mem0 的"ADD-only 不覆盖"、Graphiti 的"失效不删除"、Letta 的"文件粒度可重组"在方向上互相呼应，但**现有证据未提供任何工作把这些设计选择与基准分数做消融对照**。

---

## 八、协议与厂商侧：记忆在系统栈中的位置

**MCP 没有 memory 原语。** MCP 是基于 JSON-RPC 2.0 的开放协议，定义 Hosts（发起连接的 LLM 应用）、Clients（宿主内连接器）、Servers（提供上下文与能力的服务）三类角色，借鉴 Language Server Protocol 思路标准化地把额外上下文与工具接入 AI 应用生态[^16]。服务端向客户端提供三类原语：Resources（上下文与数据）、Prompts（模板化消息与工作流）、Tools（供模型执行的函数）；客户端可向服务端提供 Sampling、Roots、Elicitation[^16]。**规范正文中未定义名为 memory 的原语，记忆相关能力只能通过 Resources/Tools 等通用原语间接承载**[^16]。安全侧要求：Hosts 在向服务端暴露用户数据前必须获得明确同意，工具描述与注解应被视为不可信，调用任何工具前必须获得用户明确同意；规范自述 MCP 无法在协议层面强制执行这些原则，实现方 SHOULD 自行构建同意与授权流程[^16]。官方参考实现仓库明确其服务器为演示 MCP 特性与 SDK 用法的教育性示例而非生产级实现，服务器清单应查 MCP Registry[^39]。社区侧存在多个以"为 AI 提供持久记忆"为目标的 MCP 服务器（如 mcp-knowledge-graph ★886、context-portal ★765、MegaMemory ★726、MemoryMesh ★354，最后推送分布在 2025-04 至 2026-07），但**仅取得仓库元数据，未读 README 正文，不足以支撑关于其能力或数据模型的结论**[^40]。

**厂商侧只有 Anthropic 一家有据。** Claude 的 memory 功能让 Claude 记住用户及团队的项目与偏好；为完全可选功能，用户可在设置中查看并编辑；Claude 使用一份 memory summary 集中承载所有记忆，用户可通过对话更新该摘要；使用 projects 时为每个 project 创建独立记忆，项目边界用于隔离敏感对话并被官方定位为安全护栏；Incognito chats 不进入对话历史、不写入记忆[^15]。首发面向 Team 与 Enterprise，2025-10-23 扩展至 Pro 与 Max；企业管理员可随时为组织禁用；支持从其他 AI 工具导入记忆或导出备份；上线前针对敏感福祉类话题做了安全测试，关注记忆是否会强化有害模式、导致过度迎合、以及被用于绕过安全防护[^15]。**OpenAI（ChatGPT memory）与 Google（Gemini 记忆、context caching）的官方能力在现有证据中完全没有来源，三家对比无法成立。**

---

## 矛盾与不确定性

**1. 量化对比整体缺失，这是本报告最大的缺口。** 课题要求给出各代表工作在 LongMemEval、LoCoMo 上的量化结果，但 A-Mem 的 arXiv 摘要、v11 元数据与官方评测仓库 README 均只有"superior improvement against existing SOTA baselines"的定性表述，未出现任何 LoCoMo/LongMemEval 数值[^18][^20]；Mem0 论文摘要只给相对提升，HTML 抓取在 §2.2 截断未读到结果表[^21][^22]。唯一的具体数字 92.5/94.4 来自 Mem0 README 的 2026-04 新算法，且明确标注为托管平台专有优化，不能回填为 2025 年论文系统的成绩[^23]。Zep 的 DMR 94.8% 与 LongMemEval"最高提升 18.5%"是唯一在论文摘要层面给出的具体数字，但缺少绝对准确率与完整结果表[^5]。**因此"各方案孰优"在现有证据下无法判定。**

**2. 基准本身已被指出覆盖不足，且存在厂商自报口径问题。** LongMemEval 明确批评"包括较新的 LoCoMo 在内的所有长期记忆基准"都未评估对 assistant 所提供信息的回忆与对已更新用户信息的推理[^34]；LoCoMo 自身也承认 long-context LLM 在对抗性问题上比 base model 低 83%[^33]。与此同时，Mem0 README 自承分数反映托管平台、开源用户数值不同[^23]，这构成对"基准分数可比性"的实证质疑线索，但**现有证据中没有任何第三方对这些基准或厂商自报口径的公开质疑来源**，该判断只能停留在线索层面。

**3. 记忆准确率与隐私争议缺少外部来源。** 除 Anthropic 自述的安全测试（记忆强化有害模式、过度迎合、被用于绕过防护）[^15]与 MCP 规范自述"无法在协议层面强制执行安全原则"[^16]外，现有证据中没有第三方对记忆准确率、记忆污染或隐私泄露的独立评测或审计来源。

**4. 若干关键证据抓取失败，导致部分方案无法评估。** `WujiangXu/AgenticMemory` 返回 301 重定向且页面抓取失败，stars 与最近提交时间不可得[^38]；Memobase 的 GitHub README 在 main 与 master 分支均 404，其架构主张仅有文档首页的定位描述，无正文取证[^14][^35]；MCP 记忆类社区服务器仅有元数据、无 README 正文[^40]；'context compression LLM agent' 的 arXiv 检索命中 208 条但前列结果噪声极高，仅 MemPrice 一篇相关，且该文摘要不含任何具体分数[^24][^25]。

**5. 时间窗口的边界问题。** 本课题限定"2025 年以来"，但证据中多条内容落在窗口之外或跨越窗口：A-Mem 综述 v1 为 2024-04-21[^1]、LoCoMo 为 2024-02-27[^32]、LongMemEval v1 为 2024-10-14（v2 修订于 2025-03-04，宜作基准来源而非 2025 年代表性方法）[^6]；反之 MemPrice（2026-10-17）[^24]、Mem0 的 2026-04 新算法[^23]、SkeMex（NeurIPS 2026）[^27]等已明显晚于窗口。报告在正文中已逐处标注日期，但"2025 年以来主流方案"这一命题的边界因此需要读者自行按日期裁剪。

**6. 综述定义的关键缺口。** 该领域首篇系统综述声称从狭义与广义两个视角定义 agent 记忆，但抓取内容在进入定义正文前被截断，**该定义的具体表述现有公开资料不足以确认**[^2]。本报告因此只能给出功能定位与形式化执行框架，无法给出记忆本身的标准定义。

## 参考来源

[^1]: A Survey on the Memory Mechanism of Large Language Model based Agents — https://arxiv.org/abs/2404.13501v1（arxiv, 2024-04-21）
[^2]: A Survey on the Memory Mechanism of Large Language Model based Agents (HTML 全文，第 1–3.1 节) — https://arxiv.org/html/2404.13501v1（arxiv, 2024-04-21）
[^3]: A-MEM: Agentic Memory for LLM Agents — https://arxiv.org/abs/2502.12110（arxiv, 2025-02-17）
[^4]: Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory — https://arxiv.org/abs/2504.19413（arxiv, 2025-04-28）
[^5]: Zep: A Temporal Knowledge Graph Architecture for Agent Memory — https://arxiv.org/abs/2501.13956（arxiv, 2025-01-20）
[^6]: LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory — https://arxiv.org/abs/2410.10813（arxiv, 2024-10-14）
[^7]: mem0ai/mem0 — The Memory Layer for AI Agents — https://github.com/mem0ai/mem0（github, 2026-10-23）
[^8]: getzep/graphiti — A Framework for Building Temporal Knowledge Graphs — https://github.com/getzep/graphiti（github, 2026-10-25）
[^9]: letta-ai/letta — Build stateful agents with memory (README 已重定向至 letta-code) — https://github.com/letta-ai/letta（github, 2026-10-18）
[^10]: Letta Docs — Memory & dreaming (MemFS) — https://docs.letta.com/letta-code/memory（web）
[^11]: topoteretes/cognee — The Free Open-Source AI Memory Platform for Agents — https://github.com/topoteretes/cognee（github, 2026-10-23）
[^12]: langchain-ai/langmem — memory for agents that learn over time — https://github.com/langchain-ai/langmem（github, 2026-10-02）
[^14]: memodb-io/memobase — 仅取得仓库元数据（README 正文未取到） — https://github.com/memodb-io/memobase（github, 2026-01-11）
[^15]: Anthropic — Bringing memory to Claude (官方发布页，含 2025-10-23 扩展更新) — https://www.anthropic.com/news/memory（web, 2025-10-23）
[^16]: Model Context Protocol Specification 2025-06-18（官方规范正文） — https://modelcontextprotocol.io/specification/2025-06-18（web, 2025-06-18）
[^17]: nuster1128/LLM_Agent_Memory_Survey（论文官方配套仓库） — https://github.com/nuster1128/LLM_Agent_Memory_Survey（github, 2024-04-13）
[^18]: A-MEM: Agentic Memory for LLM Agents (arXiv:2502.12110v11, NeurIPS 2025) — https://arxiv.org/abs/2502.12110v11（arxiv, 2025-02-17）
[^19]: A-MEM 论文正文（HTML 版，第 1–3 节） — https://arxiv.org/html/2502.12110v11（arxiv, 2025-10-08）
[^20]: WujiangXu/AgenticMemory — A-Mem 官方评测代码仓库 README — https://raw.githubusercontent.com/WujiangXu/AgenticMemory/main/README.md（github）
[^21]: Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory (arXiv:2504.19413v1) — https://arxiv.org/abs/2504.19413v1（arxiv, 2025-04-28）
[^22]: Mem0 论文 HTML 全文（§1–§2.2 可读部分） — https://arxiv.org/html/2504.19413v1（arxiv, 2025-04-28）
[^23]: mem0ai/mem0 README（main 分支） — https://raw.githubusercontent.com/mem0ai/mem0/main/README.md（github）
[^24]: MemPrice: Online Memory Allocation via Adaptive Pricing for Long-Horizon Agents (arXiv:2610.24955) — https://arxiv.org/abs/2610.24955（arxiv, 2026-10-17）
[^25]: arXiv search listing: 'context compression LLM agent' (208 results, newest first) — https://arxiv.org/search/?query=context+compression+LLM+agent&searchtype=all&order=-announced_date_first（arxiv）
[^27]: SkeMex: Experience Makes Skillful — Enabling Generalizable Medical Agent Reasoning via Self-Evolving Skill Memory (NeurIPS 2026) — https://github.com/manglu097/SkeMex（github, 2026-10-08）
[^28]: FactorMiner: A Self-Evolving Agent with Skills and Experience Memory for Financial Alpha Discovery — https://github.com/minihellboy/factorminer（github, 2026-02-26）
[^29]: Awesome-LLM-Agent-Experience-Lifecycle (survey companion repo) — https://github.com/EvaxHe/Awesome-LLM-Agent-Experience-Lifecycle（github, 2026-10-19）
[^30]: GitHub 检索结果：'skill library agent' 前列仓库（Agent Skills 生态） — https://github.com/K-Dense-AI/scientific-agent-skills（github, 2026-10-19）
[^31]: GitHub 检索结果：'self-evolving agent memory experience' 命中仓库（含 Cognitive-Enzyme-oss） — https://github.com/7-e1even/Cognitive-Enzyme-oss（github, 2026-10-23）
[^32]: Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo) — https://arxiv.org/abs/2402.17753（arxiv, 2024-02-27）
[^33]: Evaluating Very Long-Term Conversational Memory of LLM Agents — 全文（数据集统计与实验发现） — https://arxiv.org/html/2402.17753v1（arxiv, 2024-02-27）
[^34]: LongMemEval 全文（v2）：与 LoCoMo 的评测设计差异、问题类型与实验结论 — https://arxiv.org/html/2410.10813v2（arxiv, 2025-03-04）
[^35]: Memobase Documentation — 首页（Architecture Overview 入口） — https://docs.memobase.io/（web）
[^36]: agiresearch/A-mem README — https://raw.githubusercontent.com/agiresearch/A-mem/main/README.md（github, 2025-02-25）
[^37]: WujiangXu/A-mem-sys README — https://raw.githubusercontent.com/WujiangXu/A-mem-sys/main/README.md（github, 2025-07-14）
[^38]: GitHub API 对 WujiangXu/AgenticMemory 的响应（301 重定向） — https://api.github.com/repos/WujiangXu/AgenticMemory（github）
[^39]: modelcontextprotocol/servers — MCP 官方参考实现仓库 — https://github.com/modelcontextprotocol/servers（github, 2024-11-19）
[^40]: GitHub 仓库搜索结果：MCP memory / knowledge-graph 类服务器（社区生态元数据） — https://github.com/search?q=mcp+memory+server+knowledge+graph&type=repositories（github）