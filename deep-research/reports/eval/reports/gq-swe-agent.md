# 自动修 bug 的 Coding Agent 在 SWE-bench 上的技术水平

## 摘要

SWE-bench 是自动修 bug 类 Coding Agent 的核心评测场：2,294 个任务实例取自 12 个流行 Python 仓库的真实 GitHub issue 与对应 PR，判定标准是 patch 应用后全部 fail-to-pass 测试通过，指标为 resolved 百分比[^1][^2]。技术水平可沿三条线索刻画：(1) 2023 年 10 月基准发布时，最好的非交互式 RAG 基线仅 1.96%[^1][^2]；(2) 2024 年 3 月 SWE-agent 以 agent-computer interface（ACI）把分数推到 12.47%（GPT-4 Turbo），首次证明交互式 agent 形态相对非交互式检索的跃升[^2][^9]；(3) 2025 年 7 月官方 News 记录 mini-SWE-agent 以约 100 行 Python 在 SWE-bench Verified 上达到 65%[^5][^21]。此后到 2026 年 10 月的 SOTA 轨迹在本次证据中**完全缺失**——官方排行榜页面不内联数值、Anthropic 三篇发布博客正文均未抓到 SWE-bench Verified 百分比、针对性 arXiv 检索未命中[^5][^16][^17][^18][^19]。同时，基准本身正在分叉：Verified 是 500 题人工筛选子集[^3]，SWE-Bench Pro 以 1,865 题、平均跨 4.1 文件/107.4 行的长程任务把最强模型压到 Pass@1 < 45%[^11]，而 The SWE-Bench Illusion 用两个诊断任务指认 Verified 高分部分来自记忆化[^12]。因此「当前技术水平」的诚实回答是：**Verified 上 65% 量级有官方锚点、>74% 仅有未经核验的自述；换到抗污染、高复杂度的 Pro 上，同一代模型仍在 40% 出头**。

---

## 1. 评测对象：SWE-bench 到底测什么

**任务定义（显式）**：给定一个代码库（codebase）与一段待解决 issue 的自然语言描述，语言模型需编辑代码库以解决该 issue[^1]。每个任务实例绑定一个由 PR base commit 指定的代码库版本与一个 Docker 执行环境[^1][^2]。

**数据构造流程**：从 12 个流行开源 Python 仓库收集约 90,000 个 PR[^1]；每个实例对应一个满足两条件的 PR——(1) 关联一个 issue，(2) 修改了 1 个以上测试相关文件[^2]；再经两道过滤：剔除没有任何 fail-to-pass 测试的实例、剔除导致安装或运行时错误的实例，最终缩减为 2,294 个[^1]。

**判定标准（显式）**：若 patch 成功应用且所有 fail-to-pass 测试通过，则认为解决了该 issue；基准指标为 resolved 任务实例的百分比[^1]。官方把这组「无 PR 改动时失败、PR 合并后通过」的测试称为评测的 primary signal[^2]。

**难度画像**：参考解平均修改 1.7 个文件、3.0 个函数、32.8 行（增删合计）；issue 描述平均 195 词；平均每个实例 9.1 个 fail-to-pass 测试、测试总数平均 120.8 个；代码库非测试文件平均 3,010 个、438K 行[^1]。论文明确指出解决这类 issue 常需跨多个函数、类乃至文件协调修改[^25]。

**污染控制**：训练集 SWE-bench-train 的 19,000 个实例来自另外 37 个仓库，与评测基准仓库集合不相交[^1]；采集流程可套用于任意 GitHub Python 仓库，因此能持续加入模型训练日期之后创建的 issue[^1]。

**子集与变体**：Lite 为 300 实例、覆盖 12 仓库中的 11 个[^1][^5]；Verified 为与 OpenAI 合作人工逐条审查出的 500 实例，确保问题描述清晰、测试 patch 正确、任务可解[^3]；Multimodal 增加 517 个含视觉元素的 issue，v2 保留 480 个可复现任务[^4]；Multilingual 为 300 实例、42 仓库 9 种语言[^5]。

## 2. 演进锚点：从 1.96% 到 65%

证据能支撑的时间序列只有以下几个点，且分属不同基准视图，不可混用：

| 时间 | 系统 | 基准视图 | resolved % | 证据 |
|---|---|---|---|---|
| 2023-10 | Claude 2 + BM25 检索 | SWE-bench 全量 2,294 | 1.96% | [^1][^2][^25] |
| 2023-10 | Claude 2 + oracle 检索 | SWE-bench 全量 | 4.8% | [^1] |
| 2024-03 | SWE-agent（GPT-4 Turbo） | SWE-bench 全量 2,294 | 12.47% | [^2][^9] |
| 2024-03 | SWE-agent（Claude 3 Opus） | SWE-bench 全量 | 10.5% | [^9] |
| 2024-04 | AutoCodeRover | SWE-bench Lite 300 | 19% | [^24] |
| 2024-07 | Agentless | SWE-bench Lite 300 | 32.00%（96 个正确修复），成本 $0.70 | [^23] |
| 2025-03 | SWE-agent 1.0 | SWE-bench Lite | 开源 SOTA（官方 News 未给数值） | [^5] |
| 2025-07 | mini-SWE-agent（约 100 行 Python） | SWE-bench Verified | 65% | [^5] |

几点必须注意的口径问题：

- 1.96% 是**全量集**数字，不是 Verified 子集数字[^25]；把它与后来的 Verified 分数并列比较是错的。
- SWE-agent 的 12.47% 相对「此前非交互式检索增强系统的最好成绩 3.8%」是形态级跃升，作者归因于 ACI 设计而非模型权重改动[^9]。
- Agentless 的 32.00% 走的是三阶段固定流水线（localization → repair → patch validation），不让 LLM 自主决定后续动作[^23]——说明 2024 年中期的分数提升并不全来自「agent 自主性」。
- 2025 年 7 月之后到 2026 年 10 月，**没有任何可引用的官方 resolved % 数值**。官方排行榜页面自述需通过交互式 Compare results 加载具体条目，返回内容不含任何模型的 % Resolved[^5]；针对「SWE-bench Verified leaderboard agent resolved rate cost per instance」的 arXiv 检索返回的 5 篇论文主题分别为具身策略表征对齐、test-time 推理剪枝、随机特征相变、选择性审计的 AI 辅助决策、3D 触觉传感，与该问题无关[^16]。

## 3. 系统形态：ACI 与「百行 agent」两端

**SWE-agent 的 ACI 定义（显式）**：ACI 是 LM agent 与计算机之间的抽象层，包含 agent 可用的命令以及环境反馈的格式[^9]。其组件分四类：search/navigation、file viewer、file editor、context management[^9]。每个 step 中 agent 生成一个 thought 和一个 command，命令执行的环境反馈并入上下文（ReAct 范式）[^9]。四条设计原则为：动作简单易懂、动作紧凑高效、环境反馈有信息量但简洁、guardrails 抑制错误传播并加快恢复[^10]。具体机制包括：find_file / search_file / search_dir 三个检索命令，单次查询最多返回 50 条结果，超限则不返回结果而提示 agent 写出更具体的查询[^10]；open 命令一次展示最多 100 行窗口，可用 scroll_down / scroll_up / goto 移动，并显示完整路径、总行数、省略行数与行号[^10]；edit 接收起始行、结束行、替换文本三个必需参数，单步替换该区间并自动回显更新后内容[^10]。ACI 配置通过人工检查 agent 行为 + grid search 在开发集上优化[^10]。消融显示：在 Lite 上 SWE-agent 比仅用默认 Linux shell 的 baseline agent 多解决 10.7 个百分点[^9]。

**mini-SWE-agent 的反向设计**：除 bash 外没有任何工具，不使用 LM 的 tool-calling 接口，历史完全线性，每个动作通过 subprocess.run 独立执行（无有状态 shell 会话），agent 类约 100 行 Python，支持 local / docker / podman / singularity / apptainer 环境[^21]。README 声称其在 SWE-bench Verified 上「>74%」[^21]，仓库 desc 字段同样写 >74%[^6]。**该数字不可作为结论**：README 未在该表述处给出基座模型、具体版本号、单实例成本或步数上限[^21]；README 自述当前为 v2、旧版在 v1 branch[^21]；而官方 Verified 页面明确说明 1.x 与 2.x 因 2.x 用 tool calling 调用动作、1.x 从输出字符串解析动作而「不一定可比」[^3]。同一数字在不同版本语义下不可直接比较。

**其他开源平台**：OpenHands（原 OpenDevin，MIT 许可）通过 docker 沙箱运行 bash shell、Chromium（Playwright）与 IPython 服务器，以 CodeAct 架构的通用 agent 为核心，评估框架支持 15 个基准含 SWE-bench[^22]；但该来源未给出 SWE-bench Verified 或 Lite 的具体 resolved %[^22]。

**商业模型侧**：Anthropic 称 Claude Sonnet 4.5 在 SWE-bench Verified 上达到 state-of-the-art，但博客正文未给出百分比（数字在图表中未被抓取）[^17]；Opus 4.5 博客称「state-of-the-art on tests of real-world software engineering」，正文同样不含 Verified 百分比[^19]。更值得注意的是，Sonnet 5.5（2026-09-28）的官方基准表**未出现 SWE-bench Verified**，主推指标改为 Terminal-Bench 4.0（70.6%）、FrontierCode 1.1（46.2%）、CursorBench 4.0（55.5%）等，并以「cost per task」为横轴绘制 accuracy-vs-cost 图[^18]。这构成一个间接信号：头部厂商的旗舰发布叙事正在从 SWE-bench Verified 迁出，但现有证据不足以确认这是饱和、是污染顾虑还是产品定位变化。

## 4. 官方评测纪律与成本维度

官方对 Verified 排行榜的定位值得单独指出：为便于语言模型对等比较，官方用 mini-SWE-agent 在最小 bash 环境中评测所有 LM——无工具、无特殊 scaffold，仅一个简单 ReAct 循环[^3]；温度设置上 1.x 及更早版本设为 0.0，2.x 及以后不设该参数[^3]；官方声明目标不是调参刷高分，而只做通用修复与 prompt 澄清[^3]。这意味着「Bash Only」视图（500 实例）测的是**模型本身**，而各系统自报的分数测的是**模型 + scaffold**，两者不可混谈。

成本与步数是一等指标：排行榜维度包含 resolved vs cost 散点、resolved vs average cost、resolved vs cost limit、resolved vs step limit、累计成本与累计步数分布[^5]。历史成本锚点：AutoCodeRover 平均 $0.43[^24]、Agentless $0.70[^23]。注意 Anthropic 博客给出的 $3/$15、$2/$10、$5/$25 每百万 token 是 API token 价格，不是 SWE-bench 单任务成本[^17][^18][^19]。

## 5. 基准的分叉：Verified 饱和 vs Pro 的 40% 出头

SWE-Bench Pro（Scale AI，1,865 题、41 个活跃维护仓库）对当前基准提出两条明确批评：一是易受污染——MIT/Apache 2.0/BSD 等 permissive 许可的流行开源仓库是预训练网络爬取语料的主要候选；二是任务复杂度不足[^11]。具体到数字：Verified 的 500 题中有 161 题属于相对平凡的问题，仅需 1–2 行修改，而工业界企业级工程常需跨文件、数百行修改[^11]。Pro 因此排除 1–10 行琐碎改动，参考解平均跨 4.1 个文件、107.4 行，每题至少 10 行改动，超过 100 个任务需 100 行以上修改[^11]。抗污染设计两条：public 与 held-out 集仅用强 copyleft（GPL）许可仓库（对商业训练语料构成法律障碍），以及从真实初创公司获取私有商业代码库[^11]。

统一 scaffold 下，Pro 上主流模型 Pass@1 均低于 45%；public 子集 Resolve %：Claude Sonnet 4.5 = 43.6、Claude Sonnet 4 = 42.7、GPT-5 (high) = 41.8、Claude Haiku 4.5 = 39.5，Kimi K2 Instruct 紧随其后（数值抓取被截断）[^11]。作者称 Pro 是「contamination-resistant testbed」，用于衡量 SWE-Bench Verified 饱和之后的进展[^11]。

## 6. 分数的可信度：记忆化质疑

The SWE-Bench Illusion 设计两个诊断任务探测底层知识而非解题能力：仅凭 issue 描述识别 bug 所在文件路径（不给仓库结构），以及仅凭当前文件上下文与 issue 描述复现 ground truth 函数[^12]。结果：SOTA 模型仅凭 issue 描述定位 bug 文件路径最高 76% 准确率，而在非 SWE-Bench 仓库任务上该指标最高仅 53%[^12]；函数复现任务上，Verified 与 Full 的逐字相似度（连续 5-gram 准确率）最高 35%，其他同类编码基准最高仅 18%[^12]。作者结论是 Verified 上的性能提升可能部分由记忆化而非真实问题求解能力驱动，现有评测协议可能高估真实能力，并呼吁更抗污染的基准[^12]。

## 矛盾与不确定性

**（一）核心数值缺口：2025 下半年至 2026 年 10 月的 SOTA 轨迹缺失。** 现有证据只能给出 2024-03 的 12.47% 与 2025-07 的 65% 两个官方锚点[^2][^5]，中间与之后的 Verified/Lite SOTA 演进**现有公开资料不足以确认**。原因具体：官方 swebench.com 页面不内联任何模型的 % Resolved，需交互式加载[^5]；Anthropic 三篇发布博客（Sonnet 4.5 / Opus 4.5 / Sonnet 5.5）正文均未抓到 Verified 百分比，Sonnet 5.5 的基准表甚至不含该指标[^17][^18][^19]；官方产品页 Benchmarks 章节仅作定性描述[^20]；arXiv 定向检索完全跑偏[^16]；第三方聚合仓库（ai-agent-benchmark、harness-bench、awesome-llm-bench、showdown）均只取得元数据、未读正文，且非官方来源，不能作为权威数值[^14][^15][^26][^27]；官方排行榜源仓库同样只拿到元数据[^13]。

**（二）>74% 不可采信。** mini-swe-agent 的 >74% 仅来自 README 一句表述与仓库 desc 字段，缺基座模型、版本号、harness 配置、单实例成本、pass@k 与评测日期[^6][^21]；且 v1/v2 因动作解析机制不同而官方声明「不一定可比」[^3]。augmentcode 自述「#1 open-source SWE-bench Verified implementation」属营销主张，无任何分数与配置信息[^7]；agentic-harness-engineering 的 84.7% 是 Terminal-Bench 2 而非 SWE-bench 分数[^8]。三者均不得用于回答 SOTA 问题。

**（三）未调和的表面矛盾：Verified 高分与 Pro <45%、Illusion 记忆化结论并存。** 若 Verified 已达 65–74% 量级，而 Pro 上最强模型仅 43.6%[^11]，且 Illusion 指认 Verified 提升部分源于记忆化[^12]，三者必须被解释而非并列。证据本身提供了三条可能的解释轴，但**没有任何来源做过调和性实验**：(a) 任务复杂度差异——Verified 有 161/500 题仅需 1–2 行修改，Pro 参考解平均 107.4 行、跨 4.1 文件[^11]；(b) 污染差异——Pro 用 GPL copyleft 与私有商业代码库做抗污染，Verified 的仓库来自 permissive 许可的流行开源项目[^11][^12]；(c) harness 差异——Pro 用统一 scaffold 且 Pass@1，Verified 各系统自报分数对应各自 scaffold，官方 Bash Only 视图则统一用 mini-SWE-agent[^3][^5][^11]。三者各占多少权重，现有公开资料不足以确认。

**（四）其他口径风险。** 官方 News 的「2025-03 SWE-agent 1.0 是 SWE-bench Lite 上的开源 SOTA」未给数值[^5]；Agentless 的 32.00% 与 AutoCodeRover 的 19% 均来自论文摘要，正文结果表格与实验配置（模型、采样预算、pass@1 定义）未读到[^23][^24]；Agentless 另构建了剔除 ground truth patch 或描述不足样本的 SWE-bench Lite-S 用于更严格评测，其与 Lite 原始分数的关系未在证据中展开[^23]；SWE-bench Multimodal v2 标注日期为 2026 年 9 月 1 日、重建 Docker 环境以解决依赖与浏览器漂移[^4]，说明即便是官方子集，环境漂移也会影响跨期可比性。

## 参考来源

[^1]: SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (arXiv:2310.06770v3, ICLR 2024) — https://arxiv.org/abs/2310.06770（arxiv, 2023-10-10 (v1); v3 2024-11-11）
[^2]: SWE-bench 官方站点 — Original（Overview / 评测协议） — https://www.swebench.com/original.html（web）
[^3]: SWE-bench Verified 官方页面 — https://www.swebench.com/verified.html（web）
[^4]: SWE-bench Multimodal 官方页面 — https://www.swebench.com/multimodal.html（web）
[^5]: SWE-bench Leaderboards (official site, swebench.com) — https://www.swebench.com/（web）
[^6]: SWE-agent/mini-swe-agent (repository search result metadata only) — https://github.com/SWE-agent/mini-swe-agent（github）
[^7]: augmentcode/augment-swebench-agent (repository search result metadata only) — https://github.com/augmentcode/augment-swebench-agent（github）
[^8]: china-qijizhifeng/agentic-harness-engineering (repository search result metadata only) — https://github.com/china-qijizhifeng/agentic-harness-engineering（github）
[^9]: SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering (arXiv:2405.15793v3) — https://arxiv.org/abs/2405.15793v3（arxiv, 2024-05-06 (v1); v3 revised 2024-11-11）
[^10]: SWE-agent 论文正文（HTML 全文，第 2–3 节：ACI 设计原则与命令细节） — https://arxiv.org/html/2405.15793v3（arxiv, 2024-11-11 (v3)）
[^11]: SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks? — https://arxiv.org/abs/2509.16941（arxiv, 2025-09-21 (v1); v2 2025-11-14）
[^12]: The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason — https://arxiv.org/abs/2506.12286（arxiv, 2025-06-14 (v1); v4 2025-12-01）
[^13]: SWE-bench/swe-bench.github.io — Landing page + leaderboard for SWE-Bench benchmark — https://github.com/SWE-bench/swe-bench.github.io（github）
[^14]: murataslan1/ai-agent-benchmark — AI coding agents comparison, SWE-Bench leaderboard, pricing — https://github.com/murataslan1/ai-agent-benchmark（github）
[^15]: TheLime1/harness-bench — Benchmark leaderboard for AI coding harnesses by model — https://github.com/TheLime1/harness-bench（github）
[^16]: arXiv 检索（query: SWE-bench Verified leaderboard agent resolved rate cost per instance）返回结果集 — https://arxiv.org/abs/2610.31670v1（arxiv, 2026-10-22）
[^17]: Introducing Claude Sonnet 4.5 (Anthropic 官方博客) — https://www.anthropic.com/news/claude-sonnet-4-5（web, 2025-09-29）
[^18]: Claude Sonnet 5.5 发布博客（Anthropic，2026-09-28） — https://www.anthropic.com/news/claude-sonnet-5-5（web, 2026-09-28）
[^19]: Introducing Claude Opus 4.5 (Anthropic 官方博客) — https://www.anthropic.com/news/claude-opus-4-5（web, 2025-11-24）
[^20]: Claude 模型线产品页（Anthropic，含 Sonnet 4.5 / 4.6 / 5 / 5.5 时间线） — https://www.anthropic.com/claude/sonnet（web）
[^21]: SWE-agent/mini-swe-agent README (main branch, v2) — https://raw.githubusercontent.com/SWE-agent/mini-swe-agent/main/README.md（github）
[^22]: OpenHands: An Open Platform for AI Software Developers as Generalist Agents (arXiv:2407.16741v3) — https://arxiv.org/abs/2407.16741v3（arxiv, 2024-07-23 (v1); v3 revised 2025-04-18）
[^23]: Agentless: Demystifying LLM-based Software Engineering Agents (arXiv:2407.01489v2) — https://arxiv.org/abs/2407.01489v2（arxiv, 2024-07-01）
[^24]: AutoCodeRover: Autonomous Program Improvement (arXiv:2404.05427v3) — https://arxiv.org/abs/2404.05427v3（arxiv, 2024-04-08）
[^25]: SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (arXiv:2310.06770v3) — https://arxiv.org/abs/2310.06770v3（arxiv, 2023-10-10）
[^26]: leoncuhk/awesome-llm-bench — Daily-synced Top 10 LLM leaderboards (SWE-bench Verified, Terminal-Bench, OSWorld, ARC-AGI-2, HLE) — https://github.com/leoncuhk/awesome-llm-bench（github）
[^27]: verseles/showdown — Comprehensive LLM leaderboard aggregating multiple benchmarks — https://github.com/verseles/showdown（github）