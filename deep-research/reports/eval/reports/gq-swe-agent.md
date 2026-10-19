# 自动修 Bug 的 Coding Agent 在 SWE-bench 上的技术水平

## 摘要

SWE-bench 是当前评测「自动修 bug」类 Coding Agent 的核心基准，包含 2,294 个取自 12 个流行 Python 仓库的真实 GitHub issue–PR 对[^1][^2]。其评测信号是 Fail-to-Pass 测试：未应用 PR 改动时失败、合并后通过的一组测试[^2]。2023 年 10 月发布时，最强模型 Claude 2 仅解决 1.96% 的 issue[^1][^2]；SWE-agent 作为首个面向软件工程任务的 agent 系统，把这一数字推到 12.47%[^2][^8]。此后 AutoCodeRover 在 SWE-bench Lite 上报告 19%[^9]，Agentless 在 SWE-bench Lite 上报告 32.00%[^10]。但本报告必须明确一条关键限制：**本次可用证据中没有 2025 年下半年至 2026 年 10 月的 SWE-bench Verified 排行榜条目**，因此「当前（2026 年）技术水平」无法用可核验数值回答。此外，多项研究指出 SWE-bench 存在解决方案泄漏、弱测试、记忆化等数据质量问题，可能显著高估真实能力[^15][^16]。

---

## 一、SWE-bench 是什么：任务定义与评测机制

**任务定义（显式、完整）**：SWE-bench 给定一个 codebase 和一段 issue 描述，语言模型需编辑代码库以解决该 issue[^1][^6]。每个任务实例基于一个 pull request，该 PR 需满足两个条件：(1) 关联一个 issue，(2) 修改了 1 个以上测试相关文件[^2]。每个实例构建一个执行环境（Docker 镜像），仓库安装在 PR 所基于的 commit 上[^2]。

**评测机制**：在未应用 PR 改动时，若干测试失败；PR 合并后，同一组测试通过——这些 Fail-to-Pass 测试是评测的主要信号[^2]。评测流程为：向 AI 系统提供 issue 文本，AI 系统修改代码库；完成后运行 Fail-to-Pass 测试以检查 issue 是否被成功解决[^2]。排行榜每条记录报告 % Resolved，即已解决任务实例的百分比[^3]。

**难度来源**：解决 SWE-bench 中的 issue 常需跨多个函数、类甚至文件同时理解并协调修改，要求模型与执行环境交互、处理超长上下文并进行复杂推理[^1]。

**变体与规模**：官方排行榜列出多个变体及其规模——SWE-bench 2,294 instances；Bash Only 500；Lite 300；Verified 500；Multilingual 300；Multimodal 480[^3]。其中：

- **SWE-bench Verified** 是从 SWE-bench 中人工筛选出的 500 个实例子集，与 OpenAI 合作创建；人工标注者审查每个实例，以确保问题描述清晰、测试补丁正确、且在给定信息下任务可解[^4]。2024 年 8 月 13 日正式发布，是 500 个经真实软件工程师确认可解的问题子集[^6]。
- **SWE-bench Lite** 是从完整基准中精选的 300 个任务子集，目的是降低评测成本、加快迭代[^5]。其筛选标准包括：移除含图片、外部超链接、特定 commit SHA 引用、其他 PR/issue 引用的实例；移除问题陈述少于 40 词的实例；移除修改超过 1 个文件的实例；移除 gold patch 超过 3 个 edit hunks 的实例；移除创建或删除文件的实例；移除包含带错误消息检查的测试的实例；最后从剩余候选中采样 300 个测试实例和 23 个开发实例[^5]。官方建议：当算力效率是考量时，未来系统在 SWE-bench 上评测应报告 SWE-bench Lite 的数字，而非完整集[^5]。
- **Bash Only** 是默认的 Verified 视图：所有模型在同一个 mini-SWE-agent 环境中运行[^3]。该设置下所有 LM 都在最小 bash 环境中使用 mini-SWE-agent 评测（无工具、无特殊 scaffold，仅简单 ReAct agent 循环）[^4]。

**可比性警告**：官方明确指出 release 1.x 与 2.x 结果不一定可比——2.x 使用 tool calling 调用动作，1.x 从输出字符串解析动作；1.x 及更早结果在支持时温度设为 0.0，2.x 及以后不设置温度参数[^4]。

---

## 二、技术水平的时间线：从 1.96% 到 32%（Lite）

### 2.1 起点：2023 年 10 月，1.96%

SWE-bench 于 2023 年 10 月发布，初始 RAG 基线仅得 1.96%[^2]。论文报告当时最强模型 Claude 2 仅能解决 1.96% 的 issue，微调模型 SWE-Llama 也只能解决最简单的问题[^1]。

### 2.2 SWE-agent：首个 SWE agent，12.47%

SWE-agent 是首个面向软件工程任务的 agent 系统，在 SWE-bench 上取得 12.47%[^2]。其论文报告 pass@1 = 12.5%，并称该结果与 HumanEvalFix 上的 pass@1 = 87.7% 均为当时 state-of-the-art[^8]。

**核心概念（显式定义）**：SWE-agent 的核心主张是 LM agent 是一类新的终端用户，需要专门构建的 **agent-computer interface (ACI)** 来使用软件；其定制 ACI 旨在增强 agent 创建/编辑代码文件、导航整个仓库、执行测试与其他程序的能力[^8]。论文明确研究 ACI 设计如何影响 agent 的行为与性能（即接口设计消融）[^8]。

### 2.3 AutoCodeRover：AST 路线，Lite 19%

**核心概念（显式定义）**：AutoCodeRover 将 LLM 与代码搜索能力结合，最终产出程序修改或补丁[^9]。与将项目视为文件集合的 agent 不同，AutoCodeRover 以程序表示（abstract syntax tree, AST）为工作对象[^9]。其代码搜索利用程序结构（类/方法）来增强 LLM 对 issue 根因的理解，并通过迭代搜索检索上下文[^9]。在可用测试套件时，使用基于频谱的故障定位（spectrum-based fault localization using tests）进一步锐化上下文[^9]。

结果：在 SWE-bench-lite（300 个真实 GitHub issue）上解决率 19%，高于当时报告的 SWE-agent；平均成本 $0.43 USD，低于其他基线[^9]。

### 2.4 Agentless：无 agent 三阶段，Lite 32.00%

**核心概念（显式定义）**：Agentless 采用无 agent 的三阶段流程——localization（定位）、repair（修复）、patch validation（补丁验证）[^10]。与 agent 式方法不同，Agentless 不让 LLM 决定未来动作，也不使用复杂工具[^10]。

结果：在 SWE-bench Lite 上达到 32.00%（96 个正确修复），成本 $0.70，论文称其性能最高且成本低，优于所有已有开源软件 agent[^10]。论文还称 Agentless 已被 OpenAI 采用，作为展示 GPT-4o 与新 o1 模型真实编码性能的默认方法[^10]。

### 2.5 官方 experiments 仓库可见条目

官方 experiments 仓库的 evaluation/verified 目录按日期前缀存放各次提交结果，条目命名格式为 YYYYMMDD_<agent/model>[^7]。目录中可见的条目覆盖 2023-10 至 2025-01，例如 20240402_sweagent_claude3opus、20240620_sweagent_claude3.5sonnet、20241029_OpenHands-CodeAct-2.1-sonnet-20241022、20241202_agentless-1.5_claude-3.5-sonnet-20241022、20250118_codeshellagent 等[^7]。**该目录列表在本次抓取中仅返回到 20250118_codeshellagent 为止，未显示 2025 下半年及 2026 年的条目**[^7]。

---

## 三、各系统在 SWE-bench 上报告的成绩（汇总）

| 系统 | 基准变体 | 报告成绩 | 来源 |
|---|---|---|---|
| Claude 2（基线） | SWE-bench | 1.96% | [^1][^2] |
| SWE-Llama（微调） | SWE-bench | 仅能解决最简单的问题 | [^1] |
| SWE-agent | SWE-bench | pass@1 = 12.5% / 12.47% | [^2][^8] |
| AutoCodeRover | SWE-bench Lite | 19% | [^9] |
| Agentless | SWE-bench Lite | 32.00% | [^10] |
| OpenHands | SWE-bench | **证据缺失** | — |

关于 OpenHands：论文正文（v3 HTML）确认 OpenHands 前身为 OpenDevin，是一个允许 agent 通过写代码、操作命令行、浏览网页与外部世界交互的平台[^20]；论文正文引言部分提及 SWE-bench（Jimenez et al., 2024）作为软件工程类评测任务之一[^20]。**但本次抓取仅覆盖到引言部分，未取到包含 SWE-bench resolve rate 的结果表格**[^20]。其 GitHub 仓库 README 已改版为 "Agent Canvas" 产品介绍（自托管 coding agent 控制中心），**不再包含 SWE-bench benchmark 分数表**[^19]。因此 OpenHands 在 SWE-bench 上的具体成绩在本次证据中无法确认。

OpenHands 平台本身的定位：用于开发以类似人类开发者方式与世界交互的 AI agent——写代码、操作命令行、浏览网页；支持实现新 agent、在沙箱环境中安全执行代码、多 agent 协调，以及集成评估基准[^11]。论文被 ICLR 2025 接收[^11]。以 MIT 许可发布，是横跨学术界与工业界的社区项目，有超过 188 名贡献者、2.1K+ 次贡献[^11]。仓库 OpenHands/OpenHands 创建于 2024-03-13，最近 push 于 2026-10-14，约 90,873 stars、12,044 forks，语言为 TypeScript，MIT 许可[^19]。

---

## 四、评测生态的治理变化

自 2025-11-18 起，SWE-bench Verified 与 Multilingual 仅接受来自学术团队与研究机构的提交，要求开源方法且经同行评审[^14]。提交须满足全部条件：包含 arXiv 预印本或技术报告链接；至少一位作者隶属于学术机构或成熟研究实验室（如大学、Google DeepMind、Meta FAIR、Microsoft Research、前沿实验室）[^14]。政策举例：OpenHands、SWE-RL、FrogBoss、AutoCodeRover 仍符合资格；Augment Code、Solver AI、Honeycomb.sh 不再符合资格[^14]。该日期之前的提交继续按旧规则处理；此日期之后不符合条件的提交将被关闭[^14]。SWE-bench Multimodal 排行榜仍对所有人开放提交[^14]。

官方 experiments 仓库的提交记录结构：每个提交子文件夹包含 metadata.yaml（记录谁运行了什么、产物位置）、README.md（系统描述）、logo.png 与 results/（resolved instance ids 及细分）[^13]。仓库只保存条目而非产物：预测、执行日志与推理轨迹存放在提交者自己的公开仓库中，由 metadata.yaml 的 assets 块指定[^13]。2026 年中之前的条目其 assets 指向 s3://swe-bench-submissions/[^13]。

---

## 五、矛盾与不确定性

### 5.1 核心问题无法回答：2025 下半年至 2026 年的当前技术水平

本报告最关键的结论是：**「当前（截至 2026 年 10 月）SWE-bench Verified 排行榜上排名靠前的 Coding Agent / 模型及其 resolve rate」这一问题，在本次可用证据中无法回答。** 唯一直接证据是 SWE-bench/experiments 的 evaluation/verified 目录列表，且该抓取仅返回到 20250118_codeshellagent，完全没有 2025 下半年至 2026 年 10 月的条目[^7]。因此关于当前技术水平的具体数值、发布时间、scaffold 均无任何可核验证据支撑。

### 5.2 SWE-Bench Pro 的 23% vs 70%+ 数值可靠性存疑

Scale AI 的 SWE-Bench Pro 公开排行榜页面声称：top models 在 SWE-Bench Pro public set 上得分约 23%，而在 SWE-Bench Verified 上为 70%+[^17]。该数值存在两重问题：

1. 它仅来自 Scale AI 的 leaderboard 营销页面，**未与官方 SWE-bench 排行榜或任何独立来源交叉验证**[^17]；
2. SWE-Bench Pro 官方仓库自述 2025-05-18 发现 leaderboard 存在问题、正在修复（"We have identified some issues with the leaderboard and are currently working on addressing them"），此前还移除过部分过时/误纳入的单元测试[^18]。

因此该数值作为「当前技术水平」的锚点证据不足。

SWE-Bench Pro 本身的定位与规模（供参考）：包含 1,865 个问题，来自 41 个活跃维护的仓库，覆盖业务应用、B2B 服务和开发者工具[^12]。基准划分为三个部分：公开集（11 个仓库，可公开访问）、held-out 集（12 个仓库）和商业集（18 个专有仓库，与早期创业公司有正式合作协议）[^12]。任务为长时程（long-horizon）任务，专业软件工程师可能需要数小时到数天完成，通常涉及跨多个文件的补丁和大量代码修改[^12]。主指标为 Resolve Rate：仅当补丁同时满足「修复问题（fail-to-pass 测试通过）」且「无回归（pass-to-pass 测试保持通过）」时才计为 resolved[^17]。其 V2（2025-09-22 发布）为已验证公开划分：642 个任务、覆盖 11 个仓库，每个任务以自包含 Harbor 任务目录形式提供（含 verifier、参考解、公开容器镜像），并含 HARD-51 子集与锁定协议工具[^18]。官方提供 oracle 参考补丁可解析 642/642 任务[^18]。

### 5.3 OpenHands 成绩缺失

如第三节所述，OpenHands 论文证据仅覆盖引言部分，未取到含 SWE-bench resolve rate 的结果表格[^20]；GitHub 仓库 README 已改版为 Agent Canvas 产品介绍，不再含 benchmark 分数表[^19]。因此各系统「在 SWE-bench 上报告的成绩」中 OpenHands 一项缺失。

### 5.4 基准本身的数据质量争议

多项独立研究对 SWE-bench 的有效性提出质疑：

**SWE-Bench+** 对 SWE-bench 数据集质量做了系统性人工筛查，比较 SWE-Agent+GPT-4 生成的补丁与真实 PR[^15]：
- 32.67% 的成功补丁属于「作弊」，因为解决方案直接出现在 issue 报告或评论中，作者称之为 solution leakage（解决方案泄漏）问题[^15]。
- 31.08% 通过的补丁是可疑补丁，源于弱测试用例（测试不足以验证补丁正确性）[^15]。
- 剔除这些问题实例后，SWE-Agent+GPT-4 的解决率从 12.47% 降至 3.97%[^15]。
- 同样的数据质量问题也存在于 SWE-bench Lite 和 SWE-bench Verified 两个变体中[^15]。
- 超过 94% 的 issue 创建于 LLM 知识截止日期之前，构成潜在的数据泄漏问题[^15]。

**The SWE-Bench Illusion** 提出两个诊断任务来探测模型底层知识：仅凭 issue 描述识别 buggy 文件路径；仅凭当前文件上下文与 issue 描述复现 ground truth 函数[^16]。作者给出实证证据表明 SWE-Bench-Verified 上的性能提升可能部分由记忆化（memorization）而非真实问题解决能力驱动[^16]：
- SOTA 模型仅凭 issue 描述、不访问仓库结构，识别 buggy 文件路径的准确率最高达 76%[^16]。
- 在未包含于 SWE-Bench 的仓库任务上，同一任务准确率最高仅 53%，指向可能的数据污染或记忆化[^16]。
- 函数复现任务中，SWE-Bench Verified 上的逐字相似度显著高于其他同类编码基准：SWE-Bench Verified 与 Full 上连续 5-gram 准确率最高达 35%，而其他基准任务最高仅 18%[^16]。
- 作者据此对现有评估结果的有效性提出质疑，并呼吁更稳健、抗污染的基准[^16]。

**Agentless 作者**也人工分类 SWE-bench Lite 问题，发现存在精确 ground truth patch 或 issue 描述不足/误导的问题，据此构建 SWE-bench Lite-S 以进行更严格评估[^10]。

### 5.5 版本与可比性陷阱

官方明确 release 1.x 与 2.x 结果不一定可比（tool calling vs 字符串解析动作；温度设置差异）[^4]。此外，SWE-bench 原始集（2,294）、Verified（500）、Lite（300）、Bash Only（500）是不同规模的变体，跨变体的百分比不可直接比较[^3]。官方建议在算力效率考量下报告 Lite 数字[^5]，而 Bash Only 是默认的 Verified 视图[^3]。

---

## 参考来源

[^1]: SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (arXiv:2310.06770v3) — https://arxiv.org/abs/2310.06770v3 (arxiv, 2023-10-10)
[^2]: SWE-bench (original benchmark) — Official Overview — https://www.swebench.com/original.html (web)
[^3]: SWE-bench Leaderboards — Official (variant sizes) — https://www.swebench.com/ (web)
[^4]: SWE-bench Verified — Official Overview — https://www.swebench.com/verified.html (web)
[^5]: SWE-bench Lite — Official Overview (selection criteria) — https://www.swebench.com/lite.html (web)
[^6]: SWE-bench/SWE-bench — Official Repository README — https://raw.githubusercontent.com/SWE-bench/SWE-bench/main/README.md (github)
[^7]: SWE-bench/experiments — evaluation/verified directory listing — https://github.com/SWE-bench/experiments/tree/main/evaluation/verified (github)
[^8]: SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering — https://arxiv.org/abs/2405.15793v3 (arxiv, 2024-05-06)
[^9]: AutoCodeRover: Autonomous Program Improvement — https://arxiv.org/abs/2404.05427 (arxiv, 2024-04-08)
[^10]: Agentless: Demystifying LLM-based Software Engineering Agents — https://arxiv.org/abs/2407.01489 (arxiv, 2024-07-01)
[^11]: OpenHands: An Open Platform for AI Software Developers as Generalist Agents — https://arxiv.org/abs/2407.16741 (arxiv, 2024-07-23)
[^12]: SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks? — https://arxiv.org/abs/2509.16941 (arxiv, 2025-09-21)
[^13]: SWE-bench/experiments — repository README — https://github.com/SWE-bench/experiments (github)
[^14]: SWE-bench/experiments — Verified submission policy note (11/18/2025) — https://raw.githubusercontent.com/SWE-bench/experiments/main/README.md (github, 2025-11-18)
[^15]: SWE-Bench+: Enhanced Coding Benchmark for LLMs — https://arxiv.org/abs/2410.06992 (arxiv, 2024-10-09)
[^16]: The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason — https://arxiv.org/abs/2506.12286 (arxiv, 2025-06-14)
[^17]: SWE-Bench Pro Public Leaderboard (Scale AI) — https://scale.com/leaderboard/swe_bench_pro_public (web)
[^18]: scaleapi/SWE-bench_Pro-os (official repository) — https://github.com/scaleapi/SWE-bench_Pro-os (github)
[^19]: OpenHands/OpenHands (GitHub repository) — https://github.com/OpenHands/OpenHands (github, 2024-03-13)
[^20]: OpenHands: An Open Platform for AI Software Developers as Generalist Agents (HTML full text, v3) — https://arxiv.org/html/2407.16741v3 (arxiv, 2025-04-18)