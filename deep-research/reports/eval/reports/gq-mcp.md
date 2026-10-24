# MCP 解决了什么问题、架构是什么：基于一手规范与生态证据的评估

## 摘要

Model Context Protocol（MCP）是 Anthropic 于 2024 年 11 月发布的开放标准，官方定义是「一个开放标准，用于把 AI 助手连接到数据所在的系统，包括内容仓库、业务工具和开发环境」[^1]；规范层面的定义是「一个开放协议，使 LLM 应用与外部数据源和工具之间实现无缝集成，为 LLM 提供连接其所需上下文的标准化方式」[^3]。它要解决的问题被官方表述为**集成碎片化**：每接入一个新数据源都需要各自的定制实现，使真正互联的系统难以规模化[^1]；工程侧表述为「为每一个配对（each pairing）做定制集成」[^6]。其解法是用单一协议取代碎片化集成[^1]，官方类比是「AI 应用的 USB-C 端口」[^2]，设计上受 Language Server Protocol 启发[^3][^46]。

架构上，MCP 采用 client–host–server 模型，基于 JSON-RPC 2.0，分为数据层与传输层[^4][^7]；服务器侧提供 Tools / Resources / Prompts 三类原语，分别由模型、应用、用户控制[^5][^10]，客户端侧曾提供 Sampling / Roots / Elicitation[^3][^11][^12]。

**必须前置声明的关键不确定性**：MCP 的架构在 2024-11 至 2026-07 之间发生了方向性变化——2026-07-28 版把基础协议从「有状态连接」改为「无状态、自包含请求」与「逐请求能力协商」，移除 Streamable HTTP 的协议级会话与 `Mcp-Session-Id`，弃用 Sampling 与 Roots，并以 Multi Round-Trip Requests 取代服务器主动发起的请求[^19][^20][^42][^43]。因此下文将「2025-06-18 世代架构」与「2026-07-28 世代架构」分开陈述，不做混同。

---

## 一、MCP 要解决的问题

### 1.1 官方动机：数据孤岛与逐点定制集成

Anthropic 发布博客给出的动机是双向的：一方面模型能力快速进步，另一方面「即使最复杂的模型也受限于与数据的隔离——数据被困在信息孤岛和遗留系统中」[^1]。碎片化被明确表述为「每接入一个新数据源都需要各自的定制实现」，结果是真正互联的系统难以规模化[^1]。工程博客把同一问题从实现成本角度重述为：把 agent 连接到工具和数据，传统上需要为每一个配对做定制集成，造成碎片化与重复劳动[^6]。MCP 的解法表述是提供通用开放标准，用单一协议取代碎片化的集成[^1]；对开发者的价值主张相应是「在 agent 中实现一次 MCP，即可解锁整个集成生态」[^6]，官方文档的说法是「build once and integrate everywhere」[^2]。

**术语校正**：常被用来概括这一动机的「N×M 集成爆炸」并非官方表述。在本次可用的全部证据中，Anthropic 发布博客、各版规范与官方文档、官方仓库 README 均未出现「N×M」「M×N」或「集成爆炸」字样[^1][^45][^46][^47]；最接近的只有「Every new data source requires its own custom implementation」[^1]与「a custom integration for each pairing」[^6]。该术语属二手概括，不应作为官方动机直接引用。

### 1.2 标准化对象：上下文、工具与工作流

MCP 声称标准化的不只是「工具调用」，而是三类对象的接入：数据源（如本地文件、数据库）、工具（如搜索引擎、计算器）与工作流（如专门化的 prompts）[^2]。规范把这一能力拆为三项：与语言模型共享上下文信息、向 AI 系统暴露工具与能力、构建可组合的集成与工作流[^3]。LSP 类比说明了标准化的层级：LSP 标准化了如何在整个开发工具生态中为编程语言添加支持，MCP 类似地标准化如何把额外上下文与工具集成进 AI 应用生态[^3][^46]。

### 1.3 一个未被官方列入设计目标、但被实测暴露的问题：上下文开销

Anthropic 工程博客披露了 MCP 客户端工作方式带来的副作用：多数客户端把所有工具定义预先加载进上下文窗口，工具描述占用上下文空间、增加响应时间与成本；当 agent 连接数千个工具时，需在处理请求前先处理数十万 token[^6]。此外每次工具调用的中间结果都必须经过模型——把 Google Drive 会议记录写入 Salesforce 时，完整记录会在上下文中流转两次，两小时销售会议可能额外处理约 50,000 token，更大文档可能超出上下文窗口限制[^6]。Anthropic 给出的缓解方案是让 agent 通过浏览文件系统按需发现工具、只加载当前任务所需定义，示例中 token 使用从 150,000 降至 2,000（节省 98.7%）[^6]；Cloudflare 把同类做法称为 "Code Mode"[^6]。这属于对 MCP 默认客户端模式的优化，而非协议本身的原始设计目标。

---

## 二、基本架构

### 2.1 角色模型：Host / Client / Server

MCP 采用 client–host–server 架构，每个 host 可运行多个 client 实例[^7][^45]。三方的职责边界在规范中被明确划分：

- **Host** 是容器与协调者：创建并管理多个 client 实例、控制连接权限与生命周期、执行安全策略与同意要求、处理用户授权决策、协调 AI/LLM 集成与 sampling、管理跨 client 的上下文聚合[^7]。官方文档进一步把 host 定义为「用户交互的应用」，client 定义为「协议层组件，每个 client 只负责与一个 server 的一条直接通信」[^42]。
- **Client** 由 host 创建，与某个特定 server 保持 1:1 的有状态会话，负责协议协商与能力交换、双向路由协议消息、管理订阅与通知、维持 server 之间的安全边界[^7][^45]。
- **Server** 通过 MCP 原语暴露 resources、tools、prompts，职责聚焦、可独立运行，可通过 client 接口请求 sampling，可以是本地进程或远程服务[^7]。

一条重要的隔离设计原则是：server 不应能读取整个对话，也不能「看进」其他 server；完整对话历史保留在 host，跨 server 交互由 host 控制[^7]。协议还有意限制服务端对 prompts 的可见性[^3]。

### 2.2 分层：数据层与传输层

概念上数据层是内层、传输层是外层[^4]。数据层实现基于 JSON-RPC 2.0 的交换协议，定义消息结构与语义，包含 Discovery、Server features、Client features、Utility features 四类[^4]。传输层管理通信通道与认证，处理连接建立、消息分帧与安全通信，使同一 JSON-RPC 2.0 消息格式可跨所有传输机制使用[^4]。所有 MCP 客户端与服务器之间的消息必须遵循 JSON-RPC 2.0，请求 ID 不得为 null，且不得与发送方尚未收到响应的其他请求 ID 重复[^44]。

### 2.3 服务器侧三原语及其控制权归属

服务器通过三个构建块提供功能，且控制方各不相同——这是 MCP 架构中最具设计意味的一点：

| 原语 | 控制方 | 定义 | 典型内容 |
|---|---|---|---|
| **Tools** | 模型控制（model-controlled） | LLM 可主动调用的函数，模型根据用户请求决定何时使用 | 写数据库、调用外部 API、修改文件、触发其他逻辑[^5][^10] |
| **Resources** | 应用控制（application-controlled） | 提供只读上下文访问的被动数据源，每个 resource 有唯一 URI 并声明 MIME 类型 | 文件内容、数据库 schema、API 文档、git 历史[^5][^10] |
| **Prompts** | 用户控制（user-controlled） | 预构建的指令模板，告诉模型如何配合特定工具与资源工作 | 斜杠命令、菜单选项[^5][^10] |

Resources 支持两种发现模式：Direct Resources（固定 URI）与 Resource Templates（带参数的动态 URI，如 `travel://activities/{city}/{category}`）[^5]。Resources 由应用侧决定如何使用：可选择相关片段、用 embeddings 搜索，或整体传给模型[^5]。工具相关的协议操作是 `tools/list`（发现可用工具，返回带 schema 的工具定义数组）与 `tools/call`（执行特定工具，返回执行结果）[^5]。工具执行涉及审批对话框、对某些安全操作的预授权设置，以及展示所有工具执行与结果的活动日志[^5]。

### 2.4 客户端侧原语（2025-06-18 世代）

- **Roots**：client 向 server 暴露文件系统「根」的标准化方式，用于界定 server 可在文件系统中操作的边界[^11]。支持该能力的 client 必须在初始化时声明 roots 能力，`listChanged` 表示是否在列表变化时发通知；server 发送 `roots/list` 请求获取，client 返回含 `uri`（当前规范中必须是 `file://` URI）与可选 `name` 的数组[^11]。安全要求包括只暴露有适当权限的 roots、校验 URI 防路径穿越、实施访问控制[^11]。
- **Sampling**：server 通过 client 请求 LLM 采样（completions/generations）的标准化方式，使 client 保持对模型访问、选择与权限的控制，同时让 server 无需持有任何模型 API key 即可利用 AI 能力[^12]。协议消息为 server 发送 `sampling/createMessage`，参数含 messages、modelPreferences、systemPrompt、maxTokens 等，client 返回 role、content、model、stopReason[^12]。模型选择采用偏好系统而非按名指定：三个归一化（0–1）优先级 costPriority、speedPriority、intelligencePriority，配合可作子串匹配的 model hints，最终由 client 决定[^12]。
- **Elicitation**：允许服务器在处理 `tools/call` 等请求过程中向用户索取结构化信息，支持 form 模式（带 schema 的输入表单）与 URL 模式（带外交互，数据不经客户端，适用于凭证录入、第三方 OAuth）[^42]。

### 2.5 生命周期与能力协商（2025-06-18 世代）

生命周期分三阶段：Initialization（能力协商与协议版本约定）、Operation（正常协议通信）、Shutdown（优雅终止）[^8]。初始化必须是 client 与 server 之间的第一次交互：client 必须发起，发送 `initialize` 请求，包含支持的协议版本、client 能力、clientInfo；server 必须回应自身能力与信息（capabilities、serverInfo，可选 instructions）；成功后 client 必须发送 `notifications/initialized` 通知[^8]。在 server 响应 initialize 之前，client 不应发送除 ping 以外的请求；在收到 initialized 通知之前，server 不应发送除 ping 与 logging 以外的请求[^8]。

版本协商规则是：client 发送其支持的最新版本；server 若支持该版本必须回同一版本，否则必须回另一个自己支持的版本；client 若不支持返回版本应断开连接[^8]。使用 HTTP 时，client 必须在后续所有请求中带上 `MCP-Protocol-Version` 头[^8]。能力对象可描述子能力，如 `listChanged`（prompts/resources/tools 的列表变更通知）与 `subscribe`（仅 resources，订阅单条目变更）[^8]。能力决定会话期间哪些特性与原语可用，双方必须在整个会话中遵守已声明的能力[^7]。

### 2.6 传输层

协议目前定义两种标准传输机制：stdio 与 Streamable HTTP[^9]。JSON-RPC 消息必须是 UTF-8 编码[^9]。

- **stdio**：server 可向 stderr 写 UTF-8 字符串用于日志；server 不得向 stdout 写任何非合法 MCP 消息的内容，client 不得向 server 的 stdin 写任何非合法 MCP 消息的内容[^9]。
- **Streamable HTTP**：取代了 2024-11-05 版的 HTTP+SSE 传输[^9][^14]。server 必须提供单一 HTTP 端点路径，同时支持 POST 与 GET[^9]。安全要求：server 必须校验所有入站连接的 Origin 头以防 DNS rebinding 攻击；本地运行时 server 应只绑定 127.0.0.1 而非 0.0.0.0[^9]。client 必须包含同时列出 `application/json` 与 `text/event-stream` 的 Accept 头[^9]。POST 请求体必须是单个 JSON-RPC 请求、通知或响应；若输入是请求，server 必须返回 text/event-stream 或 application/json，client 必须同时支持[^9]。SSE 流支持通过 `Last-Event-ID` 头断线重连与消息重放[^9]。会话管理上，初始化响应中 server 可返回 `Mcp-Session-Id`（全局唯一、加密安全、仅可见 ASCII 0x21–0x7E），client 必须在后续请求中携带；server 可随时终止会话，之后对该会话 ID 的请求必须返回 404，client 收到 404 必须不带会话 ID 重新 Initialize[^9]。

### 2.7 架构在 2026-07-28 版的变化（必须与上文分开阅读）

2026-07-28 版把基础协议描述由 2024-11-05 的 "Stateful connections" 改为 "Stateless, self-contained requests" 与 "Per-request capability negotiation"[^20]。具体变化包括：

- **移除协议级会话**：Streamable HTTP 中的协议级会话与 `Mcp-Session-Id` 头被移除；列表端点不再随连接变化；需要跨调用状态的服务器改用显式的 server-minted handles 作为普通工具参数传递[^19]。
- **MRTR 取代服务器主动请求**：引入 Multi Round-Trip Requests 模式，取代此前由服务器主动发起 `roots/list`、`sampling/createMessage`、`elicitation/create` 的做法——服务器返回 `InputRequiredResult`（`resultType: "input_required"`），其 `inputRequests` 字段携带所需补充信息，客户端在重试原请求时以 `inputResponses` 提供[^19]。所有结果新增必填 `resultType` 字段，普通结果为 "complete"，客户端必须将缺少该字段的旧版服务器结果视为 "complete"[^19][^44]。
- **移除 SSE 可恢复性**：`Last-Event-ID` 头与 SSE event ID 被移除，响应流中断即丢失在途请求，客户端必须以新 request ID 重新发起[^19]。
- **Sampling 与 Roots 弃用**：Sampling 自 2026-07-28 起弃用（SEP-2577），按特性生命周期政策在发布后至少 12 个月内仍保留于规范，新实现不应采用、现有实现应迁移为直接集成 LLM 提供商 API[^43]；Roots 同样自 2026-07-28 起弃用，且被明确为「仅作为协调机制而非安全边界，实际安全须由操作系统文件权限或沙箱强制执行」[^42]。2026-07-28 版客户端侧特性列表仅列 Elicitation[^20]。
- **新增 Extensions 概念**：核心协议之外的可选扩展，始终 opt-in 且需双方在初始化时显式协商；列出的 notable extensions 包括 Tasks（长时操作异步执行）、Skills over MCP、MCP Apps（会话内联渲染的交互式 UI）[^20]。
- **缓存与头部**：要求 `tools/list`、`prompts/list`、`resources/list`、`resources/read`、`resources/templates/list` 的结果通过 `CacheableResult` 携带 `ttlMs` 与 `cacheScope`[^19]；要求 Streamable HTTP POST 携带 `Mcp-Method`、`Mcp-Name` 头，并支持通过 `x-mcp-header` 从工具参数映射自定义头[^19]。

版本标识采用 `YYYY-MM-DD` 格式，表示最后一次向后不兼容变更的日期，仅做向后兼容更新时不递增版本号；修订状态分为 Draft / Current / Final，特性可被标记 Deprecated 并须保留至少十二个月（政策快速移除例外下至少九十天）后才可 Removed[^18]。

### 2.8 安全模型：协议声明边界，实现承担义务

规范的安全原则是：MCP 通过任意数据访问与代码执行路径提供强大能力，因此要求用户显式同意并理解所有数据访问与操作、用户保留对共享数据与所执行动作的控制权、Hosts 在向 servers 暴露用户数据前必须获得显式同意[^3]。工具安全上，Tools 代表任意代码执行，必须谨慎对待；工具行为描述（如 annotations）应被视为不可信，除非来自可信服务器；Hosts 在调用任何工具前必须获得用户显式同意[^3][^20]。关键限定是：规范声明 MCP 本身无法在协议层强制这些安全原则，实现者 SHOULD 构建同意与授权流程、提供安全影响文档、实施访问控制[^3]。Sampling 侧同样要求始终有人类在环并有权拒绝请求，应用应提供便于审阅的 UI、允许用户在发送前查看和编辑提示、并在交付前呈现生成结果供审阅[^12]。

---

## 三、生态落地与采纳证据

官方仓库 `modelcontextprotocol/servers` 的指标为 stars 91,534、forks 11,853，主语言 TypeScript，最近 push 2026-10-22[^25]。该仓库 README 明确警告：仓库内服务器是用于演示 MCP 特性与 SDK 用法的 reference implementations，是教学示例，**并非 production-ready 方案**，开发者需自行评估安全需求；当前 Reference Servers 仅 7 个（Everything、Fetch、Filesystem、Git、Memory、Sequential Thinking、Time），大量早期服务器（GitHub、GitLab、Google Drive、Slack、PostgreSQL、Puppeteer、Redis、Sentry、SQLite 等）已归档至 servers-archived[^25]。这与发布时「随发布提供 Google Drive、Slack、GitHub、Git、Postgres、Puppeteer 等预构建 MCP servers」的表述[^1]形成对照：早期预构建服务器已不再作为参考实现维护。

SDK 侧，官方 README 列出 C#、Go、Java、Kotlin、PHP、Python、Ruby、Rust、Swift、TypeScript 共 10 个 SDK[^25]。Python SDK 指标为 stars 24,656、forks 4,073，最近 push 2026-10-24，支持 stdio、Streamable HTTP、SSE 三种 transport，要求 Python 3.11+，许可证 MIT[^27]。TypeScript SDK 指标为 stars 13,642、forks 2,334，最近 push 2026-10-23；其 main 分支即 SDK v2（包名 `@modelcontextprotocol/server` 与 `@modelcontextprotocol/client`），实现 2026-07-28 版规范，v1.x 在 v2 发布（2026-07-27）后至少六个月内继续接收 bug 修复与安全更新，许可证 Apache-2.0[^26]。C# SDK 与 Microsoft 协作维护[^21]。云厂商侧，AWS 在 awslabs 组织下维护开源 MCP Servers 集合[^24]。

客户端侧，Claude Code 支持 stdio、remote HTTP、remote SSE（官方标注 deprecated，建议改用 HTTP）、remote WebSocket，以及仅 SDK host 可注册的 in-process `"type": "sdk"` server；并通过 `roots/list` 响应文件系统边界[^28]。Cursor 支持 stdio、SSE、Streamable HTTP 三种传输，其文档表格标注 Tools、Prompts、Resources、Roots、Elicitation、Apps（扩展）均为 Supported[^29]。VS Code 提供 "HTTP (HTTP or Server-Sent Events)" 与 "Command (stdio)" 两类连接选项[^30]。Zed 目前仅支持 Tools 与 Prompts，文档明确表示 Discovery、Sampling、Elicitation 等特性尚未支持[^31]。JetBrains AI Assistant 支持 STDIO、Streamable HTTP 与为 legacy server 保留的 SSE[^32]。

安全侧已形成独立工具品类：Snyk 维护的 agent-scan 自我描述覆盖「AI agents、MCP servers 与 agent skills」三类扫描对象[^23]；此外存在多个第三方 MCP 安全审计/扫描工具与刻意构造的易受攻击服务器用于研究[^35][^37][^38][^39]。Trail of Bits 记录了 MCP 中的 'line jumping' 漏洞——恶意服务器可在工具被实际调用之前通过工具描述注入 prompt 操纵 AI 行为，并可利用 ANSI 终端转义序列隐藏恶意指令、植入触发短语以窃取完整对话历史与敏感凭据[^34]；该机构还记录了多款 MCP 软件将第三方服务长期 API key 以明文形式存储在本地文件系统、且文件权限为全局可读的问题[^34]。

---

## 四、批评与实践者反馈

Shrivu Shankar 以「MCP 支持者」立场系统列举局限[^40]。针对「与 function calling 重复」的质疑，他承认这是初见 MCP 时最自然的反应，并判断「And it sort of is」，差别在于 MCP 额外明确了连接应用与工具服务器的网络层面细节[^40]。针对鉴权，他指出 MCP 最初未定义 auth spec，导致每个 server 各自实现「authentication」，从高摩擦到对敏感数据完全无授权机制不等，官方后续补上后他评价「things… got complicated」[^40]。协议安全上，他指出 spec 支持通过 stdio 运行本地 MCP server，使集成方常要求用户下载并运行代码，为技术能力较弱的用户创造了低摩擦的本地机器被利用路径；MCP server 实现普遍信任其输入，常见做法是直接 exec 输入代码[^40]。UI/UX 上，他指出 MCP 没有工具风险等级的概念或控制，用户可能同时挂载 `read_daily_journal`、`book_flights`、`delete_files` 等风险差异极大的工具，尽管 spec 建议应用实现确认动作，实际执行仍很薄弱[^40]。

Armin Ronacher 从必要性角度提出更窄的批评：他表示自己「barely use」MCP，理由是 Claude Code 本身就能直接运行普通工具（如 psql），MCP 只在替代方案太不可靠时才需要；具体批评是可靠性——「MCP servers themselves are sometimes not super reliable and they are an extra thing that can go wrong」；他主张保持简单，仅在浏览器自动化（playwright-mcp）这类场景才使用 MCP[^41]。

---

## 矛盾与不确定性

1. **「N×M 集成爆炸」缺乏一手依据**。全部证据中，Anthropic 发布博客、各版规范与官方文档、官方仓库 README 均未出现「N×M」「M×N」或「集成爆炸」表述[^1][^45][^46][^47]；最接近的只有「Every new data source requires its own custom implementation」[^1]与「a custom integration for each pairing」[^6]。本报告因此只使用官方原表述，未将该术语作为官方动机引用。若需使用，应标注为二手概括。

2. **架构描述存在版本断层，本报告已做切分但无法完全消解**。2025-06-18 世代的有状态会话、初始化握手与能力协商、`Mcp-Session-Id`、Sampling/Roots 为活跃原语[^7][^8][^9][^11][^12]，与 2026-07-28 世代的无状态自包含请求、逐请求能力协商、移除协议级会话、弃用 Sampling 与 Roots、以 MRTR 取代服务器主动请求[^19][^20][^42][^43]，是同一协议在两个时点的不同状态。证据未提供 2026-07-28 版的完整架构页与生命周期页，因此**该版本下初始化流程、能力协商的具体报文形态，以及无状态化后 host 如何维持跨 server 上下文聚合，现有公开资料不足以确认**。同样，`server/discover` 在 [4] 中被描述为「每个服务器都必须实现」，在 [18] 中被称为「强制 RPC」但又「该调用是可选的」，两处表述存在张力，本报告不做裁决。

3. **ChatGPT 作为 MCP Host 的支持范围无法确认**。唯一来源是 modelcontextprotocol.io/clients 概览页的一句「AI assistants like Claude and ChatGPT ... all support MCP」[^33]，属生态宣传性表述，无 OpenAI 官方文档、发布日期或支持范围（传输方式、原语、是否仅限 connector/Deep Research）佐证；该页自述未提供各客户端的传输方式或原语支持矩阵[^33]。因此本报告未把 ChatGPT 计入可核验的客户端支持矩阵，仅记录该页面声称。

4. **客户端支持矩阵存在版本滞后风险**。Cursor 文档标注 Roots、Elicitation 为 Supported[^29]，但 Roots 与 Sampling 已在 2026-07-28 版弃用[^42][^43]；Zed 文档称 Sampling、Elicitation 尚未支持[^31]，而 Elicitation 在 2026-07-28 版是唯一保留的客户端侧特性[^20]。各客户端文档未标注其对应的协议版本，**无法判断这些矩阵反映的是哪一版规范的支持状态**。

5. **批评来源的时效性与代表性不足**。两篇批评均为个人博客：sshh.io 一文无日期[^40]，lucumr.pocoo.org 一文为 2025-06-12 且作者自述该领域变化极快、博文可能很快过时[^41]，相对证据中最新的仓库 push 时间（2026-10-24）[^27]已逾 16 个月。**缺少 2026 年内、可核验日期、且针对当前规范（无状态化、MRTR、Extensions 膨胀）的系统性批评来源**，因此本报告对「社区是否认为 MCP 过度设计」不作结论。

6. **生态热度指标的可解释性有限**。多个仓库的 stars/forks/push 时间来自检索元数据而非 README 正文，证据本身已声明「仅元数据，未读 README 正文，不作为结论依据」[^21][^22][^23][^24]。本报告仅将 `servers`、`typescript-sdk`、`python-sdk` 三个已读 README 的仓库用于生态论述[^25][^26][^27]，其余仅作为「存在该类项目」的存在性证据。

## 参考来源

[^1]: Introducing the Model Context Protocol (Anthropic 官方发布博客, 2024-11) — https://www.anthropic.com/news/model-context-protocol（web, 2024-11-25）
[^2]: Model Context Protocol — 官方文档 Introduction (modelcontextprotocol.io) — https://modelcontextprotocol.io/docs/getting-started/intro（web）
[^3]: Model Context Protocol Specification 2025-06-18 — Overview (权威规范) — https://modelcontextprotocol.io/specification/2025-06-18/index（web, 2025-06-18）
[^4]: MCP 官方文档 — Architecture (数据层/传输层与核心原语) — https://modelcontextprotocol.io/docs/learn/architecture（web）
[^5]: MCP 官方文档 — Server Concepts (Tools / Resources / Prompts 三原语与控制权归属) — https://modelcontextprotocol.io/docs/learn/server-concepts（web）
[^6]: Anthropic Engineering — Code execution with MCP (碎片化动机与上下文开销) — https://www.anthropic.com/engineering/code-execution-with-mcp（web）
[^7]: MCP Specification 2025-06-18 — Architecture — https://modelcontextprotocol.io/specification/2025-06-18/architecture（web, 2025-06-18）
[^8]: MCP Specification 2025-06-18 — Lifecycle (initialization, version & capability negotiation) — https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle（web, 2025-06-18）
[^9]: MCP Specification 2025-06-18 — Transports (stdio & Streamable HTTP) — https://modelcontextprotocol.io/specification/2025-06-18/basic/transports（web, 2025-06-18）
[^10]: MCP Specification 2025-06-18 — Server primitives overview (Prompts / Resources / Tools) — https://modelcontextprotocol.io/specification/2025-06-18/server/index（web, 2025-06-18）
[^11]: MCP Specification 2025-06-18 — Client primitive: Roots — https://modelcontextprotocol.io/specification/2025-06-18/client/roots（web, 2025-06-18）
[^12]: MCP Specification 2025-06-18 — Client primitive: Sampling — https://modelcontextprotocol.io/specification/2025-06-18/client/sampling（web, 2025-06-18）
[^14]: MCP Specification 2025-03-26 — Changelog (changes since 2024-11-05) — https://modelcontextprotocol.io/specification/2025-03-26/changelog（web, 2025-03-26）
[^18]: MCP Specification — Versioning (current version 2026-07-28, revision states, negotiation) — https://modelcontextprotocol.io/specification/versioning（web）
[^19]: MCP Specification 2026-07-28 — Changelog (changes since 2025-11-25) — https://modelcontextprotocol.io/specification/2026-07-28/changelog（web, 2026-07-28）
[^20]: MCP Specification 2026-07-28 — Overview (architecture, features, extensions) — https://modelcontextprotocol.io/specification/2026-07-28（web, 2026-07-28）
[^21]: modelcontextprotocol/csharp-sdk — The official C# SDK for Model Context Protocol servers and clients. Maintained in collaboration with Microsoft. — https://github.com/modelcontextprotocol/csharp-sdk（github）
[^22]: modelcontextprotocol/ext-apps — Official repo for spec & SDK of MCP Apps protocol — https://github.com/modelcontextprotocol/ext-apps（github）
[^23]: snyk/agent-scan — Security scanner for AI agents, MCP servers and agent skills — https://github.com/snyk/agent-scan（github）
[^24]: awslabs/mcp — Open source MCP Servers for AWS — https://github.com/awslabs/mcp（github）
[^25]: modelcontextprotocol/servers — https://github.com/modelcontextprotocol/servers（github, 2026-10-22）
[^26]: modelcontextprotocol/typescript-sdk — https://github.com/modelcontextprotocol/typescript-sdk（github, 2026-10-23）
[^27]: modelcontextprotocol/python-sdk — https://github.com/modelcontextprotocol/python-sdk（github, 2026-10-24）
[^28]: Claude Code Docs — Connect Claude Code to tools via MCP — https://code.claude.com/docs/en/mcp（web）
[^29]: Cursor Docs — Customize Model Context Protocol (MCP) — https://cursor.com/docs/context/mcp（web）
[^30]: VS Code Docs — Add MCP servers in VS Code (GitHub Copilot) — https://code.visualstudio.com/docs/copilot/customization/mcp-servers（web）
[^31]: Zed Docs — Model Context Protocol — https://zed.dev/docs/ai/mcp（web）
[^32]: JetBrains Docs — Model Context Protocol (AI Assistant) — https://www.jetbrains.com/help/ai-assistant/mcp.html（web）
[^33]: Model Context Protocol 官网首页 — Broad ecosystem support — https://modelcontextprotocol.io/clients（web）
[^34]: Trail of Bits blog — MCP category index — https://blog.trailofbits.com/categories/mcp/（web）
[^35]: makalin/SecureMCP — MCP security auditing tool — https://github.com/makalin/SecureMCP（github）
[^37]: sahiloj/MCPScan — offensive MCP server auditor — https://github.com/sahiloj/MCPScan（github）
[^38]: AppiumTestDistribution/secure-hulk — MCP server & tool security scanner — https://github.com/AppiumTestDistribution/secure-hulk（github）
[^39]: IntegSec/VulnerableMCP — intentionally vulnerable MCP server for security research — https://github.com/IntegSec/VulnerableMCP（github）
[^40]: Everything Wrong with MCP (Shrivu Shankar, Substack) — https://blog.sshh.io/p/everything-wrong-with-mcp（web, 2025-04-13）
[^41]: Agentic Coding Recommendations (Armin Ronacher's Thoughts and Writings) — https://lucumr.pocoo.org/2025/6/12/agentic-coding/（web, 2025-06-12）
[^42]: Model Context Protocol — Client concepts (官方文档) — https://modelcontextprotocol.io/docs/learn/client-concepts（web）
[^43]: Sampling — MCP specification 2026-07-28 (client/sampling) — https://modelcontextprotocol.io/specification/2026-07-28/client/sampling（web, 2026-07-28）
[^44]: Base Protocol — MCP specification 2026-07-28 (basic/index) — https://modelcontextprotocol.io/specification/2026-07-28/basic/index（web, 2026-07-28）
[^45]: MCP Specification 2024-11-05 — Architecture（首个正式规范版本） — https://modelcontextprotocol.io/specification/2024-11-05/architecture（web, 2024-11-05）
[^46]: MCP Specification 2024-11-05 — Overview / Index — https://modelcontextprotocol.io/specification/2024-11-05/index（web, 2024-11-05）
[^47]: modelcontextprotocol/modelcontextprotocol — 官方规范仓库 README — https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/main/README.md（github）