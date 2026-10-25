# MCP 解决了什么问题、架构是什么：基于规范一手证据的深度分析

## 摘要

Model Context Protocol（MCP）是 Anthropic 于 2024 年 11 月开源的一项开放标准，官方定义是「an open-source standard for connecting AI applications to external systems」，即把 AI 应用连接到外部系统的开源标准[^1]。它要解决的核心问题是**集成碎片化**：模型能力持续进步，但「even the most sophisticated models are constrained by their isolation from data—trapped behind information silos and legacy systems」，而「Every new data source requires its own custom implementation, making truly connected systems difficult to scale」[^2]。MCP 的答案是「a universal, open standard for connecting AI systems with data sources, replacing fragmented integrations with a single protocol」[^2]，官方用 USB-C 端口作类比[^1]。

架构上，MCP 采用 **client–host–server** 三角色模型：Host 是发起连接的 LLM 应用，为每个 server 创建一个 1:1 的 Client，Server 提供上下文与能力[^4][^7]。协议分两层：Data layer（基于 JSON-RPC 2.0 的通信协议，含能力协商与核心原语）与 Transport layer（stdio / Streamable HTTP）[^4]。服务端三原语为 Resources、Prompts、Tools，按控制权分别归属 application、user、model[^5][^9]。

必须强调一个关键的时间维度：抓取时点的当前协议版本是 **2026-07-28**，该版本对架构做了破坏性重构——移除 initialize 握手与协议级 session、把 MCP 改为无状态协议、以 `subscriptions/listen` 取代 HTTP GET 流、以 MRTR 取代服务器主动发起请求，并弃用 Roots / Sampling / Logging[^25][^26][^27]。因此，2025-06-18 版本中广泛流传的「有状态会话 + 服务器发起请求」架构模型**已不是当前规范**。下文将分别陈述两代架构，并明确指出差异。

---

## 一、MCP 要解决的问题

### 1.1 问题陈述：数据孤岛与 N×M 集成爆炸

Anthropic 2024 年 11 月的发布公告给出了最直接的问题表述：模型能力虽快速进步，但受限于与数据的隔离，数据「trapped behind information silos and legacy systems」；每接入一个新数据源都需要各自的定制实现，导致真正互联的系统难以规模化[^2]。MCP 的定位是「a new standard for connecting AI assistants to the systems where data lives」，覆盖内容仓库、业务工具与开发环境[^2]。

从工程经济学看，这个问题的本质是连接器数量的组合爆炸：公告明确「Instead of maintaining separate connectors for each data source, developers can now build against a standard protocol」[^2]。官方文档把价值主张按角色拆开：对开发者是降低构建与集成 AI 应用的时间与复杂度；对 AI 应用是获得数据源、工具与应用的生态；对终端用户是更有能力、可代为访问数据并执行动作的应用[^1]。官方口号为 "build once and integrate everywhere"[^1]。

### 1.2 标准化的参照系：LSP

MCP 规范明确说明其借鉴 Language Server Protocol 的思路：LSP 为整个开发工具生态标准化了「如何增加对编程语言的支持」，MCP 以类似方式标准化「如何把额外上下文与工具集成进 AI 应用生态」[^3]。这解释了 MCP 的设计取向——它不试图规定 AI 应用如何思考，只规定应用与外部能力之间的接口。

### 1.3 边界限定：MCP 不做什么

一个常被误解的点是 MCP 的职责边界。官方架构文档有一句关键限定：「MCP focuses solely on the protocol for context exchange—it does not dictate how AI applications use LLMs or manage the provided context.」[^4] 也就是说，上下文拿到之后如何筛选、如何注入、如何管理，完全由应用自行决定。官方在 Resources 的说明中进一步印证这一点：应用可自行决定是选取相关片段、用 embeddings 检索，还是把原始数据整体传给模型[^5]。

### 1.4 规范定义的三项能力

规范正文把 MCP 定义为「an open protocol that enables seamless integration between LLM applications and external data sources and tools」，并列出它让应用可以做三件事：与语言模型共享上下文信息；向 AI 系统暴露工具与能力；构建可组合的集成与工作流[^3]。

---

## 二、基本架构

### 2.1 参与者模型：Host / Client / Server

MCP 采用 client–host–server 架构，每个 host 可运行多个 client 实例[^7][^27]。三者的职责界定如下：

- **MCP Host**：协调并管理一个或多个 MCP client 的 AI 应用，例如 Claude Code、Claude Desktop[^4]。Host 进程作为容器与协调者，负责：创建并管理多个 client 实例；控制 client 连接权限与生命周期；执行安全策略与用户同意要求；处理用户授权决策；协调 AI/LLM 集成与 sampling；管理跨 client 的上下文聚合[^7]。
- **MCP Client**：维护与某个 server 的连接并为 host 获取上下文[^4]。每个 client 由 host 创建，与某个特定 server 是 **1:1 关系**，负责协议协商与能力交换、双向路由协议消息、管理订阅与通知、维持 server 之间的安全边界[^7]。
- **MCP Server**：向 client 提供上下文的程序[^4]，通过 MCP 原语暴露 resources、tools、prompts；独立运行、职责聚焦；可以是本地进程或远程服务[^7]。

一条重要的隔离设计原则：server 不应能读取整个对话，也不能「看进」其他 server；完整对话历史保留在 host 侧[^7]。这既是安全边界，也是 MCP 能把「上下文交换」与「上下文管理」解耦的前提。

### 2.2 两层结构：Data layer 与 Transport layer

MCP 分为两层[^4]：

- **Data layer**：基于 JSON-RPC 的客户端-服务器通信协议，含能力与版本发现、tools/resources/prompts/notifications 等核心原语。
- **Transport layer**：通信机制与通道，含连接建立、消息分帧、授权。

协议与传输无关（transport-agnostic），可实现于任何支持双向消息交换的通道；自定义传输 MUST 保留 JSON-RPC 消息格式与 MCP 定义的生命周期要求[^13]。在 2026-07-28 版的表述中，传输是一种「binding」，定义消息如何分帧与投递、请求元数据如何携带、取消与终止如何发出信号，但**不定义消息含义**；协议语义在所有传输上一致[^28]。

### 2.3 消息层：JSON-RPC 2.0 的严格化

所有 client 与 server 之间的消息 MUST 遵循 JSON-RPC 2.0[^8][^30]。协议定义三类消息[^8]：

- **Request**：`{jsonrpc:"2.0", id: string|number, method: string, params?: object}`，可由 client 发往 server 或反向；MUST 包含字符串或整数 ID；与基础 JSON-RPC 不同，**ID MUST NOT 为 null**；同一会话内 ID MUST NOT 被请求方重复使用[^8]。
- **Response**：`{jsonrpc:"2.0", id, result?: object, error?: {code:number, message:string, data?:unknown}}`；MUST 携带与对应请求相同的 ID；result 与 error 二者 MUST 只设其一、MUST NOT 同时设置；错误码 MUST 为整数[^8]。
- **Notification**：`{jsonrpc:"2.0", method: string, params?: object}`，单向消息，接收方 MUST NOT 回复，且 MUST NOT 包含 ID[^8]。

`_meta` 字段由 MCP 保留用于附加元数据；以 `modelcontextprotocol` 或 `mcp` 为段的合法前缀（如 `modelcontextprotocol.io/`、`mcp.dev/`）为 MCP 保留[^8]。协议的完整定义以 TypeScript schema 为唯一事实来源，另有自动生成的 JSON Schema 供工具使用[^8]。

### 2.4 服务端三原语：Resources / Prompts / Tools

Server 通过 MCP 为语言模型添加上下文提供三个基础构件[^9]：

| 原语 | 定义 | 控制方 | 典型例子 |
|---|---|---|---|
| **Tools** | 「Functions that your LLM can actively call, and decides when to use them based on user requests」[^5]；可执行函数，允许模型执行动作或检索信息[^9] | Model（model-controlled）[^5][^9] | API POST 请求、文件写入[^9] |
| **Resources** | 「Passive data sources that provide read-only access to information for context」[^5]；结构化数据或内容，为模型提供附加上下文[^9] | Application（application-controlled）[^5][^9] | 文件内容、git 历史[^9] |
| **Prompts** | 「Pre-built instruction templates」[^5]；预定义模板或指令，用于引导语言模型交互[^9] | User（user-controlled）[^5][^9] | 斜杠命令、菜单选项[^9] |

这一控制权划分是 MCP 架构中最具解释力的设计：模型可自动发现并调用 Tools，但 MCP 通过 UI 展示、逐次执行审批对话框、预授权权限设置、活动日志等机制强调人类监督[^5]。Tools 使用 JSON Schema 做输入校验，协议操作为 `tools/list`（发现）与 `tools/call`（执行）[^4]。Resources 通过唯一 URI 标识并声明 MIME type，支持 Direct Resources（固定 URI）与 Resource Templates（带参数动态 URI）两种发现模式[^4][^5]。

### 2.5 客户端特性：Roots / Sampling / Elicitation（2025-06-18 版）

在 2025-06-18 版规范中，client 侧提供三项特性，用于把「服务器需要宿主配合」的场景标准化：

- **Roots**：标准化方式让 client 向 server 暴露文件系统边界，界定 server 可在文件系统中操作的目录与文件；server 可请求 roots 列表并在变化时收到通知[^10]。Root 数据类型含 `uri`（当前规范中 MUST 为 `file://` URI）与可选 `name`[^10]。
- **Sampling**：让 server 通过 client 请求语言模型进行采样（"completions" 或 "generations"），使 client 保持对模型访问、模型选择与权限的控制，而 **server 无需持有 API key**[^11]。模型选择采用偏好系统：三个归一化（0–1）优先级 `costPriority`、`speedPriority`、`intelligencePriority`，加上可选的 model hints；hints 仅为建议性，最终模型选择由 client 决定[^11]。
- **Elicitation**：让 server 在交互过程中通过 client 向用户请求额外信息，使用 JSON Schema 校验响应；`requestedSchema` 限于扁平对象与原始类型属性，有意不支持复杂嵌套结构以简化 client 实现；响应采用 accept / decline / cancel 三动作模型[^12]。

### 2.6 传输层

**stdio**：客户端将 MCP 服务器作为子进程启动，服务器从 stdin 读、向 stdout 写，消息以换行符分隔且 MUST NOT 含内嵌换行；服务器 MAY 向 stderr 写日志，但 MUST NOT 向 stdout 写任何非合法 MCP 消息的内容[^13]。2025-11-25 版进一步澄清 stderr 可用于任何日志目的（信息、调试、错误），客户端 SHOULD NOT 假定 stderr 输出表示错误状态[^20]。

**Streamable HTTP**：于 2025-03-26 引入，明确取代 2024-11-05 的 HTTP+SSE 传输[^13][^17]。服务器提供单一 HTTP 端点路径同时支持 POST 与 GET；客户端每条 JSON-RPC 消息 MUST 是一个新的 HTTP POST，且 MUST 在 Accept 头中同时列出 `application/json` 与 `text/event-stream`[^13]。安全要求包括：MUST 校验 Origin 头以防 DNS rebinding、本地运行 SHOULD 仅绑定 127.0.0.1、SHOULD 实现认证[^13]。

**旧 HTTP+SSE 传输（2024-11-05）**：服务器 MUST 提供两个端点——一个 SSE 端点供客户端接收消息，一个 HTTP POST 端点供客户端发送消息；客户端连接时服务器 MUST 发送一个 `endpoint` 事件给出 POST URI[^14]。该传输自 2025-03-26 起已被取代[^13]。

### 2.7 生命周期与能力协商（2025-06-18 版）

2025-06-18 版定义了严格的三阶段生命周期：Initialization（能力协商与版本协商）、Operation（正常通信）、Shutdown（优雅终止）[^15]。

初始化 MUST 是双方首次交互。客户端 MUST 发送 `initialize` 请求，含 `protocolVersion`、`capabilities`、`clientInfo`；服务器 MUST 返回 `protocolVersion`、`capabilities`、`serverInfo` 及可选 `instructions`；成功后客户端 MUST 发送 `notifications/initialized`[^15]。版本协商规则：客户端发送其支持版本（SHOULD 为最新），服务器支持则返回同一版本，否则返回其支持的另一版本；客户端不支持服务器返回版本时 SHOULD 断开[^15]。

能力协商决定会话期间可用的可选特性：客户端能力包括 roots、sampling、elicitation、experimental；服务器能力包括 prompts、resources、tools、logging、completions、experimental；能力对象可描述子能力如 `listChanged`、`subscribe`[^15]。双方 MUST 在整个会话中遵守已声明能力[^7]。

Shutdown 阶段未定义专门的关闭消息，由底层传输表示：stdio 下客户端 SHOULD 先关闭子进程输入流，等待退出，超时则 SIGTERM，再不退出则 SIGKILL[^15]。

### 2.8 授权（OAuth 2.1 框架）

MCP 在传输层提供授权能力，使客户端能代表资源所有者向受限 MCP 服务器发起请求；该规范面向 HTTP 类传输，对实现是 OPTIONAL 的[^16]。角色划分：受保护的 MCP 服务器充当 OAuth 2.1 resource server，MCP 客户端充当 OAuth 2.1 client，授权服务器负责与用户交互并签发令牌[^16]。

关键强制项：MCP 服务器 MUST 实现 RFC 9728 Protected Resource Metadata，客户端 MUST 使用它进行授权服务器发现；客户端 MUST 实现 RFC 8707 Resource Indicators，在授权请求与令牌请求中都 MUST 包含 `resource` 参数；令牌 MUST 通过 `Authorization: Bearer` 头传递，MUST NOT 放在 URI 查询字符串中；**禁止 token passthrough**——客户端 MUST NOT 向 MCP 服务器发送非该服务器授权服务器签发的令牌，服务器 MUST NOT 接受或中转任何其他令牌[^16]。客户端 MUST 实现 PKCE[^16]。使用 STDIO 传输的实现 SHOULD NOT 遵循该授权规范，而应从环境获取凭据[^8][^16]。

---

## 三、版本演进：从有状态握手到无状态协议

MCP 使用 `YYYY-MM-DD` 形式的字符串版本标识，表示最后一次不向后兼容变更的日期；向后兼容的更新不递增版本号[^6]。抓取时点当前版本为 2026-07-28[^6]。修订状态分为 Draft、Current、Final[^6]。每个请求通过 `_meta` 中的 `io.modelcontextprotocol/protocolVersion` 键声明所用协议版本，服务器独立接受或拒绝每个请求[^6]。

### 3.1 演进链条（关键节点）

- **2024-11-05**：初版，stdio + HTTP with SSE 两种传输[^14]。
- **2025-03-26**：新增基于 OAuth 2.1 的完整授权框架；以 Streamable HTTP 取代 HTTP+SSE；新增 JSON-RPC batching；新增工具注解（tool annotations）[^17]。
- **2025-06-18**：**移除** batching（上一版引入、本版撤销）；新增结构化工具输出；将 MCP 服务器归类为 OAuth Resource Servers 并新增 protected resource metadata；要求客户端实现 RFC 8707；新增 elicitation；新增 resource links；要求 HTTP 后续请求携带 `MCP-Protocol-Version` 头[^18]。
- **2025-11-25**：增强授权服务器发现（支持 OpenID Connect Discovery）；新增 icons 元数据；增量式 scope 同意；OAuth Client ID Metadata Documents；URL 模式 elicitation；sampling 增加工具调用支持；tasks 实验性支持；确立 JSON Schema 2020-12 为默认方言[^19]。
- **2026-07-28**：**破坏性重构**，见下节[^25]。

### 3.2 2026-07-28 的架构级变更

该版本相对 2025-11-25 的变更不是增量修补，而是架构模型的替换[^25][^26][^27]：

1. **无状态化**：移除 `initialize` / `notifications/initialized` 握手；每个请求在 `_meta` 中携带协议版本与客户端能力（`io.modelcontextprotocol/protocolVersion`、`io.modelcontextprotocol/clientCapabilities`）[^25]。规范原文：「There is no negotiation handshake.」每个请求携带其协议版本，服务器独立接受或拒绝每个请求[^26]。架构页原文：「MCP is a stateless protocol: every request is self-contained and carries its own protocol version and capabilities.」[^27]
2. **移除协议级 session 与 `Mcp-Session-Id` 头**；列表类端点（`tools/list`、`resources/list`、`prompts/list`）不再随连接变化；需要跨调用状态的服务器改用服务器签发的显式 handle，作为普通工具参数传递[^25]。
3. **新增 `server/discover`**：服务器 MUST 实现该 RPC，用于公布其支持的协议版本、能力与身份；客户端 MAY 在任何其他请求之前调用它做前置版本选择，或在 STDIO 上用作向后兼容探测[^25][^26]。
4. **以 `subscriptions/listen` 取代 HTTP GET 端点与 `resources/subscribe` / `resources/unsubscribe`**：单一长生命周期的 POST 响应流用于服务器到客户端的变更通知；客户端按类型选择订阅（`toolsListChanged`、`promptsListChanged`、`resourcesListChanged`、`resourceSubscriptions`）[^25]。
5. **MRTR（Multi Round-Trip Requests）取代服务器主动发起请求**：服务器返回 `InputRequiredResult`（`resultType: "input_required"`），其 `inputRequests` 字段携带所需补充信息，客户端在重试原请求时以 `inputResponses` 提供[^25]。所有结果现在都携带必需的 `resultType` 字段：普通结果为 `"complete"`，多轮往返的中间结果为 `"input_required"`；客户端 MUST 将缺少该字段的早期服务器结果视为 `"complete"`[^25][^30]。
6. **移除 SSE 流可恢复性与消息重投**（`Last-Event-ID` 与 SSE event ID）；响应流中断即丢失在途请求，客户端 MUST 以新请求 ID 重新发起[^25]。
7. **弃用 Roots、Sampling、Logging**：弃用窗口内仍完全可用，但新实现不应再增加支持；建议迁移方式为用工具参数/资源 URI/服务器配置替代 Roots、直接集成 LLM 供应商 API 替代 Sampling、用 stderr 或 OpenTelemetry 替代 Logging[^25]。
8. **弃用 HTTP+SSE 传输**，迁移至 Streamable HTTP[^25]。
9. **实验性 tasks 移出核心协议**，成为官方扩展 `io.modelcontextprotocol/tasks`[^25]。

### 3.3 两代兼容模型

2026-07-28 版定义了「Modern / Legacy / Dual-era」术语：Modern 指以逐请求元数据传递版本、身份与能力的协议版本（2026-07-28 及以后）；Legacy 指以 initialize 握手建立会话的版本（2025-11-25 及更早）；Dual-era 指同时支持两者的实现[^26]。兼容性矩阵要点：Modern 客户端 + Legacy 服务器 = 失败；Legacy 客户端 + Modern 服务器 = 失败；Dual-era 服务器对两类客户端均可用[^26]。Dual-era 服务器依据客户端如何开启交互来选择行为：携带 modern 逐请求 `_meta` 的请求按无状态处理，`initialize` 请求则选择 legacy 语义[^26]。

---

## 四、生态落地（有限证据）

官方称 MCP 被 Claude、ChatGPT、Visual Studio Code、Cursor、MCPJam 等客户端与服务器支持[^1]。发布时交付三部分：MCP 规范与 SDK、Claude Desktop 应用内的本地 MCP server 支持、开源 MCP servers 仓库；并提供 Google Drive、Slack、GitHub、Git、Postgres、Puppeteer 的预构建 servers[^2]。

**参考实现仓库**：`modelcontextprotocol/servers` 显示 91.5k stars、4,826 commits，最近 push 2026-10-25；README 已收窄定位，只承载 steering group 维护的少量 reference servers，完整清单应去 MCP Registry 浏览；现行 7 个 reference servers 为 Everything、Fetch、Filesystem、Git、Memory、Sequential Thinking、Time，另有 13 个已归档[^21]。README 明确的生产可用性免责声明：这些服务器是用于演示 MCP 特性与 SDK 用法的参考实现、面向开发者的教学示例，**并非 production-ready 方案**[^21]。官方 SDK 语言共 10 种：C#、Go、Java、Kotlin、PHP、Python、Ruby、Rust、Swift、TypeScript[^21]。

**SDK**：官方 Python SDK 显示 24.7k stars、1,108 commits，最近 push 2026-10-25；同时提供 server 与 client 两侧实现，支持 stdio、Streamable HTTP 与 SSE 全部标准传输，要求 Python 3.11+，README 提及「What's new in v2」与迁移指南[^22]。

**Registry**：`modelcontextprotocol/registry` 显示 7.4k stars、751 commits，实现语言 Go；定位为 MCP 服务器的「应用商店」；2025-09-08 以 preview 上线，2025-10-24 Registry API 进入 API freeze（v0.1）[^23]。官方文档说明 Registry 是面向公开可访问 MCP 服务器的集中式元数据仓库，只托管指向 npm/PyPI/Docker Hub 的元数据、不托管代码与二进制，不支持私有服务器；信任机制采用反向 DNS 命名空间认证[^41]。抓取时页面带 preview 横幅，提示 GA 前可能发生破坏性变更或数据重置[^41][^42]。

**客户端支持**：VS Code 官方博客（2025-06-12）宣布支持完整 MCP 规范，在 tools 与 workspace awareness 之外可用 authorization、prompts、resources 和 sampling[^39]；VS Code 1.102（2025-07-09）宣布 MCP support generally available，引入 MCP view 与 gallery[^40]。Cursor 文档称支持 stdio、SSE、Streamable HTTP 三种 transport，能力表列出 Tools、Prompts、Resources、Roots、Elicitation、Apps 扩展为 Supported，但**未出现 Sampling 条目**[^36]。OpenAI Agents Python SDK 文档列出四种 MCP 接入方式（HostedMCPTool、MCPServerStreamableHttp、MCPServerSse、MCPServerStdio），依赖范围为 `mcp>=1.19.0,<3`[^32]。

**SDK 分级体系（SEP-1730）**：分三级，Tier 1 为完全支持（含 sampling、elicitation 等非实验性可选能力），一致性测试通过率门槛 Tier 1 需 100%、Tier 2 需 80%、Tier 3 无最低要求；2026-01-23 一致性测试可用，2026-02-23 正式发布分级结果[^44]。官方博客称跨 Tier 1 SDK 每月下载量接近 5 亿次，TypeScript 与 Python SDK 累计下载量均突破 10 亿次[^45]。

---

## 五、矛盾与不确定性

**（1）两代架构并存造成的最主要矛盾。** 2025-06-18 与 2026-07-28 两个版本的架构描述在多个关键点上直接冲突：前者是「有状态会话协议」、以 initialize 握手建立会话、允许服务器发起 JSON-RPC 请求（sampling / elicitation / roots/list）[^7][^15]；后者是「无状态协议」、无握手、服务器不得发起请求而改用 MRTR 内嵌 input requests[^25][^27][^29]。同时，Roots / Sampling / Logging 在 2026-07-28 被弃用[^25]，而大量二手资料与客户端文档（如 Cursor 能力表仍列 Roots、Elicitation 为 Supported[^36]）仍按旧模型描述。**任何关于「MCP 架构」的陈述都必须标注所依据的协议版本**，否则会得出互相矛盾的结论。本报告已按版本分节处理，但读者需注意：抓取时点的「当前架构」是 2026-07-28 的无状态模型，而非流传更广的 2025-06-18 模型。

**（2）厂商接入时间线证据严重不足。** 两条 OpenAI 官方来源（openai.com 博客页、platform.openai.com 文档页）均返回 HTTP 403，未取得任何日期或原文表述，不得作为结论依据[^33][^34]。现有唯一 OpenAI 证据是 openai-agents-python 文档页，该页无发布日期、无公告表述，无法支撑「接入时间线」这一子问题[^32]。Google/DeepMind 侧完全没有来源。Microsoft 侧仅有 `microsoft/mcp-for-beginners` 课程仓库（17,508 stars、最近 push 2026-10-25）[^35]，无 GitHub / VS Code / Copilot 官方宣布支持 MCP 的公告或日期。Anthropic 侧只有 2024-11 发布公告[^2]，缺 Claude Desktop / Claude Code 后续 GA 与生产可用性表述。Cursor 只有当前文档，无接入时间点[^36]。**唯一时间线证据充分的是 VS Code**（2025-06-12 全规范支持[^39]、1.102 GA[^40]），可作为对照基准，但不能外推到其他厂商。

**（3）生态规模无法量化。** Registry 首页为前端渲染，静态抓取仅得 "Loading servers..."，未渲染出任何服务器条目或总数[^43]。官方博客「Tier 1 SDK 月下载近 5 亿、TS/Python 各破 10 亿」只有博客首页一句，缺一手出处（npm/PyPI 统计页或博客正文 URL）[^45]。社区 server 总量仅有 4 个高星项目样本（mcp-chrome 12,508 stars、mobile-mcp 9,960、MobileBuildMCP 6,506、gemini-notebook-mcp-cli 6,389），且活跃度差异明显（mcp-chrome 最近 push 停留在 2026-01-06）[^24]，不足以支撑「生态规模」结论。**现有公开资料不足以确认 MCP 生态的实际规模。**

**（4）生产可用性主张与官方免责声明冲突。** 官方参考实现仓库 README 明确声明这些服务器「并非 production-ready 方案」[^21]，而官方价值主张与部分客户端文档给人以「开箱即用」的印象[^1][^36]。两者并不必然矛盾（前者指参考实现，后者指协议与生态），但在评估「MCP 是否可用于生产」时必须区分：协议规范本身是权威且严格的[^3][^8]，而官方示例服务器不是生产方案[^21]。

**（5）规范仓库元数据未作为内容依据。** `modelcontextprotocol/modelcontextprotocol` 仓库仅取得元数据（约 9,528 stars、最近 push 2026-10-23），未读取 README 正文，故不作为任何规范内容的结论依据[^31]。

**（6）2026-07-28 版本下 primitives 现状的整合表述缺失。** 该版本的 changelog 与 lifecycle 页覆盖了无状态化、MRTR、subscriptions/listen 等变更[^25][^26]，architecture 页给出了 Host/Client/Server 的新职责与能力协商方式[^27]，但**没有一份证据完整重述「当前版本下三原语（Tools/Resources/Prompts）与客户端特性（Elicitation/Sampling/Roots）的最终形态」**。已知 Sampling、Roots、Logging 被弃用[^25]，Elicitation 通过 MRTR 的 input requests 承载[^29]，但 Tools/Resources/Prompts 在无状态模型下的完整语义（例如资源订阅如何与 `subscriptions/listen` 的 `resourceSubscriptions` 类型精确对应）需以 2026-07-28 版 server features 页面为准，而该页面不在本次证据集内。**此处的整合表述存在证据缺口，现有公开资料不足以完整确认。**

## 参考来源

[^1]: Model Context Protocol — Introduction (官方文档 Getting Started) — https://modelcontextprotocol.io/docs/getting-started/intro（web）
[^2]: Anthropic — Introducing the Model Context Protocol (2024 年 11 月发布公告) — https://www.anthropic.com/news/model-context-protocol（web, 2024-11-25）
[^3]: MCP Specification (revision 2025-06-18) — 规范正文 — https://modelcontextprotocol.io/specification/2025-06-18（web, 2025-06-18）
[^4]: MCP Architecture 概览（官方 docs/learn/architecture） — https://modelcontextprotocol.io/docs/learn/architecture（web）
[^5]: MCP Server Concepts（官方 docs/learn/server-concepts） — https://modelcontextprotocol.io/docs/learn/server-concepts（web）
[^6]: MCP Specification — Versioning（协议版本与修订状态） — https://modelcontextprotocol.io/specification/versioning（web）
[^7]: Architecture — Model Context Protocol Specification (2025-06-18) — https://modelcontextprotocol.io/specification/2025-06-18/architecture（web, 2025-06-18）
[^8]: Base Protocol — Model Context Protocol Specification (2025-06-18) — https://modelcontextprotocol.io/specification/2025-06-18/basic（web, 2025-06-18）
[^9]: Server Features / Primitives — Model Context Protocol Specification (2025-06-18) — https://modelcontextprotocol.io/specification/2025-06-18/server（web, 2025-06-18）
[^10]: Client Features: Roots — Model Context Protocol Specification (2025-06-18) — https://modelcontextprotocol.io/specification/2025-06-18/client（web, 2025-06-18）
[^11]: Client Features: Sampling — Model Context Protocol Specification (2025-06-18) — https://modelcontextprotocol.io/specification/2025-06-18/client/sampling（web, 2025-06-18）
[^12]: Client Features: Elicitation — Model Context Protocol Specification (2025-06-18) — https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation（web, 2025-06-18）
[^13]: MCP Specification 2025-03-26 — Transports (stdio & Streamable HTTP) — https://modelcontextprotocol.io/specification/2025-03-26/basic/transports（web, 2025-03-26）
[^14]: MCP Specification 2024-11-05 — Transports (stdio & HTTP with SSE) — https://modelcontextprotocol.io/specification/2024-11-05/basic/transports（web, 2024-11-05）
[^15]: MCP Specification 2025-06-18 — Lifecycle (initialization, capability negotiation, shutdown) — https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle（web, 2025-06-18）
[^16]: MCP Specification 2025-06-18 — Authorization (OAuth 2.1 based) — https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization（web, 2025-06-18）
[^17]: MCP Specification 2025-03-26 — Changelog (changes since 2024-11-05) — https://modelcontextprotocol.io/specification/2025-03-26/changelog（web, 2025-03-26）
[^18]: MCP Specification 2025-06-18 — Changelog (changes since 2025-03-26) — https://modelcontextprotocol.io/specification/2025-06-18/changelog（web, 2025-06-18）
[^19]: MCP Specification 2025-11-25 — Changelog (changes since 2025-06-18) — https://modelcontextprotocol.io/specification/2025-11-25/changelog（web, 2025-11-25）
[^20]: MCP Specification 2025-11-25 — Transports (stdio & Streamable HTTP, updated) — https://modelcontextprotocol.io/specification/2025-11-25/basic/transports（web, 2025-11-25）
[^21]: modelcontextprotocol/servers — Model Context Protocol Servers（官方参考实现集合） — https://github.com/modelcontextprotocol/servers（github, 2026-10-25）
[^22]: modelcontextprotocol/python-sdk — 官方 Python SDK — https://github.com/modelcontextprotocol/python-sdk（github, 2026-10-25）
[^23]: modelcontextprotocol/registry — 社区驱动的 MCP 服务器注册中心 — https://github.com/modelcontextprotocol/registry（github, 2026-10-24）
[^24]: GitHub 仓库检索结果：MCP 相关高星项目（search_repos 元数据） — https://github.com/modelcontextprotocol/typescript-sdk（github, 2026-10-23）
[^25]: MCP Specification 2026-07-28 — Changelog — https://modelcontextprotocol.io/specification/2026-07-28/changelog（web, 2026-07-28）
[^26]: MCP Specification 2026-07-28 — Lifecycle (Versioning and Compatibility) — https://modelcontextprotocol.io/specification/2026-07-28/basic/lifecycle（web, 2026-07-28）
[^27]: MCP Specification 2026-07-28 — Architecture — https://modelcontextprotocol.io/specification/2026-07-28/architecture（web, 2026-07-28）
[^28]: MCP Specification 2026-07-28 — Transports (overview) — https://modelcontextprotocol.io/specification/2026-07-28/basic/transports（web, 2026-07-28）
[^29]: MCP Specification 2026-07-28 — Streamable HTTP transport binding — https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http（web, 2026-07-28）
[^30]: MCP Specification 2026-07-28 — Base Protocol (basic/index) — https://modelcontextprotocol.io/specification/2026-07-28/basic/index（web, 2026-07-28）
[^31]: modelcontextprotocol/modelcontextprotocol — 规范与文档仓库（仅元数据） — https://github.com/modelcontextprotocol/modelcontextprotocol（github, 2026-10-23）
[^32]: Model context protocol (MCP) — OpenAI Agents SDK (Python) 官方文档 — https://openai.github.io/openai-agents-python/mcp/（web）
[^33]: OpenAI 官方博客页（未能读取正文，仅记录访问失败） — https://openai.com/index/simplifying-agentic-ai-with-mcp/（web）
[^34]: OpenAI Platform 文档：Tools / connectors / MCP 指南（未能读取正文，仅记录访问失败） — https://platform.openai.com/docs/guides/tools-connectors-mcp（web）
[^35]: microsoft/mcp-for-beginners — https://github.com/microsoft/mcp-for-beginners（github）
[^36]: Cursor 官方文档：Model Context Protocol (MCP) — https://cursor.com/docs/context/mcp（web）
[^39]: VS Code 官方博客：VS Code now supports the complete Model Context Protocol specification — https://code.visualstudio.com/blogs/2025/06/12/full-mcp-spec-support（web, 2025-06-12）
[^40]: VS Code 1.102 Release Notes (June 2025) — https://code.visualstudio.com/updates/v1_102（web, 2025-07-09）
[^41]: About the MCP Registry（官方文档） — https://modelcontextprotocol.io/registry/about（web）
[^42]: MCP Registry FAQ（官方文档） — https://modelcontextprotocol.io/registry/faq（web）
[^43]: Official MCP Registry 首页（registry.modelcontextprotocol.io） — https://registry.modelcontextprotocol.io/（web）
[^44]: SDK Tiering System（官方文档，对应 SEP-1730） — https://modelcontextprotocol.io/community/sdk-tiers（web）
[^45]: MCP Blog 首页（官方博客） — https://blog.modelcontextprotocol.io/（web）