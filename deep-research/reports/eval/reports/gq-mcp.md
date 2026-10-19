# Model Context Protocol (MCP)：问题定义与架构演进

## 摘要

MCP 是 Anthropic 于 2024-11-25 开源的开放标准，目标是连接 AI 助手与数据所在系统[^1]。它要解决的核心问题是：模型能力虽在快速进步，但受限于与数据的隔离，被困在信息孤岛与遗留系统之后；而每接入一个新数据源都需要定制实现，使真正互联的系统难以规模化——这一集成困境被概括为「N×M 数据集成问题」[^1][^5]。MCP 的解法是用单一协议取代碎片化集成，其设计直接受 Language Server Protocol (LSP) 启发：正如 LSP 标准化了跨开发工具生态添加语言支持的方式，MCP 以类似方式标准化向 AI 应用生态集成额外上下文与工具[^2]。

架构上，MCP 采用 client-host-server 三角色模型，基于 JSON-RPC 2.0，服务器通过 Resources / Prompts / Tools 三类原语暴露能力，客户端通过 Sampling / Roots / Elicitation 提供反向能力[^2][^3][^6]。协议分为 Data layer 与 Transport layer 两层，标准传输为 stdio 与 Streamable HTTP[^4][^13]。

需要特别指出：本报告所依赖的架构细节主要来自 2025-06-18 规范，而截至 2026 年 10 月，现行协议版本已推进到 2026-07-28，该版本对协议做了破坏性重构（移除 initialize 握手、移除会话、以 MRTR 取代服务器发起请求），两者在「MCP 是否为有状态会话协议」等关键点上直接冲突[^6][^7][^33]。这一矛盾在正文与末节中显式标注。

---

## 一、MCP 要解决什么问题

### 1.1 问题陈述

Anthropic 官方发布公告给出的动机链条是：模型能力快速进步，但即使最先进的模型也受限于与数据的隔离，被困在信息孤岛与遗留系统之后；每个新数据源都需要自己的定制实现，使真正互联的系统难以规模化[^1]。MCP 的定位即是用通用、开放的标准取代碎片化集成[^1]。

Wikipedia 的交叉验证补充了术语来源：在 MCP 之前，开发者常需为每个数据源或工具构建定制连接器，形成 Anthropic 所描述的「N×M」数据集成问题[^5]。同一来源指出，OpenAI 2023 年的 function-calling API 与 ChatGPT 插件框架解决了类似问题，但需要厂商特定的连接器[^5]。

### 1.2 权威定义

规范给出的定义是：MCP 是一个开放协议，实现 LLM 应用与外部数据源和工具之间的无缝集成（an open protocol that enables seamless integration between LLM applications and external data sources and tools）[^2]。其三大能力为：与语言模型共享上下文信息；向 AI 系统暴露工具与能力；构建可组合的集成与工作流[^2]。

官方文档用一个类比帮助理解：把 MCP 想象成 AI 应用的 USB-C 端口——正如 USB-C 提供连接电子设备的标准化方式，MCP 提供连接 AI 应用与外部系统的标准化方式[^4]。

### 1.3 协议边界（关键限定）

官方文档明确声明了 MCP 的边界：**MCP 只关注上下文交换协议，不规定 AI 应用如何使用 LLM 或管理所提供的上下文**[^4]。这一限定很重要——它意味着 MCP 本身不是 agent 框架，也不是模型调用规范，而是「上下文与能力如何在应用与外部系统之间流动」的协议层。

---

## 二、基本架构

### 2.1 三角色模型

MCP 采用 client-host-server 架构，每个 host 可运行多个 client 实例；协议基于 JSON-RPC，是聚焦上下文交换与 sampling 协调的有状态会话协议（stateful session protocol focused on context exchange and sampling coordination）[^3][^6]。三方角色定义如下[^2][^4]：

- **Hosts**：发起连接的 LLM 应用，即协调管理一个或多个 MCP client 的 AI 应用[^2][^4]。
- **Clients**：host 应用内的连接器，维护与 MCP server 的连接并为 host 获取上下文[^2][^4]。
- **Servers**：提供上下文与能力的服务，向 client 提供上下文，可本地或远程运行[^2][^4]。

### 2.2 各角色职责

**Host 职责**[^3][^6]：创建并管理多个 client 实例；控制 client 连接权限与生命周期；执行安全策略与同意要求；处理用户授权决策；协调 AI/LLM 集成与 sampling；跨 client 聚合上下文。

**Client 职责**[^3][^6]：由 host 创建，维护一个隔离的 server 连接；与特定 server 建立并保持一个隔离的有状态会话；负责协议协商与能力交换；双向路由消息；管理订阅与通知；维护 server 间安全边界。

**Server 职责**[^3][^6]：通过 MCP 原语暴露 resources、tools、prompts；独立运行、职责聚焦；可通过 client 接口请求 sampling；可为本地进程或远程服务。

**基数关系**：host 应用创建并管理多个 client，每个 client 与特定 server 保持 1:1 关系[^3][^6]。Wikipedia 的表述一致：host 为每个 server 创建专用 client；client 与 host 通常同机运行，servers 可本地或远程[^5]。

### 2.3 四条设计原则

规范明确列出四条设计原则[^3]：

1. servers 应极易构建；
2. servers 应高度可组合；
3. **servers 不应能读取整个对话，也不能「看进」其他 server**；
4. 功能可在 servers 与 clients 上渐进添加。

第 3 条被反复强调：完整对话历史保留在 host 侧，每个 server 连接维持隔离[^6]。

### 2.4 两层协议结构

官方文档将 MCP 划分为两层[^4]：

- **Data layer**：基于 JSON-RPC 的 client-server 通信协议，含能力与版本发现，以及 tools、resources、prompts、notifications 等核心原语。
- **Transport layer**：通信机制与通道，含连接建立、消息分帧与授权。

### 2.5 消息类型与能力协商

消息类型遵循 JSON-RPC 2.0，分三类[^3]：Requests（双向、含 method 与参数、期待响应）、Responses（成功结果或错误）、Notifications（单向、无需响应）。

能力协商在初始化时进行：双方显式声明支持的能力——servers 声明 resource subscriptions、tool support、prompt templates；clients 声明 sampling 支持与 notification handling；双方须在整个会话中遵守已声明能力[^3][^6]。规范进一步要求：已实现的 server 特性必须在 server capabilities 中声明（例如发出 resource subscription 通知需声明 subscription 支持；工具调用需声明 tool 能力）[^6]。

---

## 三、服务器端三类原语

规范定义服务器向客户端提供的三类原语为 Resources（上下文与数据）、Prompts（模板化消息与工作流）、Tools（供 AI 模型执行的函数）[^2][^35]。三者的控制主体不同，这是理解 MCP 权限模型的关键。

### 3.1 Tools（模型控制）

**定义**：Tools 允许 servers 暴露可调用函数，供语言模型调用以与外部系统交互（查询数据库、调用 API、执行计算）；每个 tool 由 name 唯一标识，并包含描述其 schema 的元数据[^8]。

**控制主体**：Tools 是 **model-controlled**——语言模型可基于其上下文理解与用户 prompt 自动发现并调用工具；协议本身不强制任何特定用户交互模型[^8]。

**能力声明**：支持 tools 的 server MUST 声明 `tools` 能力；`listChanged` 指示 server 是否会在可用工具列表变化时发出通知[^8]。

**协议方法**：`tools/list`（支持 cursor/nextCursor 分页）与 `tools/call`（参数：name、arguments）；工具列表变化时发送 `notifications/tools/list_changed`[^8]。

**工具定义字段**：name（唯一标识）、title（可选的人类可读显示名）、description、inputSchema（期望参数的 JSON Schema）、outputSchema（可选，期望输出的 JSON Schema）、annotations（可选的行为描述属性）[^8]。

**结果格式**：工具结果可含结构化或非结构化内容；非结构化内容在 `content` 字段中以多个不同类型的内容项返回（text、image、audio、resource links、embedded resources），另有 `isError` 布尔值[^8]。

**安全要求**：clients MUST 将工具注解视为不可信，除非其来自可信 server；且 SHOULD 始终有 human in the loop，并具备拒绝工具调用的能力[^8]。

### 3.2 Resources（应用控制）

**定义**：Resources 允许 servers 共享为语言模型提供上下文的数据，如文件、数据库 schema 或应用特定信息；每个 resource 由 URI 唯一标识[^9]。

**控制主体**：Resources 是 **application-driven**——host 应用根据自身需求决定如何纳入上下文（例如通过 UI 显式选择、搜索/过滤，或自动上下文包含）[^9]。

**能力声明**：支持 resources 的 server MUST 声明 `resources` 能力，其含两个可选子特性：`subscribe`（client 可订阅单个 resource 的变更通知）与 `listChanged`（server 在可用 resource 列表变化时发出通知）；server 可两者都不支持、支持其一或两者都支持[^9]。

**协议方法**：`resources/list`（分页）、`resources/read`（参数：uri；返回 `contents` 数组，含 uri、mimeType、text/blob）、`resources/templates/list`（使用 URI 模板的参数化资源，参数可通过 completion API 自动补全）；通知为 `notifications/resources/list_changed`[^9]。

**资源模板**：以 URI 模板暴露参数化资源，例如 uriTemplate 为 `file:///{path}`[^9]。

### 3.3 Prompts（用户控制）

**定义**：Prompts 允许 servers 向 clients 暴露 prompt 模板，为与语言模型交互提供结构化消息与指令；clients 可发现可用 prompts、获取其内容并提供参数以定制[^10]。

**控制主体**：Prompts 是 **user-controlled**——从 servers 暴露给 clients，意图是让用户能显式选择使用（通常通过用户发起的命令如 slash commands 触发）[^10]。

**能力声明**：支持 prompts 的 server MUST 在初始化期间声明 `prompts` 能力；`listChanged` 指示 server 是否会在可用 prompt 列表变化时发出通知[^10]。

**协议方法**：`prompts/list`（分页）与 `prompts/get`（参数：name、arguments；arguments 可通过 completion API 自动补全）；通知为 `notifications/prompts/list_changed`[^10]。

**Prompt 定义字段**：name（唯一标识）、title（可选的人类可读显示名）、description（可选）、arguments（可选的自定义参数列表，每项含 name、description、required）[^10]。

**PromptMessage 结构**：role 为 'user' 或 'assistant'；content 为若干内容类型之一（text、image 等）；所有内容类型支持可选注解，用于描述 audience、priority 与修改时间等元数据[^10]。

---

## 四、客户端侧能力

### 4.1 Sampling

**定义**：Sampling 提供一种标准化方式，让 servers 通过 clients 向语言模型请求 LLM sampling（「completions」或「generations」），使 clients 能保持对模型访问、选择与权限的控制，同时让 servers 无需任何 server API key 即可利用 AI 能力[^11]。

**用途**：Sampling 允许 servers 实现 agentic 行为——使 LLM 调用可嵌套在其他 MCP server 特性内部；servers 可请求文本、音频或图像交互，并可选地在 prompt 中包含来自 MCP servers 的上下文[^11]。

**能力声明**：支持 sampling 的 clients MUST 在初始化期间声明 `sampling` 能力[^11]。

**协议方法**：`sampling/createMessage`（server → client）；请求参数含 messages、modelPreferences、systemPrompt、maxTokens；响应含 role、content、model 与 stopReason[^11]。

**模型选择机制**：使用偏好系统，结合抽象能力优先级与可选模型提示——三个归一化优先级值（0-1）：costPriority、speedPriority、intelligencePriority[^11]。模型提示被当作可灵活匹配模型名的子串；多个提示按偏好顺序评估；clients MAY 将提示映射到等价模型[^11]。

**消息内容类型**：text、image（base64 data + mimeType）、audio（base64 data + mimeType）[^11]。

**信任与安全**：SHOULD 始终有 human in the loop 并具备拒绝 sampling 请求的能力；应用 SHOULD 让用户在发送前查看与编辑 prompt，并在交付前审查生成的响应[^11]。

### 4.2 Roots

**定义**：Roots 提供一种标准化方式，让 clients 向 servers 暴露文件系统「roots」；roots 定义 servers 可在文件系统中操作的边界，使其理解自己可访问哪些目录与文件[^12]。

**机制**：servers 可向支持 roots 的 clients 请求 roots 列表，并在列表变化时接收通知[^12]。

**能力声明**：支持 roots 的 clients MUST 在初始化期间声明 `roots` 能力；`listChanged` 指示 client 是否会在 roots 列表变化时发出通知[^12]。

**协议方法**：`roots/list`（server → client）；支持 listChanged 的 clients 在 roots 变化时发送 `notifications/roots/list_changed`[^12]。

**Root 定义字段**：uri（唯一标识；在当前规范中 MUST 为 file:// URI）与 name（可选的人类可读显示名）[^12]。

**错误处理**：clients SHOULD 返回标准 JSON-RPC 错误，例如 client 不支持 roots 时返回 -32601（Method not found），内部错误返回 -32603[^12]。

**安全要求**：clients MUST 仅暴露具有适当权限的 roots，校验所有 root URI 以防路径穿越，实施适当的访问控制，并监控 root 可达性；servers SHOULD 在操作中尊重 root 边界，并针对提供的 roots 校验所有路径[^12]。

### 4.3 Elicitation（后续版本引入）

Elicitation 使 server 可在交互中向用户请求额外信息，于 2025-06-18 版本新增[^17]。在 2026-07-28 规范中，Elicitation 被列为客户端可向服务器提供的特性（服务器发起的、向用户索取额外信息的请求）[^35]。

---

## 五、传输层

### 5.1 通用要求

MCP 使用 JSON-RPC 编码消息，JSON-RPC 消息 MUST 为 UTF-8 编码[^13]。协议定义两种标准传输机制：stdio（标准输入/输出）与 Streamable HTTP；客户端 SHOULD 尽可能支持 stdio；客户端与服务器 MAY 以可插拔方式实现自定义传输[^13]。

### 5.2 stdio 传输

stdio 传输中客户端将 MCP server 作为子进程（subprocess）启动；server 从 stdin 读 JSON-RPC 消息、向 stdout 写消息[^13]。消息以换行符分隔（delimited by newlines），且 MUST NOT 含内嵌换行[^13]。server MAY 向 stderr 写 UTF-8 字符串用于日志，客户端 MAY 捕获/转发/忽略[^13]。硬性约束：server MUST NOT 向 stdout 写任何非合法 MCP 消息；客户端 MUST NOT 向 server stdin 写任何非合法 MCP 消息[^13]。

官方文档补充了使用场景：使用 STDIO 传输的本地 MCP server 通常服务单个 client[^4]。

### 5.3 Streamable HTTP 传输

Streamable HTTP **取代**（replaces）2024-11-05 版本的 HTTP+SSE 传输[^13]。该传输中 server 作为独立进程运行，可处理多个客户端连接，使用 HTTP POST 和 GET 请求；server 可选使用 Server-Sent Events (SSE) 流式传输多条 server 消息[^13]。官方文档对应说明：使用 Streamable HTTP 传输的远程 MCP server 通常服务多个 client[^4]。

**端点要求**：server MUST 提供单一 HTTP endpoint 路径（MCP endpoint）同时支持 POST 与 GET（示例 https://example.com/mcp）[^13]。

**安全要求**：server MUST 校验所有入站连接的 Origin 头以防 DNS rebinding 攻击；本地运行时 server SHOULD 仅绑定 localhost (127.0.0.1) 而非 0.0.0.0；server SHOULD 对所有连接实现认证[^13]。

**客户端行为**：每条 JSON-RPC 消息 MUST 为对 MCP endpoint 的新 HTTP POST 请求；客户端 MUST 包含 Accept 头，同时列出 application/json 与 text/event-stream[^13]。

**服务端响应规则**：若输入为 JSON-RPC response/notification，server 接受则 MUST 返回 202 Accepted 且无 body，不接受则返回 HTTP 错误码（如 400 Bad Request）；若输入为 JSON-RPC request，server MUST 返回 Content-Type: text/event-stream（启动 SSE 流）或 Content-Type: application/json（返回单个 JSON 对象），客户端 MUST 支持两种情况[^13]。

### 5.4 历史传输（2024-11-05）

2024-11-05 版本定义两种传输：stdio 与 HTTP with Server-Sent Events (SSE)[^14]。SSE 传输要求 server 提供两个 endpoint：一个 SSE endpoint（客户端建立连接、接收 server 消息），一个普通 HTTP POST endpoint（客户端发送消息）；客户端连接时 server MUST 发送一个 endpoint event，包含客户端用于发送消息的 URI；后续客户端消息 MUST 以 HTTP POST 发往该 endpoint；server 消息以 SSE message event 发送，内容为 JSON 编码的 event data[^14]。该版本 stdio 描述与 2025-06-18 一致[^14]。

---

## 六、生命周期与版本协商（2025-06-18 语义）

规范定义三个生命周期阶段：Initialization（能力协商与协议版本约定）、Operation（正常协议通信）、Shutdown（优雅终止）[^7]。

**初始化流程**[^7]：初始化 MUST 是首次交互；client MUST 发起，发送 `initialize` 请求，包含其支持的协议版本、client capabilities 与 client 实现信息；server MUST 以其自身能力与信息（含可选 `instructions`）响应；初始化成功后 client MUST 发送 `initialized` 通知（notifications/initialized）以表明就绪。

**时序约束**[^7]：在 server 响应 initialize 请求前，client SHOULD NOT 发送除 ping 外的请求；在收到 initialized 通知前，server SHOULD NOT 发送除 ping 与 logging 外的请求。

**版本协商规则**[^7]：client MUST 发送其支持的版本（SHOULD 为最新）；若 server 支持该版本，MUST 返回相同版本，否则 MUST 返回其支持的另一版本（SHOULD 为最新）；若 client 不支持 server 返回的版本，SHOULD 断开连接。若使用 HTTP，client MUST 在后续所有请求中包含 `MCP-Protocol-Version: <protocol-version>` HTTP 头[^7]。

**能力字段**[^7]：client capabilities 包括 roots（文件系统 roots）、sampling（LLM sampling 请求）、elicitation（server elicitation 请求）、experimental（非标准特性）；server capabilities 包括 prompts（prompt 模板）、resources（可读资源）、tools（可调用工具）、logging（结构化日志消息）、completions（参数自动补全）、experimental（非标准特性）。能力对象可描述子能力，如 `listChanged`（prompts、resources、tools 的列表变更通知）与 `subscribe`（订阅单个 resource 变更）[^7]。

**版本标识规则**：版本标识采用 YYYY-MM-DD 字符串格式，表示最后一次发生向后不兼容变更的日期；只要变更保持向后兼容，协议版本号不会递增[^18]。

---

## 七、授权模型

授权在传输层提供，**仅针对 HTTP-based 传输定义授权流程**；授权对 MCP 实现是 OPTIONAL 的[^15]。使用 HTTP-based 传输的实现 SHOULD 遵循本规范；使用 STDIO 传输的实现 SHOULD NOT 遵循本规范，而应从环境变量获取凭据；使用替代传输的实现 MUST 遵循其协议既定的安全最佳实践[^15]。

**合规标准**（实现其子集）：OAuth 2.1 IETF DRAFT (draft-ietf-oauth-v2-1-13)、RFC8414、RFC7591、RFC9728[^15]。

**角色模型**[^15]：受保护的 MCP server 充当 OAuth 2.1 resource server；MCP client 充当 OAuth 2.1 client，代表 resource owner 发起受保护资源请求；authorization server 负责与用户交互并签发 access token，可托管于 resource server 或独立实体。

**具体要求**[^15]：authorization server MUST 实现 OAuth 2.1，并针对 confidential 与 public client 采取适当安全措施；authorization server 与 MCP client SHOULD 支持 RFC7591 动态客户端注册；MCP server MUST 实现 RFC9728 Protected Resource Metadata，MCP client MUST 用它做 authorization server 发现；authorization server MUST 提供 RFC8414 metadata，MCP client MUST 使用之；MCP server 返回 401 Unauthorized 时 MUST 使用 WWW-Authenticate 头指示 authorization server 位置。

---

## 八、版本迭代时间线

### 8.1 已确认的版本变更

**2024-11-05（首个版本）**：定义 stdio 与 HTTP+SSE 两种传输[^14]；三角色模型与三类服务器原语[^2][^3]；安全原则要求 host 在调用任何工具前获得用户明确同意，在向 servers 暴露用户数据前获得明确同意[^2]。

**2025-03-26**[^16]：新增基于 OAuth 2.1 的完整授权框架（PR #133）；用更灵活的 Streamable HTTP 传输取代原 HTTP+SSE 传输（PR #206）；新增 JSON-RPC batching 支持（PR #228）；新增工具注解（tool annotations），描述工具行为如是否只读/破坏性（PR #185）；ProgressNotification 增加 message 字段；新增 audio 数据类型，与既有 text、image 并列；新增 completions capability 以显式指示支持参数自动补全建议。

**2025-06-18**[^17]：移除 JSON-RPC batching 支持（PR #416）；新增结构化工具输出支持（PR #371）；将 MCP server 归类为 OAuth Resource Servers，新增 protected resource metadata 以发现对应 Authorization server（PR #338）；要求 MCP client 实现 RFC 8707 Resource Indicators，防止恶意 server 获取 access token（PR #734）；澄清授权规范与新增安全最佳实践页中的安全考量；新增 elicitation 支持（PR #382）；新增工具调用结果中的 resource links（PR #603）；要求 HTTP 后续请求通过 MCP-Protocol-Version 头指定协商后的协议版本（PR #548）；将 Lifecycle Operation 中的 SHOULD 改为 MUST；其他 schema 变更包括为更多接口类型增加 _meta 字段（PR #710）、CompletionRequest 增加 context 字段（PR #598）、增加 title 字段用于人类友好显示名（PR #663）。

**2025-11-25**[^28]：Major changes 共 9 项——支持 OpenID Connect Discovery 1.0 增强授权服务器发现（PR #797）；允许服务器为 tools/resources/resource templates/prompts 暴露 icons 元数据（SEP-973）；通过 WWW-Authenticate 实现增量 scope 同意（SEP-835）；提供工具命名指导（SEP-986）；更新 ElicitResult 与 EnumSchema 以支持 titled/untitled/单选/多选枚举（SEP-1330）；新增 URL 模式 elicitation（SEP-1036）；为 sampling 增加 tools 与 toolChoice 参数以支持工具调用（SEP-1577）；支持 OAuth Client ID Metadata Documents 作为推荐的客户端注册机制（SEP-991, PR #1296）；新增 tasks 实验性支持以通过轮询和延迟结果获取跟踪持久请求（SEP-1686）。Minor changes 包括 stdio 传输的服务器可用 stderr 记录所有类型日志（PR #670）、Streamable HTTP 传输中无效 Origin 头须返回 HTTP 403（PR #1439）、输入校验错误应作为 Tool Execution Errors 而非 Protocol Errors 返回（SEP-1303）、支持通过允许服务器主动断开来轮询 SSE 流（SEP-1699）、OAuth 2.0 Protected Resource Metadata 发现对齐 RFC 9728（SEP-985）、elicitation schema 支持所有原始类型默认值（SEP-1034）、确立 JSON Schema 2020-12 为 MCP schema 定义的默认方言（SEP-1613）。Governance 更新包括正式化 MCP 治理结构（SEP-932）、建立社区沟通规范（SEP-994）、正式化工作组与兴趣组（SEP-1302）、建立 SDK 分级制度（SEP-1730）。该版本仍使用 initialize 握手，protocolVersion 是 initialize params 的顶层字段（示例值 "2025-11-25"），而非 _meta 内字段[^29]。

**2026-07-28（破坏性重构）**[^33][^34]：主要变更 9 项——

1. 从 Streamable HTTP 传输中移除协议级会话与 Mcp-Session-Id 头；列表端点（tools/list、resources/list、prompts/list）不再随连接变化；需要跨调用状态的服务器改用显式的、由服务器铸造的 handle，作为普通工具参数传递（SEP-2567）。
2. 使 MCP 无状态化——移除 initialize / notifications/initialized 握手；每个请求在 _meta 中携带协议版本与客户端能力（io.modelcontextprotocol/protocolVersion、io.modelcontextprotocol/clientCapabilities）；客户端 SHOULD 在每个请求中标识自身（io.modelcontextprotocol/clientInfo），服务器 SHOULD 在每个结果的 _meta 中标识自身（io.modelcontextprotocol/serverInfo）；版本不匹配返回 UnsupportedProtocolVersionError（SEP-2575）。
3. 新增 server/discover——服务器 MUST 实现该 RPC 以公布其支持的协议版本、能力与身份；客户端 MAY 在任何其他请求之前调用它做前置版本选择，或在 STDIO 上用作向后兼容探测（SEP-2575）。
4. 用 subscriptions/listen 取代 HTTP GET 端点与 resources/subscribe / resources/unsubscribe——单一长生命周期的 POST-response 流用于已选择加入的服务器到客户端变更通知；客户端按类型选择加入（toolsListChanged、promptsListChanged、resourcesListChanged、resourceSubscriptions）；服务器确认并用 io.modelcontextprotocol/subscriptionId 标记通知；请求作用域的通知（如 notifications/progress、notifications/message）仍走其所属请求的响应流（SEP-2575）。
5. 移除 ping、logging/setLevel 与 notifications/roots/list_changed；日志级别改为按请求通过 _meta 中的 io.modelcontextprotocol/logLevel 设置；服务器 MUST NOT 对未包含该字段的请求发出 notifications/message（SEP-2575）。
6. 将实验性 tasks 移出核心协议，转为官方扩展（io.modelcontextprotocol/tasks）；重新设计的扩展用 tasks/get 轮询与新增的 tasks/update（客户端到服务器输入）取代阻塞式 tasks/result 方法，移除 tasks/list，并允许服务器无需逐请求选择加入即可主动返回任务 handle（SEP-2663）。
7. 引入 Multi Round-Trip Requests (MRTR) 模式，取代此前发送服务器发起请求（如 roots/list、sampling/createMessage、elicitation/create）的做法；服务器返回 InputRequiredResult（resultType: "input_required"），其 inputRequests 字段携带处理该请求所需的额外信息请求；客户端在重试原请求时以 inputResponses 提供所请求的信息（SEP-2322）。
8. 所有结果现在携带必需的 resultType 字段——普通结果为 "complete"，多轮往返请求的中间结果为 "input_required"；客户端 MUST 将省略该字段的早期协议服务器结果视为 "complete"（SEP-2322）。
9. 从 Streamable HTTP 传输中移除 SSE 流可恢复性与消息重投递（Last-Event-ID 头与 SSE event ID）；响应流中断会丢失在途请求，客户端 MUST 以新请求 ID 重新发起该请求（SEP-2575）。

次要变更包括：在 ClientCapabilities 与 ServerCapabilities 中新增 extensions 字段，以支持核心协议之外的可选扩展；为 _meta 键（traceparent、tracestate、baggage）记录 OpenTelemetry trace context 传播约定（SEP-414）[^33]。

### 8.2 版本分类与当前状态

抓取时点 versioning 页面标注的当前协议版本为 2026-07-28[^18]。draft 规范将协议版本分为两类：Modern（2026-07-28 及以后，通过 per-request 元数据传递版本、身份与能力）与 Legacy（2025-11-25 及更早，通过 initialize 握手建立会话）[^30]。draft 中不再有协商握手：每个请求携带其协议版本，服务器独立接受或拒绝每个请求；每个请求在其 _meta 字段中声明所使用的协议版本，在 HTTP 上同时通过 MCP-Protocol-Version 头传递；若服务器不实现所请求版本，必须返回 UnsupportedProtocolVersionError（错误码 -32022）并列出其支持的版本[^30]。向后兼容方面：同时支持 legacy 与 modern 的服务器可同时实现两种行为；stdio 通过 server/discover 探测并回退，Streamable HTTP 通过检查 400 Bad Request 响应体回退[^30]。

修订状态分三类：Draft（进行中）、Current（当前可用，可继续接收向后兼容变更）、Final（已完成、不再变更）[^18]。

---

## 九、生态与治理

### 9.1 治理转移

Anthropic 于 2025 年 12 月 9 日宣布将 MCP 捐赠给 Agentic AI Foundation (AAIF)[^36]。AAIF 是 Linux Foundation 下的一个 directed fund（定向基金），由 Anthropic、Block 和 OpenAI 共同创立，并获得 Google、Microsoft、Amazon Web Services (AWS)、Cloudflare、Bloomberg 支持[^36]。MCP 与 Block 的 goose、OpenAI 的 AGENTS.md 一起成为 AAIF 的创始项目[^36][^38]。AAIF 的 Platinum members 包括 Amazon Web Services、Anthropic、Block、Bloomberg、Cloudflare、Google、Microsoft 和 OpenAI[^38]。

治理安排上：Anthropic 声明 MCP 的治理模式保持不变，项目 maintainers 将继续优先考虑社区输入和透明决策[^36]。MCP 官方博客补充：AAIF Governing Board 负责战略投资、预算分配、成员招募和新项目审批等决策；单个项目（如 MCP）对其技术方向和日常运营保持完全自主权；MCP 的 maintainer 结构保持不变，协议决策者仍是原有 maintainers，通过 SEP 流程接受社区输入；Linux Foundation 提供中立归属和基础设施，使 maintainers 能独立运作，且不会主导 MCP 的技术方向[^37]。

### 9.2 采用规模

捐赠公告称 MCP 已有超过 10,000 个活跃公共 MCP 服务器，被 ChatGPT、Cursor、Gemini、Microsoft Copilot、Visual Studio Code 等平台采用；MCP 官方 SDK 在 Python 和 TypeScript 上的月度下载量超过 9700 万次[^36][^37][^38]。

### 9.3 客户端支持

官方文档列出的生态支持包括 Claude、ChatGPT、Visual Studio Code、Cursor、MCPJam 等客户端[^4]。

**OpenAI Agents SDK** 支持四种集成方式[^39]：Hosted MCP server tools（HostedMCPTool，由 OpenAI Responses API 代表模型调用公网可达的 MCP server）、Streamable HTTP（MCPServerStreamableHttp）、HTTP with SSE（MCPServerSse）、stdio（MCPServerStdio）。JS/TS SDK 还包含 MCPServerSSE 以支持旧版 Server-Sent Events transport，但文档注明 SSE 已被 MCP 项目弃用，新集成应优先使用 Streamable HTTP 或 stdio[^40]。Agents SDK 内部使用 MCP TypeScript SDK v2 客户端，但不要求用户的 MCP server 迁移到 v2 server 包；对 stdio 和 Streamable HTTP 连接会自动协商协议，并与实现了兼容协议版本的 MCP TypeScript SDK v1 server 保持兼容[^40]。

**OpenAI Responses API** 支持通过 remote MCP servers 或 Secure MCP Tunnel 为模型提供新能力；工具调用可自动允许，也可由开发者通过显式审批限制；Secure MCP Tunnel 用于连接本地或私有 MCP server 而无需将其暴露到公网[^41]。

**VS Code** 提供 MCP server gallery：在 Extensions 视图输入 @mcp 可列出可用 MCP server，可安装到 user profile 或 workspace（workspace 安装会更新 .vscode/mcp.json）[^42]。配置位置包括 workspace 的 .vscode/mcp.json（顶层 servers 对象）、workspace 可移植格式项目根目录 .mcp.json（顶层 mcpServers 对象，可跨兼容工具使用）、以及 user profile 的 mcp.json[^42]。

**Cursor** 支持三种 transport：stdio（本地，Cursor 管理，单用户）、SSE（本地/远程，作为 server 部署，多用户）、Streamable HTTP（本地/远程，作为 server 部署，多用户）[^43]。Cursor 支持的 MCP 协议能力与扩展包括 Tools、Prompts、Resources、Roots、Elicitation、Apps（extension，返回交互式 UI 视图）[^43]。

### 9.4 注册表

官方于 2025-09-08 宣布 MCP Registry 进入 preview，地址为 https://registry.modelcontextprotocol.io[^44]。该公告将 MCP Registry 定位为「publicly available MCP servers」的开放目录与 API，是 sub-registries 的「single source of truth」；preview 阶段明确不提供数据持久性保证，GA 前可能发生 breaking changes[^44]。

官方文档将 MCP Registry 定义为「the official centralized metadata repository for publicly accessible MCP servers」，由 Anthropic、GitHub、PulseMCP、Microsoft 等贡献者支持[^45]。收录范围口径：只支持公开可访问的 server（安装方式公开，如 npm 包或公共 Docker 镜像，或 server 本身公开可访问）；不支持 private server[^45]。server 元数据以标准化 server.json 格式存储，包含唯一名称（如 io.github.user/server-name）、定位方式、执行指令与发现数据[^45]。

官方 FAQ 区分四个概念：MCP Server Registry API（OpenAPI 规范）、Official MCP Registry（registry.modelcontextprotocol.io 应用）、Official MCP Registry API、以及第三方 MCP server registry[^47]。Official MCP Registry「currently only catalogs MCP servers」，未来可能扩展到 client/host apps 与 frameworks——即当前计数对象仅为 server[^47]。MCP Registry 不是 package registry：只存元数据与指向 npm/PyPI/NuGet/Docker Hub 等的引用，不托管源码或包[^47]。版本口径：每个版本有独立不可变元数据，版本字符串须唯一，旧版本仍可访问，registry 追踪「latest」版本[^47]。

registry 仓库 README 将其描述为「provides MCP clients with a list of MCP servers, like an app store for MCP servers」，并记录 2025-10-24 更新：Registry API 进入 API freeze (v0.1)，一个月以上保持稳定无 breaking changes[^46]。

### 9.5 官方 SDK 与参考实现

- **python-sdk**：stars 24,572，forks 4,043，MIT 许可，创建于 2024-09-24，最后 push 于 2026-10
-13；README 提及 v2 版本（What's new in v2）及迁移指南，表明 SDK 已进入 v2 迭代[^20]。

- **typescript-sdk**：stars 13,581，forks 2,287，创建于 2024-09-24，最后 push 于 2026-10-14；README 说明 main 分支为 SDK v2（@modelcontextprotocol/server、@modelcontextprotocol/client），是 stable release line，实现 2026-07-28 MCP 规范；v1.x（@modelcontextprotocol/sdk）位于 v1.x 分支，在 v2 发布（2026-07-27）后至少六个月内继续接收 bug fix 与安全更新[^21]。
- **csharp-sdk**：stars 4,582，forks 823，官方 C# SDK，与 Microsoft 合作维护，最后 push 于 2026-10-12[^23]。
- **servers 仓库**：stars 91,251，forks 11,791，创建于 2024-11-19，最后 push 于 2026-10-13（截至 2026 年 10 月仍高度活跃）[^19]。README 说明该仓库是 MCP 的 reference implementations 集合，并指向社区服务器列表；查找 MCP 服务器列表应浏览 MCP Registry（registry.modelcontextprotocol.io），本仓库仅收纳 MCP steering group 维护的少量 reference servers[^19]。README 警告：仓库内服务器是用于演示 MCP 特性与 SDK 用法的 reference implementations，定位为教学示例，而非生产级实现[^19]。
- **spec & docs 仓库**：stars 9,448，forks 1,868，包含 MCP 规范、协议 schema 与官方文档；schema 以 TypeScript 定义（schema/2026-07-28/schema.ts），同时提供 JSON Schema（schema/2026-07-28/schema.json）以兼容更广；MCP 由 David Soria Parra (@dsp) 与 Justin Spahr-Summers (@jspahrsummers) 创建[^22]。
- **ext-apps**：stars 2,955，MCP Apps 协议规范与 SDK 的官方仓库，该协议用于由 MCP servers 提供的、嵌入 AI 聊天机器人的 UI 标准[^24]。

发布时（2024-11-25）提供 Google Drive、Slack、GitHub、Git、Postgres、Puppeteer 等预构建 MCP servers；早期采用者包括 Block、Apollo，开发工具公司包括 Zed、Replit、Codeium、Sourcegraph[^1]。

---

## 十、安全议题

### 10.1 协议内建的安全原则

规范层面的核心安全原则包括[^2][^35]：用户同意与控制、数据隐私、工具安全（工具代表任意代码执行，必须谨慎对待）。具体到工具调用：host 必须在调用任何工具前获得用户明确同意；host 在向 servers 暴露用户数据前必须获得明确同意[^2]。clients MUST 将工具注解视为不可信，除非其来自可信 server；SHOULD 始终有 human in the loop 并具备拒绝工具调用的能力[^8]。sampling 同样 SHOULD 有 human in the loop，应用 SHOULD 让用户在发送前查看与编辑 prompt，并在交付前审查生成的响应[^11]。

### 10.2 官方识别的特有攻击：Confused Deputy Problem

官方规范设有专门的 Security Best Practices 文档，与 MCP Authorization 规范配套，面向实现授权流程的开发者、MCP server 运维者与安全评估人员[^26][^55]。文档以「攻击 + 缓解」结构组织，首个详述攻击为 Confused Deputy Problem（混淆代理问题）——攻击者可利用连接第三方 API 的 MCP proxy server，在未获用户正当同意的情况下获取授权码[^26][^55]。

该攻击成立需同时满足四个条件[^26][^55]：MCP proxy 对第三方授权服务器使用 static client ID；proxy 允许 MCP clients 动态注册（各自获得 client_id）；第三方授权服务器在首次授权后设置 consent cookie；proxy 未在转发前实现按 client 的正当同意（per-client consent）。

攻击链为[^26][^55]：用户先正常授权使第三方授权服务器在浏览器设置 consent cookie → 攻击者随后发送含恶意 redirect_uri 与新动态注册 client_id 的链接 → 授权服务器检测到 cookie 后跳过同意页 → 授权码被重定向至攻击者服务器 → 攻击者据此换取访问令牌。

### 10.3 学术研究识别的攻击面

**系统性威胁分类学**：一篇 cs.CR 论文对 MCP 进行架构与安全双视角的系统性研究，定义了 MCP server 的完整生命周期：四个阶段（creation、deployment、operation、maintenance），并进一步分解为 16 项关键活动；构建了覆盖四类攻击者（malicious developers、external attackers、malicious users、security flaws）的威胁分类学，包含 16 种不同的威胁场景；通过真实世界案例研究验证这些风险；并针对每个生命周期阶段与威胁类别提出细粒度、可操作的安全防护措施。该论文为 v3 版本，v1 提交于 2025-03-30，v3 最后修订于 2025-10-07[^48]。

**实证攻击演示**：另一篇论文论证当前 MCP 设计对终端用户存在广泛的安全风险，并演示业界领先的 LLM 可被诱导使用 MCP 工具来攻陷 AI 开发者的系统；列举的具体攻击类型包括恶意代码执行（malicious code execution）、远程访问控制（remote access control）与凭证窃取（credential theft）。论文提出并开源了 MCPSafetyScanner——据其自述为「首个用于评估任意 MCP server 安全性的 agentic 工具」，其工作流程包含三类 agent：(a) 依据 MCP server 的 tools 与 resources 自动生成对抗样本；(b) 基于这些样本检索相关漏洞与修复方案；(c) 生成详述所有发现的安全报告[^49]。

**Tool Poisoning 与 Rug Pull**：一篇论文明确指出标准 MCP 规范存在显著安全漏洞，尤其是 Tool Poisoning（工具投毒）与 Rug Pull（拉地毯）攻击，并将 tool squatting（工具抢注/仿冒）与 rug pull 并列为 MCP 需要缓解的攻击类型。该论文提出 ETDI（Enhanced Tool Definition Interface）作为 MCP 的安全扩展，包含三项机制：密码学身份验证、不可变的版本化工具定义、显式权限管理（常借助 OAuth 2.0 实现）；并进一步提出为 MCP 扩展细粒度、基于策略的访问控制，由专用策略引擎依据运行时上下文（而非静态 OAuth scope）动态评估工具能力[^50]。

### 10.4 第三方安全工具与社区生态

围绕 MCP 已形成第三方安全工具生态：

- **snyk/agent-scan**：面向 AI agents、MCP servers 与 agent skills 的安全扫描器（由 Snyk 维护），stars 3,138，最后 push 于 2026-10-14[^25]。
- **makalin/SecureMCP**：面向 MCP 应用的安全审计工具，用于检测漏洞与错误配置，自述可识别 OAuth token 泄漏、prompt injection 漏洞、rogue MCP servers 与 tool poisoning 攻击；README 列出 Prompt Injection Tester 模块（含多种注入 payload 类型、多种注入位置测试、响应分析、system prompt override 检测与 role confusion 攻击检测）与 OAuth Token Scanner 模块（含 token 格式校验、过期与 scope 分析、存储安全评估、token endpoint 校验与 JWT token 分析）。仓库约 140 stars，创建于 2025-06-06，最后 push 于 2025-06-07[^51]。
- **Agent-Threat-Rule/agent-threat-rules (ATR)**：面向 AI agent 安全威胁的开放检测规则格式，自述覆盖 prompt injection、tool poisoning、context exfiltration 与 MCP 攻击；仓库描述称该规则集已被合并进 Microsoft、Cisco、Gen Digital、MISP 与 FINOS 的开源项目（此为仓库自述主张，未在本次取证中交叉验证）。约 415 stars，创建于 2026-03-09，最后 push 于 2026-10-14[^52]。
- **intelligent-ears/Anchor**：使用 Sigstore/Rekor 为 MCP 工具 schema 提供基于透明日志（transparency-log）的来源溯源，用于检测 rug-pull、tool-poisoning 与 tool-shadowing 攻击；该仓库的存在表明 rug pull 与 tool poisoning 被工程社区视为需要专门溯源机制应对的 MCP 攻击类型。约 11 stars，最后 push 于 2026-10-02[^53]。
- **aminrj-labs/mcp-attack-labs**：面向 AI agent 的实操攻击安全实验，涵盖 MCP tool poisoning、cross-server shadowing、RAG injection、agentic memory attacks，以及一条可运行的五阶段 MCP→A2A kill chain；自述实验可在本地复现、无需云 API。约 21 stars，最后 push 于 2026-08-04[^54]。
- **Tencent/AI-Infra-Guard**：自述为 full-stack AI Red Teaming platform，通过 Agent Scan、Skills Scan、MCP scan、AI Infra scan 与 LLM jailbreak evaluation 保护 AI 生态；stars 6,865，最后 push 于 2026-10-14。注：仅取得仓库元数据与简介，未读取 README 正文，其具体扫描能力细节不作为结论依据[^59]。

### 10.5 社区治理性安全资产

Cloud Security Alliance (CSA) 社区项目「Model Context Protocol Security」由 Model Context Protocol Security Working Group 维护，治理与协作机制包含双周技术讨论与规划会（bi-weekly technical discussions and planning sessions），协作渠道为 GitHub Discussions 与主组织仓库[^56]。已产出治理性资产包括：MCP Top 10 Security Risks（分别覆盖 server 与 client 实现）、Security TTPs（战术/技术/程序数据库，含交互式矩阵视图）、Known Vulnerabilities（CVE、安全公告、事件报告）、Hardening Guide、Build Security、Operational Security、Reference Patterns、Audit & Compliance、Tools & Scripts；社区维护 Audit Database（社区维护的 MCP 安全审计结果）与 Vulnerability Database（已知安全问题与 CVE）两个数据库[^56]。

其 Build Security 页面为开发者构建 MCP server/client/tool 提供安全指南，覆盖安全架构模式、认证策略、运行时隔离与开发生命周期最佳实践[^57]。Build Security Principles 列出：Security by Design、Threat Modeling、Secure Defaults、Least Privilege、Defense in Depth[^57]。开发安全涵盖 Secure Coding、Vulnerability Prevention、Security Testing、Supply Chain Security；部署安全涵盖 Secure Configuration、Environment Hardening、Monitoring Integration、Incident Response；CI/CD 安全涵盖 Security Scanning、Dependency Checking、Security Testing、Compliance Validation；质量保证涵盖 Security Reviews、Penetration Testing、Security Audits、Vulnerability Assessment；核心主题包括 OAuth Security Patterns、Authentication Strategies、Runtime Isolation、Architecture Patterns、Tool Metadata Specification[^57]。

registry 仓库的 SECURITY.md 要求通过 GitHub Security Advisory 私有流程报告安全漏洞，明确禁止通过公开 GitHub issues、discussions 或 pull requests 报告安全漏洞；报告需包含漏洞描述、复现步骤、潜在影响、可选修复建议[^58]。

### 10.6 第三方风险主张（未交叉验证）

以下主张来自单一第三方来源，未在本次取证中交叉验证，仅作线索记录[^27]：

- 截至 2026 年 5 月公共注册表中约有 2,000+ 个 MCP servers，覆盖云厂商、数据库、SaaS 工具与开发者平台；
- 2026 年初 OpenAI 插件生态中一次 MCP server 供应链攻击影响了 47 个企业部署，未被发现的访问持续 6 个月；
- 2026 年一项针对八个开放权重模型、赋予 MCP 工具访问的研究中，多轮攻击的 prompt injection 成功率达 92%；
- MCP 已成为 agentic AI 集成的主导协议，支持范围覆盖 Claude、OpenAI 的工具使用框架、Google Gemini 及数十个第三方 AI 平台（二手转述，未取得各厂商一手声明）。

该来源还描述 MCP 部署由三部分组成：AI host（含模型的应用，如 Claude Desktop、自定义 chatbot、AI 编码助手）、MCP clients（内置于 host，管理到 MCP servers 的连接）、MCP servers（暴露具体能力的进程）；并主张 MCP servers 需广泛系统权限，形成新的特权攻击面——被攻陷或恶意的 MCP server 可指示模型外泄数据、执行未授权命令或越权操作；建议的控制措施包括按最小权限审计 MCP server 权限、输入校验防注入、记录所有 MCP 工具调用与响应的审计日志、网络分段限制 MCP server 连通性[^27]。

---

## 矛盾与不确定性

### 矛盾一：MCP 是否为有状态会话协议（未解决）

2025-06-18 规范明确称 MCP 是「stateful session protocol focused on context exchange and sampling coordination」，且 initialize 握手 MUST 为首次交互，client 与 server 之间建立隔离的有状态会话[^3][^6][^7]。而 2026-07-28 changelog 明确列出「使 MCP 无状态化——移除 initialize / notifications/initialized 握手」，并「从 Streamable HTTP 传输中移除协议级会话与 Mcp-Session-Id 头」[^33][^34]。

这两套描述直接冲突。证据中没有任何来源说明二者如何共存（例如「2025-06-18 描述适用于 legacy 版本，2026-07-28 为现行」），也未取得 2026-07-28 版本的 lifecycle / architecture 正文来确认新语义。draft versioning 页提供了部分调和线索：将协议版本分为 Modern（2026-07-28 及以后，per-request 元数据）与 Legacy（2025-11-25 及更早，initialize 握手），并说明同时支持两者的服务器可同时实现两种行为[^30]。但该页为 draft，且未直接确认 2026-07-28 已正式生效为 Current。

### 矛盾二：roots / sampling / elicitation 是否仍为服务器发起请求（未解决）

2025-06-18 规范将 roots/list、sampling/createMessage、elicitation/create 定义为服务器向客户端发起的请求[^11][^12][^17]。而 2026-07-28 changelog 称引入 MRTR 模式「取代此前发送服务器发起请求（如 roots/list、sampling/createMessage、elicitation/create）的做法」——服务器改为返回 InputRequiredResult，客户端在重试原请求时以 inputResponses 提供信息[^33][^34]。

同样，证据中没有任何来源说明这两套描述如何共存，也未取得 2026-07-28 版本的 client features 正文来确认新语义。

### 缺口一：版本时间线的结构性缺失

证据中缺失 2025-11-25 与 2026-07-28 两个版本的完整规范正文（仅取到 changelog、lifecycle 与 versioning 页）。这两个版本是当前（2026-10）实际生效的版本，但除 changelog 外没有对应版本的架构 / 传输 / 授权正文证据。因此，本报告第二节至第七节所描述的架构细节（三角色、初始化握手、能力协商、tools/resources/prompts 生命周期、sampling、roots、授权流程）均取自 2025-06-18 版本，与 2026-07-28 现行规范可能存在系统性差异。读者若需以现行规范为准，应直接查阅 2026-07-28 版本的对应章节。

### 缺口二：2025-11-25 与 2026-07-28 之间的版本关系

2025-11-25 changelog 中未出现 server/discover、_meta 中的 protocolVersion 声明或 per-request versioning 相关条目[^28]；2025-11-25 的 Lifecycle 页仍使用 initialize 握手，protocolVersion 是 initialize params 的顶层字段[^29]；2025-11-25 的 Base Protocol 页未提及 server/discover 方法[^32]。这表明 2025-11-25 仍是 legacy 握手模型，而 2026-07-28 是向 modern 无状态模型的切换点。但证据中缺乏对这次切换的动机说明与迁移路径的完整描述。

### 缺口三：注册表服务器数量的口径不一致

多个来源给出不同口径的服务器数量：Anthropic 捐赠公告称「超过 10,000 个活跃公共 MCP 服务器」[^36][^37][^38]；第三方来源称「截至 2026 年 5 月公共注册表中约有 2,000+ 个 MCP servers」[^27]。官方 MCP Registry 的 About 页、FAQ 与 registry 仓库 README 均未给出已发布 server 的数量统计[^45][^46][^47]。这些数字的口径（「活跃公共服务器」vs「公共注册表中」）与统计时点均不明确，现有公开资料不足以确认哪个数字反映实际情况。

### 缺口四：SDK v2 与协议版本的对应关系

typescript-sdk README 说明 main 分支为 SDK v2，实现 2026-07-28 MCP 规范，v1.x 在 v2 发布（2026-07-27）后至少六个月内继续接收 bug fix 与安全更新[^21]。python-sdk README 提及 v2 版本及迁移指南，但未明确其对应的协议版本[^20]。OpenAI Agents SDK 文档称其内部使用 MCP TypeScript SDK v2 客户端，但不要求用户的 MCP server 迁移到 v2 server 包，并与实现了兼容协议版本的 v1 server 保持兼容[^40]。这些信息表明 SDK 与协议版本之间存在多对多的兼容矩阵，但证据中缺乏完整的兼容性对照表。

### 缺口五：2026-07-28 之后是否有更新版本

抓取时点 versioning 页面标注的当前协议版本为 2026-07-28[^18]，draft 规范描述的是 2026-07-28 及以后的 Modern 模型[^30]。证据中未出现 2026-07-28 之后的版本号或 changelog，因此无法确认截至 2026 年 10 月是否有更新版本处于 Draft 状态。

### 缺口六：MCP Apps 扩展的规范细节

ext-apps 仓库被描述为「MCP Apps 协议规范与 SDK 的官方仓库，该协议用于由 MCP servers 提供的、嵌入 AI 聊天机器人的 UI 标准」[^24]；Cursor 文档称其支持 MCP Apps 扩展，MCP 工具可在标准工具输出之外返回交互式 UI，且 MCP Apps 遵循渐进增强——若 host 无法渲染 app UI，同一工具仍可通过普通 MCP 响应工作[^43]；2026-07-28 规范将 MCP Apps 列为可选扩展之一（会话内联渲染的交互式 UI 元素）[^35]。但证据中未取得 MCP Apps 的协议规范正文，其消息格式、渲染契约与安全模型无法在本报告中确认。

## 参考来源

[^1]: Introducing the Model Context Protocol (Anthropic 官方发布公告) — https://www.anthropic.com/news/model-context-protocol（web, 2024-11-25）
[^2]: Model Context Protocol Specification 2024-11-05 — Introduction — https://modelcontextprotocol.io/specification/2024-11-05/index（web, 2024-11-05）
[^3]: Model Context Protocol Specification 2024-11-05 — Architecture — https://modelcontextprotocol.io/specification/2024-11-05/architecture（web, 2024-11-05）
[^4]: MCP Docs — Getting Started/Intro 与 Learn/Architecture（现行官方文档） — https://modelcontextprotocol.io/docs/getting-started/intro（web）
[^5]: Wikipedia — Model Context Protocol（二手交叉验证） — https://en.wikipedia.org/wiki/Model_Context_Protocol（web）
[^6]: MCP Specification 2025-06-18 — Architecture (client-host-server) — https://modelcontextprotocol.io/specification/2025-06-18/architecture（web, 2025-06-18）
[^7]: MCP Specification 2025-06-18 — Lifecycle (initialization, version & capability negotiation) — https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle（web, 2025-06-18）
[^8]: MCP Specification 2025-06-18 — Server Features: Tools — https://modelcontextprotocol.io/specification/2025-06-18/server/tools（web, 2025-06-18）
[^9]: MCP Specification 2025-06-18 — Server Features: Resources — https://modelcontextprotocol.io/specification/2025-06-18/server/resources（web, 2025-06-18）
[^10]: MCP Specification 2025-06-18 — Server Features: Prompts — https://modelcontextprotocol.io/specification/2025-06-18/server/prompts（web, 2025-06-18）
[^11]: MCP Specification 2025-06-18 — Client Features: Sampling — https://modelcontextprotocol.io/specification/2025-06-18/client/sampling（web, 2025-06-18）
[^12]: MCP Specification 2025-06-18 — Client Features: Roots — https://modelcontextprotocol.io/specification/2025-06-18/client/roots（web, 2025-06-18）
[^13]: Transports - Model Context Protocol (Specification 2025-06-18) — https://modelcontextprotocol.io/specification/2025-06-18/basic/transports（web, 2025-06-18）
[^14]: Transports - Model Context Protocol (Specification 2024-11-05) — https://modelcontextprotocol.io/specification/2024-11-05/basic/transports（web, 2024-11-05）
[^15]: Authorization - Model Context Protocol (Specification 2025-06-18) — https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization（web, 2025-06-18）
[^16]: Key Changes - Model Context Protocol (Specification 2025-03-26 Changelog) — https://modelcontextprotocol.io/specification/2025-03-26/changelog（web, 2025-03-26）
[^17]: Key Changes - Model Context Protocol (Specification 2025-06-18 Changelog) — https://modelcontextprotocol.io/specification/2025-06-18/changelog（web, 2025-06-18）
[^18]: Versioning - Model Context Protocol Specification — https://modelcontextprotocol.io/specification/versioning（web）
[^19]: modelcontextprotocol/servers — https://github.com/modelcontextprotocol/servers（github, 2024-11-19）
[^20]: modelcontextprotocol/python-sdk — https://github.com/modelcontextprotocol/python-sdk（github, 2024-09-24）
[^21]: modelcontextprotocol/typescript-sdk — https://github.com/modelcontextprotocol/typescript-sdk（github, 2024-09-24）
[^22]: modelcontextprotocol/modelcontextprotocol (spec & docs) — https://github.com/modelcontextprotocol/modelcontextprotocol（github, 2024-09-24）
[^23]: modelcontextprotocol/csharp-sdk — https://github.com/modelcontextprotocol/csharp-sdk（github）
[^24]: modelcontextprotocol/ext-apps — https://github.com/modelcontextprotocol/ext-apps（github）
[^25]: snyk/agent-scan — https://github.com/snyk/agent-scan（github）
[^26]: MCP Specification — Security Best Practices (2026-07-28) — https://modelcontextprotocol.io/specification/2026-07-28/basic/security_best_practices（web, 2026-07-28）
[^27]: MCP Security Risks: What Security Teams Need to Know About Model Context Protocol (Decryption Digest) — https://www.decryptiondigest.com/blog/model-context-protocol-security-risks（web, 2026-05-14）
[^28]: Changelog - Model Context Protocol Specification 2025-11-25 — https://modelcontextprotocol.io/specification/2025-11-25/changelog（web, 2025-11-25）
[^29]: Lifecycle - Model Context Protocol Specification 2025-11-25 — https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle（web, 2025-11-25）
[^30]: Versioning - Model Context Protocol Specification (draft) — https://modelcontextprotocol.io/specification/draft/basic/versioning（web）
[^32]: Base Protocol - Model Context Protocol Specification 2025-11-25 — https://modelcontextprotocol.io/specification/2025-11-25/basic/index（web, 2025-11-25）
[^33]: Key Changes — Model Context Protocol (specification 2026-07-28 changelog) — https://modelcontextprotocol.io/specification/2026-07-28/changelog（web, 2026-07-28）
[^34]: modelcontextprotocol/modelcontextprotocol — docs/specification/2026-07-28/changelog.mdx (raw) — https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/main/docs/specification/2026-07-28/changelog.mdx（github, 2026-07-28）
[^35]: Specification — Model Context Protocol (2026-07-28) — https://modelcontextprotocol.io/specification/2026-07-28（web, 2026-07-28）
[^36]: Donating the Model Context Protocol and establishing the Agentic AI Foundation — https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation（web, 2025-12-09）
[^37]: MCP joins the Agentic AI Foundation | Model Context Protocol Blog — https://blog.modelcontextprotocol.io/posts/2025-12-09-mcp-joins-agentic-ai-foundation/（web, 2025-12-09）
[^38]: Linux Foundation Announces the Formation of the Agentic AI Foundation (AAIF), Anchored by New Project Contributions Including Model Context Protocol (MCP), goose and AGENTS.md — https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation（web, 2025-12-09）
[^39]: Model context protocol (MCP) — OpenAI Agents SDK (Python) — https://openai.github.io/openai-agents-python/mcp/（web）
[^40]: Model Context Protocol (MCP) — OpenAI Agents SDK (JavaScript/TypeScript) — https://openai.github.io/openai-agents-js/guides/mcp/（web）
[^41]: Tools — Remote MCP servers (OpenAI Responses API) — https://platform.openai.com/docs/guides/tools-remote-mcp（web）
[^42]: Add and manage MCP servers in VS Code — https://code.visualstudio.com/docs/copilot/chat/mcp-servers（web）
[^43]: Model Context Protocol (MCP) — Cursor Documentation — https://cursor.com/docs/context/mcp（web）
[^44]: MCP Registry Preview — official launch announcement — https://blog.modelcontextprotocol.io/posts/2025-09-08-mcp-registry-preview/（web, 2025-09-08）
[^45]: The MCP Registry — About (official docs) — https://modelcontextprotocol.io/registry/about（web）
[^46]: modelcontextprotocol/registry (GitHub repo README) — https://github.com/modelcontextprotocol/registry（github）
[^47]: Registry FAQ — MCP Registry (official FAQ) — https://modelcontextprotocol.info/tools/registry/faq/（web）
[^48]: Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions — https://arxiv.org/abs/2503.23278（arxiv, 2025-03-30）
[^49]: MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Security Exploits — https://arxiv.org/abs/2504.03767（arxiv, 2025-04-02）
[^50]: ETDI: Mitigating Tool Squatting and Rug Pull Attacks in Model Context Protocol (MCP) by using OAuth-Enhanced Tool Definitions and Policy-Based Access Control — https://arxiv.org/abs/2506.01333（arxiv, 2025-06-02）
[^51]: makalin/SecureMCP — MCP security auditing tool — https://github.com/makalin/SecureMCP（github, 2025-06-06）
[^52]: Agent-Threat-Rule/agent-threat-rules (ATR) — open detection-rule standard for AI agent security threats — https://github.com/Agent-Threat-Rule/agent-threat-rules（github, 2026-03-09）
[^53]: intelligent-ears/Anchor — transparency-log provenance for MCP tool schemas — https://github.com/intelligent-ears/Anchor（github, 2026-10-02）
[^54]: aminrj-labs/mcp-attack-labs — hands-on offensive-security labs for MCP attacks — https://github.com/aminrj-labs/mcp-attack-labs（github, 2026-08-04）
[^55]: Security Best Practices - Model Context Protocol (官方规范文档 2026-07-28) — https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices（web, 2026-07-28）
[^56]: Model Context Protocol Security — A Cloud Security Alliance Community Project (首页) — https://modelcontextprotocol-security.io/（web）
[^57]: Build Security | Model Context Protocol Security — https://modelcontextprotocol-security.io/build/（web）
[^58]: modelcontextprotocol/registry — SECURITY.md — https://raw.githubusercontent.com/modelcontextprotocol/registry/main/SECURITY.md（github）
[^59]: Tencent/AI-Infra-Guard — https://github.com/Tencent/AI-Infra-Guard（github）