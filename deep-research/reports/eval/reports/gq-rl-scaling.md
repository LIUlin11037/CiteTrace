# 推理模型的 RL scaling 与预训练 scaling：关系、test-time compute 的位置，以及证据边界

## 摘要

预训练 scaling 的经典结论是：在算力预算固定时，损失对参数量 N、数据量 D、训练算力 C 各自呈幂律关系，且存在算力最优的分配比例[^1][^26]。Chinchilla 进一步给出「模型规模与训练 token 数应等比例缩放」的结论，并指出当时的大模型普遍训练不足[^2]。这条路线的前提是「有足够数据可喂」——而 Kimi k1.5 明确把 next-token 预训练的局限表述为受可用训练数据量限制，从而把 RL scaling 定位为「新的 scaling 轴」，理由是 RL 可以通过带奖励的探索扩展训练数据，而不再受限于既有静态数据集[^12][^14]。

RL scaling（o1/R1 路线）与预训练 scaling 的关系，在现有证据下可归纳为三层，且彼此存在张力：

1. **RL 算力相对预训练算力是小量级，但它是独立可扩展的轴。** ScaleRL 估算 DeepSeek-R1-Zero 的 RL 训练用 100,000 H800 GPU-hours，约为其预训练算力的 3.75%[^8]；Kimi k1.5 则把 RL 明确定位为「新的 scaling 轴」[^12][^14]。
2. **RL 的收益上界受基座模型约束，这一派证据认为 RL 主要提升采样效率而非扩展推理边界。** 以 pass@k 为指标的系统评测发现：RLVR 模型在小 k 时优于基座，但 k 增大后基座模型在所有基准、所有模型家族上稳定反超，作者结论是 RLVR 的推理能力「源自并受限于基座模型」[^7][^10]。
3. **反方证据认为长时程 RL 能真正扩展边界。** ProRL 主张长时程 RL 能发现基座模型在大量采样下也无法触及的新推理策略，并在基座完全失败的任务上取得进展[^9]；ScaleRL 则指出并非所有 RL 配方有相同的渐近性能上限，loss 类型与 batch size 等选择可以移动该上限[^8]。

Test-time compute 之所以重要，是因为它把「推理时算力」变成与预训练算力并列的可调配资源：在 FLOPs 匹配的评估中，test-time 算力可使小模型超越大 14 倍的模型[^3]；R1-Zero 在 RL 中「自然学会用更多思考时间解推理题」，响应长度与性能在训练中途同时跃升[^5]；Qwen3 引入 thinking budget 机制，报告称增大 thinking token 预算在多项任务上带来一致性能提升[^13]。

必须前置声明的证据边界：本报告 27 条证据中**没有任何一条来自 OpenAI o1/o3 的系统卡或模型卡**，因此关于 OpenAI 推理模型的预训练算力/数据规模、RL 阶段算力占比与步数、CoT token 随难度的变化、推理时算力-准确率实测曲线，**现有公开资料不足以确认**。定价侧只有 DeepSeek 单一来源的有效快照，不足以支撑趋势判断。

---

## 一、预训练 scaling 的基线：幂律、算力最优分配与「数据墙」

### 1.1 三因子幂律的形式化定义

Kaplan et al. 把「规模」分解为三个因子：模型参数量 N（不含 embedding）、数据集大小 D、训练算力 C；在合理范围内，性能对深度/宽度等架构超参依赖很弱[^1][^26]。在未被其他两个因子瓶颈时，损失对 N、D、C 各自呈幂律关系，趋势跨越六个以上数量级，且在上端未观察到偏离迹象[^1]。具体形式为：

- 参数受限、训练至收敛：L(N) = (N_c/N)^α_N，α_N ≈ 0.076，N_c ≈ 8.8×10^13（非 embedding 参数）[^1]；
- 数据受限（大模型 + 早停）：L(D) = (D_c/D)^α_D，α_D ≈ 0.095，D_c ≈ 5.4×10^13 tokens[^1]；
- 算力受限且模型规模最优：L(C_min) = (C_c^min/C_min)^α_C^min，α_C^min ≈ 0.050，C_c^min ≈ 3.1×10^8 PF-days[^1]；
- 联合依赖与过拟合的单一方程：L(N,D) = [(N_c/N)^(α_N/α_D) + D_c/D]^α_D[^1]。

这些常数依赖词表与分词方式（Webtext2 训练集），N_c、D_c、C_c^min 的具体数值不具根本意义[^1]。

### 1.2 算力最优分配：两篇论文的分歧点

Kaplan et al. 的算力最优结论是：**训练非常大的模型、用相对适中的数据量、并显著早于收敛就停止训练**，而非把小模型训到收敛[^1][^26]。其经验分配为 N ∝ C_min^0.73、B ∝ C_min^0.24、S ∝ C_min^0.03，且 D ~ C^0.27[^1]。

Chinchilla 的结论方向相反：当前大模型显著训练不足，原因是业界扩大模型规模的同时保持训练数据量不变；算力最优下模型规模与训练 token 数应**等比例缩放**，模型规模每翻倍，训练 token 数也应翻倍[^2]。其验证方式是训练 70B 参数、4 倍数据的 Chinchilla，在与 Gopher 相同算力预算下，于大量下游任务上一致且显著优于 Gopher(280B)、GPT-3(175B)、Jurassic-1(178B)、Megatron-Turing NLG(530B)[^2]；MMLU 平均准确率 67.5%，比 Gopher 提升超过 7%[^2]；且因参数更少，微调与推理阶段算力显著更少[^2]。

这一分歧本身有方法论层面的解释（见第五节）：Epoch AI 的复现指出 Hoffmann et al. 的 A3 参数估计与其前两种方法不一致，且给出了不合常理地窄的置信区间；重做后得到的缩放律与前两种估计方法相容，并指出「训练 token 应与参数量大致同速增长」这一结论已被他人复现[^27]。

### 1.3 「数据墙」是 RL scaling 的出发点

Kimi k1.5 的摘要明确表述：next-token 预训练在 scaling compute 上有效，但受可用训练数据量限制；scaling RL 则开辟了新的改进轴，因为模型可通过带奖励的探索来扩展训练数据[^14][^12]。这是把 RL 与预训练 scaling 关联起来的最直接动机陈述：预训练 scaling 的约束变量是静态语料，RL 的约束变量则被表述为可探索的解空间。

---

## 二、RL scaling 的实证形态：sigmoid 曲线、可移动的上限与「苦涩教训」

### 2.1 用 sigmoid 而非幂律刻画 RL 训练

ScaleRL 是目前规模最大的 RL scaling 系统性研究之一，总计超过 400,000 GPU-hours（Nvidia GB200），提出用 sigmoid 型 compute-performance 曲线刻画 RL 训练[^8]：

> Reward Gain = (A − R0) × 1/(1 + (C_mid/C)^B)

其中 A 为渐近通过率、B 为决定算力效率的 scaling 指数、C_mid 为曲线中点[^8]。这个形式与预训练的幂律在结构上不同：预训练损失单调下降且上端未见偏离[^1]，而 RL 曲线有一个显式的渐近上限 A。

### 2.2 三条关键结论

- **RL 性能上限并非普适。** loss 类型与 batch size 等选择可以移动渐近上限 A[^8]。
- **苦涩教训（Bitter Lesson）在 RL 内部同样成立。** 在小算力预算下看起来更优的方法，外推到大算力区间可能更差；可通过早期训练动态估计 (A, B) 来识别可扩展方法[^8]。
- **多数常见干预只调节效率、不改上限。** loss aggregation、数据课程、长度惩罚、advantage normalization 等主要调节算力效率 B，而不显著改变性能上限[^8]。

实验规模：在 8B 稠密模型上做设计消融（单次运行最高 16,000 GPU-hours），最终在 8B 稠密模型上跑 100k GPU-hours、在 17B×16 MoE（Scout）上跑 50k GPU-hours，分别训练 7400 与 7100 步，是 ProRL 训练步数的 3.5 倍[^8]。ScaleRL 配方达到渐近奖励 A = 0.61，优于 DeepSeek(GRPO)、Qwen-2.5(DAPO)、Magistral、MiniMax-M1 等配方的拟合曲线[^8]。

### 2.3 模型规模仍在起作用

ScaleRL 的 Figure 1(b) 直接指出「scaling model size substantially improves the downstream and asymptotic RL performance」，即增大模型规模显著提升下游与渐近 RL 性能[^8]。这一点与「RL 受限于基座」的结论方向一致，但含义不同：它说明 RL 阶段的收益函数本身依赖预训练规模，而不是说 RL 无法带来新能力。

---

## 三、核心争议：RL 是扩展推理边界，还是只提升采样效率？

### 3.1 「受限于基座」一派

该论文用大 k 值的 pass@k 作为评估指标，跨多个模型家族、RL 算法以及数学/代码/视觉推理基准系统探测 RLVR 模型的推理能力边界[^7]。其方法论证是：平均分只反映平均情况、可能低估模型真实潜力，而 pass@k 反映模型在 k 次尝试内「有可能」解决的问题比例，即推理边界[^10]。核心发现：

- RLVR 模型在小 k（如 k=1）时优于基座模型，但当 k 增大时，基座模型在所有基准和所有 LLM 家族上都稳定超过 RLVR 模型[^7][^10]；
- 覆盖率与困惑度分析显示 RLVR 模型产出的推理路径已存在于基座模型的输出分布中；RLVR 主要提升采样效率，而非解决新问题[^7]；
- Figure 1（右）显示：随 RLVR 训练推进，平均性能（pass@1）提升，但可解问题覆盖率（pass@256）下降，即推理边界收缩[^10]；
- 人工检查显示，对大多数问题基座模型至少能产出一条正确 CoT，包括此前被认为只有 RLVR 模型才能解决的问题[^10]；
- 以基座模型为上界定义 sampling efficiency gap（Δ_SE），六种流行 RLVR 算法（如 PPO、GRPO、Reinforce++）表现相近，且都远未充分利用基座模型的潜力[^7][^10]；
- 蒸馏与 RLVR 本质不同：蒸馏可从更强教师引入新推理模式并真正扩展推理范围，而 RLVR 不能[^7]。

该论文发表于 NeurIPS 2025 Oral，并获 ICML 2025 AI4MATH workshop best paper[^7]。

### 3.2 「长时程 RL 扩展边界」一派

ProRL 直接挑战「RL 只是放大基座模型中已存在的高奖励输出」这一观点，主张长时程 RL 训练能发现基座模型在大量采样下也无法触及的新推理策略[^9]。方法包含三要素：KL 散度控制、reference policy resetting、多样化任务套件[^9]。实证上，RL 训练模型在广泛的 pass@k 评估上持续优于基座模型，包括基座模型无论尝试多少次都完全失败（fail entirely）的场景[^9]；推理边界的提升与基座模型的任务胜任度（task competence）和训练时长强相关，作者据此认为 RL 能随时间探索并填充解空间的新区域[^9]。论文开源了模型权重[^9]。

### 3.3 两派如何并存

两组结论的差异至少可归因于三点，且都能在证据中找到落点：(a) 训练时长——ProRL 强调「长时程」是边界扩展的前提[^9]，而 ScaleRL 指出小算力下的结论外推到大算力可能反转[^8]；(b) 指标选择——pass@k 大 k 与 pass@1 给出不同排序[^7][^10]；(c) 配方差异——ScaleRL 证明 loss 类型与 batch size 可移动渐近上限 A[^8]，因此「RL 能否扩展边界」不是单一的是非题，而依赖配方与预算。现有证据不足以判定哪一派的结论在更大规模下成立。

---

## 四、Test-time compute 为何重要

### 4.1 概念界定

按该综述的组织，test-time compute 指在推理阶段额外投入算力以提升模型表现，其机制按 System-1 / System-2 分野划分[^4]：

- **System-1 模型**中，test-time compute 通过**参数更新、输入修改、表征编辑、输出校准**四种手段应对分布偏移，提升鲁棒性与泛化[^4]；
- **System-2 模型**中，test-time compute 通过**重复采样（repeated sampling）、自纠正（self-correction）、树搜索（tree search）**增强模型解决复杂问题的推理能力[^4]。

该综述按 System-1 到 System-2 思维的演进组织，强调 test-time compute 在 System-1 → 弱 System-2 → 强 System-2 转变中的关键作用[^4]；并指出 o1 模型在复杂推理上的表现表明 test-time compute scaling 可进一步释放模型潜力，实现 System-2 思维[^4]。需注意该文作者标注为 work in progress，v1 2025-01-05、v3 2025-06-29[^4]。

### 4.2 与预训练算力的权衡：FLOPs 匹配下的比较

Snell et al. 的问题设定正是「推理时算力与预训练算力应如何权衡」：若允许 LLM 使用固定但非平凡的推理时算力，其在困难 prompt 上的性能能提升多少[^3]。论文分析两类扩展 test-time 算力的主要机制：(1) 针对稠密、基于过程的 verifier 奖励模型做搜索；(2) 在测试时依据 prompt 自适应地更新模型对响应的分布[^3]。

关键发现是：两类机制下，扩展 test-time 算力的不同方法其有效性都强烈依赖于 prompt 的难度[^3]。由此提出「compute-optimal」扩展策略——按每个 prompt 自适应地最有效分配 test-time 算力[^3]。量化结果：

- 相比 best-of-N 基线，compute-optimal 策略可将 test-time 算力扩展的效率提升 4 倍以上[^3]；
- 在 FLOPs 匹配的评估中，在较小基座模型已能达到一定非平凡成功率的问题上，test-time 算力可使小模型超越大 14 倍的模型[^3]。

这条结论的重要性在于它给出了一个可比较的兑换率：推理时算力与模型参数之间存在可替换关系，但替换的有效性以「基座模型已有非平凡成功率」为前提[^3]。

### 4.3 在 RL 训练内部，test-time compute 表现为响应长度

R1-Zero 的实证把 test-time compute 与 RL 训练直接连起来：论文明确指出 R1-Zero 在解题时倾向生成更长响应，在单条响应内包含验证、反思与替代路径探索，即「自然学会用更多思考时间解推理题」[^5]。其训练细节显示长度与性能的耦合是突变的：生成长度上限在 8.2k step 之前为 32,768 tokens，之后提升到 65,536 tokens，性能与响应长度在 8.2k step 处同时出现显著跃升[^5]。R1-Zero 的 RL 训练共 10,400 步、对应 1.6 个训练 epoch，每步 32 个不同问题、训练 batch size 为 512[^5]。

Kimi k1.5 给出同一现象的另一种表述：RL 阶段把上下文窗口扩展到 128k，并观察到随上下文长度增加性能持续提升，作者据此把 context length 认定为 LLM RL 持续 scaling 的关键维度[^12]；由于上下文可扩展，学到的 CoT 表现出 planning、reflection、correction 特性，上下文变长等价于增加搜索步数，因此无需 MCTS、value function、process reward model 等复杂技术即可取得强性能[^12]。

### 4.4 产品化：thinking budget 作为可调算力旋钮

Qwen3 把 thinking 模式与非 thinking 模式统一在同一框架内，并引入 thinking budget 机制，允许用户在推理时自适应分配算力以平衡延迟与性能[^13]；报告称增大 thinking token 的 thinking budget 会在多项任务上带来一致的性能提升[^13]。其统一框架的意义在于无需在 chat 模型与专用推理模型之间切换，可按用户 query 或 chat template 动态切换模式[^15]。

工程侧的对照物是 s1 的 budget forcing：设置 MAX_TOKENS_THINKING=32000、NUM_IGNORE=1，在思考阶段强制忽略一次结束思考 token（追加 "Wait"）以延长推理，再生成最终答案[^18]。该仓库自述为「用 1,000 条样本 + budget forcing 匹配 o1-preview 的强推理性能」的最小配方[^18]，并记录 2025-02 发布 s1.1，复用同一批 s1K 问题但推理轨迹改由 r1 生成[^18]。需注意这是项目自述主张，非独立复现。

### 4.5 为什么 test-time compute 在经济学上重要

DeepSeek API 的定价结构提供了一个直接观察点：输出 token 单价显著高于输入——以 deepseek-flash off-peak 为例，输出 $0.6 对 cache miss 输入 $0.15（约 4 倍），对 cache hit 输入 $0.003（约 200 倍）每 1M tokens[^21]。在 thinking 模式为默认模式的在售模型上（deepseek-flash 与 deepseek-v4-pro 均「Supports both non-thinking and thinking (default) modes」）[^21]，test-time compute 的扩张直接对应成本结构中最贵的那一项。两个模型上下文长度为 1M、最大输出 384K tokens[^21]。此外存在峰谷分时定价，off-peak 费率为 peak 费率的一半，peak 时段为 UTC 周一至周五 01:00-04:00 与 06:00-10:00[^21]。

---

## 五、矛盾与不确定性

### 5.1 预训练最优分配：Kaplan vs Chinchilla，以及复现层面的修正

Kaplan et al. 主张算力最优应「用非常大的模型、相对适中的数据、显著早于收敛停止」[^1][^26]，并给出 D ∝ N^0.74、模型规模每增 8 倍数据只需增约 5 倍即可避免过拟合惩罚[^1]。Chinchilla 主张等比例缩放、且当时模型普遍训练不足[^2]。Epoch AI 的复现工作提供了第三种视角：Hoffmann et al. 的 A3 参数估计与其 A1、A2 方法不一致，且无法拟合所提取的数据；作者归因于优化器因 loss scale 选择不当而提前终止，以及正文参数被四舍五入[^27]。重做后得到 A=482.01、B=2085.43、E=1.8172、α=0.3478、β=0.3658，a=β/(α+β)=0.5126，而原文正文值为 a=0.454[^27]。作者指出 β 的精度影响很大：若用四舍五入的 0.28 而非 TeX 源码中的 0.2849，在典型 D=10^11 token 下会给可约损失的有限数据项引入约 13% 的正偏差[^27]。Hoffmann et al. 的作者已确认拟合较差的原因是对不同数据点的 Huber loss 取平均而非求和，以及高 loss scale 导致优化提前终止[^27]。**含义**：把 Chinchilla 的「等比例」当作精确常数使用需谨慎，但其定性结论已被多方复现[^27]。

### 5.2 RL 是否扩展推理边界：直接对立的证据

该论文（NeurIPS 2025 Oral）结论是 RLVR 未引出根本性新推理模式，能力「源自并受限于基座模型」[^7][^10]；ProRL 主张长时程 RL 能发现基座在大量采样下也无法触及的新策略，并在基座完全失败的任务上取得进展[^9]。ScaleRL 的第三条结论（上限可被配方移动）[^8]在逻辑上为 ProRL 留出空间，但其「苦涩教训」结论又警告小算力下的优势可能不可外推[^8]。**现有公开资料不足以确认哪一方在 frontier 规模下成立**；可确认的是：指标（pass@1 vs 大 k pass@k）、训练时长、配方三者都会改变结论方向[^7][^8][^9][^10]。

### 5.3 RL 算力占比：只有一个第三方估算

ScaleRL 给出 DeepSeek-R1-Zero 的 RL 训练用 100,000 H800 GPU-hours、约为其预训练算力的 3.75%[^8]。这是论文作者的估算而非厂商披露，且本报告证据中不存在第二个独立的同口径估算。**单点估算不足以支撑「RL 算力占比的行业趋势」判断。**

### 5.4 OpenAI o1/o3 相关事实：覆盖度为 0（blocking）

27 条证据中没有任何一条来自 OpenAI o1/o3 的系统卡或模型卡。唯一涉及 OpenAI 的条目已明确自述「该来源为 DeepSeek 而非 OpenAI……不能用于回答 OpenAI 系统卡披露了哪些数字」[^11]。因此以下问题**现有公开资料不足以确认**：o1/o3 的预训练算力与数据规模量级、RL 阶段算力占比与步数、CoT token 数随难度的变化、推理时算力-准确率实测曲线。本报告中所有关于推理模型 RL 训练的量化细节均来自 DeepSeek[^5]、Kimi[^12] 与第三方研究[^8]，不能外推为 OpenAI 的披露事实。Kimi k1.5 自称 long-CoT 版本成绩「与 OpenAI o1 相当」[^12]，属作者自述对标，非 OpenAI 侧数据。

### 5.5 定价与成本：单一来源，不构成趋势

定价侧实际只有 DeepSeek API 文档一条有效来源[^21]。Anthropic 定价页抓取仅返回区域限制提示、无正文[^25]；tokencost[^22]、genai-prices[^23]、Grandet[^24] 三个仓库均只取得元数据、未读 README 或价格表正文，且 Grandet 自述为营销性主张[^24]。**单点快照不等于趋势**，本报告不给出「推理服务侧 token 消耗与定价变化趋势」的结论。同样，单次 RL 训练的实际 GPU 小时与成本缺少厂商或框架方直接披露：verl README 记录 2025/12 Mind Lab 使用 verl + Megatron-bridge 在 64 张 H800 上为万亿参数模型训练 GRPO LoRA，但未给时长与成本[^16]；OpenRLHF 的 README 正文未取证，仅有元数据与描述字段[^19]。

### 5.6 开源工具链证据的强度差异

verl 的仓库元数据为 ★24,008、forks 4,808、最近 push 2026-10-23，README 自述为「flexible, efficient and production-ready RL training library」，并主张「State-of-the-art throughput」[^16]——后者属项目自身 README 的营销性表述，需交叉验证。其 News 条目提供了可核验的落地记录：2025/03 DAPO 基于 Qwen2.5-32B 在 AIME 2024 达到 50 分、训练完全由 verl 驱动[^16]；2025/06 配合 Megatron 后端支持 DeepSeek-671B、Qwen3-235B 等大 MoE 模型[^16]。TRL 的 stars 数因 GitHub API 403 rate limit 未取得，仅有 3,911 commits 与 README 主张（GRPOTrainer 实现的 GRPO「more memory-efficient than PPO and was used to train Deepseek AI's R1」）[^17]。limit-of-RLVR 仓库仅为论文配套代码库，README 主张未取证[^20]。

### 5.7 其他需标注的不确定性

- R1-Zero 存在可读性差与语言混杂（中英文混在同一条 CoT 内）的问题，且纯规则 RL 阶段只聚焦推理任务，在写作与开放域问答上表现有限；R1 因此改为 rejection sampling + RL + SFT 的多阶段框架[^5]。这说明「纯 RL scaling」在通用能力上有已知代价。
- R1-Zero 明确不使用神经奖励模型（outcome-based 或 process-based），原因是大规模 RL 中易被 reward hacking；奖励为等权的 accuracy reward + format reward[^5]。这一选择与 test-time compute 路线中「稠密过程奖励模型 + 搜索」的机制[^3]形成对照，两条路线的相对效率现有证据不足以判定。
- Qwen3 报告称从先进教师模型蒸馏在性能与训练效率上显著优于强化学习[^13]，这与「RL 是新的 scaling 轴」的叙事[^12][^14]存在张力；该结论出现在正文摘要式段落中，具体对比数据位于后训练章节，本报告未取得该章节数据。
- DeepSeek-R1 论文 v1 提交于 2025-01-22、v2 修订于 2026-01-04，期刊引用为 Nature volume 645, pages 633-638 (2025)[^6][^11]；本报告读取的是摘要与引言层面，未读取结果表格[^11]。

## 参考来源

[^1]: Scaling Laws for Neural Language Models (Kaplan et al., 2020) — https://arxiv.org/html/2001.08361v1（arxiv, 2020-01-23）
[^2]: Training Compute-Optimal Large Language Models (Chinchilla, Hoffmann et al., 2022) — https://arxiv.org/abs/2203.15556v1（arxiv, 2022-03-29）
[^3]: Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (Snell et al., 2024) — https://arxiv.org/abs/2408.03314v1（arxiv, 2024-08-06）
[^4]: A Survey of Test-Time Compute: From Intuitive Inference to Deliberate Reasoning (Ji et al., 2025) — https://arxiv.org/abs/2501.02497（arxiv, 2025-01-05）
[^5]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (v2) — https://arxiv.org/html/2501.12948v2（arxiv, 2025-01-22）
[^6]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning — abstract/metadata page — https://arxiv.org/abs/2501.12948v2（arxiv, 2025-01-22）
[^7]: Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? — https://arxiv.org/abs/2504.13837（arxiv, 2025-04-18 (v1), v5 2025-11-24）
[^8]: The Art of Scaling Reinforcement Learning Compute for LLMs (ScaleRL) — https://arxiv.org/abs/2510.13786（arxiv, 2025-10-15）
[^9]: ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models — https://arxiv.org/abs/2505.24864（arxiv, 2025-05-30）
[^10]: Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (HTML 全文 v5) — https://arxiv.org/html/2504.13837v5（arxiv, 2025-11-24 (v5)）
[^11]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948, v2) — https://arxiv.org/abs/2501.12948（arxiv, 2025-01-22）
[^12]: Kimi k1.5: Scaling Reinforcement Learning with LLMs (arXiv:2501.12599, v4) — https://arxiv.org/html/2501.12599v4（arxiv, 2025-01-22 (v1); last revised 2025-06-03 (v4)）
[^13]: Qwen3 Technical Report (arXiv:2505.09388, v1) — https://arxiv.org/html/2505.09388v1（arxiv, 2025-05-14）
[^14]: Kimi k1.5: Scaling Reinforcement Learning with LLMs — arXiv abstract page (metadata + abstract) — https://arxiv.org/abs/2501.12599（arxiv, 2025-01-22 (v1); v4 2025-06-03）
[^15]: Qwen3 Technical Report — arXiv abstract page (metadata + abstract) — https://arxiv.org/abs/2505.09388（arxiv, 2025-05-14）
[^16]: verl-project/verl — verl/HybridFlow: A Flexible and Efficient RL Post-Training Framework — https://github.com/verl-project/verl（github, 2026-10-23）
[^17]: huggingface/trl — TRL - Transformers Reinforcement Learning — https://github.com/huggingface/trl（github）
[^18]: simplescaling/s1 — s1: Simple test-time scaling — https://github.com/simplescaling/s1（github, 2025-06-25）
[^19]: OpenRLHF/OpenRLHF — https://github.com/OpenRLHF/OpenRLHF（github, 2026-10-05）
[^20]: LeapLabTHU/limit-of-RLVR — https://github.com/LeapLabTHU/limit-of-RLVR（github, 2026-07-27）
[^21]: Models & Pricing — DeepSeek API Docs — https://api-docs.deepseek.com/quick_start/pricing（web）
[^22]: AgentOps-AI/tokencost — Easy token price estimates for 400+ LLMs（仅取得仓库元数据，未读 README 正文） — https://github.com/AgentOps-AI/tokencost（github）
[^23]: pydantic/genai-prices — Calculate prices for calling LLM inference APIs（仅取得仓库元数据，未读 README 正文） — https://github.com/pydantic/genai-prices（github）
[^24]: hanzohanzhe/Grandet — Find the cheapest API for any model（仅取得仓库元数据，未读 README 正文） — https://github.com/hanzohanzhe/Grandet（github）
[^25]: Anthropic Claude API Pricing 文档（访问受阻，仅返回区域限制提示） — https://docs.anthropic.com/en/docs/about-claude/pricing（web）
[^26]: Scaling Laws for Neural Language Models (Kaplan et al., 2020) — https://arxiv.org/abs/2001.08361v1（arxiv, 2020-01-23）
[^27]: Chinchilla Scaling: A replication attempt (Besiroglu, Erdil, Barnett, You — Epoch AI) — https://arxiv.org/html/2404.10102v2（arxiv, 2024-04-15）