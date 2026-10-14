# 项目长期约定（agent workspace）

## Git 提交风格
- 用户的习惯：**简短中文标题，无 Conventional Commits 前缀（feat/fix 等），无长正文**
  - 例：「服务端完成」「初测」「init」「加上 web 演示页，顺手修了几个 bug」
- 提交前先 `git log --format="%s" -10` 看历史风格，按用户风格写；不要默认套 Conventional Commits + 结构化正文
- 需要交代的技术细节写在 `docs/handoff.md` / README / 代码注释里，不塞进 commit message
- commit 前先确认是否已推送（`git log origin/main..main`），未推送才允许改写历史

## 分工约定（用户提出）
- 架构核心 / 易错部分由主模型做；外围、低技术力模块可按 `docs/handoff.md` 契约交低端模型
- 但涉及"先要修后端才通"的外围，主模型自己做完更划算（本次 Web 页即如此：契约先写好，最后仍是主模型实现并修掉 3 个后端 bug）

## 成本意识
- 用户在意 API 花费：全量 26 题基准约 ¥10（deepseek-chat 空闲时段 ~¥0.35/题，高峰翻倍）
- 想验证改动影响时优先"抽 2-3 题 + 与旧数据对比"，而非无脑全量重跑
- 对比实验必须控制变量：跑基准时要与旧数据条件一致（如 `DR_ENABLE_LONG_TERM_MEMORY=false`，因为旧基准跑在 M3 记忆之前）
