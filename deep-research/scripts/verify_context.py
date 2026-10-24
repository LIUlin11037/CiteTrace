"""Verify context budgeting keeps the informative parts (offline, no API key).

The property under test is not "shorter" but "shorter *without losing the
answer*": a fetched paper whose results table sits past the old prefix cut must
survive the budget.

Run: python scripts/verify_context.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deep_research.core.context import estimate_tokens, trim_document  # noqa: E402

BOILERPLATE = "\n\n".join(
    f"导航链接 {i}：首页 / 产品 / 定价 / 文档 / 博客 / 关于我们，点击了解更多服务条款与隐私政策说明。"
    for i in range(80)
)
RESULTS_TABLE = """## 表 2：蒸馏模型基准结果

| 模型 | AIME 2024 | MATH-500 | GPQA Diamond |
| --- | --- | --- | --- |
| R1-Distill-Qwen-32B | 72.6% | 94.3% | 62.1% |
| R1-Distill-Llama-70B | 70.0% | 94.5% | 65.2% |
| 同尺寸基线 Qwen2.5-32B | 26.7% | 71.2% | 41.3% |
"""
TAIL = "\n\n".join(
    f"参考文献 {i}: 略，仅列出条目与页码。" for i in range(80)
)
DOC = "标题：DeepSeek-R1 蒸馏小模型技术报告\n\n摘要：本文报告六个蒸馏模型的评测结果。\n\n" + BOILERPLATE + "\n\n" + RESULTS_TABLE + "\n\n" + TAIL


def test_table_survives_budget() -> None:
    limit = 3000
    out, trimmed = trim_document(DOC, limit)
    assert trimmed
    assert len(out) <= limit + 200, f"超预算: {len(out)}"
    # 关键断言：信息密度最高的表格必须留下（旧的"取前 N 字符"会丢掉它）
    assert "R1-Distill-Qwen-32B" in out, "结果表格被裁掉了"
    assert "72.6%" in out, "结果数字被裁掉了"
    assert "已按信息密度裁剪" in out, "缺少裁剪标记，模型会误以为读到了全文"
    old_prefix = DOC[:limit]
    print(
        f"[ok] 1. 表格存活：新策略保留基准数字；旧的取前 {limit} 字符"
        f"{'会丢掉表格' if 'R1-Distill-Qwen-32B' not in old_prefix else '恰好包含（本案）'}"
    )


def test_short_doc_untouched() -> None:
    short = "一句话结论：MLA 是一种注意力机制。"
    out, trimmed = trim_document(short, 12000)
    assert not trimmed and out == short
    print("[ok] 2. 短文档原样返回（不加标记、不裁剪）")


def test_head_kept() -> None:
    out, _ = trim_document(DOC, 2000)
    assert out.startswith("标题：DeepSeek-R1"), "标题/摘要必须保留"
    print("[ok] 3. 头部（标题/摘要）必定保留")


def test_token_estimate_sanity() -> None:
    zh = estimate_tokens("中文四个字")
    en = estimate_tokens("abcdefgh")
    assert zh >= 4 and en >= 2, (zh, en)
    print(f"[ok] 4. token 估算可用：4 个汉字≈{zh}，8 个英文字符≈{en}")


def test_ratio_report() -> None:
    out, _ = trim_document(DOC, 3000)
    print(
        f"[ok] 5. 预算效果：{len(DOC)} 字符（≈{estimate_tokens(DOC)} tokens）"
        f" → {len(out)} 字符（≈{estimate_tokens(out)} tokens）"
        f"，压缩 {1 - len(out) / len(DOC):.0%}"
    )


def main() -> None:
    test_table_survives_budget()
    test_short_doc_untouched()
    test_head_kept()
    test_token_estimate_sanity()
    test_ratio_report()
    print("\nCONTEXT BUDGET TEST PASSED")


if __name__ == "__main__":
    main()
