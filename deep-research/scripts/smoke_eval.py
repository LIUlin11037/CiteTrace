"""Smoke-test the M4 eval pipeline WITHOUT an LLM key or the network.

- run_research: returns a fabricated report + evidence
- citation verification: real parsing, mocked per-kind verifiers
- judges: canned AlignmentResult / CoverageResult
Asserts the aggregation and report artifacts are correct.

Run: python scripts/smoke_eval.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deep_research.agents.state import Evidence
from deep_research.eval.citations import extract_citations, verify_citations
from deep_research.eval.judge import AlignmentResult, ClaimVerdict, CoverageResult
from deep_research.eval.runner import run_eval

FAKE_REPORT = """# DeepSeek-R1 技术路线调研

## 摘要
DeepSeek-R1 以强化学习为核心提升推理能力[^1]。GRPO 是其关键优化算法[^2]。

## 正文
R1-Zero 展示了纯 RL 可涌现长链推理[^1]。官方开源了蒸馏小模型[^3]。
GitHub 上相关实现获得了大量关注[^4]。该引用是编造的[^5]。

## 参考来源
[^1]: DeepSeek-R1 paper — https://arxiv.org/abs/2501.12948（arxiv，2025-01）
[^2]: DeepSeekMath paper — https://arxiv.org/abs/2402.03300（arxiv，2024-02）
[^3]: DeepSeek-R1 repo — https://github.com/deepseek-ai/DeepSeek-R1（github，2025-01）
[^4]: Some blog — https://example.com/blog/r1（web，2025-02）
[^5]: Fabricated — https://arxiv.org/abs/2999.99999（arxiv，2025-03）
"""

FAKE_EVIDENCES = [
    Evidence(
        sub_question_id="q1",
        source_type="arxiv",
        title="DeepSeek-R1",
        url="https://arxiv.org/abs/2501.12948",
        key_points=["R1 以强化学习为核心提升推理能力", "GRPO 是其优化算法", "R1-Zero 纯 RL 涌现推理"],
    )
]

VERDICT_MAP = {
    "2501.12948": "exists",
    "2402.03300": "exists",
    "deepseek-ai/DeepSeek-R1": "exists",
    "https://example.com/blog/r1": "exists",
    "2999.99999": "missing",
}


async def main() -> None:
    # --- 1) citation parsing (pure, offline) ---
    cites = extract_citations(FAKE_REPORT)
    assert len(cites) == 5, f"expected 5 citations, got {len(cites)}"
    kinds = [c.kind for c in cites]
    assert kinds == ["arxiv", "arxiv", "github", "web", "arxiv"], kinds
    assert cites[2].target == "deepseek-ai/DeepSeek-R1"
    print("[ok] 1. citation parsing: 5 refs, kinds/targets correct")

    # --- 2) programmatic verification with mocked verifiers ---
    async def fake_verify(_client, target):
        return VERDICT_MAP[target]

    report = await verify_citations(
        FAKE_REPORT, verify_arxiv=fake_verify, verify_github=fake_verify, verify_web=fake_verify
    )
    assert report.total == 5 and report.exists == 4 and report.missing == 1, report.details
    assert abs(report.existence_rate - 0.8) < 1e-9
    print(f"[ok] 2. citation verification: {report.exists}/{report.total} exist "
          f"(rate {report.existence_rate:.0%}), fabricated ref flagged missing")

    # --- 3) full pipeline with injected fakes ---
    async def fake_research(_topic):
        return FAKE_REPORT, FAKE_EVIDENCES

    async def fake_citations(_report):
        return await verify_citations(
            FAKE_REPORT, verify_arxiv=fake_verify, verify_github=fake_verify,
            verify_web=fake_verify
        )

    async def fake_align(_report, _evs):
        return AlignmentResult(verdicts=[
            ClaimVerdict(claim="c1", verdict="supported", reason="r"),
            ClaimVerdict(claim="c2", verdict="partial", reason="r"),
        ])

    async def fake_cover(_report, points):
        return CoverageResult(covered=points[:3], missed=points[3:])

    with tempfile.TemporaryDirectory() as tmp:
        summary = await run_eval(
            out_dir=Path(tmp),
            run_research=fake_research,
            check_citations=fake_citations,
            align=fake_align,
            cover=fake_cover,
        )

    from deep_research.eval.dataset import load_dataset

    n_expected = len(load_dataset())
    assert summary.n_questions == n_expected, (summary.n_questions, n_expected)
    assert abs(summary.mean_citation_existence - 0.8) < 1e-9
    assert abs(summary.mean_alignment - 0.75) < 1e-9  # (1.0 + 0.5) / 2
    assert abs(summary.mean_coverage - 0.75) < 1e-9   # 3/4 covered
    assert "agent" in summary.per_subdomain and "llm" in summary.per_subdomain
    print(f"[ok] 3. pipeline: {summary.n_questions} golden questions scored, "
          f"means cite={summary.mean_citation_existence:.0%} "
          f"align={summary.mean_alignment:.0%} cover={summary.mean_coverage:.0%}")
    # --- 4) artifacts were written ---
    with tempfile.TemporaryDirectory() as tmp:
        await run_eval(
            out_dir=Path(tmp), run_research=fake_research, check_citations=fake_citations,
            align=fake_align, cover=fake_cover,
        )
        files = list(Path(tmp).glob("eval-*"))
        assert any(f.suffix == ".json" for f in files) and any(f.suffix == ".md" for f in files)
        data = json.loads(next(f for f in files if f.suffix == ".json").read_text("utf-8"))
        assert data["scores"][0]["question_id"] == "gq-agent-memory"
    print("[ok] 4. JSON + Markdown artifacts written with per-question detail")

    print("\nEVAL SMOKE TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())
