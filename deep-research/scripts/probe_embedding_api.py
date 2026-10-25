"""探针：确认 embedding / rerank 两个 API 的真实可用形态。

为什么需要它：不同平台（DashScope / 硅基流动 / 自建兼容层）的 rerank 请求体
格式不同（``input.query`` vs ``query``、``parameters.top_n`` vs ``top_n``），
猜错就是一堆 400。与其读文档，不如发一次真请求把 schema 试出来。

用法（先配好 .env 里的 DR_EMBEDDING_API_* / DR_RERANK_API_*）：
    python scripts/probe_embedding_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deep_research.core.config import get_settings  # noqa: E402


def _mask(key: str) -> str:
    return (key[:6] + "…" + key[-4:]) if len(key) > 12 else "(空)"


def probe_embedding(s: object) -> None:
    print("\n== embedding ==")
    print(f"   base={s.embedding_api_base}")
    print(f"   model={s.embedding_api_model}  key={_mask(s.embedding_api_key)}")
    if not s.embedding_api_key:
        print("   ✗ 未配置 key，跳过")
        return
    url = s.embedding_api_base.rstrip("/") + "/embeddings"
    payload = {
        "model": s.embedding_api_model,
        "input": ["Revenue for fiscal 2022 was 23,601 million."],
        "encoding_format": "float",
    }
    try:
        r = httpx.post(
            url, json=payload, headers={"Authorization": f"Bearer {s.embedding_api_key}"}, timeout=30
        )
    except httpx.HTTPError as exc:
        print(f"   ✗ 连接失败: {type(exc).__name__}: {exc}")
        return
    print(f"   HTTP {r.status_code}")
    if r.status_code != 200:
        print(f"   响应: {r.text[:300]}")
        return
    data = r.json()["data"][0]
    dim = len(data["embedding"])
    print(f"   ✓ 维度 = {dim}（把 DR_EMBEDDING_API_DIM 设成这个值）")
    print(f"   前 3 维: {data['embedding'][:3]}")
    print(f"   用量: {r.json().get('usage')}")


def probe_rerank(s: object) -> None:
    print("\n== rerank ==")
    print(f"   base={s.rerank_api_base}  model={s.rerank_api_model}  key={_mask(s.rerank_api_key)}")
    if not (s.rerank_api_key and s.rerank_api_base and s.rerank_api_model):
        print("   ✗ 未配置完整，跳过")
        return
    url = s.rerank_api_base.rstrip("/") + "/rerank"
    query = "What is the FY2018 capital expenditure for 3M?"
    docs = [
        "Capital expenditures were $1,577 million in 2018.",
        "The company employs approximately 93,000 people worldwide.",
        "Dividends paid to shareholders totaled $3.2 billion.",
    ]
    headers = {"Authorization": f"Bearer {s.rerank_api_key}"}
    # 平台差异：DashScope 风格 vs OpenAI/Cohere 风格，逐个试
    shapes = {
        "dashscope": {
            "model": s.rerank_api_model,
            "input": {"query": query, "documents": docs},
            "parameters": {"return_documents": False, "top_n": 3},
        },
        "openai_like": {
            "model": s.rerank_api_model,
            "query": query,
            "documents": docs,
            "top_n": 3,
        },
    }
    for name, payload in shapes.items():
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=30)
        except httpx.HTTPError as exc:
            print(f"   [{name}] 连接失败: {type(exc).__name__}")
            continue
        print(f"   [{name}] HTTP {r.status_code}")
        if r.status_code == 200:
            body = r.json()
            results = body.get("results") or body.get("output", {}).get("results")
            print(f"   ✓ 这个 schema 可用；返回: {str(results)[:220]}")
            print(f"   用量: {body.get('usage')}")
            return
        print(f"      响应: {r.text[:200]}")
    print("   ✗ 两种 schema 都不通，需要按响应体调整")


def main() -> None:
    s = get_settings()
    probe_embedding(s)
    probe_rerank(s)


if __name__ == "__main__":
    main()
