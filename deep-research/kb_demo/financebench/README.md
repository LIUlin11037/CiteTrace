# FinanceBench 检索基准（离线模式的外锚点）

私有语料 RAG 没有公开基准，所以用 FinanceBench 作**同构代理**：真实美国上市公司
SEC 文件（10-K / 10-Q / 8-K）+ 专家标注的 evidence（原文片段 + 页码）。

## 来源与许可

- 数据集：[patronus-ai/financebench](https://github.com/patronus-ai/financebench)
  （Patronus AI, 2023, arXiv:2311.11944）
- 开放样本：150 题 + 361 份文档元信息，**CC BY-NC 4.0（禁止商用）**
- 本仓库只提交 JSONL 元数据，**不提交 PDF**（见下）

## 目录

```
kb_demo/financebench/
├── data/                                  # 已提交（约 1MB）
│   ├── financebench_open_source.jsonl     # 150 题：question / answer / evidence(page)
│   └── financebench_document_information.jsonl
└── pdfs/                                  # 不提交，按下面步骤下载
    ├── AMD_2022_10K.pdf                   # 4.9MB / 121 页
    ├── AMERICANEXPRESS_2022_10K.pdf       # 2.4MB / 260 页
    └── BOEING_2022_10K.pdf                # 1.4MB / 190 页
```

下载（URL 来自 `data/financebench_document_information.jsonl` 的 `doc_link` 字段）：

```bash
python - <<'PY'
import json, pathlib, urllib.request
meta = pathlib.Path("kb_demo/financebench/data/financebench_document_information.jsonl")
want = {"AMD_2022_10K", "AMERICANEXPRESS_2022_10K", "BOEING_2022_10K"}
out = pathlib.Path("kb_demo/financebench/pdfs"); out.mkdir(parents=True, exist_ok=True)
for line in meta.read_text(encoding="utf-8").splitlines():
    d = json.loads(line)
    if d["doc_name"] in want:
        urllib.request.urlretrieve(d["doc_link"], out / f"{d['doc_name']}.pdf")
PY
```

## 两个会算错分的坑

1. **页码是 0-indexed**：数据集标注 `evidence_page_num=55` → PDF 第 **56** 页
   （三份 10-K 实测全部 +1）。不换算会把所有命中判成失败。
2. **切块粒度决定衡量的东西**：600 字符切片会让"命中了同一页但不是同一块"，
   `页面命中` 与 `evidence 文本命中` 会背离。PDF 默认按页切块（`split_pdf_pages`），
   只有超长页才二次切分。

## 用法

```bash
# 1) 入库（页级切块，页码作为锚点）
python scripts/ingest_kb.py kb_demo/financebench/pdfs --root . --reset

# 2) 检索层评分（零 API 成本，不调用 LLM）
python scripts/eval_financebench_retrieval.py --ks 1 5 10 20 50
```

评分口径：`页面命中@k`（检索结果里是否出现 evidence 所在页）、`同文档命中`
（是否定位到正确文件）、`evidence 文本命中`（检索片段是否包含 evidence 原文）。

## 为什么不用它做"私有化效果"的唯一证据

它是**代理**不是私有语料：README 里的结论必须写成"在 FinanceBench 开放子集上"，
并注明许可限制；真实私有语料仍需自建 10–30 题的 golden set 做校准。

实测结果（页面命中、召回上限、四组对照）见:
[`docs/检索基准-FinanceBench.md`](../../docs/检索基准-FinanceBench.md)
