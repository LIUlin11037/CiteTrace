#!/usr/bin/env bash
# 离线模式一条龙演示：建库 → 入库 → 检索 → 出报告 → 打印引用
#
# 用法：
#   ./scripts/demo_offline.sh                 # 入库 docs/ 与历史报告，跑默认题
#   ./scripts/demo_offline.sh --dry-run       # 只检查环境和切块，不写库、不花钱
#   DEMO_QID=hq-pdf-parsing ./scripts/demo_offline.sh   # 换题（库外题可演示"诚实说没有"）
#
# 设计要点：
# - 全程 DR_SOURCE_MODE=offline，只装载本地 kb 工具，没有任何联网取证；
# - 不叫 LLM 之前先把能失败的地方（PG、模型、库）全部检查一遍，
#   避免"跑了 3 分钟才告诉你数据库没起"；
# - 报告与引用直接打印，方便演示时指着说话。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QID="${DEMO_QID:-gq-react}"
DRY_RUN=0
SOURCES=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) SOURCES+=("$arg") ;;
  esac
done
if [ ${#SOURCES[@]} -eq 0 ]; then
  SOURCES=("docs" "reports/eval/reports")
fi

# ---- 解释器：优先仓库内 venv ----
if [ -x ".venv/Scripts/python.exe" ]; then PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"
else PY="python"; fi

echo "== 1/4 环境检查 =="
echo "   python: $PY"
"$PY" - <<'PY'
import sys
sys.path.insert(0, "src")
from deep_research.core.compat import use_compatible_event_loop

use_compatible_event_loop()  # 模型缓存路径 / HF 镜像 / 事件循环
from deep_research.core.config import get_settings

s = get_settings()
print(f"   信源模式: {s.source_mode}（本脚本强制 offline）")
print(f"   数据库:   {s.database_url.split('@')[-1]}")
print(f"   模型缓存: {__import__('os').environ.get('FASTEMBED_CACHE_PATH', '(默认)')}")
import psycopg

try:
    with psycopg.connect(s.database_url, connect_timeout=5) as c:
        n = c.execute("SELECT count(*) FROM checkpoints").fetchone()[0]
    print(f"   PG 连接:  OK（checkpoints {n} 行）")
except Exception as exc:  # noqa: BLE001
    print(f"   PG 连接:  失败 -> {type(exc).__name__}: {str(exc)[:80]}")
    print("   请先启动数据库：docker compose up -d postgres")
    raise SystemExit(1)
PY

echo
echo "== 2/4 入库（离线知识库）=="
if [ "$DRY_RUN" = "1" ]; then
  "$PY" scripts/ingest_kb.py "${SOURCES[@]}" --root . --dry-run
else
  "$PY" scripts/ingest_kb.py "${SOURCES[@]}" --root .
fi

if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "== dry-run 结束（未写库、未调用模型）=="
  exit 0
fi

echo
echo "== 3/4 离线跑题：$QID（无联网，仅本地库）=="
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
DR_SOURCE_MODE=offline \
DR_ENABLE_LONG_TERM_MEMORY=false \
DR_MCP_SERVERS='{"kb": {"transport": "stdio", "command": "python", "args": ["-m", "mcp_servers.kb_server"], "network": false}}' \
"$PY" -u - <<PY
import asyncio
import logging
from pathlib import Path

from deep_research.core.compat import use_compatible_event_loop

use_compatible_event_loop()
# 演示时不想看到 MCP 的每次请求日志
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
from deep_research.eval.runner import run_eval

asyncio.run(
    run_eval(question_ids=["$QID"], out_dir=Path("reports/eval/demo"), max_concurrency=1)
)
PY

echo
echo "== 4/4 产物 =="
REPORT="reports/eval/demo/reports/$QID.md"
if [ -f "$REPORT" ]; then
  echo "   报告: $REPORT（$(wc -c < "$REPORT") 字节）"
  echo "   前 3 条引用（全部指向本地文件 + 章节）："
  grep -E '^\[\^[0-9]+\]:' "$REPORT" | head -3 | cut -c1-130 | sed 's/^/     /'
  echo
  echo "   正文里的不确定性声明："
  grep -nE '知识库未收录|库内无有效证据|不足以确认' "$REPORT" | head -3 | cut -c1-110 | sed 's/^/     /' || true
else
  echo "   未找到报告（运行可能失败，见上方日志）"
  exit 1
fi
