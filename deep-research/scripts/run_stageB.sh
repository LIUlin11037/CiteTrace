set -e
PY=".venv/Scripts/python.exe"
COMMON="DR_SOURCE_MODE=offline DR_KB_TABLE=kb_chunks_qwen DR_EMBEDDING_PROVIDER=api DR_ENABLE_LONG_TERM_MEMORY=false DR_CHECKPOINT_BACKEND=postgres"
MCP='{"kb": {"transport": "stdio", "command": ".venv/Scripts/python.exe", "args": ["-m", "mcp_servers.kb_server"], "network": false}}'

$PY -c "
import asyncio
from deep_research.core.compat import use_compatible_event_loop
use_compatible_event_loop()
from deep_research.core.llm import get_chat_model
asyncio.run(get_chat_model('planner').ainvoke('ok'))
print('预检: key 可用', flush=True)
"

echo "=== exp3 剩余 10 题 $(date +%H:%M) ==="
env $COMMON DR_RERANK_PROVIDER=api DR_RERANK_CANDIDATES=50 DR_QUERY_EXPANSION=true DR_QUERY_EXPANSION_N=3 DR_MCP_SERVERS="$MCP" \
  $PY -u -c "
import asyncio
from pathlib import Path
from deep_research.core.compat import use_compatible_event_loop
use_compatible_event_loop()
from deep_research.eval.runner import run_eval
ids=[f'fb-{i:02d}' for i in range(12,22)]
asyncio.run(run_eval(dataset_path='src/deep_research/eval/golden/financebench21.json',
    out_dir=Path('reports/eval/stageB_exp3_b'), max_concurrency=1, question_ids=ids))
"
echo "=== exp3 完成 $(date +%H:%M) ==="

echo "=== exp0 全 21 题 $(date +%H:%M) ==="
env $COMMON DR_RERANK_PROVIDER=api DR_RERANK_CANDIDATES=50 DR_QUERY_EXPANSION=false DR_MCP_SERVERS="$MCP" \
  $PY -u -c "
import asyncio
from pathlib import Path
from deep_research.core.compat import use_compatible_event_loop
use_compatible_event_loop()
from deep_research.eval.runner import run_eval
asyncio.run(run_eval(dataset_path='src/deep_research/eval/golden/financebench21.json',
    out_dir=Path('reports/eval/stageB_exp0'), max_concurrency=1))
"
echo "=== exp0 完成 $(date +%H:%M) ==="
