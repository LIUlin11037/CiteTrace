FROM python:3.12-slim

# HF_ENDPOINT 必须在这里定死：容器内 huggingface.co 不可达，只能用镜像站。
# 不设的话，构建期的模型预下载会直接失败（实测踩过），运行期也要靠
# compat.ensure_hf_compat() 补救，属于依赖导入顺序的隐患。
ARG HF_ENDPOINT=https://hf-mirror.com
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HUB_DISABLE_SYMLINKS=1 \
    HF_HUB_DISABLE_XET=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    HF_ENDPOINT=${HF_ENDPOINT}

WORKDIR /app

# uv: 与本地一致的依赖管理
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- 依赖层 ----
# 只依赖 pyproject.toml，业务代码放在后面：改任何 .py 都不会让这一层失效。
# 之前把源码和安装放在同一层，导致每次改代码都要重装并重下全部依赖（慢网络下十几分钟）。
# uv 下载缓存挂成 BuildKit cache mount：即使这一层确实需要重跑，wheel 也从本地缓存取，
# 不再联网重下。`-r pyproject.toml --extra ...` 只装第三方依赖，不含本项目。
# 默认走清华 PyPI 镜像（国内直连官方源要十几分钟）；海外或企业内网可覆盖：
#   UV_INDEX_URL=https://pypi.org/simple docker compose build
ARG UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --index-url "${UV_INDEX_URL}" -r pyproject.toml \
        --extra cli --extra serve --extra pg --extra observe

# ---- 代码层 ----（改代码只重建这几层，秒级）
COPY src ./src
COPY mcp_servers ./mcp_servers
COPY scripts ./scripts

# 只装本项目本身（--no-deps：依赖已在上面的层里就位）；
# 部署形态固定 Postgres（检查点 + pgvector 长期记忆），pg extra 已含 psycopg 与 fastembed。
RUN uv pip install --system --no-deps . && mkdir -p reports

# 预下载本地 embedding 模型（bge-small-zh, ~91MB）烤进镜像：
# 离线/内网部署时运行期不能联网，模型必须随镜像一起交付。
ARG EMBED_MODEL=BAAI/bge-small-zh-v1.5
RUN python -c "from fastembed import TextEmbedding as T; list(T('${EMBED_MODEL}').embed(['warmup']))" \
    && SNAP=$(ls -d /opt/fastembed/models--*/snapshots/*/ | head -1) \
    && DEST=/opt/fastembed/fast-$(basename "${EMBED_MODEL}") \
    && mkdir -p "$DEST" && cp -L "$SNAP"* "$DEST"/ \
    && echo "offline layout:" && ls "$DEST"
# ↑ fastembed 只认 <cache>/fast-<name>/ 这种离线布局；HF 的 models--... 目录在
#   local_files_only=True 下会被判"缓存不完整"（实测），断网时直接失败。

EXPOSE 8000

# 用 serve.py 而非 uvicorn CLI：它显式指定事件循环工厂，
# 在 Windows 开发机与 Linux 容器内行为一致（见 README 说明）。
CMD ["python", "-m", "deep_research.serve", "--host", "0.0.0.0", "--port", "8000"]
