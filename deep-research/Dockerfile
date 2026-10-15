FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HUB_DISABLE_SYMLINKS=1 \
    HF_HUB_DISABLE_XET=1

WORKDIR /app

# uv: 与本地一致的依赖管理
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- 依赖层 ----
# 只依赖 pyproject.toml，业务代码放在后面：改任何 .py 都不会让这一层失效。
# 之前把源码和安装放在同一层，导致每次改代码都要重装并重下全部依赖（慢网络下十几分钟）。
# uv 下载缓存挂成 BuildKit cache mount：即使这一层确实需要重跑，wheel 也从本地缓存取，
# 不再联网重下。`-r pyproject.toml --extra ...` 只装第三方依赖，不含本项目。
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r pyproject.toml \
        --extra cli --extra serve --extra pg --extra observe

# ---- 代码层 ----（改代码只重建这几层，秒级）
COPY src ./src
COPY mcp_servers ./mcp_servers
COPY scripts ./scripts

# 只装本项目本身（--no-deps：依赖已在上面的层里就位）；
# 部署形态固定 Postgres（检查点 + pgvector 长期记忆），pg extra 已含 psycopg 与 fastembed。
RUN uv pip install --system --no-deps . && mkdir -p reports

EXPOSE 8000

# 用 serve.py 而非 uvicorn CLI：它显式指定事件循环工厂，
# 在 Windows 开发机与 Linux 容器内行为一致（见 README 说明）。
CMD ["python", "-m", "deep_research.serve", "--host", "0.0.0.0", "--port", "8000"]
