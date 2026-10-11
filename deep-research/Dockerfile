FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# uv: 与本地一致的依赖管理
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 只拷贝依赖声明先做缓存层
COPY pyproject.toml README.md ./
RUN uv pip install --system -e ".[cli,serve]"

COPY src ./src
COPY mcp_servers ./mcp_servers
COPY scripts ./scripts

RUN mkdir -p reports
EXPOSE 8000
CMD ["uvicorn", "deep_research.api:app", "--host", "0.0.0.0", "--port", "8000"]
