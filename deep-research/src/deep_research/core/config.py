"""Application settings, loaded from environment / .env.

Design notes:
- Per-role model config: cheap fast model for high-volume Researcher calls,
  stronger model for Critic/Writer. All roles default to the same provider
  but can be overridden independently via env vars.
- MCP servers are declared as JSON in env (MCP_SERVERS), keeping core
  agnostic to how servers are launched (stdio command or remote SSE).
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

RoleName = Literal["planner", "researcher", "critic", "writer"]


class ModelConfig(BaseModel):
    """OpenAI-compatible endpoint config for one role."""

    model: str
    base_url: str
    api_key: str
    temperature: float = 0.3
    max_tokens: int = 4096
    request_timeout: float = 120.0


class MCPServerConfig(BaseModel):
    """One MCP server entry.

    transport="stdio": launched locally via command/args.
    transport="sse":   remote server reachable at url.
    """

    transport: Literal["stdio", "sse"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- default provider (OpenAI-compatible: DeepSeek / Qwen / OpenAI) ----
    default_base_url: str = "https://api.deepseek.com/v1"
    default_api_key: str = ""
    default_model: str = "deepseek-chat"

    # ---- per-role overrides (fall back to default_*) ----
    planner_model: str | None = None
    researcher_model: str | None = None
    critic_model: str | None = None
    writer_model: str | None = None

    # Per-role endpoint overrides — this is what makes per-role routing
    # cross-vendor: e.g. run the token-heavy researcher on a cheap model while
    # the critic sits on a stronger model from another provider. Every endpoint
    # must speak the OpenAI chat-completions protocol.
    planner_base_url: str = ""
    planner_api_key: str = ""
    researcher_base_url: str = ""
    researcher_api_key: str = ""
    critic_base_url: str = ""
    critic_api_key: str = ""
    writer_base_url: str = ""
    writer_api_key: str = ""

    researcher_temperature: float = 0.2
    planner_temperature: float = 0.5
    critic_temperature: float = 0.0
    writer_temperature: float = 0.5

    # ---- research behaviour ----
    max_sub_questions: int = 6
    max_critique_rounds: int = 2
    max_researcher_steps: int = 8
    evidence_content_limit: int = 4000  # chars kept per evidence piece

    # ---- MCP servers: JSON object, e.g.
    # DR_MCP_SERVERS={"arxiv": {"transport": "stdio", "command": "python",
    #                          "args": ["-m", "deep_research_mcp.arxiv"]}}
    mcp_servers: str = "{}"

    # ---- persistence ----
    checkpoint_db_path: str = "checkpoints.sqlite3"
    # sqlite（开发默认，零依赖）或 postgres（M3 生产形态）
    checkpoint_backend: Literal["sqlite", "postgres"] = "sqlite"
    database_url: str = (
        "postgresql://deepresearch:deepresearch@localhost:5432/deepresearch"
    )

    # ---- long-term memory (requires postgres + pgvector) ----
    # Off by default: everything runs without PG. Flip on once PG is up.
    enable_long_term_memory: bool = False
    memory_namespace: str = "research"
    memory_embedding_model: str = "BAAI/bge-small-zh-v1.5"  # fastembed, 512d
    memory_embedding_dim: int = 512
    memory_recall_limit: int = 5
    memory_min_score: float = 0.35  # cosine similarity floor for recall

    # ---- observability (Langfuse) ----
    # Auto-detected: tracing turns on as soon as a public+secret key pair is
    # configured. Set langfuse_enabled=False to force-off, True to require it.
    langfuse_enabled: bool | None = None
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def tracing_active(self) -> bool:
        if self.langfuse_enabled is False:
            return False
        configured = bool(self.langfuse_public_key and self.langfuse_secret_key)
        if self.langfuse_enabled is True and not configured:
            raise RuntimeError(
                "DR_LANGFUSE_ENABLED=true but DR_LANGFUSE_PUBLIC_KEY / DR_LANGFUSE_SECRET_KEY missing"
            )
        return configured

    def model_config_for(self, role: RoleName) -> ModelConfig:
        override = getattr(self, f"{role}_model")
        return ModelConfig(
            model=override or self.default_model,
            base_url=getattr(self, f"{role}_base_url") or self.default_base_url,
            api_key=getattr(self, f"{role}_api_key") or self.default_api_key,
            temperature=getattr(self, f"{role}_temperature"),
        )

    def parse_mcp_servers(self) -> dict[str, MCPServerConfig]:
        raw: dict[str, Any] = json.loads(self.mcp_servers)
        return {name: MCPServerConfig(**cfg) for name, cfg in raw.items()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
