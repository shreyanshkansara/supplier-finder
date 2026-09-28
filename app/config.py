from functools import lru_cache
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _with_query(url: str, scheme: str, keep: dict) -> str:
    parts = urlsplit(url)
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(keep), parts.fragment))


class Settings(BaseSettings):
    """All configuration comes from environment variables (or a local .env file)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- core ---
    app_name: str = "Sourcing Agent"
    company_name: str = "Rohida Furniture Works"
    api_key: str = "dev-key-change-me"
    log_level: str = "INFO"

    # --- database ---
    database_url: str = "postgresql+asyncpg://sourcing:sourcing@localhost:5432/sourcing"
    # "postgres" keeps paused runs across restarts; "memory" is for unit tests only.
    checkpointer: Literal["postgres", "memory"] = "postgres"

    # --- models (Google Gemini) ---
    gemini_api_key: str | None = None  # from Google AI Studio
    gemini_model: str | None = None  # e.g. a Flash / Flash-Lite model id, copied from AI Studio
    gemini_embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 1536  # must match the Vector() size in the migration
    llm_timeout_s: float = 30.0
    llm_retry_attempts: int = 4  # SDK retries on 5xx / network errors
    # Free-tier pacing (requests per minute per model). The app waits for a slot instead of
    # burning quota on 429s; on a 429 it waits as long as the API asks, up to llm_max_rate_wait_s.
    gemini_rpm: int = 14
    gemini_embed_rpm: int = 90
    llm_max_rate_wait_s: float = 90.0
    typesafe_api_key: str | None = None
    typesafe_model: str | None = None  # None -> the SDK default ("jev-latest")
    jev_batch_size: int = 25  # candidates judged per Jev request

    # --- ranking policy (see app/agent/nodes.py RankPolicy; tune with eval/tune_thresholds.py) ---
    jev_shortlist_at: float = 0.35  # tuned on the 40-request eval set (eval/results/thresholds.json)
    jev_review_at: float = 0.25
    jev_quotable_at: float = 0.5
    rank_w_similarity: float = 0.3
    rank_w_spec: float = 0.5
    rank_w_fit: float = 0.2

    # --- agent behaviour ---
    retrieval_top_k: int = 25
    max_vendors: int = 10  # most vendors listed in a result
    max_clarify_rounds: int = 3


    # --- deployment ---
    # Load and embed data/vendors.json in the background at startup if the vendors table is empty.
    # Handy on hosts without a shell (e.g. Render's free tier). Idempotent: does nothing once seeded.
    seed_on_start: bool = False

    @field_validator("database_url")
    @classmethod
    def _asyncpg_url(cls, v: str) -> str:
        """Accept the URL exactly as a host gives it (postgres:// or postgresql://, sslmode=require,
        channel_binding=...) and turn it into what SQLAlchemy's asyncpg driver expects."""
        parts = urlsplit(v)
        if parts.scheme not in ("postgres", "postgresql", "postgresql+asyncpg", "postgresql+psycopg"):
            return v
        query = dict(parse_qsl(parts.query))
        query.pop("channel_binding", None)  # libpq-only; asyncpg rejects it
        if "sslmode" in query:  # asyncpg spells it `ssl`
            query["ssl"] = query.pop("sslmode")
        return _with_query(v, "postgresql+asyncpg", query)

    @property
    def checkpointer_url(self) -> str:
        """LangGraph's Postgres saver uses psycopg (libpq), which wants postgresql:// and `sslmode`."""
        query = dict(parse_qsl(urlsplit(self.database_url).query))
        if "ssl" in query:
            query["sslmode"] = query.pop("ssl")
        return _with_query(self.database_url, "postgresql", query)


@lru_cache
def get_settings() -> Settings:
    return Settings()
