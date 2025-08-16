import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_WEBHOOK_SECRET = re.compile(r"[A-Za-z0-9_-]{1,256}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bot_token: SecretStr
    allowed_user_ids: Annotated[list[int], NoDecode] = Field(default_factory=list)

    webhook_url: str | None = None
    webhook_path: str = "/telegram"
    webhook_secret: SecretStr | None = None
    webapp_host: str = "0.0.0.0"
    webapp_port: int = Field(default=8080, ge=1, le=65535)

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr = SecretStr("ollama")
    llm_model: str = "qwen2.5:7b-instruct"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=64, le=32768)
    llm_timeout: float = Field(default=90.0, gt=0)

    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: str | None = None
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    # e5 models expect these prefixes; set both to "" for models that don't (e.g. bge-m3).
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "

    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_path: Path = Path("data/qdrant")
    qdrant_collection: str = "askdocs_chunks"

    chunk_size: int = Field(default=1000, ge=200, le=2000)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, ge=1, le=20)
    fetch_k: int = Field(default=20, ge=1, le=100)
    score_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    score_margin: float = Field(default=0.06, ge=0.0, le=1.0)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    max_context_chars: int = Field(default=6000, ge=500, le=50000)

    max_file_size_mb: float = Field(default=10.0, gt=0, le=20)
    max_docs_per_chat: int = Field(default=20, ge=1, le=30)
    max_chunks_per_doc: int = Field(default=2000, ge=1)
    max_concurrent_uploads: int = Field(default=2, ge=1)
    max_concurrent_answers: int = Field(default=4, ge=1)
    rate_limit_requests: int = Field(default=10, ge=1)
    rate_limit_period: float = Field(default=60.0, gt=0)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def _split_ids(cls, value: object) -> object:
        if isinstance(value, str):
            return [part for part in value.split(",") if part.strip()]
        return value

    @field_validator(
        "webhook_url",
        "webhook_secret",
        "qdrant_url",
        "qdrant_api_key",
        "embedding_device",
        mode="before",
    )
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("webhook_path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("webhook_path must start with '/'")
        return value

    @field_validator("webhook_secret")
    @classmethod
    def _check_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not _WEBHOOK_SECRET.fullmatch(value.get_secret_value()):
            raise ValueError("webhook_secret may contain only A-Z, a-z, 0-9, _ and - (1-256 chars)")
        return value

    @model_validator(mode="after")
    def _check_ranges(self) -> "Settings":
        if self.chunk_overlap * 2 > self.chunk_size:
            raise ValueError("chunk_overlap must be at most half of chunk_size")
        if self.fetch_k < self.top_k:
            raise ValueError("fetch_k must be >= top_k")
        if self.webhook_url and self.webhook_secret is None:
            raise ValueError("webhook_secret is required when webhook_url is set")
        return self

    @property
    def max_file_size_bytes(self) -> int:
        return int(self.max_file_size_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
