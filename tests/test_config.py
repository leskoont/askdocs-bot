from pathlib import Path

import pytest
from pydantic import ValidationError

from askdocs.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "BOT_TOKEN",
        "ALLOWED_USER_IDS",
        "QDRANT_URL",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "LOG_LEVEL",
        "WEBHOOK_URL",
        "WEBHOOK_PATH",
        "WEBHOOK_SECRET",
        "QDRANT_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BOT_TOKEN", "123456:test-token")


def load(**env) -> Settings:
    return Settings(_env_file=None, **env)


def test_defaults():
    settings = load()
    assert settings.bot_token.get_secret_value() == "123456:test-token"
    assert settings.allowed_user_ids == []
    assert settings.qdrant_url is None
    assert settings.embedding_model == "intfloat/multilingual-e5-small"
    assert settings.max_file_size_bytes == 10 * 1024 * 1024


def test_token_is_not_leaked_in_repr():
    assert "test-token" not in repr(load())


def test_allowed_ids_from_comma_separated_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_USER_IDS", "111, 222,333")
    assert load().allowed_user_ids == [111, 222, 333]


def test_single_and_empty_allowed_ids(monkeypatch):
    monkeypatch.setenv("ALLOWED_USER_IDS", "42")
    assert load().allowed_user_ids == [42]
    monkeypatch.setenv("ALLOWED_USER_IDS", "")
    assert load().allowed_user_ids == []


def test_empty_qdrant_url_means_embedded_mode(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "")
    assert load().qdrant_url is None


def test_empty_secrets_are_none(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", "")
    monkeypatch.setenv("QDRANT_API_KEY", " ")
    settings = load()
    assert settings.webhook_secret is None
    assert settings.qdrant_api_key is None


def test_example_env_file_is_valid():
    settings = Settings(_env_file=Path(__file__).parent.parent / ".env.example")
    assert settings.embedding_query_prefix == "query: "
    assert settings.allowed_user_ids == []
    assert settings.qdrant_url is None


def test_env_file_is_read(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_TOKEN")
    env = tmp_path / ".env"
    env.write_text("BOT_TOKEN=1:from-file\nLLM_MODEL=qwen2.5:3b\nTOP_K=3\n", encoding="utf-8")
    settings = Settings(_env_file=env)
    assert settings.bot_token.get_secret_value() == "1:from-file"
    assert settings.llm_model == "qwen2.5:3b"
    assert settings.top_k == 3


def test_missing_token(monkeypatch):
    monkeypatch.delenv("BOT_TOKEN")
    with pytest.raises(ValidationError):
        load()


@pytest.mark.parametrize(
    "env",
    [
        {"chunk_size": 400, "chunk_overlap": 300},
        {"top_k": 10, "fetch_k": 5},
        {"max_file_size_mb": 50},
        {"log_format": "xml"},
        {"log_level": "verbose"},
        {"score_margin": 1.5},
        {"max_concurrent_answers": 0},
        {"webhook_path": "telegram"},
        {"webhook_secret": "has space"},
        {"webhook_secret": "x" * 257},
        {"webhook_url": "https://example.com"},
    ],
)
def test_invalid_values(env):
    with pytest.raises(ValidationError):
        load(**env)


def test_log_level_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert load().log_level == "DEBUG"


def test_webhook_with_secret():
    settings = load(webhook_url="https://example.com", webhook_secret="abc_DEF-123")
    assert settings.webhook_secret.get_secret_value() == "abc_DEF-123"
