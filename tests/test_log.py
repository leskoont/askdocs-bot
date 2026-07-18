import json
import logging

import pytest
import structlog

from askdocs.log import setup_logging

TOKEN = "1234567890:AAH" + "x" * 32
API_KEY = "sk-or-v1-" + "f" * 40


@pytest.fixture(autouse=True)
def restore_logging():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    structlog.reset_defaults()


def fail_with_secrets():
    token = TOKEN  # noqa: F841 - must not show up even though it is a local
    raise RuntimeError(f"POST https://api.telegram.org/bot{TOKEN}/getFile failed, key {API_KEY}")


@pytest.mark.parametrize("fmt", ["console", "json"])
def test_secrets_are_redacted(fmt, capsys):
    setup_logging("INFO", fmt, secrets=[API_KEY, "ollama"])
    log = structlog.get_logger("test")
    log.info("starting", url=f"https://api.telegram.org/bot{TOKEN}/getMe")
    logging.getLogger("aiohttp").warning("request to /bot%s/sendMessage", TOKEN)
    try:
        fail_with_secrets()
    except RuntimeError:
        log.exception("upload failed")

    err = capsys.readouterr().err
    assert TOKEN not in err
    assert TOKEN.split(":")[1] not in err
    assert API_KEY not in err
    assert "***" in err
    assert "upload failed" in err
    if fmt == "json":
        for line in err.splitlines():
            json.loads(line)


def test_short_values_are_not_treated_as_secrets(capsys):
    setup_logging("INFO", "console", secrets=["ollama"])
    structlog.get_logger("test").info("model loaded", backend="ollama")
    assert "ollama" in capsys.readouterr().err


def test_level_filters_and_accepts_lowercase(capsys):
    setup_logging("warning", "console")
    log = structlog.get_logger("test")
    log.info("hidden")
    log.warning("shown")
    err = capsys.readouterr().err
    assert "shown" in err
    assert "hidden" not in err
