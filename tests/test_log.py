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


def test_level_filters_and_accepts_lowercase(capsys):
    setup_logging("warning", "console")
    log = structlog.get_logger("test")
    log.info("hidden")
    log.warning("shown")
    err = capsys.readouterr().err
    assert "shown" in err
    assert "hidden" not in err
