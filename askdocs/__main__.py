import asyncio
import sys

import structlog
from aiogram.exceptions import TelegramUnauthorizedError
from aiogram.utils.token import TokenValidationError
from pydantic import ValidationError

from askdocs.bot.app import run
from askdocs.config import get_settings
from askdocs.log import setup_logging
from askdocs.rag.store import StoreConfigError


def main() -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration error:", file=sys.stderr)
        for error in exc.errors(include_input=False, include_url=False):
            field = ".".join(str(part) for part in error["loc"]).upper() or "settings"
            print(f"  {field}: {error['msg']}", file=sys.stderr)
        return 2

    setup_logging(settings.log_level, settings.log_format)
    log = structlog.get_logger("askdocs")

    try:
        asyncio.run(run(settings))
    except (TokenValidationError, TelegramUnauthorizedError) as exc:
        log.error("telegram rejected the bot token, check BOT_TOKEN", error=str(exc))
        return 1
    except StoreConfigError as exc:
        log.error("cannot open vector store", error=str(exc))
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
