import logging
import sys

import structlog


def setup_logging(level: str = "INFO", fmt: str = "console") -> None:
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    # Tracebacks never include local variables: they would carry message texts and the token.
    if fmt == "json":
        render = [
            structlog.processors.ExceptionRenderer(
                structlog.tracebacks.ExceptionDictTransformer(show_locals=False)
            ),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ]
    else:
        render = [
            structlog.dev.ConsoleRenderer(
                colors=sys.stderr.isatty(), exception_formatter=structlog.dev.plain_traceback
            )
        ]

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *render,
        ],
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())

    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "openai"):
        logging.getLogger(noisy).setLevel(max(root.level, logging.WARNING))
