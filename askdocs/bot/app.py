import asyncio
import contextlib
import signal

import structlog
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand, ErrorEvent
from aiohttp import web
from qdrant_client import QdrantClient

from askdocs.bot import texts
from askdocs.bot.handlers import build_router
from askdocs.bot.middlewares import (
    AccessMiddleware,
    RateLimiter,
    ThrottlingMiddleware,
)
from askdocs.config import Settings
from askdocs.rag.embeddings import SentenceTransformerEmbedder
from askdocs.rag.llm import LLMClient
from askdocs.rag.service import RagService
from askdocs.rag.store import ChunkStore, StoreConfigError

log = structlog.get_logger(__name__)

COMMANDS = [
    BotCommand(command="docs", description="Список документов"),
    BotCommand(command="ask", description="Задать вопрос"),
    BotCommand(command="clear", description="Удалить все документы"),
    BotCommand(command="help", description="Справка"),
]


def create_bot(settings: Settings) -> Bot:
    return Bot(
        token=settings.bot_token.get_secret_value(),
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML,
            link_preview_is_disabled=True,
            allow_sending_without_reply=True,
        ),
    )


def create_dispatcher(settings: Settings, rag: RagService) -> Dispatcher:
    dp = Dispatcher(settings=settings, rag=rag)

    access = AccessMiddleware(set(settings.allowed_user_ids))
    dp.message.outer_middleware(access)
    dp.callback_query.outer_middleware(access)
    limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_period)
    dp.message.middleware(ThrottlingMiddleware(limiter))

    dp.include_router(build_router())
    dp.errors.register(on_error)
    return dp


async def on_error(event: ErrorEvent) -> bool:
    log.error(
        "unhandled error",
        update_id=event.update.update_id,
        exc_info=event.exception,
    )
    try:
        if event.update.message is not None:
            await event.update.message.answer(texts.ERROR)
        elif event.update.callback_query is not None:
            await event.update.callback_query.answer(texts.ERROR, show_alert=True)
    except Exception:
        log.warning("failed to report error to user", exc_info=True)
    return True


def create_rag(
    settings: Settings, qdrant: QdrantClient, embedder: SentenceTransformerEmbedder
) -> RagService:
    store = ChunkStore(qdrant, settings.qdrant_collection, embedder.dimension)
    llm = LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
    )
    return RagService(store, embedder, llm, settings)


def create_qdrant(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        return QdrantClient(url=settings.qdrant_url, api_key=api_key)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    try:
        return QdrantClient(path=str(settings.qdrant_path))
    except RuntimeError as exc:
        raise StoreConfigError(f"{settings.qdrant_path} is used by another process") from exc


async def load_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return await SentenceTransformerEmbedder.load(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        query_prefix=settings.embedding_query_prefix,
        passage_prefix=settings.embedding_passage_prefix,
    )


async def run(settings: Settings) -> None:
    bot = create_bot(settings)
    try:
        # Check the token before spending a minute on loading the embedding model.
        me = await bot.get_me()
        log.info("authorized", bot=me.username)

        qdrant = create_qdrant(settings)
        embedder = await load_embedder(settings)
        rag = create_rag(settings, qdrant, embedder)
        await rag.store.setup(create_indexes=bool(settings.qdrant_url))

        dp = create_dispatcher(settings, rag)
        await bot.set_my_commands(COMMANDS)
        try:
            if settings.webhook_url:
                await _run_webhook(bot, dp, settings)
            else:
                await bot.delete_webhook()
                log.info("starting polling")
                await dp.start_polling(bot, close_bot_session=False)
        finally:
            await rag.llm.close()
            rag.store.close()
    finally:
        await bot.session.close()


async def _run_webhook(bot: Bot, dp: Dispatcher, settings: Settings) -> None:
    from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

    secret = settings.webhook_secret.get_secret_value()
    url = settings.webhook_url.rstrip("/") + settings.webhook_path

    async def on_startup() -> None:
        await bot.set_webhook(
            url, secret_token=secret, allowed_updates=dp.resolve_used_update_types()
        )

    dp.startup.register(on_startup)

    app = web.Application()
    # before the request handler: its shutdown hook waits for running updates
    setup_application(app, dp, bot=bot)
    SimpleRequestHandler(dispatcher=dp, bot=bot, secret_token=secret).register(
        app, path=settings.webhook_path
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, settings.webapp_host, settings.webapp_port)
    await site.start()
    log.info("webhook server started", url=url, port=settings.webapp_port)
    try:
        await stop.wait()
        log.info("stopping")
    finally:
        await runner.cleanup()
