import time
from contextlib import suppress
from html import escape

import aiohttp
import structlog
from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import Document, InlineKeyboardMarkup, Message

from askdocs.bot import texts
from askdocs.bot.handlers.questions import answer_question
from askdocs.config import Settings
from askdocs.rag.parsing import (
    DocumentTooLargeError,
    EmptyDocumentError,
    ParseError,
    is_supported,
)
from askdocs.rag.service import (
    DocumentLimitError,
    DuplicateDocumentError,
    RagService,
)
from askdocs.rag.store import DocumentInfo

log = structlog.get_logger(__name__)
router = Router(name="documents")

DOWNLOAD_TIMEOUT = 120
PROGRESS_INTERVAL = 3.0
LIST_NAME_LIMIT = 60


@router.message(F.document.file_name.func(is_supported), flags={"rate_limit": True})
async def on_document(message: Message, bot: Bot, rag: RagService, settings: Settings) -> None:
    document = message.document
    name = document.file_name

    if document.file_size and document.file_size > settings.max_file_size_bytes:
        await message.reply(
            texts.UPLOAD_TOO_BIG.format(
                size=f"{document.file_size / 1024 / 1024:.1f}",
                limit=f"{settings.max_file_size_mb:g}",
            )
        )
        return

    status = await message.reply(texts.UPLOAD_PROCESSING.format(name=escape(name)))
    try:
        reply, ready = await _index(message.chat.id, document, name, status, bot, rag)
    except Exception:
        log.exception("indexing failed", file_name=name, chat_id=message.chat.id)
        reply, ready = texts.ERROR, False
    try:
        await _safe_edit(status, reply, None)
    except TelegramAPIError as exc:
        log.warning("status edit failed", error=str(exc))
        with suppress(TelegramAPIError):
            await message.answer(reply)

    question = (message.caption or "").strip()
    if ready and message.chat.type == ChatType.PRIVATE and question:
        await answer_question(message, question, bot, rag)


# People share all kinds of files in groups; only complain in a private chat.
@router.message(F.document, F.chat.type == ChatType.PRIVATE, flags={"rate_limit": True})
async def on_unsupported(message: Message) -> None:
    await message.reply(texts.UPLOAD_UNSUPPORTED)


@router.message(F.migrate_to_chat_id)
async def on_migrate(message: Message, rag: RagService) -> None:
    await rag.move_chat(message.chat.id, message.migrate_to_chat_id)


async def _index(
    chat_id: int, document: Document, name: str, status: Message, bot: Bot, rag: RagService
) -> tuple[str, bool]:
    # status text, and whether the document can be asked about now
    try:
        buffer = await bot.download(document, timeout=DOWNLOAD_TIMEOUT)
        data = buffer.read()
    except (TelegramAPIError, aiohttp.ClientError, TimeoutError) as exc:
        log.warning("download failed", file_name=name, error=str(exc))
        return texts.UPLOAD_DOWNLOAD_FAILED, False

    last_update = time.monotonic()

    async def progress(done: float) -> None:
        nonlocal last_update
        if done >= 1 or time.monotonic() - last_update < PROGRESS_INTERVAL:
            return
        last_update = time.monotonic()
        text = texts.UPLOAD_PROGRESS.format(name=escape(name), percent=int(done * 100))
        try:
            await _safe_edit(status, text, None)
        except TelegramAPIError as exc:
            log.debug("progress update failed", error=str(exc))

    replacing = any(d.file_name == name for d in await rag.list_documents(chat_id))
    try:
        doc = await rag.add_document(chat_id, name, data, progress)
    except DuplicateDocumentError as exc:
        return texts.UPLOAD_DUPLICATE.format(name=escape(exc.existing.file_name)), True
    except DocumentLimitError as exc:
        return texts.UPLOAD_LIMIT.format(limit=exc.limit), False
    except DocumentTooLargeError as exc:
        if exc.unpacked:
            return texts.UPLOAD_TOO_HEAVY, False
        return texts.UPLOAD_TOO_LONG.format(limit=rag.settings.max_chunks_per_doc), False
    except EmptyDocumentError as exc:
        return (texts.UPLOAD_SCANNED if exc.scanned else texts.UPLOAD_EMPTY), False
    except ParseError as exc:
        log.info("parse failed", file_name=name, error=str(exc))
        return texts.UPLOAD_BROKEN, False
    done = texts.UPLOAD_REPLACED if replacing else texts.UPLOAD_DONE
    return done.format(name=escape(doc.file_name), chunks=doc.chunk_count), True


@router.message(Command("docs"))
async def cmd_docs(message: Message, rag: RagService) -> None:
    docs = await rag.list_documents(message.chat.id)
    if not docs:
        await message.answer(texts.NO_DOCS)
        return
    await message.answer(docs_view(docs))


def docs_view(docs: list[DocumentInfo]) -> str:
    lines = [texts.DOCS_HEADER.format(count=len(docs))]
    for number, doc in enumerate(docs, start=1):
        lines.append(
            texts.DOCS_ITEM.format(
                number=number,
                name=escape(_shorten(doc.file_name, LIST_NAME_LIMIT)),
                chunks=doc.chunk_count,
                date=doc.uploaded_at.strftime("%d.%m.%Y"),
            )
        )
    lines.append("")
    lines.append(texts.DOCS_FOOTER)
    return "\n".join(lines)


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def _safe_edit(message: Message, text: str, markup: InlineKeyboardMarkup | None) -> None:
    try:
        await message.edit_text(text, reply_markup=markup)
    except TelegramBadRequest as exc:
        if "not modified" not in str(exc):
            raise
