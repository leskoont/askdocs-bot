import time
from contextlib import suppress
from html import escape

import aiohttp
import structlog
from aiogram import Bot, F, Router
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import CallbackQuery, Chat, Document, InlineKeyboardMarkup, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

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
    ChatBusyError,
    DocumentLimitError,
    DuplicateDocumentError,
    RagService,
)
from askdocs.rag.store import DocumentInfo

log = structlog.get_logger(__name__)
router = Router(name="documents")

DOWNLOAD_TIMEOUT = 120
PROGRESS_INTERVAL = 3.0
# Telegram cuts callback popups at 200 characters.
POPUP_NAME_LIMIT = 150
LIST_NAME_LIMIT = 60


class DeleteDoc(CallbackData, prefix="del"):
    doc_id: str


class ClearDocs(CallbackData, prefix="clear"):
    confirm: bool


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
    text, markup = docs_view(docs)
    await message.answer(text, reply_markup=markup)


@router.callback_query(DeleteDoc.filter())
async def on_delete(
    callback: CallbackQuery, callback_data: DeleteDoc, bot: Bot, rag: RagService
) -> None:
    if not isinstance(callback.message, Message):
        await callback.answer()
        return
    chat = callback.message.chat
    if not await _is_admin(bot, chat, callback.from_user.id):
        await callback.answer(texts.ADMIN_ONLY, show_alert=True)
        return

    docs = await rag.list_documents(chat.id)
    doc = next((d for d in docs if d.doc_id == callback_data.doc_id), None)
    try:
        deleted = doc is not None and await rag.delete_document(chat.id, doc.doc_id)
    except ChatBusyError:
        await callback.answer(texts.BUSY, show_alert=True)
        return
    if not deleted:
        await callback.answer(texts.DELETE_NOT_FOUND)
    else:
        await callback.answer(texts.DELETED.format(name=_shorten(doc.file_name, POPUP_NAME_LIMIT)))

    remaining = [d for d in docs if d.doc_id != callback_data.doc_id]
    if remaining:
        text, markup = docs_view(remaining)
        await _safe_edit(callback.message, text, markup)
    else:
        await _safe_edit(callback.message, texts.NO_DOCS, None)


@router.message(Command("clear"))
async def cmd_clear(message: Message, bot: Bot, rag: RagService) -> None:
    # An anonymous group admin writes on behalf of the group itself.
    anonymous_admin = message.sender_chat is not None and message.sender_chat.id == message.chat.id
    user_id = message.from_user.id if message.from_user else 0
    if not anonymous_admin and not await _is_admin(bot, message.chat, user_id):
        await message.answer(texts.ADMIN_ONLY)
        return

    docs = await rag.list_documents(message.chat.id)
    if not docs:
        await message.answer(texts.NO_DOCS)
        return
    builder = InlineKeyboardBuilder()
    builder.button(text=texts.CLEAR_YES, callback_data=ClearDocs(confirm=True))
    builder.button(text=texts.CLEAR_NO, callback_data=ClearDocs(confirm=False))
    await message.answer(
        texts.CLEAR_CONFIRM.format(count=len(docs)), reply_markup=builder.as_markup()
    )


@router.callback_query(ClearDocs.filter())
async def on_clear(
    callback: CallbackQuery, callback_data: ClearDocs, bot: Bot, rag: RagService
) -> None:
    if not isinstance(callback.message, Message):
        await callback.answer()
        return
    chat = callback.message.chat
    if not await _is_admin(bot, chat, callback.from_user.id):
        await callback.answer(texts.ADMIN_ONLY, show_alert=True)
        return
    if not callback_data.confirm:
        await callback.answer()
        await _safe_edit(callback.message, texts.CLEAR_CANCELLED, None)
        return
    if rag.is_busy(chat.id):
        await callback.answer(texts.BUSY, show_alert=True)
        return

    # A big chat takes longer to clear than Telegram waits for the callback answer.
    await callback.answer()
    try:
        await _safe_edit(callback.message, texts.CLEAR_PROGRESS, None)
    except TelegramAPIError as exc:
        log.debug("clear progress update failed", error=str(exc))
    try:
        text = texts.CLEAR_DONE.format(count=await rag.clear(chat.id))
    except ChatBusyError:
        text = texts.BUSY
    except Exception:
        log.exception("clear failed", chat_id=chat.id)
        text = texts.ERROR
    try:
        await _safe_edit(callback.message, text, None)
    except TelegramAPIError as exc:
        log.warning("clear status edit failed", error=str(exc))
        with suppress(TelegramAPIError):
            await callback.message.answer(text)


def docs_view(docs: list[DocumentInfo]) -> tuple[str, InlineKeyboardMarkup]:
    lines = [texts.DOCS_HEADER.format(count=len(docs))]
    builder = InlineKeyboardBuilder()
    for number, doc in enumerate(docs, start=1):
        lines.append(
            texts.DOCS_ITEM.format(
                number=number,
                name=escape(_shorten(doc.file_name, LIST_NAME_LIMIT)),
                chunks=doc.chunk_count,
                date=doc.uploaded_at.strftime("%d.%m.%Y"),
            )
        )
        builder.button(
            text=texts.DELETE_BUTTON.format(number=number, name=_shorten(doc.file_name, 40)),
            callback_data=DeleteDoc(doc_id=doc.doc_id),
        )
    builder.adjust(1)
    lines.append("")
    lines.append(texts.DOCS_FOOTER)
    return "\n".join(lines), builder.as_markup()


async def _is_admin(bot: Bot, chat: Chat, user_id: int) -> bool:
    if chat.type == ChatType.PRIVATE:
        return True
    member = await bot.get_chat_member(chat.id, user_id)
    return member.status in (ChatMemberStatus.CREATOR, ChatMemberStatus.ADMINISTRATOR)


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def _safe_edit(message: Message, text: str, markup: InlineKeyboardMarkup | None) -> None:
    try:
        await message.edit_text(text, reply_markup=markup)
    except TelegramBadRequest as exc:
        if "not modified" not in str(exc):
            raise
