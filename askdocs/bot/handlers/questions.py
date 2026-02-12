import asyncio
from contextlib import suppress

import structlog
from aiogram import Bot, F, Router
from aiogram.enums import ChatType, ContentType
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from askdocs.bot import texts
from askdocs.bot.formatting import render_answer
from askdocs.rag.llm import LLMError, LLMOutOfTokensError, LLMTimeoutError
from askdocs.rag.service import NoDocumentsError, RagService

log = structlog.get_logger(__name__)
router = Router(name="questions")

MAX_QUESTION_LENGTH = 1000

_MEDIA = {
    ContentType.PHOTO,
    ContentType.VIDEO,
    ContentType.VIDEO_NOTE,
    ContentType.VOICE,
    ContentType.AUDIO,
    ContentType.STICKER,
    ContentType.ANIMATION,
    ContentType.CONTACT,
    ContentType.LOCATION,
    ContentType.POLL,
}


@router.message(Command("ask"), flags={"rate_limit": True})
async def cmd_ask(message: Message, command: CommandObject, bot: Bot, rag: RagService) -> None:
    question = (command.args or "").strip()
    if not question:
        await message.answer(texts.ASK_USAGE)
        return
    await answer_question(message, question, bot, rag)


@router.message(F.text.startswith("/"), F.chat.type == ChatType.PRIVATE)
async def unknown_command(message: Message) -> None:
    await message.answer(texts.UNKNOWN_COMMAND)


# In groups the bot only reacts to /ask, otherwise it would answer every message in the chat.
@router.message(F.text, F.chat.type == ChatType.PRIVATE, flags={"rate_limit": True})
async def on_text(message: Message, bot: Bot, rag: RagService) -> None:
    await answer_question(message, message.text.strip(), bot, rag)


@router.message(F.content_type.in_(_MEDIA), F.chat.type == ChatType.PRIVATE)
async def on_media(message: Message) -> None:
    await message.answer(texts.MEDIA_NOT_SUPPORTED)


async def answer_question(message: Message, question: str, bot: Bot, rag: RagService) -> None:
    if len(question) > MAX_QUESTION_LENGTH:
        await message.reply(texts.QUESTION_TOO_LONG.format(limit=MAX_QUESTION_LENGTH))
        return

    try:
        async with ChatActionSender.typing(chat_id=message.chat.id, bot=bot):
            answer = await rag.answer(message.chat.id, question)
    except NoDocumentsError:
        await message.reply(texts.NO_DOCS)
        return
    except LLMError as exc:
        log.warning("llm failed", chat_id=message.chat.id, error=str(exc))
        if isinstance(exc, LLMTimeoutError):
            await message.reply(texts.LLM_TIMEOUT)
        elif isinstance(exc, LLMOutOfTokensError):
            await message.reply(texts.LLM_OUT_OF_TOKENS)
        else:
            await message.reply(texts.LLM_FAILED)
        return
    except asyncio.CancelledError:
        with suppress(Exception):
            await message.reply(texts.RESTARTING)
        raise

    if not answer.found:
        await message.reply(texts.NOT_FOUND)
        return
    for part in render_answer(answer):
        await message.reply(part)
