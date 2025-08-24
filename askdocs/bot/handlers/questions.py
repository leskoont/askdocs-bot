
import structlog
from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from askdocs.bot import texts
from askdocs.bot.formatting import render_answer
from askdocs.rag.llm import LLMError, LLMOutOfTokensError, LLMTimeoutError
from askdocs.rag.service import NoDocumentsError, RagService

log = structlog.get_logger(__name__)
router = Router(name="questions")

MAX_QUESTION_LENGTH = 1000


@router.message(F.text, F.chat.type == ChatType.PRIVATE, flags={"rate_limit": True})
async def on_text(message: Message, bot: Bot, rag: RagService) -> None:
    await answer_question(message, message.text.strip(), bot, rag)


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

    if not answer.found:
        await message.reply(texts.NOT_FOUND)
        return
    for part in render_answer(answer):
        await message.reply(part)
