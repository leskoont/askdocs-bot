from aiogram import Router
from aiogram.enums import ChatType
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from askdocs.bot import texts
from askdocs.config import Settings

router = Router(name="common")


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        texts.START if message.chat.type == ChatType.PRIVATE else texts.START_GROUP
    )


@router.message(Command("help"))
async def cmd_help(message: Message, settings: Settings) -> None:
    await message.answer(
        texts.HELP.format(
            max_size=f"{settings.max_file_size_mb:g}", max_docs=settings.max_docs_per_chat
        )
    )
