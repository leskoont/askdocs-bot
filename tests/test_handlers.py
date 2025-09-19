import itertools
from datetime import UTC, datetime

import aiohttp
import pytest
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.methods import (
    AnswerCallbackQuery,
    EditMessageText,
    GetChatMember,
    GetFile,
    SendMessage,
)
from aiogram.types import CallbackQuery, Chat, Document, Message, PhotoSize, Update, User
from helpers import FakeLLM, MockedSession, make_pdf

from askdocs.bot import texts
from askdocs.bot.app import create_dispatcher
from askdocs.bot.handlers import common, documents, questions
from askdocs.bot.handlers.documents import ClearDocs, DeleteDoc
from askdocs.config import Settings
from askdocs.rag import service
from askdocs.rag.llm import LLMError, LLMOutOfTokensError, LLMTimeoutError
from askdocs.rag.service import ChatBusyError
from askdocs.rag.store import DocumentInfo

USER_ID = 10
GROUP = {"chat_type": "supergroup", "chat_id": -100}
CONTRACT = (
    "# Оплата\n\nЗаказчик оплачивает услуги в течение десяти рабочих дней после "
    "подписания акта.\n\n# Ответственность\n\nЗа просрочку оплаты начисляется пеня "
    "0,1 процента за каждый день.\n"
).encode()

_ids = itertools.count(1)


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, bot_token="42:TEST", **overrides)


@pytest.fixture
def session() -> MockedSession:
    return MockedSession()


@pytest.fixture
def bot(session) -> Bot:
    return Bot("42:TEST", session=session, default=DefaultBotProperties(parse_mode=ParseMode.HTML))


@pytest.fixture
def make_dp(rag):
    def factory(**overrides):
        return create_dispatcher(make_settings(**overrides), rag)

    yield factory
    # Handler routers are module-level singletons; detach them so the next test
    # can include them into a fresh dispatcher.
    for router in (common.router, documents.router, questions.router):
        router._parent_router = None


@pytest.fixture
def dp(make_dp):
    return make_dp()


def user(user_id: int = USER_ID) -> User:
    return User(id=user_id, is_bot=False, first_name="Иван")


def message_update(
    text: str | None = None,
    *,
    document: Document | None = None,
    caption: str | None = None,
    chat_type: str = "private",
    chat_id: int = USER_ID,
    user_id: int = USER_ID,
    **fields,
) -> Update:
    n = next(_ids)
    return Update(
        update_id=n,
        message=Message(
            message_id=n,
            date=datetime.now(UTC),
            chat=Chat(id=chat_id, type=chat_type),
            from_user=user(user_id),
            text=text,
            document=document,
            caption=caption,
            **fields,
        ),
    )


def callback_update(
    data: str, *, chat_type: str = "private", chat_id: int = USER_ID, user_id: int = USER_ID
) -> Update:
    n = next(_ids)
    return Update(
        update_id=n,
        callback_query=CallbackQuery(
            id=str(n),
            from_user=user(user_id),
            chat_instance="ci",
            data=data,
            message=Message(
                message_id=500,
                date=datetime.now(UTC),
                chat=Chat(id=chat_id, type=chat_type),
                text="list",
            ),
        ),
    )


def document(name: str, data: bytes, session: MockedSession) -> Document:
    file_id = f"file{next(_ids)}"
    session.files[file_id] = data
    return Document(file_id=file_id, file_unique_id=file_id, file_name=name, file_size=len(data))


async def upload(dp, bot, session, name: str, data: bytes, **kwargs) -> None:
    await dp.feed_update(bot, message_update(document=document(name, data, session), **kwargs))


def popups(session: MockedSession) -> list[AnswerCallbackQuery]:
    return [r for r in session.requests if isinstance(r, AnswerCallbackQuery)]


async def test_start(dp, bot, session):
    await dp.feed_update(bot, message_update("/start"))
    await dp.feed_update(bot, message_update("/start", **GROUP))
    assert session.texts() == [texts.START, texts.START_GROUP]


async def test_help_shows_limits(make_dp, bot, session):
    dp = make_dp(max_file_size_mb=5, max_docs_per_chat=7)
    await dp.feed_update(bot, message_update("/help"))
    reply = session.texts()[0]
    assert "до 5 МБ" in reply
    assert "может быть до 7" in reply


async def test_upload_and_ask(dp, bot, session, llm: FakeLLM):
    await upload(dp, bot, session, "договор.md", CONTRACT)
    assert session.texts() == [
        texts.UPLOAD_PROCESSING.format(name="договор.md"),
        texts.UPLOAD_DONE.format(name="договор.md", chunks=2),
    ]

    llm.reply = "Пеня — 0,1 процента в день [1]."
    await dp.feed_update(bot, message_update("Какая пеня за просрочку?"))
    assert session.texts()[-1] == (
        "Пеня — 0,1 процента в день [1].\n\n"
        "<b>Источники:</b>\n[1] договор.md, раздел «Ответственность»"
    )


async def test_replies_are_threaded(dp, bot, session, llm):
    # In a busy group an answer without a reply is hard to match with its question.
    await upload(dp, bot, session, "a.md", CONTRACT, **GROUP)
    await dp.feed_update(bot, message_update("/ask Когда заказчик оплачивает услуги?", **GROUP))
    sent = [r for r in session.requests if isinstance(r, SendMessage)]
    assert len(sent) == 2
    assert all(r.reply_parameters is not None for r in sent)


async def test_caption_is_a_question(dp, bot, session, llm):
    llm.reply = "В течение десяти рабочих дней [1]."
    doc = document("a.md", CONTRACT, session)
    await dp.feed_update(
        bot, message_update(document=doc, caption="Когда заказчик оплачивает услуги?")
    )
    assert session.texts()[-2] == texts.UPLOAD_DONE.format(name="a.md", chunks=2)
    assert session.texts()[-1].startswith("В течение десяти рабочих дней [1].")
    assert len(llm.calls) == 1


async def test_caption_after_duplicate_still_answers(dp, bot, session, llm):
    await upload(dp, bot, session, "a.md", CONTRACT)
    doc = document("a.md", CONTRACT, session)
    await dp.feed_update(
        bot, message_update(document=doc, caption="Когда заказчик оплачивает услуги?")
    )
    assert texts.UPLOAD_DUPLICATE.format(name="a.md") in session.texts()
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    "error",
    [
        TelegramRetryAfter(EditMessageText(text="x"), "Too Many Requests", 30),
        TelegramBadRequest(EditMessageText(text="x"), "message to edit not found"),
    ],
    ids=["flood", "deleted"],
)
async def test_status_edit_failure_falls_back_to_new_message(dp, bot, session, llm, error):
    make_request = session.make_request

    async def no_edits(bot, method, **kwargs):
        if isinstance(method, EditMessageText):
            raise error
        return await make_request(bot, method, **kwargs)

    session.make_request = no_edits
    llm.reply = "В течение десяти рабочих дней [1]."
    doc = document("a.md", CONTRACT, session)
    await dp.feed_update(
        bot, message_update(document=doc, caption="Когда заказчик оплачивает услуги?")
    )
    sent = [r.text for r in session.requests if isinstance(r, SendMessage)]
    assert sent[1] == texts.UPLOAD_DONE.format(name="a.md", chunks=2)
    assert sent[2].startswith("В течение десяти рабочих дней [1].")


async def test_caption_is_ignored_in_groups(dp, bot, session, llm):
    doc = document("a.md", CONTRACT, session)
    await dp.feed_update(
        bot, message_update(document=doc, caption="Когда заказчик оплачивает услуги?", **GROUP)
    )
    assert llm.calls == []


async def test_file_name_is_escaped(dp, bot, session):
    await upload(dp, bot, session, "R&D <v2>.txt", "Текст о бюджете на исследования.".encode())
    assert session.texts()[-1] == texts.UPLOAD_DONE.format(name="R&amp;D &lt;v2&gt;.txt", chunks=1)


async def test_unsupported_file_is_not_downloaded(dp, bot, session):
    await upload(dp, bot, session, "table.xlsx", b"PK\x03\x04")
    assert session.texts() == [texts.UPLOAD_UNSUPPORTED]
    assert not any(isinstance(r, GetFile) for r in session.requests)


async def test_new_version_of_a_file(dp, bot, session, rag):
    await upload(dp, bot, session, "a.md", CONTRACT)
    await upload(dp, bot, session, "a.md", CONTRACT + "Приложение 1.\n".encode())
    assert session.texts()[-1] == texts.UPLOAD_REPLACED.format(name="a.md", chunks=2)
    assert len(await rag.list_documents(USER_ID)) == 1


async def test_too_big_file(make_dp, bot, session):
    dp = make_dp(max_file_size_mb=1)
    doc = Document(file_id="big", file_unique_id="big", file_name="a.pdf", file_size=3 * 1024**2)
    await dp.feed_update(bot, message_update(document=doc))
    assert session.texts() == [texts.UPLOAD_TOO_BIG.format(size="3.0", limit="1")]
    assert not any(isinstance(r, GetFile) for r in session.requests)


async def test_scanned_pdf(dp, bot, session):
    await upload(dp, bot, session, "scan.pdf", make_pdf([[], []]))
    assert session.texts()[-1] == texts.UPLOAD_SCANNED


async def test_broken_docx(dp, bot, session):
    await upload(dp, bot, session, "broken.docx", b"not a zip")
    assert session.texts()[-1] == texts.UPLOAD_BROKEN


async def test_duplicate_upload(dp, bot, session):
    await upload(dp, bot, session, "a.md", CONTRACT)
    await upload(dp, bot, session, "b.md", CONTRACT)
    assert session.texts()[-1] == texts.UPLOAD_DUPLICATE.format(name="a.md")


async def test_too_long_document(make_dp, bot, session, rag):
    dp = make_dp()
    text = "\n\n".join(f"Абзац {i}. " + "текст " * 50 for i in range(80))
    await upload(dp, bot, session, "big.txt", text.encode())
    assert session.texts()[-1] == texts.UPLOAD_TOO_LONG.format(limit=50)


async def test_download_failure(dp, bot, session, monkeypatch):
    async def broken(*args, **kwargs):
        raise aiohttp.ClientPayloadError("connection reset")
        yield b""

    monkeypatch.setattr(session, "stream_content", broken)
    await upload(dp, bot, session, "a.md", CONTRACT)
    assert session.texts()[-1] == texts.UPLOAD_DOWNLOAD_FAILED


async def test_unexpected_indexing_error_updates_status(dp, bot, session, rag, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("store is down")

    monkeypatch.setattr(rag, "add_document", boom)
    await upload(dp, bot, session, "a.md", CONTRACT)
    # the "processing" status is edited instead of being left hanging
    assert session.texts() == [texts.UPLOAD_PROCESSING.format(name="a.md"), texts.ERROR]
    assert isinstance(session.requests[-1], EditMessageText)


async def test_upload_progress(dp, bot, session, monkeypatch):
    monkeypatch.setattr(documents, "PROGRESS_INTERVAL", 0)
    monkeypatch.setattr(service, "_EMBED_BATCH", 1)
    await upload(dp, bot, session, "a.md", CONTRACT)
    assert session.texts() == [
        texts.UPLOAD_PROCESSING.format(name="a.md"),
        texts.UPLOAD_PROGRESS.format(name="a.md", percent=50),
        texts.UPLOAD_DONE.format(name="a.md", chunks=2),
    ]


async def test_question_without_documents(dp, bot, session, llm):
    await dp.feed_update(bot, message_update("Что в договоре?"))
    assert session.texts() == [texts.NO_DOCS]
    assert llm.calls == []


async def test_question_not_covered_by_documents(dp, bot, session, llm):
    await upload(dp, bot, session, "a.md", CONTRACT)
    await dp.feed_update(bot, message_update("рецепт борща"))
    assert session.texts()[-1] == texts.NOT_FOUND
    assert llm.calls == []


@pytest.mark.parametrize(
    ("error", "reply"),
    [
        (LLMError("down"), texts.LLM_FAILED),
        (LLMTimeoutError("slow"), texts.LLM_TIMEOUT),
        (LLMOutOfTokensError("reasoning"), texts.LLM_OUT_OF_TOKENS),
    ],
    ids=["failed", "timeout", "out-of-tokens"],
)
async def test_llm_failures(dp, bot, session, llm, error, reply):
    await upload(dp, bot, session, "a.md", CONTRACT)
    llm.error = error
    await dp.feed_update(bot, message_update("Когда заказчик оплачивает услуги?"))
    assert session.texts()[-1] == reply


async def test_long_answer_is_split(dp, bot, session, llm):
    await upload(dp, bot, session, "a.md", CONTRACT)
    llm.reply = "\n\n".join(f"Пункт {i} [1]. " + "текст " * 100 for i in range(20))
    await dp.feed_update(bot, message_update("Когда заказчик оплачивает услуги?"))
    answers = [r.text for r in session.requests if isinstance(r, SendMessage)][1:]
    assert len(answers) > 1
    assert all(len(t) <= 4096 for t in answers)
    assert "<b>Источники:</b>" in answers[-1]


async def test_ask_command_in_group(dp, bot, session, llm):
    await upload(dp, bot, session, "a.md", CONTRACT, **GROUP)
    llm.reply = "В течение десяти рабочих дней [1]."

    await dp.feed_update(bot, message_update("Когда оплата?", **GROUP))
    assert len(llm.calls) == 0, "plain text in a group must be ignored"

    await dp.feed_update(bot, message_update("/ask Когда заказчик оплачивает услуги?", **GROUP))
    assert session.texts()[-1].startswith("В течение десяти рабочих дней [1].")


async def test_ask_without_question(dp, bot, session):
    await dp.feed_update(bot, message_update("/ask"))
    assert session.texts() == [texts.ASK_USAGE]


async def test_unknown_command(dp, bot, session, llm):
    await dp.feed_update(bot, message_update("/foo"))
    assert session.texts() == [texts.UNKNOWN_COMMAND]
    assert llm.calls == []


async def test_group_ignores_other_commands_and_media(dp, bot, session, llm):
    photo = [PhotoSize(file_id="p", file_unique_id="p", width=10, height=10)]
    await dp.feed_update(bot, message_update("/foo@other_bot", **GROUP))
    await dp.feed_update(bot, message_update(photo=photo, **GROUP))
    assert session.texts() == []


async def test_docs_list_and_delete_button(dp, bot, session, rag):
    await upload(dp, bot, session, "a.md", CONTRACT)
    await upload(dp, bot, session, "b.txt", "Второй документ про доставку.".encode())

    await dp.feed_update(bot, message_update("/docs"))
    listing = session.requests[-1]
    assert isinstance(listing, SendMessage)
    assert "1. a.md — фрагментов: 2" in listing.text
    assert "2. b.txt — фрагментов: 1" in listing.text
    buttons = [row[0] for row in listing.reply_markup.inline_keyboard]
    assert [b.text for b in buttons] == ["Удалить 1. a.md", "Удалить 2. b.txt"]

    await dp.feed_update(bot, callback_update(buttons[0].callback_data))
    assert popups(session)[0].text == texts.DELETED.format(name="a.md")
    edited = session.requests[-1]
    assert isinstance(edited, EditMessageText)
    assert "1. b.txt" in edited.text
    assert [d.file_name for d in await rag.list_documents(USER_ID)] == ["b.txt"]


async def test_delete_popup_fits_telegram_limit(dp, bot, session):
    name = "очень длинное имя файла " * 20 + ".txt"
    await upload(dp, bot, session, name, "Текст про доставку.".encode())
    await dp.feed_update(bot, message_update("/docs"))
    button = session.requests[-1].reply_markup.inline_keyboard[0][0]
    await dp.feed_update(bot, callback_update(button.callback_data))
    assert len(popups(session)[0].text) <= 200


def test_docs_list_fits_one_message():
    docs = [DocumentInfo(f"d{i}", "имя" * 85, 50, "0" * 64, datetime.now(UTC)) for i in range(30)]
    text, markup = documents.docs_view(docs)
    assert len(text) <= 4096
    assert len(markup.inline_keyboard) == 30


@pytest.mark.parametrize("button", ["delete", "clear"])
async def test_buttons_during_upload(dp, bot, session, rag, monkeypatch, button):
    async def busy(*args):
        raise ChatBusyError

    await upload(dp, bot, session, "a.md", CONTRACT)
    [doc] = await rag.list_documents(USER_ID)
    monkeypatch.setattr(rag, "delete_document", busy)
    monkeypatch.setattr(rag, "clear", busy)
    data = DeleteDoc(doc_id=doc.doc_id) if button == "delete" else ClearDocs(confirm=True)
    await dp.feed_update(bot, callback_update(data.pack()))
    assert [(p.text, p.show_alert) for p in popups(session)] == [(texts.BUSY, True)]
    assert len(await rag.list_documents(USER_ID)) == 1


async def test_delete_button_for_missing_document(dp, bot, session):
    await dp.feed_update(bot, callback_update(DeleteDoc(doc_id="gone").pack()))
    assert popups(session)[0].text == texts.DELETE_NOT_FOUND


async def test_clear_with_confirmation(dp, bot, session, rag):
    await upload(dp, bot, session, "a.md", CONTRACT)
    await dp.feed_update(bot, message_update("/clear"))
    assert session.texts()[-1] == texts.CLEAR_CONFIRM.format(count=1)

    await dp.feed_update(bot, callback_update(ClearDocs(confirm=False).pack()))
    assert session.texts()[-1] == texts.CLEAR_CANCELLED
    assert len(await rag.list_documents(USER_ID)) == 1

    await dp.feed_update(bot, callback_update(ClearDocs(confirm=True).pack()))
    assert session.texts()[-1] == texts.CLEAR_DONE.format(count=1)
    assert await rag.list_documents(USER_ID) == []


async def test_group_members_cannot_delete(dp, bot, session, rag):
    await upload(dp, bot, session, "a.md", CONTRACT, **GROUP)
    [doc] = await rag.list_documents(-100)

    await dp.feed_update(bot, message_update("/clear", **GROUP))
    assert session.texts()[-1] == texts.ADMIN_ONLY

    await dp.feed_update(bot, callback_update(ClearDocs(confirm=True).pack(), **GROUP))
    await dp.feed_update(bot, callback_update(DeleteDoc(doc_id=doc.doc_id).pack(), **GROUP))
    assert [(p.text, p.show_alert) for p in popups(session)] == [(texts.ADMIN_ONLY, True)] * 2
    assert len(await rag.list_documents(-100)) == 1


async def test_group_admin_can_clear(dp, bot, session, rag):
    session.admins.add(USER_ID)
    await upload(dp, bot, session, "a.md", CONTRACT, **GROUP)
    await dp.feed_update(bot, message_update("/clear", **GROUP))
    assert session.texts()[-1] == texts.CLEAR_CONFIRM.format(count=1)
    await dp.feed_update(bot, callback_update(ClearDocs(confirm=True).pack(), **GROUP))
    assert await rag.list_documents(-100) == []


async def test_anonymous_group_admin_can_clear(dp, bot, session):
    await upload(dp, bot, session, "a.md", CONTRACT, **GROUP)
    group = Chat(id=-100, type="supergroup")
    await dp.feed_update(bot, message_update("/clear", sender_chat=group, **GROUP))
    assert session.texts()[-1] == texts.CLEAR_CONFIRM.format(count=1)
    assert not any(isinstance(r, GetChatMember) for r in session.requests)


async def test_media_is_rejected(dp, bot, session):
    photo = [PhotoSize(file_id="p", file_unique_id="p", width=10, height=10)]
    await dp.feed_update(bot, message_update(photo=photo))
    assert session.texts() == [texts.MEDIA_NOT_SUPPORTED]


async def test_unexpected_error_is_reported(dp, bot, session, rag, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("store is down")

    monkeypatch.setattr(rag, "list_documents", boom)
    await dp.feed_update(bot, message_update("/docs"))
    assert session.texts() == [texts.ERROR]

    await dp.feed_update(bot, callback_update(DeleteDoc(doc_id="x").pack()))
    assert [(p.text, p.show_alert) for p in popups(session)] == [(texts.ERROR, True)]


async def test_group_upgraded_to_supergroup(dp, bot, session, rag):
    await upload(dp, bot, session, "a.md", CONTRACT, chat_type="group", chat_id=-5)
    await dp.feed_update(
        bot, message_update(chat_type="group", chat_id=-5, migrate_to_chat_id=-1005)
    )
    assert await rag.list_documents(-5) == []
    assert [d.file_name for d in await rag.list_documents(-1005)] == ["a.md"]
