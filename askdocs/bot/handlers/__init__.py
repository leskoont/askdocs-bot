from aiogram import Router

from askdocs.bot.handlers import common, questions


def build_router() -> Router:
    router = Router(name="root")
    # questions last: its plain-text handler catches everything that is left
    router.include_routers(common.router, questions.router)
    return router
