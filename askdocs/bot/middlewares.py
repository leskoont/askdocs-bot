import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from aiogram import BaseMiddleware
from aiogram.dispatcher.flags import get_flag
from aiogram.types import CallbackQuery, Message, TelegramObject, User

from askdocs.bot import texts

log = structlog.get_logger(__name__)

Handler = Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]]


class AccessMiddleware(BaseMiddleware):
    def __init__(self, allowed_ids: set[int]):
        self.allowed_ids = allowed_ids

    async def __call__(self, handler: Handler, event: TelegramObject, data: dict[str, Any]) -> Any:
        user: User | None = data.get("event_from_user")
        if not self.allowed_ids or (user is not None and user.id in self.allowed_ids):
            return await handler(event, data)

        log.info("access denied", user_id=user.id if user else None)
        if isinstance(event, Message) and user is not None and event.chat.type == "private":
            await event.answer(texts.ACCESS_DENIED.format(user_id=user.id))
        elif isinstance(event, CallbackQuery):
            await event.answer(texts.ACCESS_DENIED_SHORT, show_alert=True)
        return None


class RateLimiter:
    def __init__(self, limit: int, period: float, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.period = period
        self._clock = clock
        self._hits: dict[int, deque[float]] = {}

    def hit(self, key: int) -> float:
        """Register a hit. Returns 0 if allowed, otherwise seconds until the next free slot."""
        now = self._clock()
        window = self._hits.setdefault(key, deque())
        while window and now - window[0] >= self.period:
            window.popleft()
        if len(window) >= self.limit:
            return self.period - (now - window[0])
        window.append(now)
        return 0.0


class ThrottlingMiddleware(BaseMiddleware):
    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter

    async def __call__(self, handler: Handler, event: TelegramObject, data: dict[str, Any]) -> Any:
        user: User | None = data.get("event_from_user")
        if user is None or not get_flag(data, "rate_limit"):
            return await handler(event, data)

        wait = self.limiter.hit(user.id)
        if wait <= 0:
            return await handler(event, data)

        log.info("rate limited", user_id=user.id, wait=round(wait, 1))
        if isinstance(event, Message) and event.chat.type == "private":
            await event.answer(texts.RATE_LIMITED.format(seconds=math.ceil(wait)))
        return None
