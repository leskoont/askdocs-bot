import asyncio

from askdocs.bot.middlewares import InFlightMiddleware, RateLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_rate_limiter_allows_up_to_limit():
    clock = Clock()
    limiter = RateLimiter(3, 60, clock=clock)
    assert [limiter.hit(1) for _ in range(3)] == [0, 0, 0]
    assert limiter.hit(1) == 60


def test_rate_limiter_window_slides():
    clock = Clock()
    limiter = RateLimiter(2, 10, clock=clock)
    limiter.hit(1)
    clock.now += 4
    limiter.hit(1)
    clock.now += 3
    assert limiter.hit(1) == 3  # first hit expires in 3 seconds
    clock.now += 3
    assert limiter.hit(1) == 0


def test_rate_limiter_is_per_user():
    limiter = RateLimiter(1, 60, clock=Clock())
    assert limiter.hit(1) == 0
    assert limiter.hit(2) == 0
    assert limiter.hit(1) > 0


def test_rejected_hits_do_not_extend_the_window():
    clock = Clock()
    limiter = RateLimiter(1, 10, clock=clock)
    limiter.hit(1)
    for _ in range(5):
        clock.now += 1
        limiter.hit(1)
    clock.now += 5
    assert limiter.hit(1) == 0


async def test_drain_waits_for_finished_and_cancels_stuck_updates():
    inflight = InFlightMiddleware()
    done = []

    async def quick(event, data):
        await asyncio.sleep(0.05)
        done.append("quick")

    async def stuck(event, data):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            done.append("cancelled")
            raise

    tasks = [asyncio.create_task(inflight(h, None, {})) for h in (quick, stuck)]
    await asyncio.sleep(0)
    assert len(inflight.tasks) == 2

    await inflight.drain(0.2)
    await asyncio.gather(*tasks, return_exceptions=True)
    assert done == ["quick", "cancelled"]
    assert inflight.tasks == set()


async def test_drain_without_updates_returns_at_once():
    await asyncio.wait_for(InFlightMiddleware().drain(10), timeout=1)
