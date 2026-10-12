"""The global LLM budget must hold under concurrency, against a real Postgres."""

import asyncio
import os
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from grounded.db import repositories
from grounded.db.models import LLMDailyUsage

DAY = date(2026, 1, 15)


async def _reserve_concurrently(attempts: int, limit: int, day: date = DAY) -> list[bool]:
    """Fire `attempts` reservations at once, each in its own connection and transaction."""
    engine = create_async_engine(os.environ["DATABASE_URL"], pool_size=attempts, max_overflow=0)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    start = asyncio.Event()

    async def attempt() -> bool:
        async with factory() as session:
            await session.connection()  # open the connection first, so all start together
            await start.wait()
            allowed = await repositories.reserve_llm_call(session, day, limit)
            await session.commit()
            return allowed

    tasks = [asyncio.create_task(attempt()) for _ in range(attempts)]
    await asyncio.sleep(0.2)
    start.set()
    results = await asyncio.gather(*tasks)
    await engine.dispose()
    return list(results)


async def _calls_on(day: date) -> int | None:
    engine = create_async_engine(os.environ["DATABASE_URL"])
    async with AsyncSession(engine) as session:
        calls = await session.scalar(select(LLMDailyUsage.calls).where(LLMDailyUsage.day == day))
    await engine.dispose()
    return calls


def test_concurrent_reservations_never_exceed_the_limit() -> None:
    """25 simultaneous requests race for 10 slots: exactly 10 win, the counter stops at 10.

    A count-then-call implementation fails this: every request reads the same
    count before any of them writes, so all 25 get through.
    """
    results = asyncio.run(_reserve_concurrently(attempts=25, limit=10))

    assert results.count(True) == 10
    assert asyncio.run(_calls_on(DAY)) == 10


def test_budget_is_per_utc_day() -> None:
    asyncio.run(_reserve_concurrently(attempts=3, limit=3))
    next_day = date(2026, 1, 16)

    assert asyncio.run(_reserve_concurrently(attempts=1, limit=3)) == [False]
    assert asyncio.run(_reserve_concurrently(attempts=1, limit=3, day=next_day)) == [True]
    assert asyncio.run(_calls_on(next_day)) == 1


def test_lowering_the_limit_blocks_immediately() -> None:
    asyncio.run(_reserve_concurrently(attempts=5, limit=5))

    assert asyncio.run(_reserve_concurrently(attempts=2, limit=3)) == [False, False]
    assert asyncio.run(_calls_on(DAY)) == 5, "a rejected reservation must not change the count"
