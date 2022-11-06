from collections.abc import AsyncIterable, AsyncIterator, Iterable
from typing import TypeVar

T = TypeVar('T')
async def aiter(iterable: Iterable[T]) -> AsyncIterator[T]:
    for item in iterable:
        yield item


async def aenumerate(aiterable: AsyncIterable[T], start: int = 0) -> AsyncIterator[tuple[int, T]]:
    count = start
    async for item in aiterable:
        yield count, item
        count += 1
