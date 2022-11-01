from collections.abc import AsyncIterator, Iterable
from typing import TypeVar

T = TypeVar('T')
async def aiter(iterable: Iterable[T]) -> AsyncIterator[T]:
    for item in iterable:
        yield item
