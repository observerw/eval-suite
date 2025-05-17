import asyncio as aio
from collections import OrderedDict
from collections.abc import Iterator


class OrderedSet[T]:
    _inner: OrderedDict[T, None]

    def add(self, item: T) -> None:
        self._inner[item] = None

    def __iter__(self) -> Iterator[T]:
        return iter(self._inner.keys())

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._inner.keys())


async def queue_counter[T](queue: aio.Queue[T | None]):
    count = 0
    while data := await queue.get():
        count += 1
        yield data, count
