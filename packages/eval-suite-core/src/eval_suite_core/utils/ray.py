from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Self, cast

from ray.util.queue import Queue


@dataclass
class RayQueue[T]:
    """Enhanced ray queue"""

    queue: Queue

    @classmethod
    def create(cls, *, maxsize: int = 0, actor_options: dict | None = None) -> Self:
        return cls(queue=Queue(maxsize=maxsize, actor_options=actor_options))

    async def get(self) -> T | None:
        return cast(T | None, await self.queue.get_async())

    async def put(self, item: T | None):
        await self.queue.put_async(item)

    async def get_batch(self, batch_size: int = 512) -> Sequence[T]:
        items: list[T] = []

        while (item := await self.queue.get_async()) and (len(items) < batch_size):
            items.append(item)

        return items

    async def put_batch(self, items: Sequence[T]):
        for item in items:
            await self.queue.put_async(item)

    async def __aiter__(self) -> AsyncGenerator[T, None]:
        while value := await self.queue.get_async():
            yield value
