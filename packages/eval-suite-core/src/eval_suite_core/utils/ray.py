from collections.abc import Sequence
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

    async def get(self) -> T:
        return cast(T, await self.queue.get_async())

    async def put(self, item: T):
        await self.queue.put_async(item)

    async def get_batch(self, batch_size: int = 512) -> Sequence[T]:
        items: list[T] = []

        # TODO

        while len(items) < batch_size:
            item = await self.queue.get_async()
            items.append(item)

        return items

    async def put_batch(self, items: Sequence[T]):
        for item in items:
            await self.queue.put_async(item)
