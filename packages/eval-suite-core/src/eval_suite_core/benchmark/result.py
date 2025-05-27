import asyncio as aio
from collections.abc import Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, create_model

from eval_suite_core.client.schema import Message
from eval_suite_core.metric.base import AnyMetric
from eval_suite_core.metric.id import EvalID
from eval_suite_core.metric.result import ResultMap
from eval_suite_core.utils.ray import RayQueue


class EvalCache(BaseModel):
    """Utility class to load and update the cache of a single sample."""

    model_config = {"validate_assignment": True}

    eval_path: Path
    generation: Message

    @classmethod
    def create(cls, metrics: Iterable[AnyMetric]) -> "type[EvalCache]":
        return create_model(
            "DerivedCache",
            __base__=cls,
            field_definitions={
                metric.id: (metric._Result | None)  #
                for metric in metrics
            },
        )


@dataclass
class EvalResultCollector:
    tg: aio.TaskGroup

    base_path: Path

    generation_queue: RayQueue[tuple[EvalID, Message] | None] = RayQueue.create()
    result_queue: RayQueue[ResultMap | None] = RayQueue.create()

    groups: dict[EvalID, ResultMap] = {}

    @asynccontextmanager
    @classmethod
    async def create(cls, base_path: Path):
        async with aio.TaskGroup() as tg:
            yield cls(
                base_path=base_path,
                tg=tg,
            )

    async def spawn_generation(self):
        while generation := await self.generation_queue.get():
            raise NotImplementedError

    async def spawn_result(self):
        while result := await self.result_queue.get():
            self.groups.setdefault(result._eval_id, result)

    async def spawn(self):
        self.tg.create_task(self.spawn_generation())

        raise NotImplementedError("Not implemented yet")


@dataclass
class EvalItemLoader:
    tg: aio.TaskGroup

    base_path: Path

    collector: EvalResultCollector
    sink_metrics: list[AnyMetric]

    retry_queue: RayQueue[EvalID | None] = RayQueue.create()

    @asynccontextmanager
    @classmethod
    async def create(
        cls,
        collector: EvalResultCollector,
        sink_metrics: Sequence[AnyMetric],
        base_path: Path,
    ):
        async with aio.TaskGroup() as tg:
            yield cls(
                tg=tg,
                base_path=base_path,
                collector=collector,
                sink_metrics=[*sink_metrics],
            )

    async def spawn(self):
        while eval_id := await self.retry_queue.get():
            raise NotImplementedError
