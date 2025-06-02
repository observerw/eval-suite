from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, create_model, model_validator

from eval_suite_core.benchmark.config import EvalConfig
from eval_suite_core.client.schema import Message
from eval_suite_core.metric.base import AnyMetric
from eval_suite_core.metric.id import EvalID, ItemID
from eval_suite_core.metric.item import ChatItemBase
from eval_suite_core.metric.result import ExceptionResult, ResultMap
from eval_suite_core.prompt.schema import ChatSequence
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

    @model_validator(mode="after")
    def _save(self):
        raise NotImplementedError


type MetricGraphResult = ResultMap | ExceptionResult


class MetricGraphResultGroups(dict[ItemID, list[MetricGraphResult]]):
    def total_count(self, item_id: ItemID) -> int:
        return len(self.get(item_id, []))

    def result_count(self, item_id: ItemID) -> int:
        return len(
            [
                result
                for result in self.get(item_id, [])
                if not isinstance(result, ExceptionResult)
            ]
        )

    def exc_result_count(self, item_id: ItemID) -> int:
        return self.total_count(item_id) - self.result_count(item_id)


@dataclass
class Manager:
    base_path: Path
    config: EvalConfig
    dataset: list[ChatItemBase]

    generation_queue: RayQueue[tuple[EvalID, Message]] = RayQueue.create()
    result_queue: RayQueue[MetricGraphResult] = RayQueue.create()
    retry_queue: RayQueue[ItemID] = RayQueue.create()

    groups: MetricGraphResultGroups = MetricGraphResultGroups()
    histories: dict[ItemID, ChatSequence] = {}

    def is_finished(self, item_id: ItemID) -> bool:
        """Check if item has sufficient results (>= n_samples) or excceeded max retries (>= max_n_samples)."""

        result_count = self.groups.result_count(item_id)
        exc_result_count = self.groups.exc_result_count(item_id)

        if result_count >= self.config.n_samples:
            return True

        if (max_n := self.config.max_n_samples) and exc_result_count >= max_n:
            return True

        return False

    async def item_stream(self):
        """Stream items from the dataset and retry queue."""

        max_n = self.config.max_n_samples
        item_lookup = {item.item_id: item for item in self.dataset}

        for sample_id, item in enumerate(self.dataset, start=1):
            yield item.model_copy(update={"_sample_id": sample_id})

        async for item_id in self.retry_queue:
            result_count = self.groups.result_count(item_id)
            if not (max_n and result_count >= max_n):
                yield item_lookup[item_id].model_copy(
                    update={"_sample_id": result_count + 1}
                )

    async def spawn(self):
        raise NotImplementedError
