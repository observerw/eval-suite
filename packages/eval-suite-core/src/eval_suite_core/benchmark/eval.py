import asyncio as aio
from typing import Self

import ray
from ray.util.queue import Queue

from eval_suite_core.metric.base import (
    AsyncBatchMetricBase,
    AsyncBatchMetricDefault,
    AsyncMetricBase,
    BatchMetricBase,
    BatchMetricDefault,
    MetricDefault,
    MetricID,
    _MetricBase,
)
from eval_suite_core.metric.result import (
    EvalResultBase,
    EvalResultMap,
    ToResultArgs,
    ToResultArgsBase,
)


@ray.remote(max_retries=3)
def eval_worker(
    metric: MetricDefault,
    input: ToResultArgsBase,
    *args: tuple[MetricID, EvalResultBase],
) -> tuple[_MetricBase, EvalResultBase]:
    prec = EvalResultMap({k: v for k, v in args if k in metric.prec})
    eval_path, item, generation = input
    result = metric.to_result(eval_path, item, generation, prec)
    return metric, result


@ray.remote(max_retries=3)
async def async_eval_worker(
    m: AsyncMetricBase, input: ToResultArgsBase, *args: tuple[MetricID, EvalResultBase]
) -> EvalResultBase:
    prec = EvalResultMap({k: v for k, v in args if k in m.prec})
    eval_path, item, generation = input
    result = await m.to_result(eval_path, item, generation, prec)
    return result


class ResultEvent(aio.Event):
    type Result = EvalResultBase | BaseException

    args: ToResultArgs
    result: Result | None = None

    @classmethod
    def create(cls, args: ToResultArgs) -> Self:
        instance = cls()
        instance.args = args
        return instance

    def set_result(self, result: Result) -> None:
        self.result = result
        self.set()

    async def wait_result(self) -> Result:
        await self.wait()
        if not (result := self.result):
            raise RuntimeError("Result is not set")

        return result


@ray.remote(max_retries=3)
async def batch_eval_receiver(
    m: BatchMetricBase | AsyncBatchMetricBase,
    input: ToResultArgsBase,
    exec_queue: Queue,
    *args: tuple[MetricID, EvalResultBase],
) -> EvalResultBase:
    """Reciver do not actually execute the `to_result`, it just send the item to the real batch worker and wait for result, so that the batch execution can also be a part of the DAG."""

    prec = EvalResultMap({k: v for k, v in args if k in m.prec})

    event = ResultEvent.create(
        args=ToResultArgs(
            eval_path=input.eval_path,
            item=input.item,
            generation=input.generation,
            prec=prec,
        )
    )
    await exec_queue.put_async(event)

    if isinstance(result := await event.wait_result(), BaseException):
        raise result

    return result


@ray.remote(max_retries=3)
def batch_eval_worker(
    m: BatchMetricDefault,
    exec_queue: Queue,
):
    event_batch: list[ResultEvent] = []

    while len(event_batch) < m.config.batch_size and (event := exec_queue.get()):
        assert isinstance(event, ResultEvent)
        event_batch.append(event)

    results = m.to_result(
        [event.args for event in event_batch if event.args.eval_path in m.prec]
    )

    for event, result in zip(event_batch, results):
        event.set_result(result)


@ray.remote(max_retries=3)
async def async_batch_eval_worker(
    m: AsyncBatchMetricDefault,
    exec_queue: Queue,
):
    event_batch: list[ResultEvent] = []

    while len(event_batch) < m.config.batch_size and (event := await exec_queue.get()):
        assert isinstance(event, ResultEvent)
        event_batch.append(event)

    results = await m.to_result(
        [event.args for event in event_batch if event.args.eval_path in m.prec]
    )

    for event, result in zip(event_batch, results):
        event.set_result(result)
