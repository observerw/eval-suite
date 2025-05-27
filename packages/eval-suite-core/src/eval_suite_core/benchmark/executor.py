import asyncio as aio
from contextlib import asynccontextmanager
from dataclasses import dataclass

from eval_suite_core.benchmark.base import BenchmarkBase
from eval_suite_core.benchmark.eval import MetricGraph
from eval_suite_core.client.base import AnyClient
from eval_suite_core.metric.item import ItemBase


@dataclass
class BenchmarkExecutor:
    benchmark: BenchmarkBase
    client: AnyClient

    metric_graph: MetricGraph

    tg: aio.TaskGroup

    @classmethod
    @asynccontextmanager
    async def create(cls, benchmark: BenchmarkBase, client: AnyClient):
        async with (
            MetricGraph.create(
                ordered_metrics=benchmark.ordered_metrics,
                sink_metrics=benchmark.sink_metrics,
            ) as metric_graph,
            aio.TaskGroup() as tg,
        ):
            yield cls(
                benchmark=benchmark,
                client=client,
                metric_graph=metric_graph,
                tg=tg,
            )

    async def generate_worker(self):
        item_batch: list[ItemBase] = []

        raise NotImplementedError("Not implemented yet")

    async def eval_worker(self):
        raise NotImplementedError("Not implemented yet")

    # async def item_stream(self):
    #     items = [
    #         self.benchmark._Item.model_validate(raw_item)  #
    #         for raw_item in self.benchmark.dataset
    #     ]

    #     def with_sample_id(item: ItemBase, sample_id: SampleID) -> ItemBase:
    #         ret = item.model_copy()
    #         ret._sample_id = sample_id
    #         return ret

    #     # n_samples not reached, simply yield
    #     n_samples = self.benchmark.config.n_samples
    #     for sample_id, item in product(range(1, n_samples + 1), items):
    #         yield with_sample_id(item, SampleID(sample_id))

    #     item_lookup = {item.item_id: item for item in items}
    #     max_n_samples = self.benchmark.config.max_n_samples

    #     # n_samples reached, wait for retry
    #     while eval_id := await self.regenerate_queue.get():
    #         input_id = eval_id.item_id
    #         prev_sample_id = eval_id.sample_id
    #         curr_sample_id = SampleID(prev_sample_id + 1)

    #         # if exceeded max_n_samples, skip
    #         if not (max_n_samples and curr_sample_id > max_n_samples):
    #             yield with_sample_id(item_lookup[input_id], curr_sample_id)

    #         self.regenerate_queue.task_done()

    async def run(self):
        async with aio.TaskGroup() as tg:
            gen_worker = tg.create_task(self.generate_worker())
            eval_worker = tg.create_task(self.eval_worker())

        raise NotImplementedError
