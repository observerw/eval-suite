import asyncio as aio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from itertools import product

from eval_suite_core.benchmark.base import BenchmarkBase
from eval_suite_core.benchmark.eval import MetricGraph
from eval_suite_core.client.base import OfflineClientBase, OnlineClientBase, _ClientBase
from eval_suite_core.metric.base import MetricID
from eval_suite_core.metric.item import EvalID, EvalItemBase, SampleID
from eval_suite_core.metric.result import ToResultArgsBase
from eval_suite_core.utils.collections import queue_counter


@dataclass
class BenchmarkExecutor:
    benchmark: BenchmarkBase
    client: _ClientBase

    metric_graph: MetricGraph

    item_queue: aio.Queue[EvalItemBase | None] = aio.Queue()
    eval_queue: aio.Queue[ToResultArgsBase | None] = aio.Queue()
    regenerate_queue: aio.Queue[EvalID | None] = aio.Queue()

    @classmethod
    @asynccontextmanager
    async def create(cls, benchmark: BenchmarkBase, client: _ClientBase):
        async with MetricGraph.create(
            ordered_metrics=benchmark.ordered_metrics,
            sink_metrics=benchmark.metrics,
        ) as metric_graph:
            yield cls(
                benchmark=benchmark,
                client=client,
                metric_graph=metric_graph,
            )

    async def generate_worker(self, tg: aio.TaskGroup):
        item_batch: list[EvalItemBase] = []

        while item := await self.item_queue.get():
            match self.client:
                case OnlineClientBase() as cli:
                    raise NotImplementedError
                case OfflineClientBase() as cli:
                    item_batch.append(item)
                    raise NotImplementedError

        raise NotImplementedError("Not implemented yet")

    async def eval_worker(self, tg: aio.TaskGroup):
        async def execute(input: ToResultArgsBase):
            results = await self.metric_graph.execute(input)
            # TODO send results to result collection
            self.eval_queue.task_done()

        batch_size_lookup: dict[int, list[MetricID]] = {}

        async for input, count in queue_counter(self.eval_queue):
            tg.create_task(execute(input))
            # TODO when input enough, create a batch_eval_worker task

        self.eval_queue.task_done()
        await self.eval_queue.join()

        raise NotImplementedError("Not implemented yet")

    async def item_stream(self):
        items = [
            self.benchmark._Item.model_validate(raw_item)  #
            for raw_item in self.benchmark.dataset
        ]

        def with_sample_id(item: EvalItemBase, sample_id: SampleID) -> EvalItemBase:
            ret = item.model_copy()
            ret._sample_id = sample_id
            return ret

        # n_samples not reached, simply yield
        n_samples = self.benchmark.config.n_samples
        for sample_id, item in product(range(1, n_samples + 1), items):
            yield with_sample_id(item, SampleID(sample_id))

        item_lookup = {item.item_id: item for item in items}
        max_n_samples = self.benchmark.config.max_n_samples

        # n_samples reached, wait for retry
        while eval_id := await self.regenerate_queue.get():
            input_id = eval_id.item_id
            prev_sample_id = eval_id.sample_id
            curr_sample_id = SampleID(prev_sample_id + 1)

            # if exceeded max_n_samples, skip
            if not (max_n_samples and curr_sample_id > max_n_samples):
                yield with_sample_id(item_lookup[input_id], curr_sample_id)

            self.regenerate_queue.task_done()

    async def run(self):
        async with aio.TaskGroup() as tg:
            gen_worker = tg.create_task(self.generate_worker(tg))
            eval_worker = tg.create_task(self.eval_worker(tg))

            async for item in self.item_stream():
                await self.item_queue.put(item)

        raise NotImplementedError
