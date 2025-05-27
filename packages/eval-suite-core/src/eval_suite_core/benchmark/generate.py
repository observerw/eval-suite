from collections.abc import Sequence

import ray

from eval_suite_core.client.base import OfflineClientBase, OnlineClientBase
from eval_suite_core.metric.item import ItemBase


@ray.remote
async def online_generate_worker(client: OnlineClientBase, item: ItemBase):
    return await client.generate(item.format([]))  # TODO history


@ray.remote
def offline_generate_worker(
    client: OfflineClientBase, item_batch: Sequence[ItemBase]
):
    return client.generate(item.format([]) for item in item_batch)  # TODO history
