import os

from pydantic import BaseModel


class RayOptions(BaseModel):
    num_cpus: int = 0
    num_gpus: int = 0
    resources: dict[str, int] = {}


class MetricConfig(BaseModel):
    ray_options: RayOptions
    batch_size: int


class SyncMetricConfig(MetricConfig):
    ray_options: RayOptions = RayOptions(
        num_cpus=os.cpu_count() or 1,
        num_gpus=0,
        resources={},
    )

    batch_size: int = os.cpu_count() or 1


class AsyncMetricConfig(MetricConfig):
    ray_options: RayOptions = RayOptions(
        num_cpus=0,
        num_gpus=0,
        resources={},
    )

    batch_size: int = 1024
