from typing import Any

from pydantic import BaseModel


class MetricConfig(BaseModel):
    num_cpus: int = 0
    num_gpus: int = 0
    resources: dict[str, int] = {}

    batch_size: int = 1024

    @property
    def ray_options(self) -> dict[str, Any]:
        return {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "resources": self.resources,
        }
