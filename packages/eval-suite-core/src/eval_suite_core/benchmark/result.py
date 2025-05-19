import json
from pathlib import Path
from typing import Any, Self

import pydantic
import ray
from pydantic import BaseModel, create_model

from eval_suite_core.client.schema import Message
from eval_suite_core.metric.result import EvalResultBase
from eval_suite_core.utils.ray import RayQueue


class EvalCache(BaseModel):
    """Utility class to load and update the cache of a single sample."""

    model_config = {"validate_assignment": True}

    eval_path: Path
    generation: Message

    @classmethod
    def create_schema(cls, defs: dict[str, type[EvalResultBase]]) -> type[Self]:
        return create_model(
            "DerivedEvalCache",
            __base__=cls,
            field_definitions={
                metric: (Result, None) for metric, Result in defs.items()
            },
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        data: dict[str, Any] = {
            "eval_path": path,
        }

        for key in cls.model_fields.keys():
            if not (value_path := path / f".{key}.json").exists():
                continue

            value = json.loads(value_path.read_text())
            data[key] = value

        return cls.model_validate(data)

    @classmethod
    def try_load(cls, path: Path) -> Self | None:
        try:
            return cls.load(path)
        except pydantic.ValidationError:
            return

    def update(self, metric: str, result: EvalResultBase) -> Self:
        if not hasattr(self, metric):
            raise ValueError(f"Metric {metric} not found in cache.")

        (self.eval_path / f".{metric}.json").write_text(result.model_dump_json())
        setattr(self, metric, result)

        return self


@ray.remote
class EvalResultCollection:
    queue: RayQueue[EvalResultBase]
