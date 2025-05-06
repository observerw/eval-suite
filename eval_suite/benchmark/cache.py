import json
from pathlib import Path
from typing import Any, Self

import pydantic
from pydantic import BaseModel, Field, PrivateAttr

from eval_suite.benchmark.result import EvalResultBase, ExceptionEvalResult
from eval_suite.benchmark.schema import EvalID, EvalOutputBase
from eval_suite.client import Message


class EvalCache[Output: EvalOutputBase, Result: EvalResultBase](BaseModel):
    _eval_id: EvalID = PrivateAttr()

    gen: Message | None = None
    output: Output | None = None
    result: Result | ExceptionEvalResult | None = Field(
        default=None,
        discriminator="result_type",
    )

    @property
    def finished(self) -> bool:
        return self.result is not None

    @classmethod
    def load(cls, path: Path) -> Self:
        data: dict[str, Any] = {}

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

    def dump(self, path: Path) -> None:
        for key, value in self:
            if not value:
                continue

            (path / f".{key}.json").write_text(value.json())


class EvalCachePool:
    _lookup: dict[EvalID, EvalCache] = {}

    def __init__(
        self,
        # type-specified `EvalCache` class
        schema: type[EvalCache],
        base_path: Path,
    ) -> None:
        self._schema = schema
        self._base_path = base_path

    def __getitem__(self, eval_id: EvalID) -> EvalCache:
        if cache := self._lookup.get(eval_id):
            assert cache._eval_id == eval_id
            return cache

        cache = self._schema.try_load(self._base_path / str(eval_id)) or self._schema()
        cache._eval_id = eval_id

        self._lookup[eval_id] = cache
        return cache

    def update(self, cache: EvalCache):
        assert cache._eval_id

        self._lookup[cache._eval_id] = cache
        cache.dump(self._base_path / str(cache._eval_id))

    def update_field(self, eval_id: EvalID, **kwargs: Any):
        if not (cache := self._lookup.get(eval_id)):
            return

        for key, value in kwargs.items():
            if not hasattr(cache, key):
                continue

            setattr(cache, key, value)

        self._lookup[eval_id] = cache
