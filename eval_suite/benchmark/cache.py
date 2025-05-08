import json
from pathlib import Path
from typing import Any, NotRequired, Self, TypedDict, Unpack

import pydantic
from pydantic import BaseModel, Field, PrivateAttr

from eval_suite.client import Message
from eval_suite.metric.result import EvalResultBase, ExceptionEvalResult
from eval_suite.metric.schema import EvalID, EvalOutputBase


class EvalCacheDict(TypedDict):
    gen: NotRequired[Message | None]
    output: NotRequired[EvalOutputBase | None]
    result: NotRequired[EvalResultBase | ExceptionEvalResult | None]


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


class EvalCachePool(BaseModel):
    type _EvalCache = EvalCache[EvalOutputBase, EvalResultBase]

    cache_schema: type[EvalCache]  # Cache with concrete types
    base_path: Path

    _lookup: dict[EvalID, _EvalCache] = {}

    def __getitem__(self, eval_id: EvalID) -> _EvalCache:
        if cache := self._lookup.get(eval_id):
            assert cache._eval_id == eval_id
            return cache

        cache = (
            self.cache_schema.try_load(self.base_path / str(eval_id))
            or self.cache_schema()
        )
        cache._eval_id = eval_id

        self._lookup[eval_id] = cache
        return cache

    def update(self, cache: _EvalCache):
        assert cache._eval_id

        self._lookup[cache._eval_id] = cache
        cache.dump(self.base_path / str(cache._eval_id))

    def update_field(self, eval_id: EvalID, **kwargs: Unpack[EvalCacheDict]):
        cache = self[eval_id]

        for key, value in kwargs.items():
            if not hasattr(cache, key):
                continue

            setattr(cache, key, value)

        self.update(cache)
