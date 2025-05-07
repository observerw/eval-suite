import json
from abc import ABC
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import RootModel


def set_value(attr: str, value: Any, rets: Iterable[Any]):
    for ret in rets:
        setattr(ret, attr, value)


class TypeWrapper[T](RootModel[T]):
    """
    HACK access the real type of generic type parameter with the magic of pydantic

    see <https://docs.pydantic.dev/latest/concepts/models/#generic-models> for details
    """

    root: T

    @classmethod
    def type(cls) -> type[T]:
        field = cls.model_fields["root"]
        assert (anno := field.annotation)
        return anno

    @classmethod
    def create(cls, value: Any) -> T:
        return cls.model_validate(value).root


def method_resolve[Base: ABC](
    cls: type[Any],
    methods: Sequence[str],
    *,
    base: type[Base] | None = None,
) -> str | None:
    """Get the first implemented method according to the mro."""

    for cls in cls.mro():
        # only check subclass of `Base`
        if base and not issubclass(cls, base):
            continue

        if override_method := next(
            (
                method  #
                for method in methods
                if method in cls.__dict__
            ),
            None,
        ):
            return override_method


def dump_json(data: dict, path: Path):
    """
    Dump the data to a json file.
    """

    if not path.suffix == ".json":
        raise ValueError("Only json files are supported")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
