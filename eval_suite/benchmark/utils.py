import json
from pathlib import Path
from typing import Any

from pydantic import RootModel


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


def dump_json(data: dict, path: Path):
    """
    Dump the data to a json file.
    """

    if not path.suffix == ".json":
        raise ValueError("`path` must be a valid json file path")

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
