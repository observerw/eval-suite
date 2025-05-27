from typing import ClassVar, TypeVar, override

from pydantic import BaseModel
from pydantic._internal._generics import get_model_typevars_map


class TypeVarsMixin:
    _unresolved_typevars_map: ClassVar[dict[TypeVar, TypeVar]] = {}
    _resolved_typevars_map: ClassVar[dict[TypeVar, type]] = {}

    @classmethod
    def typevars_map(cls) -> dict[TypeVar, type]:
        return cls._resolved_typevars_map

    @classmethod
    def _resolve_typevar(cls, typevar: TypeVar) -> type:
        return cls._resolved_typevars_map[typevar]

    @override
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not issubclass(cls, BaseModel):
            return

        if not (curr_map := get_model_typevars_map(cls)):
            return

        if not cls._unresolved_typevars_map:
            cls._unresolved_typevars_map = {
                k: v for k, v in curr_map.items() if isinstance(v, TypeVar)
            }
            cls._resolved_typevars_map = {
                k: v for k, v in curr_map.items() if not isinstance(v, TypeVar)
            }

            return

        _unresolved_reverse_map = {
            v: k for k, v in cls._unresolved_typevars_map.items()
        }
        for typevar, var_or_type in curr_map.items():
            match var_or_type:
                case TypeVar():
                    cls._unresolved_typevars_map[typevar] = var_or_type
                case _:
                    unresolved_typevar = _unresolved_reverse_map[typevar]
                    cls._unresolved_typevars_map.pop(unresolved_typevar, None)
                    cls._resolved_typevars_map[unresolved_typevar] = var_or_type
