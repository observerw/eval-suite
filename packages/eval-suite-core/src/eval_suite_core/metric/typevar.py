from inspect import isclass
from typing import Any, ClassVar, TypeVar, override

from pydantic import BaseModel, PrivateAttr
from pydantic._internal._generics import get_model_typevars_map


class TypeVarMixin:
    _typevar_map: ClassVar[dict[TypeVar, Any]] = PrivateAttr()

    @override
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not issubclass(cls, BaseModel) or cls is BaseModel:
            return

        if not (curr_map := get_model_typevars_map(cls)):
            return

        if all_map := cls._typevar_map:
            for typevar, typ in all_map.items():
                if isinstance(typ, TypeVar) and (realtype := curr_map.get(typ)):
                    all_map[typevar] = realtype
        else:
            cls._typevar_map = curr_map

    def _resolve_typevar(self, typevar: TypeVar) -> type:
        assert isclass(typ := self._typevar_map.get(typevar)), (
            f"Type {typ} is not a class"
        )
        return typ
