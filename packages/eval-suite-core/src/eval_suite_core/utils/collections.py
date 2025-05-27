from collections import OrderedDict
from collections.abc import Iterator


class OrderedSet[T]:
    _inner: OrderedDict[T, None]

    def add(self, item: T) -> None:
        self._inner.setdefault(item, None)

    def __iter__(self) -> Iterator[T]:
        return iter(self._inner.keys())

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._inner.keys())
