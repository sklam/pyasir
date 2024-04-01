from dataclasses import dataclass
from typing import Callable
from pprint import pformat


class LazyRepr:
    __slots__ = ["_fn"]

    def __init__(self, fn):
        self._fn = fn

    def __repr__(self):
        return pformat(self._fn())

    def __str__(self):
        return pformat(self._fn())
