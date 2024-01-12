from __future__ import annotations

import abc
from typing import Any, Callable, Type
import operator

from dataclasses import dataclass
from . import nodes as _df


def ensure_type(ty: DataTypeLike) -> DataType:
    if isinstance(ty, type) and issubclass(ty, DataType):
        return ty()
    else:
        assert isinstance(ty, DataType)
        return ty


class TypeOpError(ValueError):
    pass


class DataType(abc.ABC):
    __singleton = None

    def __new__(cls):
        if cls.__singleton is None:
            cls.__singleton = object.__new__(cls)
        return cls.__singleton

    @abc.abstractmethod
    def get_binop(self, op: str, lhs: DataType, rhs: DataType) -> OpTrait:
        ...

    @abc.abstractmethod
    def get_cast(self, valtype: DataType) -> OpTrait:

    def attribute_lookup(self, attr: str) -> AttrOp:
        raise AttributeError(attr)

    def __repr__(self):
        return self.__class__.__name__


DataTypeLike = DataType | Type[DataType]


@dataclass(frozen=True)
class OpTrait:
    result_type: DataType

    def __post_init__(self):
        assert isinstance(self.result_type, DataType)


@dataclass(frozen=True)
class MakeOp(OpTrait):
    ...


@dataclass(frozen=True)
class AttrOp(OpTrait):
    base_type: DataType
    attr: str
    index: int
