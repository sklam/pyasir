from __future__ import annotations

import abc
from typing import Any, Callable, Type
import operator

from dataclasses import dataclass
from . import nodes as _df


def ensure_type(ty: DataType | Type[DataType]) -> DataType:
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
        ...

    def __repr__(self):
        return self.__class__.__name__


@dataclass(frozen=True)
class OpTrait:
    result_type: DataType


@dataclass(frozen=True)
class BaseConstTrait:
    datatype: DataType


@dataclass(frozen=True)
class IntBinop(OpTrait):
    py_impl: Callable


INT_BINOPS = {
    "<=": lambda restype: IntBinop(restype, operator.le),
    ">=": lambda restype: IntBinop(restype, operator.ge),
    ">": lambda restype: IntBinop(restype, operator.gt),
    "<": lambda restype: IntBinop(restype, operator.lt),
    "+": lambda restype: IntBinop(restype, operator.add),
    "-": lambda restype: IntBinop(restype, operator.sub),
    "*": lambda restype: IntBinop(restype, operator.mul),
}


class IntegerType(DataType):
    bitwidth: int

    def get_binop(self, op: str, lhs: DataType, rhs: DataType) -> IntBinop:
        optrait = INT_BINOPS[op](self)
        if lhs != self or rhs != self:
            raise TypeOpError(f"unsupported op for {op}({lhs, rhs})")
        return optrait

    def get_cast(self, valtype: DataType) -> OpTrait:
        raise NotImplementedError


class Int64(IntegerType):
    bitwidth = 64


class BooleanType(DataType):
    def get_binop(self, op: str, lhs: DataType, rhs: DataType) -> OpTrait:
        raise NotImplementedError

    def get_cast(self, valtype: DataType) -> OpTrait:
        raise NotImplementedError


Bool = BooleanType


@dataclass(frozen=True)
class FloatBinop(OpTrait):
    py_impl: Callable


@dataclass(frozen=True)
class IntToFloatCast(OpTrait):
    py_impl: Callable
    from_type: IntegerType
    to_type: FloatType


FLT_BINOPS = {
    "<=": lambda restype: FloatBinop(restype, operator.le),
    ">=": lambda restype: FloatBinop(restype, operator.ge),
    ">": lambda restype: FloatBinop(restype, operator.gt),
    "<": lambda restype: FloatBinop(restype, operator.lt),
    "+": lambda restype: FloatBinop(restype, operator.add),
    "-": lambda restype: FloatBinop(restype, operator.sub),
    "*": lambda restype: FloatBinop(restype, operator.mul),
}


class FloatType(DataType):
    bitwidth: int

    def get_binop(self, op: str, lhs: DataType, rhs: DataType) -> OpTrait:
        if lhs != self or rhs != self:
            raise TypeOpError(f"unsupported op for {op}({lhs, rhs})")
        optrait = FLT_BINOPS[op](self)
        return optrait

    def get_cast(self, valtype: DataType) -> OpTrait:
        assert isinstance(valtype, IntegerType)
        optrait = IntToFloatCast(
            self, py_impl=float, from_type=valtype, to_type=self
        )
        return optrait


class Float64(FloatType):
    bitwidth = 64


class _PackedType(DataType):
    def get_binop(self, op: str, lhs: DataType, rhs: DataType) -> OpTrait:
        raise NotImplementedError

    def get_cast(self, valtype: DataType) -> OpTrait:
        raise NotImplementedError
