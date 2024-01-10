from __future__ import annotations

from typing import Any, Callable, Type
import operator

from dataclasses import dataclass
from . import nodes as _df



class TypeOpError(ValueError):
    pass



class DataType:
    def __init__(self):
        raise AssertionError("cannot instantiate")

    @classmethod
    def get_binop(cls, op, lhs, rhs):
        raise NotImplementedError

    @classmethod
    def get_const_trait(cls):
        return cls.ConstTrait(cls)


@dataclass(frozen=True)
class OpTrait:
    name: str

@dataclass(frozen=True)
class BaseConstTrait:
    datatype: Type[DataType]


@dataclass(frozen=True)
class IntBinop(OpTrait):
    py_impl: Callable


INT_BINOPS = {
    "<=": IntBinop("int.le", operator.le),
    ">=": IntBinop("int.ge", operator.ge),
    ">": IntBinop("int.gt", operator.gt),
    "<": IntBinop("int.lt", operator.lt),
    "+": IntBinop("int.add", operator.add),
    "-": IntBinop("int.sub", operator.sub),
    "*": IntBinop("int.mul", operator.mul),
}

class IntegerType(DataType):

    class ConstTrait(BaseConstTrait):
        pass

    @classmethod
    def get_binop(cls, op: str, lhs: _df.ValueNode, rhs: _df.ValueNode) -> _df.ValueNode:
        optrait = INT_BINOPS[op]
        if lhs.datatype != cls or rhs.datatype != cls:
            raise TypeOpError(f"unsupported op for {op}({lhs.datatype, rhs.datatype})")
        return _df.ExprNode(cls, op=optrait, args=(lhs, rhs))


class Int64(IntegerType):
    bitwidth = 64



class BooleanType(DataType):
    pass

Bool = BooleanType


@dataclass(frozen=True)
class PackedType(DataType):
    pass


