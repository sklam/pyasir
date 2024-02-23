from __future__ import annotations


import operator
from dataclasses import dataclass
from typing import Callable

from llvmlite import ir

from ..datatypes import DataType, OpTrait, TypeOpError
from ..dispatchables.be_llvm import emit_llvm_type, emit_llvm_const, emit_llvm
from ..interpret import eval_op


@dataclass(frozen=True)
class BoolBinop(OpTrait):
    py_impl: Callable


BOOL_BINOPS = {
    "&": lambda restype: BoolBinop(restype, operator.and_),
    "|": lambda restype: BoolBinop(restype, operator.or_),
}


class BooleanType(DataType):
    def get_binop(self, op: str, lhs: DataType, rhs: DataType) -> BoolBinop:
        optrait = BOOL_BINOPS[op](self)
        if lhs != self or rhs != self:
            raise TypeOpError(f"unsupported op for {op}({lhs, rhs})")
        return optrait


Bool = BooleanType


@eval_op.register
def _(op: BoolBinop, lhs, rhs):
    return op.py_impl(lhs, rhs)


@emit_llvm_const.register
def _(datatype: Bool, builder: ir.IRBuilder, value: ir.Value):
    return ir.Constant(ir.IntType(1), value)


@emit_llvm.register
def _(op: BoolBinop, builder: ir.IRBuilder, lhs: ir.Value, rhs: ir.Value):
    opimpl = op.py_impl

    if opimpl == operator.and_:
        return builder.and_(lhs, rhs)
    elif opimpl == operator.or_:
        return builder.or_(lhs, rhs)
    else:
        raise AssertionError(f"not supported {op}")
