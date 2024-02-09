from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable
from ctypes import c_int64

from llvmlite import ir

from ..dispatchables.be_llvm import emit_llvm_type, emit_llvm_const, emit_llvm
from ..dispatchables.ctypes import emit_c_type
from ..interpret import eval_op
from ..datatypes import TypeOpError, DataType, OpTrait


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


class Int64(IntegerType):
    bitwidth = 64


@eval_op.register
def _(op: IntBinop, lhs, rhs):
    return op.py_impl(lhs, rhs)


@emit_llvm_type.register
def _(datatype: IntegerType, module: ir.Module):
    return ir.IntType(datatype.bitwidth)


@emit_c_type.register
def _(datatype: Int64):
    return c_int64


@emit_llvm_const.register
def _(datatype: Int64, builder: ir.IRBuilder, value: ir.Value):
    return ir.Constant(ir.IntType(datatype.bitwidth), value)


@emit_llvm.register
def _(op: IntBinop, builder: ir.IRBuilder, lhs: ir.Value, rhs: ir.Value):
    opimpl = op.py_impl

    if opimpl == operator.le:
        return builder.icmp_signed("<=", lhs, rhs)
    elif opimpl == operator.ge:
        return builder.icmp_signed(">=", lhs, rhs)
    elif opimpl == operator.lt:
        return builder.icmp_signed("<", lhs, rhs)
    elif opimpl == operator.gt:
        return builder.icmp_signed(">", lhs, rhs)
    elif opimpl == operator.sub:
        return builder.sub(lhs, rhs)
    elif opimpl == operator.add:
        return builder.add(lhs, rhs)
    elif opimpl == operator.mul:
        return builder.mul(lhs, rhs)
    else:
        raise AssertionError(f"not supported {op}")
