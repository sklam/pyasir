from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable
from ctypes import c_double

from llvmlite import ir

from ..dispatchables.be_llvm import emit_llvm_type, emit_llvm_const, emit_llvm
from ..dispatchables.ctypes import emit_c_type
from ..interpret import eval_op
from ..datatypes import TypeOpError, DataType, OpTrait
from . import integers


@dataclass(frozen=True)
class FloatBinop(OpTrait):
    py_impl: Callable


@dataclass(frozen=True)
class IntToFloatCast(OpTrait):
    py_impl: Callable
    from_type: integers.IntegerType
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
        assert isinstance(valtype, integers.IntegerType)
        optrait = IntToFloatCast(
            self, py_impl=float, from_type=valtype, to_type=self
        )
        return optrait


class Float64(FloatType):
    bitwidth = 64


@eval_op.register
def _(op: IntToFloatCast, value):
    return op.py_impl(value)


@eval_op.register
def _(op: FloatBinop, lhs, rhs):
    return op.py_impl(lhs, rhs)


@emit_llvm_type.register
def _(datatype: FloatType, module: ir.Module):
    return ir.DoubleType()


@emit_c_type.register
def _(datatype: Float64):
    return c_double


@emit_llvm_const.register
def _(datatype: Float64, builder: ir.IRBuilder, value: ir.Value):
    return ir.Constant(ir.DoubleType(), value)


@emit_llvm.register
def _(op: IntToFloatCast, builder: ir.IRBuilder, value: ir.Value):
    to_width = op.to_type.bitwidth
    to_type = {32: ir.FloatType(), 64: ir.DoubleType()}[to_width]
    return builder.sitofp(value, to_type)


@emit_llvm.register
def _(op: FloatBinop, builder: ir.IRBuilder, lhs: ir.Value, rhs: ir.Value):
    opimpl = op.py_impl

    if opimpl == operator.le:
        return builder.fcmp_signed("<=", lhs, rhs)
    elif opimpl == operator.ge:
        return builder.fcmp_signed(">=", lhs, rhs)
    elif opimpl == operator.lt:
        return builder.fcmp_signed("<", lhs, rhs)
    elif opimpl == operator.gt:
        return builder.fcmp_signed(">", lhs, rhs)
    elif opimpl == operator.sub:
        return builder.fsub(lhs, rhs)
    elif opimpl == operator.add:
        return builder.fadd(lhs, rhs)
    elif opimpl == operator.mul:
        return builder.fmul(lhs, rhs)
    else:
        raise AssertionError(f"not supported {op}")
