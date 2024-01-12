from __future__ import annotations

from functools import singledispatch
from pyasir import datatypes as _dt
from llvmlite import ir


@singledispatch
def emit_llvm_type(datatype: _dt.DataType, module: ir.Module):
    raise NotImplementedError(datatype)


@singledispatch
def emit_llvm_const(
    datatype: _dt.DataType, builder: ir.IRBuilder, value: ir.Value
):
    raise NotImplementedError(datatype)


@singledispatch
def emit_llvm(op: _dt.OpTrait, builder: ir.IRBuilder, *args: ir.Value):
    raise NotImplementedError(op)
