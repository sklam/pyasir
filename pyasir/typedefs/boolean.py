from __future__ import annotations

from llvmlite import ir

from dataclasses import dataclass

from ..datatypes import DataType, OpTrait
from ..dispatchables.be_llvm import emit_llvm_type, emit_llvm_const, emit_llvm

class BooleanType(DataType):
    pass


Bool = BooleanType


@emit_llvm_const.register
def _(datatype: Bool, builder: ir.IRBuilder, value: ir.Value):
    return ir.Constant(ir.IntType(1), value)
