from __future__ import annotations

# default implementation
from ..datatypes import ZeroOpTrait
from ..dispatchables.be_llvm import emit_llvm, emit_llvm_type
from ..dispatchables.interpret import eval_op


@emit_llvm.register
def _(op: ZeroOpTrait, builder):
    llty = emit_llvm_type(op.result_type, builder.module)
    return llty(None)


@eval_op.register
def _(op: ZeroOpTrait):
    return op.result_type.zero_value()
