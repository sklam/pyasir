from __future__ import annotations

# default implementation
from ..datatypes import ZeroOpTrait
from ..dispatchables.be_llvm import emit_llvm, emit_llvm_type


@emit_llvm.register
def _(op: ZeroOpTrait, builder):
    llty = emit_llvm_type(op.result_type, builder.module)
    return llty(None)
