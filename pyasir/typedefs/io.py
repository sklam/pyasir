from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable
from ctypes import c_double

from llvmlite import ir

from ..dispatchables.be_llvm import emit_llvm_type, emit_llvm_const, emit_llvm
from ..dispatchables.ctypes import emit_c_type
from ..interpret import eval_op
from ..datatypes import TypeOpError, DataType, OpTrait, AttrOp
from .. import nodes as _df



class IO(DataType):
    pass


@dataclass(frozen=True)
class SyncOp(OpTrait):
    ...


@dataclass(frozen=True)
class StateOp(OpTrait):
    ...


def sync(io, *args) -> _df.DFNode:
    io, *args = _df.as_node_args((io, *args))
    states = [io, *args[:-1]]
    # for st in states:
    #     assert isinstance(st.datatype, IO)
    value = args[-1]
    dt = value.datatype
    return _df.ExprNode(dt, SyncOp(dt), args=(*states, value))




def seq() -> _df.DFNode:
    return _df.ExprNode(IO(), StateOp(IO()), args=())



@eval_op.register
def _(op: SyncOp, *args):
    return args[-1]



class DummyState:
    pass


@eval_op.register
def _(op: StateOp, *args):
    return DummyState()


# -----------------------------------------------------------------------------


@emit_llvm.register
def _(op: StateOp, builder: ir.IRBuilder):
    return ir.Constant(ir.IntType(32), 0)

@emit_llvm.register
def _(op: SyncOp, builder: ir.IRBuilder, *args: ir.Value):
    return args[-1]
