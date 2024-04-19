from __future__ import annotations

import ctypes
import operator
from functools import cache
from dataclasses import dataclass, make_dataclass
from typing import Type, Callable
from inspect import Signature, Parameter

from llvmlite import ir

from pyasir.typing import get_annotations
from pyasir import datatypes as _dt
from pyasir import nodes as _df
import pyasir

from pyasir.interpret import eval_op
from pyasir.dispatchables.be_llvm import (
    emit_llvm_type,
    emit_llvm_const,
    emit_llvm,
)
from pyasir.dispatchables.ctypes import emit_c_type
from ..datatypes import OpTrait, define_op, TypeOpError
from .integers import Int64
from .integers import Bool
from .io import IO


@dataclass(frozen=True)
class PointerBinop(OpTrait):
    py_impl: Callable


PTR_BINOPS = {
    "==": lambda restype: PointerBinop(restype, operator.eq),
    "!=": lambda restype: PointerBinop(restype, operator.ne),
}


class Pointer(_dt.DataType):
    def get_binop(
        self, op: str, lhs: _dt.DataType, rhs: _dt.DataType
    ) -> PointerBinop:
        optrait = PTR_BINOPS[op](Bool())
        if lhs != self or rhs != self:
            raise TypeOpError(f"unsupported op for {op}({lhs, rhs})")
        return optrait


@dataclass(frozen=True)
class PointerAlloc(OpTrait):
    pass


@define_op(PointerAlloc(Pointer()))
def alloc(n: pyasir.Int64) -> Pointer:
    pass


@dataclass(frozen=True)
class PointerFree(OpTrait):
    pass


@dataclass(frozen=True)
class IntToPtr(OpTrait):
    pass


@define_op(PointerFree(IO()))
def free(p: Pointer) -> IO:
    pass


@dataclass(frozen=True)
class PointerLoad(OpTrait):
    pass


@dataclass(frozen=True)
class PointerStore(OpTrait):
    item_type: _dt.DataType


def load(dt: _dt.DataType, ptr: Pointer):
    dt = _dt.ensure_type(dt)
    return _df.wrap(_df.ExprNode(dt, PointerLoad(dt), args=(_df.as_node(ptr),)))


def store(ptr: Pointer, item) -> IO:
    return _df.wrap(
        _df.ExprNode(
            IO(),
            PointerStore(IO(), item_type=item.datatype),
            args=_df.as_node_args((ptr, item)),
        )
    )


def as_pointer(ptr: Pointer) -> _df.ValueNode:
    return _df.wrap(
        _df.ExprNode(Pointer(), IntToPtr(Pointer()), args=(_df.as_node(ptr),))
    )


# -----------------------------------------------------------------------------


@eval_op.register
def eval_op_PointerAlloc(op: PointerAlloc, n):
    cdll = ctypes.CDLL(None)
    cdll.malloc.restype = ctypes.c_void_p
    out = cdll.malloc(n)
    assert out is not None
    print("malloc", hex(out))
    return out


@eval_op.register
def eval_op_PointerFree(op: PointerFree, p):
    print("free", hex(p))
    cdll = ctypes.CDLL(None)
    free = cdll.free
    free.argtypes = [ctypes.c_void_p]
    free.restype = None
    free(p)
    return 0


@eval_op.register
def eval_op_IntToPtr(op: IntToPtr, p):
    return p


@eval_op.register
def eval_op_PointerLoad(op: PointerLoad, ptr):
    from pyasir.dispatchables.ctypes import emit_c_type

    ct_result = emit_c_type(op.result_type)
    castedptr = ctypes.cast(ptr, ctypes.POINTER(ct_result))
    print("load", hex(ptr))
    # prevent using a reference
    out = ct_result.from_buffer_copy(ct_result.from_address(ptr))
    print("  loaded", " ".join([f"{b:02x}" for b in bytes(out)]))
    return out


@eval_op.register
def eval_op_PointerStore(op: PointerStore, ptr, item):
    from pyasir.dispatchables.ctypes import emit_c_type

    ct_item = emit_c_type(op.item_type)

    castedptr = ctypes.cast(ctypes.c_void_p(ptr), ctypes.POINTER(ct_item))
    print("store", hex(ptr))
    castedptr[0] = item


@emit_c_type.register
def _(datatype: Pointer):
    # Use c_size_t type to prevent conversion to None
    return ctypes.c_size_t


@eval_op.register
def _(op: PointerBinop, lhs, rhs):
    assert op.py_impl == operator.ne
    # print("!=", lhs, rhs)
    return op.py_impl(lhs, rhs)


# -----------------------------------------------------------------------------


@emit_llvm_type.register
def _(datatype: Pointer, module: ir.Module):
    return ir.IntType(8).as_pointer()


@emit_llvm.register
def _(op: PointerAlloc, builder: ir.IRBuilder, n: ir.Value):
    module: ir.Module = builder.module
    try:
        fn = module.get_global("malloc")
    except KeyError:
        fnty = ir.FunctionType(ir.IntType(8).as_pointer(), [ir.IntType(64)])
        fn = ir.Function(module, fnty, name="malloc")

    return builder.call(fn, [n])


@emit_llvm.register
def _(op: PointerFree, builder: ir.IRBuilder, ptr: ir.Value):
    module: ir.Module = builder.module
    try:
        fn = module.get_global("free")
    except KeyError:
        fnty = ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer()])
        fn = ir.Function(module, fnty, name="free")

    return builder.call(fn, [ptr])


@emit_llvm.register
def _(op: PointerStore, builder: ir.IRBuilder, ptr: ir.Value, item: ir.Value):
    builder.store(item, builder.bitcast(ptr, item.type.as_pointer()))
    return ir.Constant(ir.IntType(32), 0)


@emit_llvm.register
def _(op: PointerLoad, builder: ir.IRBuilder, ptr: ir.Value):
    item_type = emit_llvm_type(op.result_type, builder.module)
    return builder.load(builder.bitcast(ptr, item_type.as_pointer()))


@emit_llvm.register
def _(op: IntToPtr, builder: ir.IRBuilder, n: ir.Value):
    return builder.inttoptr(n, ir.IntType(8).as_pointer())


@emit_llvm.register
def _(op: PointerBinop, builder: ir.IRBuilder, lhs: ir.Value, rhs: ir.Value):
    cmpstr = {
        operator.eq: "==",
        operator.ne: "!=",
    }[op.py_impl]
    return builder.icmp_unsigned(cmpstr, lhs, rhs)
