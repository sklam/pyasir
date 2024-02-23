from __future__ import annotations

from functools import cache
from dataclasses import dataclass, make_dataclass
from typing import Type
from inspect import Signature, Parameter

from llvmlite import ir

from pyasir.typing import get_annotations
from pyasir import datatypes as _dt
from pyasir import nodes as _df

from pyasir.interpret import eval_op
from pyasir.dispatchables.be_llvm import (
    emit_llvm_type,
    emit_llvm_const,
    emit_llvm,
)
from pyasir.dispatchables.ctypes import emit_c_type


def Struct(cls: Type) -> StructType:
    ann = get_annotations(cls)
    g = {"__struct_annotations__": tuple(map(tuple, ann.items()))}
    typ = type(cls.__name__, (StructType,), g)
    return typ


class StructType(_dt.DataType):
    __struct_annotations__: tuple[tuple[str, _dt.DataType], ...]

    def get_make(self, args, kwargs):
        sig = Signature(
            Parameter(k, Parameter.POSITIONAL_OR_KEYWORD)
            for k, _t in self.__struct_annotations__
        )
        ba = sig.bind(*args, **kwargs)
        args = ba.arguments
        return args

    @cache
    def attribute_lookup(self, attr: str) -> _dt.AttrOp:
        for i, (k, t) in enumerate(self.__struct_annotations__):
            if k == attr:
                return _dt.AttrOp(t, self, attr, i)
        raise AttributeError(attr)

    def get_fields(self) -> tuple[tuple[str, _dt.DataType], ...]:
        return self.__struct_annotations__


@eval_op.register
def _(op: _dt.MakeOp, *args):
    assert isinstance(op.result_type, StructType)
    cstruct = emit_c_type(op.result_type)
    # TODO: Make sure this is read-only
    return cstruct(*args)


@eval_op.register
def _(op: _dt.AttrOp, value):
    return getattr(value, op.attr)


@emit_llvm.register
def _(op: _dt.MakeOp, builder: ir.IRBuilder, *args: ir.Value):
    assert isinstance(op.result_type, StructType)
    llty = emit_llvm_type(op.result_type, builder.module)
    st = llty(None)
    for i, v in enumerate(args):
        st = builder.insert_value(st, v, i)
    return st


@emit_llvm.register
def _(op: _dt.AttrOp, builder: ir.IRBuilder, st: ir.Value):
    return builder.extract_value(st, op.index)


@emit_llvm_type.register
def _(datatype: StructType, module: ir.Module):
    fields = [emit_llvm_type(t, module) for k, t in datatype.get_fields()]
    return ir.LiteralStructType(fields)


@emit_c_type.register
def _struct_ctype(datatype: StructType):
    import ctypes

    if datatype in _struct_ctype.cache:
        return _struct_ctype.cache[datatype]

    fields = [(k, emit_c_type(t)) for k, t in datatype.get_fields()]

    class c_struct(ctypes.Structure):
        _fields_ = fields

    _struct_ctype.cache[datatype] = c_struct
    return c_struct


_struct_ctype.cache = {}
