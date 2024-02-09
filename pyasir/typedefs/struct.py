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

    def _make_dataclass(self):
        fields = self.__struct_annotations__
        return make_dataclass(self.__class__.__name__, fields, frozen=True)


@eval_op.register
def _(op: _dt.MakeOp, *args):
    assert isinstance(op.result_type, StructType)
    dc = op.result_type._make_dataclass()
    return dc(*args)


@eval_op.register
def _(op: _dt.AttrOp, value):
    return getattr(value, op.attr)


@emit_llvm.register
def _(op: _dt.MakeOp, builder: ir.IRBuilder, *args: ir.Value):
    assert isinstance(op.result_type, StructType)
    elems = [
        emit_llvm_type(t, builder.module)
        for _k, t in op.result_type.__struct_annotations__
    ]
    llty = ir.LiteralStructType(elems)
    st = llty(None)
    for i, v in enumerate(args):
        st = builder.insert_value(st, v, i)
    return st


@emit_llvm.register
def _(op: _dt.AttrOp, builder: ir.IRBuilder, st: ir.Value):
    return builder.extract_value(st, op.index)
