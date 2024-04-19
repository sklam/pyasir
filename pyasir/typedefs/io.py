from __future__ import annotations

import operator
from functools import reduce
from dataclasses import dataclass
from typing import Callable
from ctypes import c_double
from types import FunctionType, CellType

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


def prepare_closure_as_args(fn):
    closures = fn.__closure__
    freevars = fn.__code__.co_freevars
    cells = {
        k: cell.cell_contents
        for k, cell in zip(freevars, closures, strict=True)
    }
    # map values to our scope
    arg_names = [f"cell_{k}" for k in cells]
    arg_nodes = {
        k: _df.ArgNode(v.datatype, name=k)
        for k, v in zip(arg_names, cells.values())
    }
    scope = _df.Scope(zip(arg_nodes.values(), cells.values()))

    closure = tuple([CellType(an) for an in scope])
    # rebuild function with new variables
    newfn = FunctionType(
        fn.__code__,
        globals=fn.__globals__,
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=closure,
    )

    return newfn, scope


class RAII:
    def __init__(self, state: IO) -> None:
        self._state = state
        self._cleanups = []
        self._expects = [_df.as_node(True)]

    def do(self, expr_region, *, expect=None, cleanup=None) -> _df.ValueNode:
        # prelude
        if expect is None:
            expect = lambda x: True

        # body
        state = self._state

        fn, scope = prepare_closure_as_args(expr_region)

        predicate = self._expects[-1]

        # manually build up a case expression
        # ok case
        ok_case_res = fn()
        enter_ok = _df.EnterNode.make(ok_case_res, scope=scope)

        # bad case
        bad_case_res = _df.zeroinit(enter_ok.datatype)
        enter_bad = _df.EnterNode.make(bad_case_res, scope=scope)

        assert bad_case_res.datatype == ok_case_res.datatype

        valexpr = _df.CaseExprNode(
            ok_case_res.datatype,
            predicate,
            (enter_ok, enter_bad),
            _df.as_node_args((True, False)),
        )

        # epilog
        state, valexpr = _df.unpack(_df.pack(state, valexpr))

        self._expects.append(self._expects[-1] & _df.as_node(expect(valexpr)))

        if cleanup is not None:
            self._cleanups.append((valexpr, cleanup))
        self._state = state
        return valexpr

    def complete(self, retval: _df.ValueNode) -> _df.ValueNode:
        seq = self._state
        args = [seq]
        for val, cleanup in self._cleanups:
            args.append(cleanup(val))
        args.append(retval)
        [*_before, last] = _df.unpack(_df.pack(*args))
        return last


def sync(io, *args) -> _df.DFNode:
    io, *args = _df.as_node_args((io, *args))
    states = [io, *args[:-1]]
    # for st in states:
    #     assert isinstance(st.datatype, IO)
    value = args[-1]
    dt = value.datatype
    return _df.wrap(_df.ExprNode(dt, SyncOp(dt), args=(*states, value)))


def seq() -> _df.DFNode:
    return _df.wrap(_df.ExprNode(IO(), StateOp(IO()), args=()))


@eval_op.register
def _(op: SyncOp, *args):
    return args[-1]


class DummyState:
    pass


@eval_op.register
def _(op: StateOp, *args):
    return DummyState()


# -----------------------------------------------------------------------------


@emit_llvm_type.register
def _(ty: IO, builder: ir.IRBuilder):
    return ir.IntType(32)


@emit_llvm.register
def _(op: StateOp, builder: ir.IRBuilder):
    return ir.Constant(emit_llvm_type(op.result_type, builder.module), 0)


@emit_llvm.register
def _(op: SyncOp, builder: ir.IRBuilder, *args: ir.Value):
    return args[-1]
