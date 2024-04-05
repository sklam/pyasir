from __future__ import annotations

from typing import Any, Callable, Iterable
from types import SimpleNamespace
from dataclasses import dataclass

from pyasir.nodes import (
    RegionNode,
    EnterNode,
    DFNode,
    ArgNode,
    ValueNode,
    UnpackNode,
    LoopExprNode,
    node_replace_attrs,
    Scope,
)
import pyasir.nodes as _df

from pyasir import datatypes as _dt
import pyasir.typedefs as _tys
import pyasir
from .registry import registry
from pyasir.interpret import eval_node, Context, Data
from pyasir.nodes import custom_pprint
from pyasir.dispatchables.be_llvm import emit_llvm
from pyasir.dialects.transforms import lift, lift_and_inline

PyDialect = SimpleNamespace()


@custom_pprint
@dataclass(frozen=True)
class ForLoopNode(RegionNode):
    region_func: Callable

    @classmethod
    def make(cls, func):
        return cls(func)

    def __call__(self, *args: Any, **kwargs: Any):
        [args, kwargs] = self._pre_call(args, kwargs)
        assert len(args)
        indarg = node_replace_attrs(args[0], datatype=args[0].datatype.element)
        scope, loopbody = self._call_region(
            self.region_func, (indarg, *args[1:]), kwargs
        )

        return ForLoopExprNode(
            loopbody.datatype,
            iterator=args[0],
            body=EnterNode.make(loopbody, scope),
        )


_dummy = lambda: None


@custom_pprint
@dataclass(frozen=True, order=False)
class ForLoopExprNode(ValueNode):
    iterator: ValueNode
    body: EnterNode

    def __hash__(self):
        return id(self)

    def dialect_lower(self):
        @_df.LoopExprNode.template
        def new_loop(iterator, indvar_init, *args):
            rangeiter_ok, rangeiter_ind = _df.unpack(_df.call(advance, iterator, indvar_init))
            @_df.CaseExprNode.template(rangeiter_ok)
            def cases(rangeiter_ind, *args):
                case1 = lift_and_inline(self.body, (rangeiter_ind, *args))
                yield (True, case1)
                case0 = _df.pack(rangeiter_ind, *args)
                yield (False, case0)
            ind, *other = _df.unpack(cases(rangeiter_ind, *args))
            body_data = _df.pack(iterator, ind, *other)
            return _df.pack(rangeiter_ok, body_data)

        range_res = _df.call(forloop_iter, self.iterator)
        scope = self.body.scope
        args = list(scope.values())
        out = new_loop(range_res, _df.zeroinit(args[0].datatype), *args[1:])
        from pprint import pprint
        pprint(out)

        # clean up iterator
        _iterator, *other = _df.unpack(out)
        return _df.pack(*other)


PyDialect.forloop = ForLoopNode.make


registry["py"] = PyDialect


# ---------------------------------------------------------------------------


@eval_node.register
def _eval_node_ForLoopExprNode(node: ForLoopExprNode, ctx: Context):
    scope = node.body.scope
    inner_scope = {k: ctx.eval(v) for k, v in scope.items()}
    scope_values = Data(tuple(v.value for v in inner_scope.values()))
    iterator = scope_values.value[0]

    loop_keys = list(inner_scope.keys())

    init_ind = None
    while True:
        ok, ind = advance(iterator, init_ind)
        if not ok:
            scope_values = Data((ind, *scope_values.value[1:]))
            break
        else:
            init_ind = ind
            loop_values = map(Data, [ind, *scope_values.value[1:]])
            scope = dict(zip(loop_keys, loop_values))
            packed_values = ctx.nested_call(node.body.body, scope=scope)
            scope_values = packed_values

    return scope_values



# ------


from pyasir.typedefs.functions import CallOp, Function, RangeCallOp


def advance(it: Iterable, init: Any) -> tuple[bool, Any]:
    if isinstance(it, _ForLoopIter):
        return it.advance(init)
    else:
        try:
            ind = next(it)
        except StopIteration:
            return False, init
        else:
            return True, ind


class _ForLoopIter:
    def __init__(self, it):
        self._it = iter(it)

    def advance(self, init):
        try:
            ind = next(self._it)
        except StopIteration:
            return False, init
        else:
            return True, ind


def forloop_iter(it: Iterable) -> tuple[bool, Any]:
    return _ForLoopIter(it)


@dataclass(frozen=True)
class ForLoopIterCallOp(CallOp):
    pass

@Function.register(forloop_iter)
class ForLoopIterFunction(Function):
    function = forloop_iter

    def get_call(self, args, kwargs):
        [arg] = args
        assert isinstance(arg.datatype, _tys.Int64Iterator), arg.datatype
        return ForLoopIterCallOp(result_type=arg.datatype, function=forloop_iter)


@dataclass(frozen=True)
class IterAdvanceCallOp(CallOp):
    pass


@Function.register(advance)
class AdvanceFunction(Function):
    function = advance

    def get_call(self, args, kwargs):
        [iterator, init] = args
        assert isinstance(iterator.datatype, _tys.Int64Iterator), iterator.datatype
        assert iterator.datatype.element == init.datatype
        restype = _tys.Packed[_tys.Bool, iterator.datatype.element]()
        return IterAdvanceCallOp(result_type=restype, function=advance)


from ..interpret import eval_op


@eval_op.register
def _(op: IterAdvanceCallOp, args):
    args, kwargs = args
    return advance(*args, **kwargs)


@eval_op.register
def _(op: ForLoopIterCallOp, args):
    args, kwargs = args
    return forloop_iter(*args, **kwargs)


from llvmlite import ir

@emit_llvm.register
def _(op: RangeCallOp, builder: ir.IRBuilder, n: ir.Value):
    i64 = ir.IntType(64)
    st = ir.LiteralStructType([i64] * 3)(None)
    st = builder.insert_value(st, i64(0), 0)
    st = builder.insert_value(st, n, 1)
    st = builder.insert_value(st, i64(1), 2)
    return st

@emit_llvm.register
def _(op: ForLoopIterCallOp, builder: ir.IRBuilder, range_obj: ir.Value):
    with builder.goto_entry_block():
        loop_storage = builder.alloca(range_obj.type)
    builder.store(range_obj, loop_storage)
    return loop_storage


@emit_llvm.register
def _(op: IterAdvanceCallOp, builder: ir.IRBuilder, range_ptr: ir.Value, init_ind: ir.Value):
    # TODO: Non-unit step is not implemented yet.
    keys = ['start', 'stop', 'step']
    field_ptrs = {}
    intty = ir.IntType(32)
    for i, k in enumerate(keys):
        field_ptrs[k] = x = builder.gep(range_ptr, [intty(0), intty(i)], inbounds=True,
                                        name=f'ptr.{k}')

    ind = builder.load(field_ptrs['start'], name='ind')
    stop = builder.load(field_ptrs['stop'], name='stop')
    step = builder.load(field_ptrs['step'], name='step')

    next_ind = builder.add(ind, step, name='next_ind')
    ok = builder.icmp_signed("<", ind, stop, name='ok')
    last_ind = builder.sub(ind, step, name='last_ind')

    new_start = builder.select(ok, next_ind, init_ind)
    builder.store(new_start, field_ptrs['start'])

    out_struct = ir.LiteralStructType([ok.type, ind.type])(None)
    out_struct = builder.insert_value(out_struct, ok, 0)
    out_struct = builder.insert_value(out_struct, builder.select(ok, ind, last_ind), 1)

    return out_struct
