from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import operator
from inspect import signature
from functools import singledispatch
from pprint import pformat

from . import nodes as _df
from . import datatypes as _dt

def interpret(funcdef: _df.FuncDef, *args: Any, **kwargs: Any) -> Any:
    data: Data = _eval_funcdef(funcdef, *args, **kwargs)
    return data.value


def _eval_funcdef(funcdef: _df.FuncDef, *args: Any, **kwargs: Any) -> Data:
    sig = signature(funcdef.func)
    ba = sig.bind(*args, **kwargs)
    args = {_df.ArgNode(t, k): v for (k, v), t in zip(ba.arguments.items(), funcdef.argtys)}
    ctx = Context(scope=args, cache={})
    res = ctx.eval(funcdef.node)
    return res

def _normalize(value: Any) -> _df.DFNode:
    if isinstance(value, _df.DFNode):
        return value
    return _df.LiteralNode(value)


@dataclass(frozen=True)
class Data:
    value: Any

    def __post_init__(self):
        value = self.value
        assert not isinstance(value, Data)
        assert not isinstance(value, _df.DFNode)


@dataclass(frozen=True)
class Context:
    scope: dict[str, Any]
    cache: dict[_df.DFNode, Data]

    def eval(self, node: _df.DFNode) -> Data:
        if node in self.cache:
            res = self.cache[node]
        else:
            data = eval_node(node, self)
            self.cache[node] = data
            res = data
        assert isinstance(res, Data)
        return res

    def nested_call(self, node: _df.DFNode, scope: dict[str, Any]) -> Data:
        nested = Context(scope=scope, cache={})
        return nested.eval(node)

    def do_loop(
        self, *values: _df.DFNode, scope: dict[str, Any]
    ) -> tuple[Data, ...]:
        nested = Context(scope=scope, cache={})
        return tuple([nested.eval(v) for v in values])


@singledispatch
def eval_node(node: _df.DFNode, ctx: Context):
    raise NotImplementedError(f"{type(node)}:\n{pformat(node)}")


@eval_node.register(_df.ArgNode)
def _eval_node_ArgNode(node: _df.ArgNode, ctx: Context):
    v = ctx.scope[node]
    if isinstance(v, Data):
        return v
    return Data(v)


@eval_node.register(_df.LiteralNode)
def _eval_node_LiteralNode(node: _df.LiteralNode, ctx: Context):
    return Data(node.py_value)


@eval_node.register(_df.EnterNode)
def _eval_node_EnterNode(node: _df.EnterNode, ctx: Context):
    scope = {k: ctx.eval(v) for k, v in node.scope.items()}
    return ctx.nested_call(node.node, scope)


@eval_node.register(_df.CaseExprNode)
def _eval_node_CaseExprNode(node: _df.CaseExprNode, ctx: Context):
    pred = ctx.eval(node.pred).value
    for case in node.cases:
        if pred == case.region.case_pred.py_value:
            return ctx.eval(case.node)
    raise AssertionError(f"no matching case for pred={pred}:\n{pformat(node)}")


@eval_node.register(_df.ExprNode)
def _eval_node_ExprNode(node: _df.ExprNode, ctx: Context):
    evaled = [ctx.eval(arg).value for arg in node.args]
    return Data(eval_op(node.op, *evaled))


@eval_node.register(_df.UnpackNode)
def _eval_node_UnpackNode(node: _df.UnpackNode, ctx: Context):
    values = ctx.eval(node.producer)
    return values.value[node.index]


@eval_node.register(_df.LoopBodyNode)
def _eval_node_LoopBodyNode(node: _df.LoopBodyNode, ctx: Context):
    scope = node.scope
    inner_scope = {k: ctx.eval(v) for k, v in scope.items()}
    while True:
        pred, *values = ctx.do_loop(*node.values, scope=inner_scope)
        if pred.value:
            inner_scope = dict(zip(scope.keys(), values))
        else:
            break
    return Data(values)


@eval_node.register(_df.CallNode)
def _eval_node_CallNode(node: _df.CallNode, ctx: Context):
    return _eval_funcdef(
        node.func.build_node(),
        *[ctx.eval(v) for v in node.args],
        **{k: ctx.eval(v) for k, v in node.kwargs.items()},
    )


# -----------------------------eval_op----------------------------------------

@singledispatch
def eval_op(op: _dt.OpTrait, *args: Any):
    raise NotImplementedError(f"no eval_op is implemented for {op}")


@eval_op.register(_dt.IntBinop)
def _(op: _dt.IntBinop, lhs, rhs):
    return op.py_impl(lhs, rhs)

