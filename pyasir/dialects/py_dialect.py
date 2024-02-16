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
    node_replace_attrs,
    Scope,
)

from pyasir import datatypes as _dt
import pyasir
from .registry import registry
from pyasir.interpret import eval_node, Context, Data

PyDialect = SimpleNamespace()


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

        return ForLoopExprNode(loopbody.datatype,
                               region=self, loopbody=loopbody, scope=scope)


@dataclass(frozen=True, order=False)
class ForLoopExprNode(ValueNode):
    region: ForLoopNode
    loopbody: ValueNode
    scope: Scope

    def __hash__(self):
        return id(self)


PyDialect.forloop = ForLoopNode.make


registry["py"] = PyDialect


# ---------------------------------------------------------------------------


@eval_node.register
def _eval_node_ForLoopExprNode(node: ForLoopExprNode, ctx: Context):
    scope = node.scope
    inner_scope = {k: ctx.eval(v) for k, v in scope.items()}
    scope_values = list(inner_scope.values())
    iterator = iter(scope_values[0].value)

    def advance(it: Iterable) -> tuple[bool, Any]:
        try:
            ind = next(it)
        except StopIteration:
            return False, None
        else:
            return True, ind

    loop_keys = list(inner_scope.keys())

    while True:
        ok, ind = advance(iterator)
        if not ok:
            break

        loop_values = [Data(ind), *scope_values[1:]]
        scope = dict(zip(loop_keys, loop_values))
        packed_values = ctx.do_loop(node.loopbody, scope=scope)
        scope_values = packed_values.value

    return Data(tuple(scope_values))
