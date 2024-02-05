from __future__ import annotations

from typing import Any, Callable, Iterable
from types import SimpleNamespace
from dataclasses import dataclass
from inspect import signature
from pprint import pprint

from pyasir.nodes import RegionNode, EnterNode, DFNode, ArgNode, ValueNode, UnpackNode
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
        from ..nodes import as_node

        [args, kwargs, sig, ty_args] = self._pre_call(self.region_func, args, kwargs)

        nodes = self._call_region(self.region_func, (ty_args[0].element, *ty_args[1:]))
        ind, *values = nodes
        body_values = tuple(map(as_node, [ind, *values]))

        scope = self._prepare_scope(sig, args, kwargs)
        loopbody = ForLoopBodyNode(pyasir.Packed(), body_values, scope)
        value_nodes = tuple(
            [
                UnpackNode(loopbody.values[i].datatype, loopbody, i)
                for i in range(len(body_values))
            ]
        )
        return ForLoopExprNode(self, values=value_nodes)


@dataclass(frozen=True, order=False)
class ForLoopBodyNode(DFNode):
    datatype: _dt.DataType
    values: tuple[ValueNode, ...]
    scope: dict

    def __hash__(self):
        return id(self)


@dataclass(frozen=True, order=False)
class ForLoopExprNode(DFNode):
    parent: ForLoopNode
    values: tuple[ValueNode, ...]

    def __iter__(self):
        return iter(self.values)



PyDialect.forloop = ForLoopNode.make


registry['py'] = PyDialect



# ---------------------------------------------------------------------------

@eval_node.register
def _eval_node_ForLoopBodyNode(node: ForLoopBodyNode, ctx: Context):
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
    first_key = loop_keys[0]
    loop_keys[0] = ArgNode(datatype=first_key.datatype.element,
                           name=first_key.name)

    while True:
        ok, ind = advance(iterator)
        if not ok:
            break

        loop_values = [Data(ind), *scope_values[1:]]
        scope = dict(zip(loop_keys, loop_values))
        scope_values = ctx.do_loop(*node.values, scope=scope)

    return Data(tuple(scope_values))

