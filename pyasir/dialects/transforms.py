from __future__ import annotations

from dataclasses import dataclass, field
from pprint import pprint
from functools import singledispatch
from typing import Any, Callable

import pyasir.nodes as _df



def lift(expr: _df.DFNode) -> tuple[_df.DFNode, dict[_df.ArgNode, _df.ArgNode]]:
    """Lift expression to return a copy of the `expr` with all
    argument node replaced with fresh ones.

    Returns
    -------
    (result, arg_map)
        result: The lifted nodes
        arg_map: Mapping from old ArgNode to new ArgNode.
    """
    new_args = []
    arg_map = {}

    def transformer(node: _df.DFNode):
        if isinstance(node, _df.ArgNode):
            repl: _df.ArgNode = arg_map.get(node)
            assert repl is not node
            if repl is None:
                repl = node.clone()
                new_args.append(repl)
                arg_map[node] = repl
            return repl
        else:
            return None   # to descent

    ctx = TransformerContext(transformer=transformer)
    result = transform_visitor(expr, ctx)
    return result, arg_map


def dialect_lower(root: _df.FuncDef):
    def transformer(node: _df.DFNode):
        transform = getattr(node, "dialect_lower", None)
        if transform is None:
            return None
        return transform()

    ctx = TransformerContext(transformer=transformer)
    result = ctx.visit(root)
    return result


@dataclass(frozen=True)
class TransformerContext:
    transformer: Callable[[_df.DFNode], _df.DFNode | None]
    repl_cache: dict[int, Any] = field(default_factory=dict)


    def visit(self, node: _df.FuncDef):
        return transform_visitor(node, self)

    def visit_child(self, node: _df.DFNode, name: str, parent: _df.DFNode):
        print(f"--visit[{type(node)}] {type(parent)}.{name}")
        key = id(node)
        if key in self.repl_cache:
            out = self.repl_cache[key]
        else:
            out = transform_visitor(node, self)
            self.repl_cache[key] = out
            if out is not node:
                print("<-replaced", type(out))
        return out


@singledispatch
def transform_visitor(node: object, ctx: TransformerContext):
    raise TypeError(type(node))


@transform_visitor.register
def _(node: tuple, ctx: TransformerContext):
    return tuple(transform_visitor(item, ctx) for item in node)

@transform_visitor.register
def _(node: _df.DFNode, ctx: TransformerContext):
    result = ctx.transformer(node)
    if result is None:
        # Recursively apply to children
        repl = {}
        for name, child in node.get_child_nodes().items():
            repl[name] = ctx.visit_child(child, name=name, parent=node)
        return node.replace_child_nodes(repl)
    else:
        return result

