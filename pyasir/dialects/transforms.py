from __future__ import annotations

from dataclasses import dataclass, field
from pprint import pprint
from functools import singledispatch
from typing import Any

import pyasir.nodes as _df


def transform(root: _df.FuncDef):
    ctx = TransformerContext()
    result = ctx.visit(root)
    return result


@dataclass(frozen=True)
class TransformerContext:
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
    repl = {}
    for name, child in node.get_child_nodes().items():
        repl[name] = ctx.visit_child(child, name=name, parent=node)
    return node.replace_child_nodes(repl)


@transform_visitor.register
def _(node: _df.DialectMixin, ctx: TransformerContext):
    transform = getattr(node, "transform", None)
    if transform is None:
        raise TypeError("dialectic undefined", type(node))
    return transform()

