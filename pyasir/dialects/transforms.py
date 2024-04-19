from __future__ import annotations

from dataclasses import dataclass, field
from pprint import pprint
from functools import singledispatch
from typing import Any, Callable

import pyasir.nodes as _df


def lift_and_inline(expr: _df.EnterNode, args: _df.ValueNode):
    assert isinstance(expr, _df.EnterNode)
    lifted, argmap = lift(expr.body, expr.scope)
    return _df.EnterNode.make(
        lifted, _df.Scope(dict(zip(argmap.values(), _df.as_node_args(args), strict=True)))
    )


def lift(expr: _df.DFNode, old_scope: _df.Scope):
    arg_map = {k: k.clone() for k in old_scope}

    def transformer(node: _df.DFNode):
        if isinstance(node, _df.ArgNode):
            return arg_map[node]
        else:
            return None  # to descent

    ctx = TransformerContext(transformer=transformer)
    result = transform_visitor(expr, ctx)
    return result, arg_map


def dialect_lower(root: _df.FuncDef):
    def transformer(node: _df.DFNode):
        transform = getattr(node, "dialect_lower", None)
        if transform is None:
            return None
        return transform()

    ctx = TransformerContext(transformer=transformer, recurse=True)
    result = ctx.visit(root)
    return result


@dataclass(frozen=True)
class TransformerContext:
    transformer: Callable[[_df.DFNode], _df.DFNode | None]
    repl_cache: dict[Any, Any] = field(default_factory=dict)
    recurse: bool = False

    def visit(self, node: _df.FuncDef):
        return transform_visitor(node, self)

    def visit_child(self, node: _df.DFNode, name: str, parent: _df.DFNode):
        print(f"--visit[{type(node)}] {type(parent)}.{name}")
        if node in self.repl_cache:
            out = self.repl_cache[node]
        else:
            old = node
            while True:
                out = transform_visitor(node, self)
                if out != node:
                    print("<-replaced", type(out))
                    node = out
                    if not self.recurse:
                        break
                else:
                    out = node
                    break
            self.repl_cache[old] = out
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
        repl_count = 0
        for name, child in node.get_child_nodes().items():
            repl[name] = new = ctx.visit_child(child, name=name, parent=node)
            if child != new:
                repl_count += 1
        return node.replace_child_nodes(repl) if repl_count else node
    else:
        return result

@transform_visitor.register
def _(node: _df.Scope, ctx: TransformerContext):
    result = ctx.transformer(node)
    if result is None:
        # Recursively apply to children
        repl = {}
        repl_count = 0
        for k, v in node.items():
            repl[k] = new = ctx.visit_child(v, name=k, parent=node)
            if new is not v:
                repl_count += 1

        return _df.Scope(repl) if repl_count else node
    else:
        return result
