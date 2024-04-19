from __future__ import annotations

from dataclasses import dataclass, field
from pprint import pprint
from functools import singledispatch
from typing import Any, Callable
import logging

import pyasir.nodes as _df


_logger = logging.getLogger(__name__)


def lift_and_inline(expr: _df.EnterNode, args: _df.ValueNode):
    assert isinstance(expr, _df.EnterNode)
    lifted, argmap = lift(expr.body, expr.scope)
    return _df.EnterNode.make(
        lifted,
        _df.Scope(
            dict(zip(argmap.values(), _df.as_node_args(args), strict=True))
        ),
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


@dataclass
class TransformerContext:
    transformer: Callable[[_df.DFNode], _df.DFNode | None]
    """Transformer function to apply
    """
    recurse: bool = False
    """Recursively apply transformation
    """
    # Internal states
    _memo: dict[Any, Any] = field(default_factory=dict)
    """Memoize replacements
    """
    _run_depth: int = 0
    """Run depth for debugging
    """

    def __post_init__(self):
        _logger.debug("Transform %s @ %s", self, hex(id(self)))

    def visit(self, node: _df.FuncDef):
        return transform_visitor(node, self)

    def visit_child(self, node: _df.DFNode, name: str, parent: _df.DFNode):
        self._run_depth += 1
        depth = self._run_depth
        try:
            _logger.debug(
                "%s [%d] visit %s, parent %s, field %s",
                hex(id(self)),
                depth,
                type(node).__name__,
                type(parent).__name__,
                name,
            )
            if node in self._memo:
                out = self._memo[node]

                _logger.debug(
                    "%s [%d] ... (cache) replaced with %s",
                    hex(id(self)),
                    depth,
                    type(out).__name__,
                )
            else:
                old = node
                while True:
                    out = transform_visitor(node, self)
                    if out != node:
                        _logger.debug(
                            "%s [%d] ... replaced with %s",
                            hex(id(self)),
                            depth,
                            type(out).__name__,
                        )
                        node = out
                        if not self.recurse:
                            break
                    else:
                        out = node
                        break
                self._memo[old] = out
        finally:
            self._run_depth -= 1
        return out


@singledispatch
def transform_visitor(node: object, ctx: TransformerContext):
    raise TypeError(type(node))


@transform_visitor.register
def _(node: tuple, ctx: TransformerContext):
    return tuple(
        ctx.visit_child(item, name=str(i), parent=node)
        for i, item in enumerate(node)
    )


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
            repl[k] = new = ctx.visit_child(v, name=k.name, parent=node)
            if new is not v:
                repl_count += 1

        return _df.Scope(repl) if repl_count else node
    else:
        return result
