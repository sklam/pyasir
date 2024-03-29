from __future__ import annotations

from functools import singledispatch


import graphviz as gv
import dataclasses as _dc

import pyasir.nodes as _df
import pyasir.datatypes as _dt


def node_as_graphviz(node: _df.DFNode):
    g = gv.Digraph()
    ctx = Context()
    gv_node(node, g, ctx=ctx)
    ctx.finish(g)
    return g


def get_node_name(node) -> str:
    return f"{node.__class__.__name__}@{hex(id(node))}"


@_dc.dataclass(frozen=True)
class Context:
    cache: set[int] = _dc.field(default_factory=set)
    edges: list[tuple] = _dc.field(default_factory=list)

    def check_cache(self, node_id: str):
        cache = self.cache
        if node_id in cache:
            return False
        else:
            cache.add(node_id)
            return True

    def add_edge(self, *args, **kwargs):
        self.edges.append((args, kwargs))

    def finish(self, g):
        for args, kwargs in self.edges:
            g.edge(*args, **kwargs)



@singledispatch
def gv_node(node, g: gv.Digraph, ctx: Context):
    this = get_node_name(node)
    if ctx.check_cache(this):
        g.node(this, label=gv.escape(repr(node)), shape='oval')


@gv_node.register
def _(node: _df.DFNode, g: gv.Digraph, ctx):
    this = get_node_name(node)
    if ctx.check_cache(this):
        if isinstance(node, _df.CaseExprNode):
            g.node(this, label=f"CaseExprNode", shape='rect')
            items = enumerate(node.cases)
            render_items(this, items, g, ctx)
            return

        elif isinstance(node, _df.PackNode):
            g.node(this, label=f"PackNode", shape='rect')
            items = enumerate(node.values)
            render_items(this, items, g, ctx)
            return

        elif isinstance(node, _df.CallNode):
            g.node(this, label=f"CallNode(func={node.func})", shape='rect')
            items = enumerate(node.args)
            render_items(this, items, g, ctx)
            items = node.kwargs.items()
            render_items(this, items, g, ctx)
            return

        elif isinstance(node, _df.LiteralNode):
            g.node(this, label=repr(node), shape='rect')
            return

        elif isinstance(node, _df.ExprNode):
            g.node(this, label=f"{node.__class__.__name__} {node.op}", shape='rect')
            items = enumerate(node.args)
            render_items(this, items, g, ctx)
            return

        elif isinstance(node, _df.UnpackNode):
            g.node(this, label=f"{node.__class__.__name__} {node.index}", shape='rect')
            items = [('producer', node.producer)]
            render_items(this, items, g, ctx)
            return

        elif isinstance(node, _df.ArgNode):
            g.node(this, label=repr(node), shape='rect')
            return

        else:
            g.node(this, label=f"{node.__class__.__name__}", shape='rect')

            items = [(fd.name, getattr(node, fd.name)) for fd in _dc.fields(node)]
            if isinstance(node, _df.EnterNode):
                sub_items = dict(items)
                scope_item = sub_items.pop('scope')
                render_items(this, [('scope', scope_item)], g, ctx)
                with g.subgraph(name=f"cluster_{this}",
                                graph_attr={'style': 'dashed'}) as g:
                    render_items(this, list(sub_items.items()), g, ctx)
            else:
                render_items(this, items, g, ctx)


@gv_node.register
def _(node: _df.Scope, g: gv.Digraph, ctx):
    this = get_node_name(node)
    if ctx.check_cache(this):
        g.node(this, label=f"{node.__class__.__name__}", shape='rect')

        items = list(node.items())
        render_items(this, [(k.name, v) for k, v in items], g, ctx)

        for argnode, argvalue in items:
            ctx.add_edge(get_node_name(argnode), get_node_name(argvalue), style='dotted')


def render_items(this, items, g, ctx):
    for k, v in items:
        if not isinstance(v, _dt.DataType):
            gv_node(v, g, ctx)
            ctx.add_edge(this, get_node_name(v), taillabel=gv.escape(str(k)))

