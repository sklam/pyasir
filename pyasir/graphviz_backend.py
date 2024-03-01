from __future__ import annotations

from functools import singledispatch


import graphviz as gv
import dataclasses as _dc

import pyasir.nodes as _df
import pyasir.datatypes as _dt


def node_as_graphviz(node: _df.DFNode):
    g = gv.Digraph()
    gv_node(node, g, cache=set())
    return g


def get_node_name(node) -> str:
    return f"{node.__class__}@{hex(id(node))}"



def check_cache(cache: set, node_id: str):
    if node_id in cache:
        return False
    else:
        cache.add(node_id)
        return True



@singledispatch
def gv_node(node, g: gv.Digraph, cache: set):
    this = get_node_name(node)
    if check_cache(cache, this):
        g.node(this, label=gv.escape(repr(node)), shape='oval')


@gv_node.register
def _(node: tuple, g: gv.Digraph, cache):
    this = get_node_name(node)
    if check_cache(cache, this):
        g.node(this, label="tuple", shape='rect')
        items = enumerate(node)
        render_items(this, items, g, cache)


@gv_node.register
def _(node: _df.DFNode, g: gv.Digraph, cache):
    this = get_node_name(node)
    if check_cache(cache, this):
        if isinstance(node, _df.LiteralNode):
            g.node(this, label=repr(node), shape='rect')
            return

        elif isinstance(node, _df.ExprNode):
            g.node(this, label=f"{node.__class__.__name__} {node.op}", shape='rect')
            items = enumerate(node.args)
            render_items(this, items, g, cache)
            return

        elif isinstance(node, _df.UnpackNode):
            g.node(this, label=f"{node.__class__.__name__} {node.index}", shape='rect')
            items = [('producer', node.producer)]
            render_items(this, items, g, cache)
            return

        elif isinstance(node, _df.ArgNode):
            g.node(this, label=repr(node), shape='rect')
            return

        else:
            g.node(this, label=f"{node.__class__.__name__}", shape='rect')

            items = [(fd.name, getattr(node, fd.name)) for fd in _dc.fields(node)]
            if isinstance(node, _df.EnterNode):
                with g.subgraph(name=f"cluster_{this}") as g:
                    render_items(this, items, g, cache)
            else:
                render_items(this, items, g, cache)


@gv_node.register
def _(node: _df.Scope, g: gv.Digraph, cache):
    this = get_node_name(node)
    if check_cache(cache, this):
        g.node(this, label=f"{node.__class__.__name__}", shape='rect')

        items = node.items()
        render_items(this, items, g, cache)


def render_items(this, items, g, cache):
    for k, v in items:
        if not isinstance(v, _dt.DataType):
            gv_node(v, g, cache)
            g.edge(this, get_node_name(v), taillabel=gv.escape(str(k)))

