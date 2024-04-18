# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pyasir
from pyasir import nodes as pir
from pyasir.interpret import interpret
from pyasir.be_llvm import generate
from pprint import pprint

# %% [markdown]
# ## Prelude
#

# %%
from IPython.display import Image, display
def render_digraph_as_png(g):
    return display(Image(g.pipe(format='png')))


# %% [markdown]
# ## ControlFlow: Switch-Case

# %%
@pir.func
def get_bigger(a: pyasir.Int64, b: pyasir.Int64) -> pyasir.Int64:
    @pir.switch(a > b)    # switch ( a > b ) {
    def swt(a, b):
        @pir.case(True)   # case 1: return a;
        def case1(a, b):
            return a
        yield case1
        
        @pir.case(False)
        def case2(a, b):  # case 0: return b;
            return b      # }
        yield case2

    return swt(a, b)

tree_get_bigger = get_bigger.build_node()

# %%
pprint(tree_get_bigger)

# %%
render_digraph_as_png(tree_get_bigger.to_graphviz())

# %%
print("bigger (123, 321)?", interpret(tree_get_bigger, 123, 321))
print("bigger (654, 456)?", interpret(tree_get_bigger, 654, 456))

# %%
jf_get_bigger = generate(tree_get_bigger, optlevel=0)
render_digraph_as_png(jf_get_bigger.to_graphviz())

# %%
jf_get_bigger(123, 321), jf_get_bigger(654, 456)


# %% [markdown]
# ## ControlFlow: Loop

# %%
@pir.func
def loop_between(begin: pyasir.Int64, end: pyasir.Int64) -> pyasir.Int64:
    c = 0
    it = begin
    
    @pir.loop   # a do-while loop
    def loop(it, c, end):
        # do {
        c += it
        it += 1
        # } while ( it < end )  # nextround: it, c, end
        return it < end, pir.pack(it, c, end)

    [it, c, _end] = pir.unpack(loop(it, c, end))
    return c

tree_loop_between = loop_between.build_node()
tree_loop_between

# %%
render_digraph_as_png(tree_loop_between.to_graphviz())

# %%
got = interpret(tree_loop_between, 2, 17)
got

# %%
jf_loop_between = generate(tree_loop_between, optlevel=0)
jf_loop_between.to_graphviz()

# %%
jf_loop_between(2, 17)

# %%

# %% [markdown]
# ## Under-the-hood

# %% [markdown]
# ### Graph is constructed by evaluation

# %%
one = pir.as_node(1)
pprint(one)
render_digraph_as_png(one.to_graphviz())

# %%
two = one + one
pprint(two)
render_digraph_as_png(two.to_graphviz())


# %%
 
@pir.loop   # a do-while loop
def loop(it, c, end):
    c += it
    it += 1
    return it < end, pir.pack(it, c, end)

looped = loop(one, one, two)
pprint(looped)

# %%
render_digraph_as_png(looped.to_graphviz())


# %% [markdown]
# ## ControlFlow: For-loop
#
# * For-loop can be expressed by a combination of do-while and if-else

# %%
@pir.func
def forloop(n: pyasir.Int64) -> pyasir.Int64:
    c = 0
    iterator = pir.call(range, n)  # for i in range(n):

    @pir.dialect.py.forloop
    def loop(i, c):                #      c = c + i
        return pir.pack(i, c + i)

    (i, c) = pir.unpack(loop(iterator, c))
    return i * c


# %%
tree_forloop = forloop.build_node()
render_digraph_as_png(tree_forloop.to_graphviz())

# %%
from pyasir.dialects.transforms import dialect_lower

tree_lowered_forloop = dialect_lower(tree_forloop)
render_digraph_as_png(tree_lowered_forloop.to_graphviz())

# %%
jf_lowered_forloop = generate(tree_lowered_forloop)
jf_lowered_forloop.to_graphviz()

# %% [markdown]
# ### Dialect Lowering

# %%
import inspect

from IPython.display import Markdown

from pyasir.dialects import py_dialect

lines, line_start = inspect.getsourcelines(py_dialect.ForLoopExprNode)
source = ''.join(lines)

Markdown(f"""
```python
{source}
```
""")

# %% [markdown]
# ## Stateful Ops

# %%
from pyasir.typedefs import pointer as pointer_api
from pyasir.typedefs import io


@pyasir.Struct
class DataNode:
    next: pyasir.Pointer #["DataNode"]
    data: pyasir.Int64


@pir.func
def process_list(n: pyasir.Int64) -> pyasir.Int64:
    struct_sizeof = 16
    null = pointer_api.as_pointer(0)

    ptr = null = pointer_api.as_pointer(0)  #null

    # Init loop
    @pir.loop
    def loop(ptr, i, n):
        node = pir.make(DataNode,  next=ptr, data=i)  # DataNode(next=ptr, data=i)
        ptr = io.sync(  # monad
            newptr := pointer_api.alloc(struct_sizeof), # newptr = malloc
            pointer_api.store(newptr, node),            # *newptr = node
            newptr,
        )
        i += 1
        return i < n, pir.pack(ptr, i, n)

    (ptr, i, n) = pir.unpack(loop(ptr, 0, n))

    # Calc loop
    c = 0

    @pir.loop
    def loop(ptr, c):
        null = pointer_api.as_pointer(0)

        cur = io.sync(  # do monad
            cur := pointer_api.load(DataNode, ptr),
            pointer_api.free(ptr), # -> IO
            cur,
        )
        ptr = cur.next
        c += cur.data
        return ptr != null, pir.pack(ptr, c)

    (ptr, c) = pir.unpack(loop(ptr, c))

    return c

    
tree_process_list = process_list.build_node()
render_digraph_as_png(tree_process_list.to_graphviz())

# %%
jf_process_list = generate(tree_process_list)
jf_process_list.to_graphviz()

# %%
jf_process_list(10)

# %%
interpret(tree_process_list, 10)

# %%
