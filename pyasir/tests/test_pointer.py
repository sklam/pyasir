import pyasir
from pyasir import nodes as pir
from pyasir.interpret import interpret
from pyasir.be_llvm import generate
from pprint import pprint
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
        ptr = cur.attrs.next
        c += cur.attrs.data
        return ptr != null, pir.pack(ptr, c)

    (ptr, c) = pir.unpack(loop(ptr, c))

    return c


def test_pointer_interpret():
    tree = process_list.build_node()
    res = interpret(tree, 10)
    assert res == sum(range(10))



def test_pointer_llvm():
    tree = process_list.build_node()
    jf = generate(tree)
    res = jf(10)
    assert res == sum(range(10))


