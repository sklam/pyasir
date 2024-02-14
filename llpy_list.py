from llpyfe.types import Int64, Pointer
from llpyfe import aot, struct

from pyasir.typedefs import pointer as pointer_api
import pyasir.nodes as _df
from pyasir.typedefs import io

@struct
class List:
    # singlely linked list
    next_ptr: Pointer["List"]
    data: Int64



@aot
def testme(n: Int64) -> Int64:
    struct_sizeof = 16

    head = _df.make(List,  next_ptr=pointer_api.alloc(struct_sizeof), data=123)
    orig_head = head
    node1 = _df.make(List, next_ptr=0, data=123 + n)
    curnode = io.sync(
        pointer_api.store(head.next_ptr, node1),
        head,
    )

    out = 0
    cont = True
    while cont:

        out += curnode.data
        cont = False

        if curnode.next_ptr != 0:
            curnode = pointer_api.load(List, curnode.next_ptr)
            cont = io.sync(curnode, True)


    return io.sync(out, pointer_api.free(orig_head.next_ptr), out)


