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
    null = pointer_api.as_pointer(0)
    alloc = pointer_api.alloc(struct_sizeof)
    head = _df.make(List,  next_ptr=alloc, data=123)
    node1 = _df.make(List, next_ptr=null, data=123 + n)
    pointer_api.store(head.next_ptr, node1)
    curnode = head
    out = 0
    cont = True

    while cont:
        out = out + curnode.data
        cont = curnode.next_ptr != null
        if cont:
            curnode = pointer_api.load(List, curnode.next_ptr)

    pointer_api.free(head.next_ptr)
    return out

