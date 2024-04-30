from llpyfe.types import Int64,
from llpyfe import aot, struct

from pyasir.typedefs import pointer as pointer_api
import pyasir.nodes as _df
from pyasir.typedefs import io

@aot
def testme(n: Int64) -> Int64:
    struct_sizeof = 16
    null = pointer_api.as_pointer(0)
    res1 = pointer_api.alloc(struct_sizeof)

    if res1 == null:
        pass
    else:
        res2 = pointer_api.alloc(struct_sizeof)
        if res2 == null:
            pointer_api.free(res1)
        else:
            pointer_api.free(res2)
            pointer_api.free(res1)

    return 0


@aot
def exit(res1: Int64, res2: Int64) -> IO:
    pointer_api.free(res2)
    pointer_api.free(res1)
    return io.sync()


@aot
def task_1(args) -> IO:

    null = pointer_api.as_pointer(0)
    res1 = pointer_api.alloc(struct_sizeof)

    return io.sync()

def testme(n: Int64) -> Int64:
    res1 = io.do(task_1, n).raii(pointer_api.free)
    res2 = io.do(task_2, n, res1).raii(pointer_api.free)

    return 0

