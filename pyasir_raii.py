from pprint import pprint

from pyasir.typedefs import pointer as pointer_api
import pyasir.nodes as _df
from pyasir.typedefs import io, Int64, Packed, Pointer
from pyasir.interpret import interpret
from pyasir.be_llvm import generate


free = lambda x: pointer_api.free(x)

@_df.func
def testme(n: Int64) -> Int64:
    state = io.seq()
    null = pointer_api.as_pointer(0)

    lifetime = io.RAII(state)

    non_null = lambda x: x != null
    free = lambda x: pointer_api.free(x)

    res1 = lifetime.do(
        lambda: pointer_api.alloc(n),
        cleanup=free,
        expect=non_null
    )
    lifetime.do(
        lambda: pointer_api.store(res1, _df.cast(123, Int64))
    )

    res2 = lifetime.do(
        lambda: pointer_api.alloc(n),
        cleanup=free,
        expect=non_null
    )

    lifetime.do(
        lambda: pointer_api.store(res2, _df.cast(234, Int64))
    )

    res3 = lifetime.do(
        lambda: pointer_api.alloc(n),
        cleanup=free,
        expect=non_null
    )


    lifetime.do(
        lambda: pointer_api.store(res3, _df.cast(345, Int64))
    )

    res4 = lifetime.do(
        lambda: pointer_api.alloc(n),
        cleanup=free,
        expect=non_null
    )

    lifetime.do(
        lambda: pointer_api.store(res4, _df.cast(456, Int64))
    )

    out = lifetime.do(
        lambda: (
            pointer_api.load(Int64, res1) + pointer_api.load(Int64, res2)
            + pointer_api.load(Int64, res3) + pointer_api.load(Int64, res4)
        )
    )

    return lifetime.complete(out)



def test_testeme():
    traced = testme.build_node()
    got = interpret(traced, 8)
    print('got', got)
    jf = generate(traced)
    print('llvm', jf(8))


if __name__ == "__main__":
    test_testeme()
