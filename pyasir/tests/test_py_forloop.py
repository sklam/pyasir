from pyasir import nodes as pir
from pyasir import Int64
from pyasir.interpret import interpret
from pyasir.be_llvm import generate
from pprint import pprint
from pyasir.dialects.transforms import transform


@pir.func
def forloop(n: Int64) -> Int64:
    c = 0
    iterator = pir.call(range, n)

    @pir.dialect.py.forloop
    def loop(i, c):
        return pir.pack(i, c + i)

    (i, c) = pir.unpack(loop(iterator, c))
    return i * c


def test_forloop_once():
    arg = 5
    traced = forloop.build_node()
    pprint(traced)
    # traced.to_graphviz().view()
    res = interpret(traced, arg)
    expected = sum(range(arg)) * (arg - 1)
    assert res == expected


    transformed = transform(traced)
    transformed.to_graphviz().view()

    res = interpret(transformed, arg)
    assert res == expected

    jf = generate(transformed)
    res = jf(arg)
    assert res == expected

if __name__ == "__main__":
    test_forloop_once()
