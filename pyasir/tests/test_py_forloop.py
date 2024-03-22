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
    traced = forloop.build_node()
    pprint(traced)
    # traced.to_graphviz().view()
    res = interpret(traced, 5)
    assert res == sum(range(5)) * 4


    transformed = transform(traced)
    transformed.to_graphviz().view()

    got = interpret(transformed, 5)
    print("GOT", got, res)

    jf = generate(transformed)
    print(jf)
    print("LLVM GOT", jf(5), res)


if __name__ == "__main__":
    test_forloop_once()
