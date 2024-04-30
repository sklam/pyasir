from pyasir import nodes as pir
from pyasir import Int64
from pyasir.interpret import interpret
from pyasir.be_llvm import generate
from pprint import pprint
from pyasir.dialects.transforms import dialect_lower


import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger('pyasir').setLevel(logging.DEBUG)


@pir.func
def forloop(n: Int64) -> Int64:
    c = 0
    iterator = pir.call(range, n)

    @pir.dialect.py.forloop
    def loop(i, c):
        return pir.pack(i, c + i)

    out = loop(iterator, c)
    (i, c) = pir.unpack(out)
    return i * c


def forloop_expected(n):
    c = 0
    for i in range(n):
        c += i
    return i * c


def test_forloop_once():
    arg = 1
    traced = forloop.build_node()
    pprint(traced)
    # traced.to_graphviz().view()
    res = interpret(traced, arg)
    print(res)
    expected = forloop_expected(arg)
    print("res", res, '??', expected)
    assert res == expected


    transformed = dialect_lower(traced)
    # # transformed.to_graphviz().view()

    res = interpret(transformed, arg)
    print("res", res, '??', expected)
    assert res == expected

    jf = generate(transformed)
    res = jf(arg)
    assert res == expected
    transformed.to_graphviz().view()
    jf.get_cfg().view()

if __name__ == "__main__":
    test_forloop_once()
