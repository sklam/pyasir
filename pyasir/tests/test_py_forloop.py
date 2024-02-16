from pyasir import nodes as pir
from pyasir import Int64
from pyasir.interpret import interpret
from pprint import pprint


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
    res = interpret(traced, 5)
    assert res == sum(range(5)) * 4


if __name__ == "__main__":
    test_forloop_once()
