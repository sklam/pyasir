from pyasir import nodes as df
from pyasir import datatypes as dt
from pyasir.interpret import interpret
from pprint import pprint


def fac_py(n):
    y = 1
    while n > 1:
        y *= n
        n -= 1
    return y


fac_expect = expect = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]


def test_fac_py():
    for i, expect in enumerate(fac_expect):
        got = fac_py(i)
        assert expect == got


@df.func
def fac_ir(n: dt.Int64) -> dt.Int64:
    y = 1

    @df.switch(n > 1)
    def swt(n, y):
        @df.case(1)
        def do_loop(n, y):
            @df.loop
            def loop(n, y):
                y *= n
                n -= 1
                return n > 1, (n, y)

            n, y = loop(n, y)
            return y

        @df.case(0)
        def dont(n, y):
            return y

        yield do_loop
        yield dont

    y = swt(n, y)
    return y


def test_fac_ir_once():
    traced = fac_ir.build_node()
    pprint(traced)
    assert interpret(traced, 5) == fac_expect[5]


def test_fac_ir():
    traced = fac_ir.build_node()
    for i, expect in enumerate(fac_expect):
        got = interpret(traced, i)
        assert expect == got


if __name__ == "__main__":
    test_fac_ir_once()
