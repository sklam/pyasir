from pyasir import nodes as df
from pyasir import datatypes as pyasir
from pyasir.interpret import interpret
import pyasir
from pprint import pprint


def fib_py(n):
    if n <= 1:
        return 1
    else:
        return fib_py(n - 1) + fib_py(n - 2)


fib_expect = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]


def test_fib_py():
    for i, expect in enumerate(fib_expect):
        got = fib_py(i)
        assert expect == got


@df.func
def fib_ir(n: pyasir.Int64) -> pyasir.Int64:
    @df.switch(n <= 1)
    def swt(n):
        @df.case(1)
        def case0(n):
            return 1

        @df.case(0)
        def case1(n):
            return fib_ir(n - 1) + fib_ir(n - 2)

        yield case0
        yield case1

    r = swt(n)
    return r


def test_fib_ir_once():
    traced = fib_ir.build_node()
    pprint(traced)
    res = interpret(traced, 5)
    assert res == fib_expect[5]


def test_fib_ir():
    traced = fib_ir.build_node()

    for i, expect in enumerate(fib_expect):
        got = interpret(traced, i)
        assert expect == got


if __name__ == "__main__":
    test_fib_ir()
