from pprint import pprint

from pyasir.be_llvm import generate
from pyasir.interpret import interpret
from pyasir import nodes as df
from pyasir import datatypes as dt

from .test_fib import fib_ir, fib_expect


def test_fib_llvm():
    node = fib_ir.build_node()
    print(type(node))
    pprint(node)

    jf = generate(node)

    for x, y in enumerate(fib_expect):
        got = jf(x)
        print(f"fib({x}) = {got}")
        assert got == y
        assert interpret(node, x) == y


def fib_tail(n, accum1=1, accum2=1):
    # from https://eli.thegreenplace.net/2017/on-recursion-continuations-and-trampolines/
    if n < 2:
        return accum1
    else:
        return fib_tail(n - 1, accum1 + accum2, accum1)


@df.func
def _fib_tail_ir(n: dt.Int64, accum1: dt.Int64, accum2: dt.Int64) -> dt.Int64:
    # IR version of fib_tail
    @df.switch(n < 2)
    def swt(n, accum1, accum2):
        @df.case(True)
        def case1(n, accum1, accum2):
            return accum1

        @df.case(False)
        def case0(n, accum1, accum2):
            return _fib_tail_ir(n - 1, accum1 + accum2, accum1)

        yield case1
        yield case0

    return swt(n, accum1, accum2)


@df.func
def fib_tail_ir(n: dt.Int64) -> dt.Int64:
    return _fib_tail_ir(n, 1, 1)


def test_fib_tail_recur_llvm():
    node = fib_tail_ir.build_node()
    print(type(node))
    pprint(node)

    jf = generate(node)

    for x, y in enumerate(fib_expect):
        got = jf(x)
        print(f"fib({x}) = {got}")
        assert got == y

        assert interpret(node, x) == y


if __name__ == "__main__":
    test_fib_tail_recur_llvm()
