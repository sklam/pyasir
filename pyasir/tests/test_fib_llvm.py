from pprint import pprint

import pyasir
from pyasir.be_llvm import generate
from pyasir.interpret import interpret
from pyasir import nodes as df

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
    # From https://eli.thegreenplace.net/2017/on-recursion-continuations-and-trampolines/
    # License: public domain
    # See https://eli.thegreenplace.net/pages/about
    # > These days I manage my open-source projects on GitHub; they all have
    # > very liberal licenses. Some of the blog posts contain code; unless
    # > otherwise stated, all of it is in the public domain.
    if n < 2:
        return accum1
    else:
        return fib_tail(n - 1, accum1 + accum2, accum1)


@df.func
def _fib_tail_ir(
    n: pyasir.Int64, accum1: pyasir.Int64, accum2: pyasir.Int64
) -> pyasir.Int64:
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
def fib_tail_ir(n: pyasir.Int64) -> pyasir.Int64:
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
    test_fib_llvm()
    # test_fib_tail_recur_llvm()
