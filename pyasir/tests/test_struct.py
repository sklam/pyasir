import pyasir
from pyasir import nodes as df
from pyasir.interpret import interpret
from pyasir.be_llvm import generate

from pyasir import Struct
from typing import no_type_check


@Struct
class MyStruct:
    x: pyasir.Int64
    y: pyasir.Float64


@df.func
def udt(x: pyasir.Int64) -> pyasir.Float64:
    y = df.cast(x, pyasir.Float64) * 0.1
    struct = df.make(MyStruct, x=x, y=y)
    return df.cast(struct.x, pyasir.Float64) + struct.y


def test():
    node = udt.build_node()

    arg = 123
    expect = interpret(node, arg)
    print(expect)
    jt = generate(node)
    got = jt(arg)
    print(got)
    assert expect == got


if __name__ == "__main__":
    test()
