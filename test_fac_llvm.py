from pprint import pprint

from pyasir.interpret import interpret
from pyasir.be_llvm import generate

from test_fac import fac_ir, fac_expect


def test_fac_llvm():
    node = fac_ir.build_node()
    pprint(node)
    jf = generate(node)
    for x, y in enumerate(fac_expect):
        got = jf(x)
        print(f"fac({x}) = {got}")
        assert got == y
        assert interpret(node, x) == y


if __name__ == "__main__":
    test_fac_llvm()
