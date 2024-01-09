from pprint import pprint

from test_fac import fac_ir, fac_expect


def test_fac_llvm():
    nodes = fac_ir.build_node()
    pprint(nodes)


if __name__ == "__main__":
    test_fac_llvm()
