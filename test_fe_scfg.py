from pprint import pprint

import llpyfe
import llpyfe.types
from llpyfe.parser import parse_file
from llpyfe.translate import Translator
from pyasir.interpret import interpret
from pyasir.be_llvm import generate


the_file = "./llpy_scfg.py"
the_module = "llpy_scfg"
the_func = "sum1d"


def fe_compile():
    module_graph = parse_file(the_file, the_module, the_func)
    print(module_graph.tree)

    translator = Translator()
    source = translator.translate_file(module_graph.tree)
    return source


def run_function(source, args):
    with open("test_source.py", "w") as fout:
        print(source, file=fout)

    from test_source import sum1d as func
    node = func.build_node()
    node.to_graphviz().view()
    # pprint(node)

    print("BEGIN INTERPRET")
    res = interpret(node, *args)
    print("END INTERPRET")
    print("res", res)

    print("BEGIN GENERATE")
    jf = generate(node, optlevel=1)
    print("END GENERATE")
    jf.get_cfg().view()
    print("LLVM RUNNING")
    llvm_res = jf(*args)
    print(jf.llmod)
    assert llvm_res == res
    return res


def test():
    source = fe_compile()
    args = (10,)
    res = run_function(source, args)
    print("res =", res)




if __name__ == "__main__":
    test()
