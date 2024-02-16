from pprint import pprint

import llpyfe
import llpyfe.types
from llpyfe.parser import parse_file
from llpyfe.translate import Translator
from pyasir.interpret import interpret
from pyasir.be_llvm import generate


the_file = "./llpy_list.py"
the_module = "llpy_list"
the_func = "testme"


def fe_compile():
    module_graph = parse_file(the_file, the_module, the_func)
    print(module_graph.tree)

    translator = Translator()
    source = translator.translate_file(module_graph.tree)
    print(source)
    return source


def run_function(source, args):
    global_dict = local_dict = {}
    exec(source, global_dict, local_dict)
    func = global_dict[the_func]
    node = func.build_node()
    # pprint(node)

    res = interpret(node, *args)

    jf = generate(node)
    llvm_res = jf(*args)
    assert llvm_res == res
    return res


def test():
    source = fe_compile()
    args = (5,)
    res = run_function(source, args)
    print("res =", res)


    # from llpy_hello import do_sum
    # assert res == do_sum(*args)


if __name__ == "__main__":
    test()
