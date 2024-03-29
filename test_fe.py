from pprint import pprint

import llpyfe
import llpyfe.types
from llpyfe.parser import parse_file
from llpyfe.translate import Translator
from pyasir.interpret import interpret


the_file = "./llpy_hello.py"
the_module = "llpy_hello"
the_func = "do_sum"


def fe_compile():
    module_graph = parse_file(the_file, the_module, the_func)

    funcdef = module_graph.tree.names[the_func]
    print(funcdef.node)

    src_buffer = []

    translator = Translator()
    src_buffer.append(translator.get_import_lines())
    src_buffer.append(translator.translate(funcdef.node))

    source = '\n'.join(src_buffer)
    print(source)
    return source


def run_function(source, args):
    global_dict = local_dict = {}
    exec(source, global_dict, local_dict)
    func = global_dict[the_func]
    node = func.build_node()
    # pprint(node)

    res = interpret(node, *args)
    return res


def test():
    source = fe_compile()
    args = (5,)
    res = run_function(source, args)

    from llpy_hello import do_sum
    assert res == do_sum(*args)


if __name__ == "__main__":
    test()
