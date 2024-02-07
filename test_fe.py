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
    funcdef = parse_file(the_file, the_module, the_func)
    print(funcdef.node)

    src_buffer = []

    translator = Translator()
    src_buffer.append(translator.get_import_lines())
    src_buffer.append(translator.translate(funcdef.node))

    source = '\n'.join(src_buffer)
    print(source)
    return source


def run_function(source):
    global_dict = local_dict = {}
    exec(source, global_dict, local_dict)
    func = global_dict[the_func]
    node = func.build_node()
    # pprint(node)

    res = interpret(node, 5)
    return res


def test():
    source = fe_compile()
    res = run_function(source)
    assert res == sum(range(5)) + 100


if __name__ == "__main__":
    test()
