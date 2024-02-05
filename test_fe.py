from llpyfe.parser import parse_file
from llpyfe.translate import Translator


the_file = "./llpy_hello.py"
the_module = "llpy_hello"
the_func = "do_sum"

funcdef = parse_file(the_file, the_module, the_func)
print(funcdef.node)


translator = Translator()
source = translator.translate(funcdef.node)


