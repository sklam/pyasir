from mypy import build
from mypy.main import process_options
from pathlib import Path


def parse_file(the_file, the_module, the_func):
    files, opt = process_options([the_file])
    # # Ensure that the ASTs are preserved
    opt.preserve_asts = True
    # opt.show_traceback = True
    opt.export_types = True
    # # Revert fine_grained_incremental to False to avoid using cached ASTs
    opt.fine_grained_incremental = True

    # Build the result object which contains the ASTs
    result = build.build(files, options=opt)
    # Access the AST for a specific module
    module_graph = result.graph[the_module]
    fn = module_graph.tree.names[the_func]
    return fn
