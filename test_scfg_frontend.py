from __future__ import annotations
import sys
import ast
from typing import Type
from textwrap import dedent
import functools

from numba_rvsdg.core.datastructures.ast_transforms import (
    unparse_code,
    AST2SCFGTransformer,
    SCFG2ASTTransformer,
    AST2SCFG,
    SCFG2AST,
)



from llpyfe.types import Int64


def sum1d(n: Int64) -> Int64:
    c = 0
    for i in range(n):
        c += i
    return c


class InsertAOTDecorator(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.Node:
        if node.name == "transformed_function":
            node.decorator_list.append(ast.Name("aot", ctx=ast.Load()))
        return node


class BailRewrite(Exception):
    pass


def query(node: ast.AST, **kwargs: dict[str, Type]):
    for attr, tyspec in kwargs.items():
        field = getattr(node, attr)
        if not isinstance(field, tyspec):
            raise BailRewrite(f"{node}.{attr} is a {type(field)} not a {tyspec}")
        node = field
    return node


def ast_parse_expr(source: str):
    tree = ast.parse(source, mode='eval')
    assert isinstance(tree, ast.Expression)
    return tree.body


def recover(fn):
    @functools.wraps(fn)
    def wrap(self, node: ast.AST) -> ast.AST:
        try:
            return fn(self, node)
        except BailRewrite as e:
            print('[Bailed]', e)
            return node
    return wrap


def maybe(fn):
    @functools.wraps(fn)
    def wrap(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BailRewrite as e:
            print('[Bailed]', e)
            return None
    return wrap


class RewriteNext(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.__body_stack = []
        self.__marked_adv_unpack = set()

    @recover
    def visit_Call(self, node: ast.Call) -> ast.Node:
        funcname: str | None = maybe(query)(node, func=ast.Name, id=str)
        if funcname == "next":
            node = self.rewrite_next_call(node)
        elif (funcname == "iter" and len(node.args) == 1
              and isinstance(node.args[0], ast.Call)
              and query(node.args[0], func=ast.Name, id=str) == 'range'):
            # iter(range(...))
            range_args = map(ast.unparse, node.args[0].args)
            node = ast_parse_expr(
                f"__pir__.call(forloop_iter, "
                f"__pir__.call(range, {', '.join(range_args)}))"
            )
        return node

    @recover
    def visit_Assign(self, node: ast.Assign) -> ast.Node:
        if maybe(query)(node, value=ast.Name, id=str) == "__scfg_undefined__":
            # rewrite ? = __scfg_undefined__
            node.value = ast_parse_expr("__pir__.zeroinit(Int64)")
            return node

        node = super().generic_visit(node)
        if (isinstance(node, ast.Assign)
                and node.value in self.__marked_adv_unpack):
            # rewrite advance() call to unpack
            orig_targets = node.targets
            node.targets = [ast.Name('__adv__', ctx=ast.Store())]
            return [
                node,
                ast.Assign([ast.Name("__adv_ok__", ctx=ast.Store())],
                           ast_parse_expr("__adv__[0]")),
                ast.Assign(orig_targets,
                           ast_parse_expr("__adv__[1]")),
            ]
        return node

    def rewrite_next_call(self, node: ast.Call) -> ast.Node:
        if query(node.args[1], value=str) != '__sentinel__':
            raise BailRewrite(f"expecting __sentinel__ got {node.args[1]}")
        arg0, arg1 = map(ast.unparse, node.args)
        magic = "'__sentinel__'"
        last_instr = self.get_parent_offset(-1)
        if not isinstance(last_instr, ast.Assign):
            raise BailRewrite(f"expect assign but got {last_instr}")
        if len(last_instr.targets) != 1:
            raise BailRewrite(f"expect single target but got {last_instr}")
        next_instr = self.get_parent_offset(+1)
        if not isinstance(next_instr, ast.If):
            raise BailRewrite(f"expect next instr to be a if but got {next_instr}")
        next_if_cmp: ast.Compare = next_instr.test
        [cmp] = next_if_cmp.comparators
        if cmp.value != '__sentinel__':
            raise BailRewrite(f"expect {magic} but got {next_instr}")
        sentinel = last_instr.targets[0].id
        arg1 = sentinel
        fmt = f"__pir__.unpack(__pir__.call(advance, {arg0}, {arg1}))"
        tree = ast_parse_expr(fmt)
        # change the next `if ? != '__sentinel__'` to `if ok`
        next_instr.test = ast.Name("__adv_ok__", ctx=ast.Load())
        self.__marked_adv_unpack.add(tree)
        return tree

    def get_parent_offset(self, offset) -> ast.Node | None:
        try:
            tos, base_idx = self.__body_stack[-1]
            return tos[base_idx + offset]
        except IndexError:
            return None

    def generic_visit(self, node):
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                self.__body_stack.append([old_value, None])
                for i, value in enumerate(old_value):
                    self.__body_stack[-1][1] = i
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                else:
                    self.__body_stack.pop()
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

class RenameFunction(ast.NodeTransformer):
    def __init__(self, orig_name: str, repl_name: str):
        self.orig_name = orig_name
        self.repl_name = repl_name

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.Node:
        if node.name == self.orig_name:
            node.name = self.repl_name
        return node


def _workaround_by_round_tripping(node: ast.AST):
    """Workaround issues that the AST produced by SCFG2ASTTransformer contains
    nested lists in code blocks.

    This issue doesn't affect ast.unparse, but can affect ast.NodeTransformer

    Fix the issue by simply rebuilding the AST from unparsed source.
    """
    what = ast.unparse(node)
    return ast.parse(what)



def restructure_source(function):
    ast2scfg_transformer = AST2SCFGTransformer(function)
    astcfg = ast2scfg_transformer.transform_to_ASTCFG()
    scfg = astcfg.to_SCFG()
    scfg.restructure()
    scfg2ast = SCFG2ASTTransformer()
    original_ast = unparse_code(function)[0]
    transformed_ast = scfg2ast.transform(original=original_ast, scfg=scfg)
    transformed_ast = _workaround_by_round_tripping(transformed_ast)
    InsertAOTDecorator().visit(transformed_ast)
    ast.fix_missing_locations(transformed_ast)
    print(ast.unparse(transformed_ast))
    RewriteNext().visit(transformed_ast)
    ast.fix_missing_locations(transformed_ast)
    RenameFunction("transformed_function", "sum1d").visit(transformed_ast)
    ast.fix_missing_locations(transformed_ast)
    return ast.unparse(transformed_ast)


def main(out_filename: str):
    source = restructure_source(sum1d)

    body = dedent(f"""
        from llpyfe.types import Int64
        from llpyfe import aot

        from pyasir.dialects.py_dialect import advance, forloop_iter
        import pyasir.nodes as _df

        from pyasir import nodes as __pir__
        """)
    body += f"\n{source}\n"

    with open(out_filename, "w") as fout:
        print(body, file=fout)


if __name__ == "__main__":
    main(out_filename=sys.argv[1])