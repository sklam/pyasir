from __future__ import annotations

import importlib
import mypy.nodes as _mypy
import mypy.types as _mypyt
import textwrap
import re
import ast
from dataclasses import dataclass
from functools import singledispatch
from typing import NamedTuple



_REGEX_AST_TEMPLATE = re.compile(r"\$[a-zA-Z_0-9]+")


@dataclass(frozen=True)
class IndentedBlock:
    statements: list[ast.AST]

    def __post_init__(self):
        copied = self.statements.copy()
        self.statements.clear()
        for stmt in copied:
            if isinstance(stmt, IndentedBlock):
                self.statements.extend(stmt.statements)
            else:
                self.statements.append(stmt)


_ASTLike = IndentedBlock | ast.AST
_TemplateReplVal = str | IndentedBlock | ast.AST


def ast_parse_eval(text: str) -> _ASTLike:
    return ast.parse(text, mode="eval")


def ast_template(source: str, repl: dict[str, _TemplateReplVal]) -> _ASTLike:
    def match(m: str):
        out = repl[m.group(0)]
        if isinstance(out, ast.AST):
            return ast.unparse(out)
        elif isinstance(out, IndentedBlock):
            INDENT = " " * 4
            return textwrap.indent(
                "\n".join(map(ast.unparse, out.statements)), INDENT
            ).lstrip()
        else:
            assert isinstance(out, str), type(out)
            return out

    replaced = _REGEX_AST_TEMPLATE.sub(match, source)
    tree = ast.parse(replaced.replace("$", ""))

    if isinstance(tree, ast.Module):
        if len(tree.body) == 1:
            [stmt] = tree.body
            return stmt
        else:
            return IndentedBlock(tree.body)
    return tree


class Translator:
    def translate(self, fndef: _mypy.FuncDef) -> str:
        tree = mypy_to_ast(fndef)
        print("ast.dump".center(80, "-"))
        print(ast.dump(tree, indent=4))
        print("=" * 80)
        source = ast.unparse(tree)
        print("source".center(80, "-"))
        print(source)
        print("=" * 80)
        return source

    def get_import_lines(self) -> str:
        return """
import pyasir
from pyasir import nodes as __pir__
"""


def mypy_to_ast(tree: _mypy.Node) -> _ASTLike:
    out = _mypy_to_ast(tree)
    assert isinstance(out, (IndentedBlock, ast.AST))
    assert not isinstance(out, ast.Module), f"---- {type(tree)}"
    return out




class FindLoadedNames(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.__modified_vars = set()

    def get(self) -> set[str]:
        return frozenset(self.__modified_vars)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            self.__modified_vars.add(node.id)



def find_loaded_names(block: list[ast.stmt]) -> frozenset[str]:
    finder = FindLoadedNames()
    for b in block:
        finder.visit(b)
    return finder.get()


def load_pyasir_type(tyname: str):
    assert isinstance(tyname, str)
    module_path, obj_name = tyname.rsplit('.', 1)
    module = importlib.import_module(module_path)
    tyclass = getattr(module, obj_name)
    return tyclass.__pyasir_type__


@singledispatch
def _mypy_to_ast(tree: _mypy.Node) -> _ASTLike:
    raise NotImplementedError(tree)


@_mypy_to_ast.register
def _(tree: _mypy.FuncDef) -> _ASTLike:
    arg_pos = get_funcdef_args(tree)

    calltype: _mypyt.CallableType = tree.type

    prepared_args = []
    for arg in arg_pos:
        ty = calltype.argument_by_name(arg).typ
        pirty = load_pyasir_type(str(ty))
        prepared_args.append(f"{arg}: {pirty}")

    pir_retty = load_pyasir_type(str(calltype.ret_type))

    repl = {
        "$name": tree.name,
        "$args": ", ".join(prepared_args),
        "$body": mypy_to_ast(tree.body),
        "$retty": pir_retty,
    }
    nodes = ast_template(
        """
@__pir__.func
def $name($args) -> $retty:
    $body
""",
        repl,
    )
    return nodes


@_mypy_to_ast.register
def _(tree: _mypy.Block) -> _ASTLike:
    ast_stmt_list = [mypy_to_ast(stmt) for stmt in tree.body]


    return IndentedBlock(ast_stmt_list)


@_mypy_to_ast.register
def _(tree: _mypy.AssignmentStmt) -> _ASTLike:
    [lhs] = tree.lvalues
    rhs = tree.rvalue
    tree = ast_template(
        f"$lhs = $rhs",
        {"$lhs": lhs.name, "$rhs": mypy_to_ast(rhs)},
    )
    return tree


def prepare_iterator_call(iter_expr: _mypy.CallExpr) -> tuple[ast.AST, list[ast.AST]]:
    callee = iter_expr.callee
    args = iter_expr.args
    return mypy_to_ast(callee), list(map(mypy_to_ast, args))


@_mypy_to_ast.register
def _(tree: _mypy.ForStmt) -> _ASTLike:
    index = tree.index
    index_name = index.name
    iter_expr = tree.expr
    assert not tree.else_body
    body_block = mypy_to_ast(tree.body)

    modified_names = find_loaded_names(body_block.statements)
    names = {index_name, *modified_names}
    names.remove(index_name)
    more = tuple(names)

    iter_callee, iter_args = prepare_iterator_call(iter_expr)

    repl = {
        "$args": ', '.join([index_name, *more]),
        "$loopargs": ', '.join(["iterator", *more]),
        "$iter_callee": iter_callee,
        "$iter_args": ', '.join(ast.unparse(iter_args)),
        "$iter": mypy_to_ast(iter_expr),
        "$body": body_block,
    }
    tree = ast_template(
        """
iterator = __pir__.call($iter_callee, $iter_args)

@__pir__.dialect.py.forloop
def loop($args):
    $body
    return $args

$args = loop($loopargs)
""",
        repl,
    )
    return tree


@_mypy_to_ast.register
def _(tree: _mypy.IfStmt) -> _ASTLike:
    [pred_tree] = tree.expr
    [body_tree] = tree.body
    pred_expr = mypy_to_ast(pred_tree)
    body_block = mypy_to_ast(body_tree)
    assert tree.else_body is None

    modified_names = find_loaded_names(body_block.statements)
    repl = {
        "$args": ', '.join(modified_names),
        "$pred": pred_expr,
        "$body": body_block,
    }

    return ast_template("""
@__pir__.switch($pred)
def switch($args):
    @__pir__.case(1)
    def ifblk($args):
        $body
        return $args

    @__pir__.case(0)
    def elseblk($args):
        return $args

    yield ifblk
    yield elseblk

$args = switch($args)
        """,
        repl

    )



@_mypy_to_ast.register
def _(tree: _mypy.WhileStmt) -> _ASTLike:
    pred_tree = tree.expr
    body_tree = tree.body
    pred_expr = mypy_to_ast(pred_tree)
    body_block = mypy_to_ast(body_tree)
    assert tree.else_body is None

    modified_names = find_loaded_names(body_block.statements)
    repl = {
        "$args": ', '.join(modified_names),
        "$pred": pred_expr,
        "$body": body_block,
    }

    return ast_template("""
@__pir__.dialect.py.whileloop($pred)
def loop_region($args):
    $body
    return $args

$args = loop_region($args)
        """,
        repl

    )


@_mypy_to_ast.register
def _(tree: _mypy.ExpressionStmt) -> _ASTLike:
    return mypy_to_ast(tree.expr)


@_mypy_to_ast.register
def _(tree: _mypy.ComparisonExpr) -> _ASTLike:
    [opstr] = tree.operators
    [lhs, rhs] = map(mypy_to_ast, tree.operands)
    repl = {
        "$lhs": lhs,
        "$rhs": rhs,
    }
    return ast_template(
        f"$lhs {opstr} $rhs",
        repl,
    )


@_mypy_to_ast.register
def _(tree: _mypy.OperatorAssignmentStmt) -> _ASTLike:
    opstr = tree.op + "="
    return ast_template(
        f"$lhs {opstr} $rhs",
        {
            "$lhs": tree.lvalue.name,
            "$rhs": mypy_to_ast(tree.rvalue),
        },
    )


@_mypy_to_ast.register
def _(tree: _mypy.CallExpr) -> _ASTLike:
    callee: _mypy.NameExpr = tree.callee
    argbuf = ", ".join([ast.unparse(mypy_to_ast(arg)) for arg in tree.args])
    return ast_template(
        f"$callee($args)",
        {
            "$callee": callee.name,
            "$args": argbuf,
        },
    )


@_mypy_to_ast.register
def _(tree: _mypy.ReturnStmt) -> _ASTLike:
    repl = {"$value": mypy_to_ast(tree.expr)}
    return ast_template(f"return $value", repl)


@_mypy_to_ast.register
def _(tree: _mypy.NameExpr) -> _ASTLike:
    return ast_parse_eval(tree.name)


@_mypy_to_ast.register
def _(tree: _mypy.StrExpr) -> _ASTLike:
    return ast_parse_eval(f"{tree.value!r}")


@_mypy_to_ast.register
def _(tree: _mypy.IntExpr) -> _ASTLike:
    return ast_parse_eval(f"{tree.value}")


def get_funcdef_args(fndef: _mypy.FuncDef) -> tuple[str]:
    arg_pos: list[str] = []
    for akind, aname in zip(fndef.arg_kinds, fndef.arg_names):
        if akind == _mypy.ArgKind.ARG_POS:
            assert isinstance(aname, str)
            arg_pos.append(aname)
        else:
            assert False
    return tuple(arg_pos)
