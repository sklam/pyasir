from __future__ import annotations
import copy
import importlib
import mypy.nodes as _mypy
import mypy.types as _mypyt
import textwrap
import re
import ast
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, NamedTuple


_REGEX_AST_TEMPLATE = re.compile(r"( *)(\$[a-zA-Z_0-9]+)")



def auto_format_source(code: str) -> str:
    try:
        import black
    except ImportError:
        return code
    else:
        return black.format_str(code, mode=black.FileMode())



@dataclass(frozen=True)
class SourceGen:
    def generate_source(self) -> str:
        return ast.unparse(self.generate_ast())

    def generate_ast(self) -> ast.AST:
        raise NotImplementedError(type(self))


@dataclass(frozen=True)
class SourceBlock(SourceGen):
    statements: list[SourceGen]

    def __post_init__(self):
        copied = self.statements.copy()
        self.statements.clear()
        for stmt in copied:
            if isinstance(stmt, SourceBlock):
                self.statements.extend(stmt.statements)
            elif isinstance(stmt, SourceGen):
                self.statements.append(stmt)
            else:
                assert not isinstance(stmt, ast.AST)
        assert self.statements

    def generate_ast(self) -> ast.AST:
        nodes = [stmt.generate_ast() for stmt in self.statements]
        return ast.Module(body=nodes, type_ignores=())


@dataclass(frozen=True)
class ASTWrapper(SourceGen):
    astree: ast.AST

    def __post_init__(self):
        assert isinstance(self.astree, ast.AST)

    def _generate_source_details(self) -> str:
        return ast.unparse(self.astree)

    @staticmethod
    def from_source(source: str) -> ASTWrapper:
        try:
            tree = ast.parse(source)
        except Exception:
            print(source)
            breakpoint()
            raise
        return ASTWrapper(tree)

    def generate_ast(self) -> ast.AST:
        return self.astree


def normalize_ast_module(tree: ast.AST) -> ast.expr:
    if isinstance(tree, ast.Module):
        if len(tree.body) == 1:
            [tree] = tree.body
        else:
            raise ValueError(type(tree))
    if isinstance(tree, ast.Expr):
        tree = tree.value
    elif isinstance(tree, ast.Expression):
        tree = tree.body
    assert isinstance(tree, ast.expr), ast.dump(tree)
    return tree



_TemplateReplVal = str | SourceBlock


def ast_parse_eval(text: str) -> SourceGen:
    return ASTWrapper(ast.parse(text, mode="eval"))


def ast_template(source: str, repl: dict[str, _TemplateReplVal]) -> SourceGen:
    def match(m):
        spaces: str = m.group(1)
        key: str = m.group(2)
        out = repl[key]

        if isinstance(out, SourceGen):
            out = out.generate_source()
        else:
            assert isinstance(out, str)
        return textwrap.indent(out, prefix=spaces)

    replaced = _REGEX_AST_TEMPLATE.sub(match, source)
    source = replaced.replace("$", "")
    tree = ASTWrapper.from_source(source)

    if isinstance(tree, ast.Module):
        if len(tree.body) == 1:
            [stmt] = tree.body
            return ASTWrapper(stmt)
        else:
            return SourceBlock(tree.body)
    return tree


class Translator:

    def translate_file(self, tree: _mypy.MypyFile) -> str:
        buf: list[str] = []
        buf.append(self.get_import_lines())
        for defn in tree.defs:
            if isinstance(defn, (_mypy.Decorator, _mypy.FuncDef)):
                buf.append(self.translate(defn))
            elif isinstance(defn, (_mypy.ImportFrom, _mypy.Import)):
                buf.append(self.translate_import(defn))
            elif isinstance(defn, _mypy.ClassDef):
                buf.append(self.translate_classdef(defn))
            else:
                raise NotImplementedError(type(defn), defn)
        return '\n'.join(buf)

    def translate_classdef(self, tree: _mypy.ClassDef) -> str:
        out: list[str] = []

        [decor] = tree.decorators
        assert decor.node.fullname == "llpyfe.struct"
        assert not tree.base_type_exprs
        assert not tree.metaclass
        fields: str = []
        for dfn in tree.defs.body:
            assert isinstance(dfn, _mypy.AssignmentStmt)
            assert isinstance(dfn.rvalue, _mypy.TempNode)
            [lval] = dfn.lvalues
            name = lval.name
            ty = load_pyasir_type(str(dfn.type))
            fields.append(f"    {name}: {ty}")

        out.append(f"@pyasir.Struct")
        out.append(f"class {tree.name}:")
        out.extend(fields)
        return '\n'.join(out)

    def translate_import(self, tree: _mypy.ImportFrom | _mypy.Import) -> str:
        out: list[str] = []
        if isinstance(tree, _mypy.ImportFrom):
            basepath = str(tree.id)
            for target, alias in tree.names:
                alias = alias or target
                out.append(f"from {basepath} import {target} as {alias}")
        elif isinstance(tree, _mypy.Import):
            for target, alias in tree.ids:
                if alias:
                    out.append(f"import {target} as {alias}")
                else:
                    out.append(f"import {target}")

        else:
            raise NotImplementedError(type(tree), tree)

        # write import lines
        return '\n'.join(out)


    def translate(self, fndef: _mypy.FuncDef) -> str:
        tree = PostProcessSeq().visit(mypy_to_ast(fndef).generate_ast())
        # print("ast.dump".center(80, "-"))
        # print(ast.dump(tree, indent=4))
        print("=" * 80)
        source = ast.unparse(tree)
        print("source".center(80, "-"))
        print(source)
        print("=" * 80)
        return auto_format_source(source)

    def get_import_lines(self) -> str:
        return """
import pyasir
from pyasir import nodes as __pir__
"""


class PostProcessSeq(ast.NodeTransformer):
    def_level = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.def_level += 1
        body = _post_proc_body(node.body)
        out = copy.copy(node)
        out.body = [self.visit(n) for n in body]
        if self.def_level == 1:
            # is top level
            stmt = _assign_to_seq(ast.parse("io.seq()", mode='eval').body)
            out.body.insert(0, stmt)
        # print("<<<<")
        # print(ast.unparse(node))
        # print('------')
        # print(ast.dump(out))
        # print(ast.unparse(out))
        # print("<>>>>")
        self.def_level -= 1
        return out

    def visit_Return(self, node: ast.Return) -> ast.Return:
        if self.def_level == 1:
            # is top level
            seq = _make_load_seq()
            ret = copy.copy(node)
            ret.value = _wrap_io_seq([seq, node.value])
            return ret
        return node


def _post_proc_body(nodes):
    assert isinstance(nodes, list)
    wr = BlockSyncWriter()

    for node in nodes:
        if isinstance(node, ast.Assign):
            wr.append_assign(node)
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Yield):
                wr.append_other(node)
            else:
                wr.append_expr(node)
        elif isinstance(node, (ast.FunctionDef, ast.Return)):
            wr.append_other(node)
        else:
            raise TypeError(type(node), ast.unparse(node))

    body = wr.write_body()
    return body




def mypy_to_ast(tree: _mypy.Node) -> SourceGen:
    out = _mypy_to_ast(tree)
    assert isinstance(out, SourceGen), f"{type(tree)} didn't return SourceGen"
    return out


def _make_load_seq() -> ast.AST:
    return ast.Name(id="__pir_seq__", ctx=ast.Load())


def _wrap_io_seq(args) -> ast.AST:
    # XXX: fix namespace
    callee = ast.parse("io.sync", mode='eval').body
    callnode = ast.Call(callee, args=args, keywords=[])
    return ast.fix_missing_locations(callnode)


def _assign_to_seq(sync_node: ast.Call) -> ast.AST:
    callnode = ast.Assign(
        targets=[ast.Name(id='__pir_seq__', ctx=ast.Store())],
        value=sync_node,
    )
    return ast.fix_missing_locations(callnode)


class BlockSyncWriter:
    _exprs: list[ast.expr]
    _body: list[ast.AST]

    def __init__(self):
        self._exprs = []
        self._body = []

    def append_assign(self, node: ast.Assign):
        [target] = node.targets
        if isinstance(target, ast.List):
            self.flush()
            self._body.append(node)
        else:
            node = ast.NamedExpr(target=target, value=node.value)
            self._exprs.append(node)

    def append_expr(self, expr: ast.expr):
        self._exprs.append(expr)

    def append_other(self, node: ast.AST):
        self.flush()
        self._body.append(node)

    def flush(self):
        if self._exprs:
            name_seq = _make_load_seq()
            args = [*self._exprs, name_seq]
            self._exprs.clear()
            self._body.append(_assign_to_seq(_wrap_io_seq(args)))

    def write_body(self) -> list[ast.AST]:
        self.flush()
        return self._body



class FindLoadedNames(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.__modified_vars = set()

    def get(self) -> set[str]:
        return frozenset(self.__modified_vars)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            self.__modified_vars.add(node.id)

    def handle(self, srcgen: SourceGen):
        if isinstance(srcgen, SourceBlock):
            for stmt in srcgen.statements:
                self.handle(stmt)
        elif isinstance(srcgen, ASTWrapper):
            self.visit(srcgen.astree)
        else:
            raise TypeError(type(srcgen))



def find_loaded_names(block: list[SourceGen]) -> frozenset[str]:
    finder = FindLoadedNames()
    for b in block:
        finder.handle(b)
    return finder.get()


def load_pyasir_type(tyname: str):
    assert isinstance(tyname, str)
    module_path, obj_name = tyname.rsplit('.', 1)
    module = importlib.import_module(module_path)
    tyclass = getattr(module, obj_name)
    return tyclass.__pyasir_type__


@singledispatch
def _mypy_to_ast(tree: _mypy.Node) -> SourceGen:
    raise NotImplementedError(tree)


@_mypy_to_ast.register
def _(tree: _mypy.FuncDef) -> SourceGen:
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
def _(tree: _mypy.Decorator) -> SourceGen:
    [decor] = tree.decorators
    if decor.node.fullname == "llpyfe.aot":
        repl = {
            "$decor": "",
            "$func": mypy_to_ast(tree.func),
        }
    else:
        repl = {
            "$decor": "@" + mypy_to_ast(decor),
            "$func": mypy_to_ast(tree.func),
        }
    return ast_template(
        """$decor\n$func
""",
        repl
    )


@_mypy_to_ast.register
def _(tree: _mypy.Block) -> SourceGen:
    ast_stmt_list = [mypy_to_ast(stmt) for stmt in tree.body]


    return SourceBlock(ast_stmt_list)


@_mypy_to_ast.register
def _(tree: _mypy.AssignmentStmt) -> SourceGen:
    [lhs] = tree.lvalues
    rhs = tree.rvalue
    tree = ast_template(
        f"$lhs = $rhs",
        {"$lhs": lhs.name,
         "$rhs": mypy_to_ast(rhs)},
    )
    return tree


def prepare_iterator_call(iter_expr: _mypy.CallExpr) -> tuple[ast.AST, list[ast.AST]]:
    callee = iter_expr.callee
    args = iter_expr.args
    return mypy_to_ast(callee), list(map(mypy_to_ast, args))


@_mypy_to_ast.register
def _(tree: _mypy.ForStmt) -> SourceGen:
    index = tree.index
    index_name = index.name
    iter_expr = tree.expr
    assert not tree.else_body
    body_block = mypy_to_ast(tree.body)

    modified_names = find_loaded_names(body_block.statements) | {'__pir_seq__'}
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
def _(tree: _mypy.IfStmt) -> SourceGen:
    [pred_tree] = tree.expr
    [body_tree] = tree.body
    pred_expr = mypy_to_ast(pred_tree)
    body_block = mypy_to_ast(body_tree)
    assert tree.else_body is None

    modified_names = find_loaded_names(body_block.statements) | {'__pir_seq__'}
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
        return __pir__.pack($args)

    @__pir__.case(0)
    def elseblk($args):
        return __pir__.pack($args)

    yield ifblk
    yield elseblk

[$args] = __pir__.unpack(switch($args))
        """,
        repl

    )



@_mypy_to_ast.register
def _(tree: _mypy.WhileStmt) -> SourceGen:
    pred_tree = tree.expr
    body_tree = tree.body
    pred_expr = mypy_to_ast(pred_tree)
    body_block = mypy_to_ast(body_tree)
    assert tree.else_body is None

    modified_names = find_loaded_names(body_block.statements) | {'__pir_seq__'}
    repl = {
        "$args": ', '.join(modified_names),
        "$pred": pred_expr,
        "$body": body_block,
    }
    assert modified_names, breakpoint()
    return ast_template("""
@__pir__.switch($pred)
def swt_while($args) :
    @__pir__.case(1)
    def case1($args):
        @__pir__.loop
        def loop($args):
            $body
            return $pred, __pir__.pack($args)
        [$args] = __pir__.unpack(loop($args))
        return __pir__.pack($args)

    @__pir__.case(0)
    def case0($args):
        return __pir__.pack($args)

    yield case1
    yield case0

[$args] = __pir__.unpack(swt_while($args))
__pir_seq__ = io.sync($args, __pir_seq__)
        """,
        repl

    )

@_mypy_to_ast.register
def _(tree: _mypy.AssignmentExpr) -> SourceGen:
    lhs = ast.Name(id=str(tree.target.name), ctx=ast.Store())
    rhs = mypy_to_ast(tree.value).generate_ast()
    rhs = normalize_ast_module(rhs)
    return ASTWrapper(ast.NamedExpr(target=lhs, value=rhs))


@_mypy_to_ast.register
def _(tree: _mypy.ExpressionStmt) -> SourceGen:
    return mypy_to_ast(tree.expr)


@_mypy_to_ast.register
def _(tree: _mypy.OpExpr) -> SourceGen:
    op = tree.op
    assert op == "+"

    return ast_template(
        f"$left {op} $right", {"$left": mypy_to_ast(tree.left),
                               "$right": mypy_to_ast(tree.right)})


@_mypy_to_ast.register
def _(tree: _mypy.ComparisonExpr) -> SourceGen:
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
def _(tree: _mypy.OperatorAssignmentStmt) -> SourceGen:
    opstr = tree.op + "="
    return ast_template(
        f"$lhs {opstr} $rhs",
        {
            "$lhs": tree.lvalue.name,
            "$rhs": mypy_to_ast(tree.rvalue),
        },
    )


@_mypy_to_ast.register
def _(tree: _mypy.CallExpr) -> SourceGen:
    callee: _mypy.NameExpr = tree.callee
    argbuf = ", ".join([mypy_to_ast(arg).generate_source() for arg in tree.args])
    return ast_template(
        f"$callee($args)",
        {
            "$callee": mypy_to_ast(callee),
            "$args": argbuf,
        },
    )


@_mypy_to_ast.register
def _(tree: _mypy.MemberExpr) -> SourceGen:
    repl = {
        "$base": mypy_to_ast(tree.expr),
    }
    return ast_template(f"$base.{tree.name}", repl)


@_mypy_to_ast.register
def _(tree: _mypy.ReturnStmt) -> SourceGen:
    repl = {"$value": mypy_to_ast(tree.expr)}
    return ast_template(f"return $value", repl)


@_mypy_to_ast.register
def _(tree: _mypy.NameExpr) -> SourceGen:
    return ast_parse_eval(tree.name)


@_mypy_to_ast.register
def _(tree: _mypy.StrExpr) -> SourceGen:
    return ast_parse_eval(f"{tree.value!r}")


@_mypy_to_ast.register
def _(tree: _mypy.IntExpr) -> SourceGen:
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
