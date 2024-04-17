from __future__ import annotations

import logging
from typing import Any
from dataclasses import dataclass, replace as _dc_replace
from functools import singledispatch
from pprint import pformat
from ctypes import CFUNCTYPE

from llvmlite import ir
from llvmlite import binding as llvm

from . import nodes as _df

from pyasir.dispatchables.be_llvm import (
    emit_llvm,
    emit_llvm_type,
    emit_llvm_const,
)
from pyasir.dispatchables.ctypes import emit_c_type


_logger = logging.getLogger(__name__)


def generate(funcdef: _df.FuncDef):
    mod = ir.Module()
    be = LLVMBackend(module=mod, scope={}, cache={})
    fn = be.emit_funcdef(funcdef)
    print(mod)
    # Make a jit and bind to ctypes
    lljit = make_llvm_jit(mod)
    # optimize
    pmb = llvm.create_pass_manager_builder()
    pmb.opt_level = 2
    pmb.inlining_threshold = 200
    pm = llvm.create_module_pass_manager()
    pmb.populate(pm)
    llmod = llvm.parse_assembly(str(mod))
    pm.run(llmod)
    dbginfo = {}
    dbg_llvm_optimized = str(llmod)
    dbginfo["dbg_llvm_optimized"] = dbg_llvm_optimized
    _logger.debug("Optimized LLVM\n%s", dbg_llvm_optimized)

    tm = llvm.Target.from_default_triple().create_target_machine()
    dbg_asm = tm.emit_assembly(llmod)
    dbginfo["dbg_asm"] = dbg_asm
    _logger.debug("Assembly\n%s", dbg_asm)

    # bind
    jitlib = (
        llvm.JITLibraryBuilder()
        .add_ir(llmod)
        .add_current_process()
        .export_symbol(fn.name)
    )
    jitres = jitlib.link(lljit, repr(funcdef))
    addr = jitres[fn.name]

    c_argtys = [emit_c_type(ty) for ty in funcdef.argtys]
    c_retty = emit_c_type(funcdef.retty)

    proto = CFUNCTYPE(c_retty, *c_argtys)
    ccall = proto(addr)
    jf = JittedFunction(
        _resource=jitres,
        address=addr,
        ccall=ccall,
        llmod=llmod,
        fname=fn.name,
        dbginfo=dbginfo,
    )
    return jf


@dataclass(frozen=True)
class JittedFunction:
    _resource: llvm.ResourceTracker
    address: int
    ccall: callable
    llmod: llvm.ModuleRef
    fname: str
    dbginfo: dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.ccall(*args, **kwargs)

    def get_cfg(self):
        llmod = self.llmod
        g = llvm.view_dot_graph(
            llvm.get_function_cfg(llmod.get_function(self.fname))
        )
        return g


def make_llvm_jit(mod: ir.Module):
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    target = llvm.Target.from_default_triple()
    tm = target.create_target_machine()

    return llvm.create_lljit_compiler(tm)


inttype = ir.IntType(64)


@dataclass(frozen=True)
class LLVMBackend:
    module: ir.Module
    scope: dict[str, ir.Value]
    cache: dict[_df.DFNode, ir.Value]
    builder: ir.IRBuilder | None = None

    def emit_funcdef(self, funcdef: _df.FuncDef):
        func = funcdef.func
        fname = func.__name__
        try:
            fn = self.module.get_global(fname)
        except KeyError:
            llargtys = tuple(
                [emit_llvm_type(t, self.module) for t in funcdef.argtys]
            )
            llretty = emit_llvm_type(funcdef.retty, self.module)
            fnty = ir.FunctionType(llretty, llargtys)
            fn = ir.Function(self.module, fnty, name=fname)
            fn.calling_convention = "fastcc"
            for i, an in enumerate(funcdef.argnodes):
                fn.args[i].name = an.name

            # define the function body
            builder = ir.IRBuilder(fn.append_basic_block("entry"))

            scope = funcdef.bind_scope(fn.args, {})
            be = LLVMBackend(
                module=self.module,
                scope=scope,
                builder=builder,
                cache={},
            )
            res = be.emit(funcdef.node)
            be.builder.ret(res)
        return fn

    def emit(self, node: _df.DFNode) -> ir.Value:
        if node in self.cache:
            res = self.cache[node]
        else:
            # print(node.dump_shorten())
            data = emit_node(node, self)
            # print(f"emit {str(node)[:100]}")
            # print("-->", data.type)
            # print(self.builder.function)
            self.cache[node] = data
            res = data
        assert isinstance(res, ir.Value)
        return res

    def nested_call(
        self, node: _df.DFNode, scope: dict[_df.ArgNode, ir.Value]
    ) -> ir.Value:
        nested = _dc_replace(self, scope=scope, cache={})
        return nested.emit(node)


@singledispatch
def emit_node(node: _df.DFNode, be: LLVMBackend):
    raise NotImplementedError(f"{type(node)}:\n{pformat(node)}")


@emit_node.register(_df.ArgNode)
def _emit_node_ArgNode(node: _df.ArgNode, be: LLVMBackend):
    v = be.scope[node]
    return v


@emit_node.register(_df.EnterNode)
def _emit_node_EnterNode(node: _df.EnterNode, be: LLVMBackend):
    assert node.scope is not None
    scope = {k: be.emit(v) for k, v in node.scope.items()}
    return be.nested_call(node.body, scope)


@emit_node.register(_df.CaseExprNode)
def _emit_node_CaseExprNode(node: _df.CaseExprNode, be: LLVMBackend):
    value = be.emit(node.pred)
    bb_default = be.builder.append_basic_block("swt_default")
    bb_after = be.builder.append_basic_block("swt_after")
    with be.builder.goto_block(bb_default):
        be.builder.unreachable()
    # evaluate the arguments first
    first_case = node.cases[0]
    first_scope = first_case.scope
    for v_val in first_scope.values():
        be.emit(v_val)

    cases = []
    case_phis = []
    for case, case_pred in zip(node.cases, node.case_predicates, strict=True):
        bb_case = be.builder.append_basic_block(f"case_{case_pred.py_value}")
        cases.append((case_pred.py_value, bb_case))
        with be.builder.goto_block(bb_case):
            case_output = be.emit(case)
            be.builder.branch(bb_after)
            case_phis.append((be.builder.block, case_output))

    print("------")
    print(node.pred)
    print(value.type)
    swt = be.builder.switch(value, bb_default)
    for case_val, bb_case in cases:
        swt.add_case(case_val, bb_case)

    be.builder.position_at_end(bb_after)
    phi = be.builder.phi(case_phis[0][1].type)
    for bb, val in case_phis:
        phi.add_incoming(val, bb)
    return phi


@emit_node.register(_df.ExprNode)
def _emit_node_ExprNode(node: _df.ExprNode, be: LLVMBackend):
    emitted = [be.emit(arg) for arg in node.args]
    return emit_llvm(node.op, be.builder, *emitted)


@emit_node.register(_df.CallNode)
def _emit_node_CallNode(node: _df.CallNode, be: LLVMBackend):
    from .typedefs.functions import CallOp

    args = [be.emit(v) for v in node.args]
    kwargs = {k: be.emit(v) for k, v in node.kwargs.items()}
    if isinstance(node.func, _df.FuncNode):
        funcdef: _df.FuncDef = node.func.build_node()
        inner_be = _dc_replace(be, scope={}, cache={})
        fn = inner_be.emit_funcdef(funcdef)

        scope = funcdef.bind_scope(args, kwargs)
        res = be.builder.call(fn, scope.values())
        return res
    elif isinstance(node.func, CallOp):
        out = emit_llvm(node.func, be.builder, *args, **kwargs)
        return out
    else:
        raise NotImplementedError


@emit_node.register(_df.LiteralNode)
def _emit_node_LiteralNode(node: _df.LiteralNode, be: LLVMBackend):
    val = node.py_value
    return emit_llvm_const(node.datatype, be.builder, val)


@emit_node.register
def _eval_node_PackNode(node: _df.PackNode, be: LLVMBackend):
    values = [be.emit(v) for v in node.values]
    struct = ir.Constant(ir.LiteralStructType([v.type for v in values]), None)
    for i, v in enumerate(values):
        struct = be.builder.insert_value(struct, v, i)
    return struct


@emit_node.register
def _emit_node_UnpackNode(node: _df.UnpackNode, be: LLVMBackend):
    values = be.emit(node.producer)
    return be.builder.extract_value(values, node.index)


@emit_node.register
def _emit_node_LoopExprNode(node: _df.LoopExprNode, be: LLVMBackend):
    scope = node.body.scope

    incoming_values = {k: (be.emit(v), be.builder.block) for k, v in scope.items()}

    bb_loop = be.builder.append_basic_block("loop")
    bb_endloop = be.builder.append_basic_block("endloop")
    be.builder.branch(bb_loop)
    be.builder.position_at_end(bb_loop)

    # phi and incoming
    phis = {
        k: be.builder.phi(v.type, name=f"phi.{k.name}")
        for k, (v, bb) in incoming_values.items()
    }
    for phi, (lv, bb) in zip(phis.values(), incoming_values.values()):
        phi.add_incoming(lv, bb)

    loopbody = be.nested_call(node.body.body, scope=phis)
    loop_pred = be.builder.extract_value(loopbody, 0)
    loop_values_packed = be.builder.extract_value(loopbody, 1)
    loop_values = [
        be.builder.extract_value(loop_values_packed, i)
        for i in range(len(node.datatype.elements))
    ]

    bb_out_loop = be.builder.block
    # phi loop back
    for phi, lv in zip(phis.values(), loop_values):
        phi.add_incoming(lv, bb_out_loop)

    # phi
    be.builder.cbranch(
        be.builder.trunc(loop_pred, ir.IntType(1)), bb_loop, bb_endloop
    )

    be.builder.position_at_end(bb_endloop)
    return loop_values_packed


def _printf(builder: ir.IRBuilder, fmtstring: str, *args: ir.Value):
    module: ir.Module = builder.module
    try:
        printf = module.get_global("printf")
    except KeyError:
        fnty = ir.FunctionType(
            ir.IntType(32), [ir.IntType(8).as_pointer()], var_arg=True
        )
        printf = ir.Function(module, fnty, "printf")

    fmt = bytearray((fmtstring + "\0").encode())
    fmt_type = ir.ArrayType(ir.IntType(8), len(fmt))

    name = module.get_unique_name("conststr")
    gv = ir.GlobalVariable(module, fmt_type, name=name)
    gv.initializer = ir.Constant(fmt_type, fmt)
    gv.global_constant = True

    gv.linkage = "internal"
    return builder.call(
        printf, [builder.bitcast(gv, printf.type.pointee.args[0]), *args]
    )
