from __future__ import annotations

import operator
from typing import Any
from dataclasses import dataclass, replace as _dc_replace
from inspect import signature
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


def generate(funcdef: _df.FuncDef):
    mod = ir.Module()
    be = LLVMBackend(module=mod, scope={}, cache={})
    fn = be.emit_funcdef(funcdef)
    print(mod)
    # Make a jit and bind to ctypes
    lljit = make_llvm_jit(mod)
    # optimize
    pmb = llvm.create_pass_manager_builder()
    pmb.opt_level = 3
    pmb.inlining_threshold = 200
    pm = llvm.create_module_pass_manager()
    pmb.populate(pm)

    llmod = llvm.parse_assembly(str(mod))
    pm.run(llmod)
    print(llmod)

    tm = llvm.Target.from_default_triple().create_target_machine()
    print(tm.emit_assembly(llmod))

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
    jf = JittedFunction(_resource=jitres, address=addr, ccall=ccall)
    return jf


@dataclass(frozen=True)
class JittedFunction:
    _resource: llvm.ResourceTracker
    address: int
    ccall: callable

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.ccall(*args, **kwargs)


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
        sig = signature(func)

        try:
            fn = self.module.get_global(fname)
        except KeyError:
            sig = signature(func)
            llargtys = tuple(
                [emit_llvm_type(t, self.module) for t in funcdef.argtys]
            )
            llretty = emit_llvm_type(funcdef.retty, self.module)
            fnty = ir.FunctionType(llretty, llargtys)
            fn = ir.Function(self.module, fnty, name=fname)
            fn.calling_convention = "fastcc"
            for i, k in enumerate(sig.parameters):
                fn.args[i].name = k

            # define the function body
            builder = ir.IRBuilder(fn.append_basic_block("entry"))

            ba = sig.bind(*fn.args)
            args = {
                _df.ArgNode(t, k): v
                for (k, v), t in zip(ba.arguments.items(), funcdef.argtys)
            }
            be = LLVMBackend(
                module=self.module, scope=args, builder=builder, cache={}
            )
            res = be.emit(funcdef.node)
            be.builder.ret(res)
        return fn

    def emit(self, node: _df.DFNode) -> ir.Value:
        if node in self.cache:
            res = self.cache[node]
        else:
            data = emit_node(node, self)
            self.cache[node] = data
            res = data
        assert isinstance(res, ir.Value)
        return res

    def nested_call(
        self, node: _df.DFNode, scope: dict[_df.ArgNode, ir.Value]
    ) -> ir.Value:
        nested = _dc_replace(self, scope=scope, cache={})
        return nested.emit(node)

    def do_loop(
        self, *values: _df.DFNode, scope: dict[str, Any]
    ) -> tuple[ir.Value, tuple[ir.Value, ...]]:
        nested = _dc_replace(self, scope=scope, cache={})
        pred, *values = [nested.emit(v) for v in values]
        return pred, values


@singledispatch
def emit_node(node: _df.DFNode, be: LLVMBackend):
    raise NotImplementedError(f"{type(node)}:\n{pformat(node)}")


@emit_node.register(_df.ArgNode)
def _emit_node_ArgNode(node: _df.ArgNode, be: LLVMBackend):
    v = be.scope[node]
    return v


@emit_node.register(_df.EnterNode)
def _emit_node_EnterNode(node: _df.EnterNode, be: LLVMBackend):
    if node.scope is None:
        scope = be.scope
    else:
        scope = {k: be.emit(v) for k, v in node.scope.items()}
    return be.nested_call(node.node, scope)


@emit_node.register(_df.CaseExprNode)
def _emit_node_CaseExprNode(node: _df.CaseExprNode, be: LLVMBackend):
    bb_current = be.builder.basic_block
    pred = be.emit(node.pred)

    bb_default = be.builder.append_basic_block("default")
    bb_after = be.builder.append_basic_block("after")

    bb_case_preds = {}
    bb_case_outs = {}

    for case in node.cases:
        bb_case = be.builder.append_basic_block(f"case")

        be.builder.position_at_end(bb_current)
        case_pred = ir.Constant(pred.type, case.region.case_pred.py_value)

        be.builder.position_at_end(bb_case)
        case_outs = be.emit(case)

        bb_case_end = be.builder.basic_block
        bb_case_preds[bb_case] = case_pred
        bb_case_outs[bb_case_end] = case_outs

        be.builder.branch(bb_after)

    be.builder.position_at_end(bb_default)
    be.builder.unreachable()

    be.builder.position_at_end(bb_current)
    swt = be.builder.switch(pred, bb_default)
    for bb, case_val in bb_case_preds.items():
        swt.add_case(case_val, bb)

    be.builder.position_at_end(bb_after)
    phi = be.builder.phi(inttype)
    for bb, case_outs in bb_case_outs.items():
        phi.add_incoming(case_outs, bb)
    return phi


@emit_node.register(_df.ExprNode)
def _emit_node_ExprNode(node: _df.ExprNode, be: LLVMBackend):
    emitted = [be.emit(arg) for arg in node.args]
    return emit_llvm(node.op, be.builder, *emitted)


@emit_node.register(_df.CallNode)
def _emit_node_CallNode(node: _df.CallNode, be: LLVMBackend):
    args = [be.emit(v) for v in node.args]
    kwargs = {k: be.emit(v) for k, v in node.kwargs.items()}
    funcdef: _df.FuncDef = node.func.build_node()
    inner_be = _dc_replace(be, scope={}, cache={})
    fn = inner_be.emit_funcdef(funcdef)

    ba_args = signature(node.func.func).bind(*args, **kwargs).arguments
    res = be.builder.call(fn, ba_args.values())
    return res


@emit_node.register(_df.LiteralNode)
def _emit_node_LiteralNode(node: _df.LiteralNode, be: LLVMBackend):
    val = node.py_value
    return emit_llvm_const(node.datatype, be.builder, val)


@emit_node.register(_df.UnpackNode)
def _emit_node_UnpackNode(node: _df.UnpackNode, be: LLVMBackend):
    values = be.emit(node.producer)
    return be.builder.extract_value(values, node.index)


@emit_node.register(_df.LoopBodyNode)
def _emit_node_LoopBodyNode(node: _df.LoopBodyNode, be: LLVMBackend):
    scope = node.scope

    bb_head = be.builder.basic_block
    incoming_values = {k: be.emit(v) for k, v in scope.items()}

    bb_loop = be.builder.append_basic_block("loop")
    bb_endloop = be.builder.append_basic_block("endloop")
    be.builder.branch(bb_loop)
    be.builder.position_at_end(bb_loop)

    # phi and incoming
    phis = {k: be.builder.phi(v.type) for k, v in incoming_values.items()}
    for phi, lv in zip(phis.values(), incoming_values.values()):
        phi.add_incoming(lv, bb_head)

    loop_pred, loop_values = be.do_loop(*node.values, scope=phis)

    # phi loop back
    for phi, lv in zip(phis.values(), loop_values):
        phi.add_incoming(lv, bb_loop)

    # phi
    be.builder.cbranch(
        be.builder.trunc(loop_pred, ir.IntType(1)), bb_loop, bb_endloop
    )

    be.builder.position_at_end(bb_endloop)
    struct = ir.Constant(
        ir.LiteralStructType([v.type for v in loop_values]), None
    )
    for i, v in enumerate(loop_values):
        struct = be.builder.insert_value(struct, v, i)
    return struct


def _printf(builder: ir.IRBuilder, fmtstring: str, *args: ir.Value):
    module: ir.Module = builder.module
    try:
        printf = module.get_global("printf")
    except KeyError:
        fnty = ir.FunctionType(
            ir.IntType(32), [ir.IntType(8).as_pointer()], var_arg=True
        )
        printf = ir.Function(module, fnty, "printf")

    fmt = bytearray(fmtstring.encode())
    fmt_type = ir.ArrayType(ir.IntType(8), len(fmt))

    name = module.get_unique_name("conststr")
    gv = ir.GlobalVariable(module, fmt_type, name=name)
    gv.initializer = ir.Constant(fmt_type, fmt)
    gv.global_constant = True

    gv.linkage = "internal"
    return builder.call(
        printf, [builder.bitcast(gv, printf.type.pointee.args[0]), *args]
    )
