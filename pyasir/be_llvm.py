from __future__ import annotations

from typing import Any
from dataclasses import dataclass, replace as _dc_replace
from inspect import signature
from functools import singledispatch
from pprint import pformat
from ctypes import CFUNCTYPE, c_int64

from llvmlite import ir
from llvmlite import binding as llvm

from . import nodes as _df



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
    jitlib =llvm.JITLibraryBuilder().add_ir(llmod).export_symbol(fn.name)
    jitres = jitlib.link(lljit, repr(funcdef))
    addr = jitres[fn.name]
    proto = CFUNCTYPE(c_int64, c_int64)
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
            nargs = len(sig.parameters)
            fnty = ir.FunctionType(inttype, [inttype] * nargs)
            fn = ir.Function(self.module, fnty, name=fname)
            fn.calling_convention = "fastcc"
            for i, k in enumerate(sig.parameters):
                fn.args[i].name = k

            # define the function body
            builder = ir.IRBuilder(fn.append_basic_block("entry"))

            ba = sig.bind(*fn.args)
            args = {_df.ArgNode(k): v for k, v in ba.arguments.items()}
            be = LLVMBackend(module=self.module, scope=args, builder=builder, cache={})
            res = be.emit(funcdef.node)
            be.builder.ret(res)
        return fn

    def emit(self, node: _df.DFNode):
        if node in self.cache:
            res = self.cache[node]
        else:
            data = emit_node(node, self)
            self.cache[node] = data
            res = data
        assert isinstance(res, ir.Value)
        return res

    def nested_call(self, node: _df.DFNode, scope: dict[_df.ArgNode, ir.Value]):
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
        bb_case_preds[bb_case] = case_pred

        be.builder.position_at_end(bb_case)
        case_outs = be.emit(case)
        bb_case_outs[bb_case] = case_outs
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



@emit_node.register(_df.OpNode)
def _emit_node_OpNode(node: _df.OpNode, be: LLVMBackend):
    lhs = be.emit(node.left)
    rhs = be.emit(node.right)
    if node.op == "<=":
        return be.builder.icmp_signed("<=", lhs, rhs)
    elif node.op == ">=":
        return be.builder.icmp_signed(">=", lhs, rhs)
    elif node.op == "<":
        return be.builder.icmp_signed("<", lhs, rhs)
    elif node.op == ">":
        return be.builder.icmp_signed(">", lhs, rhs)
    elif node.op == "-":
        return be.builder.sub(lhs, rhs)
    elif node.op == "+":
        return be.builder.add(lhs, rhs)
    else:
        raise AssertionError(f"not supported {node}")


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
    assert isinstance(val, int)
    return inttype(val)
