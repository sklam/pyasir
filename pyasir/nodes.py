from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import partial, partialmethod
from typing import Any
from inspect import signature, Signature, BoundArguments
from types import FrameType


def _generic_binop(self, other, *, op):
    return OpNode.make(op, self, other)


@dataclass(frozen=True)
class DFNode:

    __add__ = partialmethod(_generic_binop, op="+")
    __sub__ = partialmethod(_generic_binop, op="-")
    __mul__ = partialmethod(_generic_binop, op="*")

    __le__ = partialmethod(_generic_binop, op="<=")
    __gt__ = partialmethod(_generic_binop, op=">")


@dataclass(frozen=True)
class RegionNode(DFNode):
    pass


def _call_func_no_post(func):
    sig = signature(func)
    args = sig.bind(*(ArgNode(k) for k in sig.parameters.keys()))
    ret = func(**args.arguments)
    return ret


def _call_func(func):
    ret = _call_func_no_post(func)
    return _normalize(ret)


@dataclass(frozen=True)
class FuncNode(RegionNode):
    func: callable

    @classmethod
    def make(cls, func):
        return cls(func)

    def build_node(self) -> FuncDef:
        node = _call_func(self.func)
        return FuncDef(self.func, node)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return CallNode.make(self, *args, **kwargs)


@dataclass(frozen=True)
class FuncDef:
    func: callable
    node: DFNode


@dataclass(frozen=True)
class SwitchNode(RegionNode):
    pred_node: DFNode
    region_func: callable

    @classmethod
    def make(cls, pred, func=None):
        if func is None:
            return partial(cls.make, pred)
        return cls(pred, func)

    def __call__(self, *args: Any, **kwargs: Any):
        gen = _call_func_no_post(self.region_func)
        nodes = tuple(gen)
        sig = signature(self.region_func)
        ba = sig.bind(*args, **kwargs)
        scope = {ArgNode(k): _normalize(v) for k, v in ba.arguments.items()}
        return EnterNode.make(
            self, CaseExprNode(self.pred_node, nodes), scope=scope
        )


@dataclass(frozen=True)
class LoopNode(RegionNode):
    region_func: callable

    @classmethod
    def make(cls, func):
        return cls(func)

    def __call__(self, *args: Any, **kwargs: Any):
        nodes = _call_func_no_post(self.region_func)
        cont, *values = nodes

        sig = signature(self.region_func)
        ba = sig.bind(*args, **kwargs)
        scope = {ArgNode(k): v for k, v in ba.arguments.items()}

        loopbody = LoopBodyNode(
            tuple([_normalize(cont), *tuple(values)]), scope
        )
        pred_node = UnpackNode(loopbody, 0)
        value_nodes = tuple(
            [UnpackNode(loopbody, i) for i in range(len(values))]
        )
        return LoopExprNode(self, pred=pred_node, values=value_nodes)


@dataclass(frozen=True)
class CaseExprNode(DFNode):
    pred: DFNode
    cases: tuple[EnterNode]


@dataclass(frozen=True)
class CaseNode(RegionNode):
    case_pred: DFNode

    @classmethod
    def make(cls, case_pred: Any):
        return CaseNode(_normalize(case_pred))

    def __call__(self, func) -> Any:
        nodes = _call_func(func)
        return EnterNode.make(self, nodes, scope=None)


@dataclass(frozen=True)
class LoopBodyNode(DFNode):
    values: tuple[DFNode]
    scope: dict

    def __hash__(self):
        return id(self)


@dataclass(frozen=True)
class LoopExprNode(DFNode):
    parent: LoopNode
    pred: DFNode
    values: tuple[DFNode]

    def __iter__(self):
        return iter(self.values)


@dataclass(frozen=True)
class UnpackNode(DFNode):
    producer: DFNode
    index: int


func = FuncNode.make
switch = SwitchNode.make
case = CaseNode.make
loop = LoopNode.make


@dataclass(frozen=True)
class LiteralNode(DFNode):
    py_value: object

    @classmethod
    def make(self, val):
        return LiteralNode(py_value=val)


def _normalize(val):
    if isinstance(val, DFNode):
        return val
    assert isinstance(val, int), f"{type(val)} - {val}"
    return LiteralNode.make(val)


def _normalize_args(args):
    return tuple([_normalize(v) for v in args])


def _normalize_kwargs(kwargs):
    return {k: _normalize(v) for k, v in kwargs.items()}


@dataclass(frozen=True)
class ArgNode(DFNode):
    name: str


@dataclass(frozen=True)
class OpNode(DFNode):
    op: str
    left: Any
    right: Any

    @classmethod
    def make(cls, op, lhs, rhs):
        return cls(op=op, left=_normalize(lhs), right=_normalize(rhs))


@dataclass(frozen=True)
class CallNode(DFNode):
    func: FuncNode
    args: tuple
    kwargs: dict

    @classmethod
    def make(cls, func, *args, **kwargs):
        return cls(
            func=func,
            args=_normalize_args(args),
            kwargs=_normalize_kwargs(kwargs),
        )

    def __hash__(self):
        return id(self)


@dataclass(frozen=True)
class EnterNode(DFNode):
    region: RegionNode
    node: DFNode
    scope: dict[ArgNode, DFNode]

    @classmethod
    def make(cls, region, node, scope):
        _sentry_scope(scope)
        return cls(region=region, node=node, scope=scope)

    def __iter__(self):
        if isinstance(self.node, LoopExprNode):
            unpacked = [
                UnpackNode(self.node, i) for i in range(len(self.node.values))
            ]
            return iter(unpacked)
        else:
            return NotImplemented

    def __hash__(self):
        return id(self)


def _sentry_scope(scope: dict[ArgNode, DFNode] | None):
    if scope is not None:
        for k, v in scope.items():
            assert isinstance(k, ArgNode)
            assert isinstance(v, DFNode)