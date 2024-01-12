from __future__ import annotations

from dataclasses import dataclass
from functools import partial, partialmethod, singledispatch
from typing import Any, Callable
from inspect import signature

import pyasir
from . import datatypes as _dt
from . import typing as _tp


def _generic_binop(self, other, *, op):
    other = as_node(other)
    optrait = self.datatype.get_binop(op, self.datatype, other.datatype)
    return ExprNode(optrait.result_type, optrait, args=(self, other))


@dataclass(frozen=True)
class DFNode:
    ...


@dataclass(frozen=True)
class ValueNode(DFNode):
    datatype: _dt.DataType

    __add__ = partialmethod(_generic_binop, op="+")
    __sub__ = partialmethod(_generic_binop, op="-")
    __mul__ = partialmethod(_generic_binop, op="*")

    __le__ = partialmethod(_generic_binop, op="<=")
    __ge__ = partialmethod(_generic_binop, op=">=")
    __gt__ = partialmethod(_generic_binop, op=">")
    __lt__ = partialmethod(_generic_binop, op="<")


@dataclass(frozen=True)
class RegionNode(DFNode):
    pass


def _call_func_no_post(func, argtys):
    sig = signature(func)
    args = sig.bind(
        *(
            ArgNode(datatype=t, name=k)
            for k, t in zip(sig.parameters.keys(), argtys)
        )
    )
    ret = func(**args.arguments)
    return ret


def _call_func(func, argtys):
    ret = _call_func_no_post(func, argtys)
    return as_node(ret)


@dataclass(frozen=True)
class FuncNode(RegionNode):
    func: Callable
    argtys: tuple[_dt.DataType, ...]
    retty: _dt.DataType

    @classmethod
    def make(cls, func):
        typehints = _tp.get_annotations(func)
        retty = typehints.pop("return")
        argtys = tuple(typehints.values())
        return cls(func, argtys, retty)

    def build_node(self) -> FuncDef:
        node = _call_func(self.func, self.argtys)
        if node.datatype != self.retty:
            raise _dt.TypeOpError(
                f"FuncDef returned {node.datatype} instead of {self.retty}"
            )
        return FuncDef(self.func, self.argtys, self.retty, node)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return CallNode.make(self, *args, **kwargs)


@dataclass(frozen=True)
class FuncDef:
    func: Callable
    argtys: tuple[_dt.DataType, ...]
    retty: _dt.DataType
    node: DFNode

    def __post_init__(self):
        for x in self.argtys:
            assert isinstance(x, _dt.DataType)
        assert isinstance(self.retty, _dt.DataType)


def _bind__dt(sig, *args: ValueNode, **kwargs: ValueNode):
    return sig.bind(
        *(a.datatype for a in args),
        **{k: v.datatype for k, v in kwargs.items()},
    )


@dataclass(frozen=True)
class SwitchNode(RegionNode):
    pred_node: DFNode
    region_func: Callable

    @classmethod
    def make(cls, pred, func=None):
        if func is None:
            return partial(cls.make, pred)
        return cls(pred, func)

    def __call__(self, *args: Any, **kwargs: Any):
        args = as_node_args(args)
        kwargs = as_node_kwargs(kwargs)
        sig = signature(self.region_func)
        ba_types = _bind__dt(sig, *args, **kwargs)

        ty_args = tuple(ba_types.arguments.values())
        case_nodes = _call_func_no_post(self.region_func, ty_args)
        nodes = tuple(
            [
                EnterNode.make(
                    cn, _call_func(cn.case_func, ty_args), scope=None
                )
                for cn in case_nodes
            ]
        )

        ba = sig.bind(*args, **kwargs)
        scope = {
            ArgNode(v.datatype, k): as_node(v) for k, v in ba.arguments.items()
        }
        for n in nodes[1:]:
            if nodes[0].datatype != n.datatype:
                raise _dt.TypeOpError(
                    f"incompatible type {n.datatype}; expect {nodes[0].datatype}"
                )
        return EnterNode.make(
            self,
            CaseExprNode(nodes[0].datatype, self.pred_node, nodes),
            scope=scope,
        )


@dataclass(frozen=True)
class LoopNode(RegionNode):
    region_func: Callable

    @classmethod
    def make(cls, func):
        return cls(func)

    def __call__(self, *args: Any, **kwargs: Any):
        sig = signature(self.region_func)
        ba_types = _bind__dt(sig, *args, **kwargs)
        ty_args = tuple(ba_types.arguments.values())
        nodes = _call_func_no_post(self.region_func, ty_args)
        cont, values = nodes

        sig = signature(self.region_func)
        ba = sig.bind(*args, **kwargs)
        scope = {ArgNode(v.datatype, k): v for k, v in ba.arguments.items()}

        body_values = tuple(map(as_node, [cont, *values]))
        loopbody = LoopBodyNode(pyasir.Packed(), body_values, scope)
        pred_node = UnpackNode(loopbody.values[0].datatype, loopbody, 0)
        value_nodes = tuple(
            [
                UnpackNode(loopbody.values[i + 1].datatype, loopbody, i)
                for i in range(len(values))
            ]
        )
        return LoopExprNode(self, pred=pred_node, values=value_nodes)


@dataclass(frozen=True)
class CaseExprNode(DFNode):
    datatype: _dt.DataType
    pred: DFNode
    cases: tuple[EnterNode, ...]


@dataclass(frozen=True)
class CaseNode(RegionNode):
    case_pred: DFNode
    case_func: Callable

    @classmethod
    def make(cls, case_pred: Any):
        return lambda fn: CaseNode(as_node(case_pred), fn)


@dataclass(frozen=True)
class LoopBodyNode(DFNode):
    datatype: _dt.DataType
    values: tuple[ValueNode, ...]
    scope: dict

    def __hash__(self):
        return id(self)


@dataclass(frozen=True)
class LoopExprNode(DFNode):
    parent: LoopNode
    pred: ValueNode
    values: tuple[ValueNode, ...]

    def __iter__(self):
        return iter(self.values)


@dataclass(frozen=True)
class UnpackNode(ValueNode):
    producer: DFNode
    index: int


func = FuncNode.make
switch = SwitchNode.make
case = CaseNode.make
loop = LoopNode.make


@dataclass(frozen=True)
class LiteralNode(ValueNode):
    py_value: Any


@singledispatch
def as_node(val) -> ValueNode:
    raise NotImplementedError(type(val))


@as_node.register
def _(val: ValueNode):
    return val


@as_node.register
def _(val: int):
    return LiteralNode(pyasir.Int64(), val)


@as_node.register
def _(val: float):
    return LiteralNode(pyasir.Float64(), val)


@as_node.register
def _(val: bool):
    return LiteralNode(pyasir.Bool(), val)


def as_node_args(args) -> tuple[ValueNode, ...]:
    return tuple([as_node(v) for v in args])


def as_node_kwargs(kwargs) -> dict[str, ValueNode]:
    return {k: as_node(v) for k, v in kwargs.items()}


@dataclass(frozen=True)
class ArgNode(ValueNode):
    name: str


@dataclass(frozen=True)
class ExprNode(ValueNode):
    op: _dt.OpTrait
    args: tuple[ValueNode, ...]


@dataclass(frozen=True)
class CallNode(ValueNode):
    func: FuncNode
    args: tuple
    kwargs: dict

    @classmethod
    def make(cls, func: FuncNode, *args, **kwargs):
        return cls(
            datatype=func.retty,
            func=func,
            args=as_node_args(args),
            kwargs=as_node_kwargs(kwargs),
        )

    def __hash__(self):
        return id(self)


@dataclass(frozen=True)
class EnterNode(ValueNode):
    region: RegionNode
    node: DFNode
    scope: dict[ArgNode, DFNode]

    @classmethod
    def make(cls, region, node, scope):
        _sentry_scope(scope)
        return cls(
            datatype=node.datatype, region=region, node=node, scope=scope
        )

    def __iter__(self):
        if isinstance(self.node, LoopExprNode):
            unpacked = [
                UnpackNode(self.node.values[i].datatype, self.node, i)
                for i in range(len(self.node.values))
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


def cast(value: ValueNode, to_type: _dt.DataType) -> DFNode:
    to_type = _dt.ensure_type(to_type)
    op = to_type.get_cast(value.datatype)
    return ExprNode(op.result_type, op, args=tuple([value]))
