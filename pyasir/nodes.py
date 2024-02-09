from __future__ import annotations

import uuid
from dataclasses import dataclass, replace as _dc_replace
from functools import partial, partialmethod, singledispatch
from typing import Any, Callable, Iterable
from inspect import signature, isgenerator, Signature
from collections.abc import Mapping
from pprint import PrettyPrinter

import pyasir
from . import datatypes as _dt
from . import typing as _tp
from . import dialect  # reexport


def _generic_binop(self, other, *, op):
    other = as_node(other)
    optrait = self.datatype.get_binop(op, self.datatype, other.datatype)
    return ExprNode(optrait.result_type, optrait, args=(self, other))


@dataclass(frozen=True)
class DFNode:
    ...


def node_replace_attrs(node: DFNode, **attrs):
    return _dc_replace(node, **attrs)


@dataclass(frozen=True)
class ValueNode(DFNode):
    datatype: _dt.DataType

    def __post_init__(self):
        assert isinstance(self.datatype, _dt.DataType), type(self.datatype)

    __add__ = partialmethod(_generic_binop, op="+")
    __sub__ = partialmethod(_generic_binop, op="-")
    __mul__ = partialmethod(_generic_binop, op="*")

    __le__ = partialmethod(_generic_binop, op="<=")
    __ge__ = partialmethod(_generic_binop, op=">=")
    __gt__ = partialmethod(_generic_binop, op=">")
    __lt__ = partialmethod(_generic_binop, op="<")

    def __getattr__(self, attr: str):
        attrop = self.datatype.attribute_lookup(attr)
        return ExprNode(attrop.result_type, attrop, (self,))


class Scope(Mapping):
    _values: dict[ArgNode, ValueNode]

    def __init__(self, values):
        self._values = dict(values)

    def __repr__(self):
        out = []
        for k, v in self._values.items():
            out.append(f"{k.name}:{k.datatype} = {type(v).__qualname__}")
        return f"Scope({', '.join(out)})"

    def __getitem__(self, k: ArgNode) -> ValueNode:
        return self._values[k]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def _pprint_Scope(self, object, stream, indent, allowance, context, level):
    assert isinstance(object, Scope)
    write = stream.write
    write("Scope{")
    if self._indent_per_level > 1:
        write((self._indent_per_level - 1) * " ")
    length = len(object)
    if length:
        items = [(k, v) for k, v in object.items()]
        self._format_dict_items(
            items, stream, indent, allowance + 1, context, level
        )
    write("}")


PrettyPrinter._dispatch[Scope.__repr__] = _pprint_Scope


class ArgBinder:
    __sig: Signature
    __argnodes: dict[str, ArgNode]
    __valnodes: dict[str, ValueNode]

    def __init__(self, sig_or_func):
        if isinstance(sig_or_func, Signature):
            sig = sig_or_func
        else:
            sig = signature(sig_or_func)
        self.__sig = sig

    def bind_to_datatypes(self, args, kwargs):
        ba = self.__sig.bind(*args, **kwargs)
        self.__argnodes = {
            k: ArgNode(dt, name=k) for k, dt in ba.arguments.items()
        }

    def bind_to_valuenodes(self, args, kwargs):
        ba = self.__sig.bind(*args, **kwargs)
        self.__valnodes = ba.arguments
        self.__argnodes = {
            k: ArgNode(vn.datatype, name=k) for k, vn in ba.arguments.items()
        }

    def get_scope(self):
        argnodes = self.__argnodes
        return {argnodes[k]: vn for k, vn in self.__valnodes.items()}

    def get_argnodes(self):
        return self.__argnodes


@dataclass(frozen=True)
class RegionNode(DFNode):
    def _pre_call(self, args: tuple[ValueNode], kwargs: dict[str, ValueNode]):
        args = as_node_args(args)
        kwargs = as_node_kwargs(kwargs)
        return args, kwargs

    def _call_region(
        self, region_func, args: tuple[ValueNode], kwargs: dict[str, ValueNode]
    ) -> tuple[Scope, ValueNode | Iterable[ValueNode]]:
        ab = ArgBinder(region_func)
        ab.bind_to_valuenodes(args, kwargs)
        arguments = ab.get_argnodes()
        scope = ab.get_scope()

        res = region_func(**arguments)
        if not isgenerator(res):
            res = as_node(res)
        return scope, res


def _call_func_no_post(func, argtys):
    ab = ArgBinder(func)
    ab.bind_to_datatypes(argtys, {})
    argnodes = ab.get_argnodes()
    res = func(**argnodes)
    return tuple(argnodes.values()), res


def _call_func(func, argtys):
    argnodes, ret = _call_func_no_post(func, argtys)
    return argnodes, as_node(ret)


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
        argnodes, node = _call_func(self.func, self.argtys)
        if node.datatype != self.retty:
            raise _dt.TypeOpError(
                f"FuncDef returned {node.datatype} instead of {self.retty}"
            )

        return FuncDef(self.func, self.argtys, self.retty, argnodes, node)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return CallNode.make(self, *args, **kwargs)


@dataclass(frozen=True)
class FuncDef:
    func: Callable
    argtys: tuple[_dt.DataType, ...]
    retty: _dt.DataType
    argnodes: tuple[ArgNode, ...]
    node: DFNode

    def __post_init__(self):
        for x in self.argtys:
            assert isinstance(x, _dt.DataType)
        assert isinstance(self.retty, _dt.DataType)

    def bind_scope(self, args, kwargs) -> Scope:
        ba = signature(self.func).bind(*args, **kwargs)
        return Scope({an: ba.arguments[an.name] for an in self.argnodes})


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
        [args, kwargs] = self._pre_call(args, kwargs)
        case_nodes: Iterable[CaseNode]
        _scope, case_nodes = self._call_region(self.region_func, args, kwargs)
        # Get case nodes
        nodes = []
        for cn in case_nodes:
            case_scope, case_res = self._call_region(cn.case_func, args, kwargs)
            nodes.append(EnterNode.make(cn, case_res, scope=case_scope))

        for n in nodes[1:]:
            if nodes[0].datatype != n.datatype:
                raise _dt.TypeOpError(
                    f"incompatible type {n.datatype}; expect {nodes[0].datatype}"
                )
        out = EnterNode.make(
            self,
            CaseExprNode(nodes[0].datatype, self.pred_node, tuple(nodes)),
            scope={},
        )
        return out


@dataclass(frozen=True)
class LoopNode(RegionNode):
    region_func: Callable

    @classmethod
    def make(cls, func):
        return cls(func)

    def __call__(self, *args: Any, **kwargs: Any):
        [args, kwargs] = self._pre_call(args, kwargs)
        scope, nodes = self._call_region(self.region_func, args, kwargs)
        cont, values = nodes
        body_values = tuple([cont, *values])

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

    def __post_init__(self):
        assert isinstance(self.cases, tuple)


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
    scope: Scope

    def __hash__(self):
        return id(self)


@dataclass(frozen=True)
class LoopExprNode(DFNode):
    parent: LoopNode
    pred: ValueNode
    values: tuple[ValueNode, ...]

    def __iter__(self):
        return iter(self.values)


@dataclass(frozen=True, order=False)
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


@as_node.register
def _(val: tuple):
    return tuple(map(as_node, val))


def as_node_args(args) -> tuple[ValueNode, ...]:
    return tuple([as_node(v) for v in args])


def as_node_kwargs(kwargs) -> dict[str, ValueNode]:
    return {k: as_node(v) for k, v in kwargs.items()}


def _argnode_uuid():
    return str(uuid.uuid4())


@dataclass(frozen=True)
class ArgNode(ValueNode):
    name: str

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    def __repr__(self):
        return f"ArgNode({self.name!r} at {hex(id(self))})"


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
    scope: Scope

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


def _sentry_scope(scope: Scope[ArgNode, DFNode] | None):
    if scope is not None:
        for k, v in scope.items():
            assert isinstance(k, ArgNode)
            assert isinstance(v, DFNode)


def cast(value: ValueNode, to_type: _dt.DataType) -> DFNode:
    to_type = _dt.ensure_type(to_type)
    op = to_type.get_cast(value.datatype)
    return ExprNode(op.result_type, op, args=tuple([value]))


def make(__dt: _dt.DataType, *args, **kwargs) -> DFNode:
    dt = _dt.ensure_type(__dt)
    args = dt.get_make(args, kwargs)
    return ExprNode(dt, _dt.MakeOp(dt), args=tuple(args.values()))


def call(__func: Callable, *args, **kwargs) -> DFNode:
    from .typedefs import Function

    assert callable(__func)
    fty = Function.lookup(__func)
    op = fty.get_call(args, kwargs)
    return CallNode(op.result_type, op, args, kwargs)
