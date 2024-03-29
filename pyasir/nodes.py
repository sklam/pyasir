from __future__ import annotations

import weakref
from dataclasses import (
    dataclass,
    replace as _dc_replace,
    fields as _dc_fields,
)
from functools import partial, partialmethod, singledispatch
from typing import Any, Callable, Iterable, Annotated, Union
from inspect import signature, isgenerator, Signature
from collections.abc import Mapping
from pprint import PrettyPrinter, pprint

import pyasir
from . import datatypes as _dt
from . import typing as _tp
from . import dialect  # reexport
from .details import props as _props

def _generic_binop(self, other, *, op):
    other = as_node(other)
    optrait = self.datatype.get_binop(op, self.datatype, other.datatype)
    return ExprNode(optrait.result_type, optrait, args=(self, other))


ChildTypes = Union["DFNode", tuple["DFNode", ...]]


_pprint_cache = {}


def _format_namespace_items(self: PrettyPrinter, items, stream, indent, allowance, context, level):
    # Adapted from builtin PrettyPrinter._format_namespace_items
    # to be more compact
    write = stream.write
    write('\n' + ' ' * indent)
    delimnl = ',\n' + ' ' * indent
    last_index = len(items) - 1
    for i, (key, ent) in enumerate(items):
        last = i == last_index
        write(key)
        write('=')
        if id(ent) in context:
            # Special-case representation of recursion to match standard
            # recursive dataclass repr.
            write(f"...[{hex(id(ent))}]")
        else:
            self._format(ent, stream, indent + len(key) + 1,
                            allowance if last else 1,
                            context, level)

            write(delimnl)


def _pformat(self: PrettyPrinter, object, stream, indent, allowance, context, level):
    # Adapted from builtin PrettyPrinter._pprint_dataclass
    # to be more compact
    objid = id(object)
    if objid in _pprint_cache:
        stream.write(_pprint_cache[objid])
        return

    first = not _pprint_cache

    ref = 1 + len(_pprint_cache)
    _pprint_cache[objid] = f"<@{ref} : {type(object).__name__}>"
    try:
        cls_name = object.__class__.__name__
        indent += 2 #len(cls_name) + 1
        items = [(f.name, getattr(object, f.name))
                for f in _dc_fields(object) if f.repr]
        stream.write(cls_name + f"@{ref}" + '(')
        _format_namespace_items(self, items, stream, indent, allowance, context, level)
        stream.write(')')
    finally:
        if first:
            _pprint_cache.clear()


def _pprint_Scope(self, object, stream, indent, allowance, context, level):
    assert isinstance(object, Scope)
    write = stream.write
    write("Scope{")
    if self._indent_per_level > 1:
        write((self._indent_per_level - 1) * " ")
    length = len(object)
    if length:
        items = [(k.name, v) for k, v in object.items()]
        _format_namespace_items(
            self,
            items, stream, indent, allowance + 1, context, level
        )
    write("}")


def custom_pprint(cls):
    PrettyPrinter._dispatch[cls.__repr__] = _pformat
    return cls


@custom_pprint
@dataclass(frozen=True)
class DFNode:

    def dump_shorten(self) -> str:
        fields = _dc_fields(self)
        buf = []
        buf.append(f"{self.__class__.__name__} [{hex(id(self))}] (")
        for fd in fields:
            buf.append(
                f"  {fd.name:10}: {fd.type:50} [{hex(id(getattr(self, fd.name)))}]"
            )
        buf.append(f")")
        return "\n".join(buf)

    def to_graphviz(self):
        from .graphviz_backend import node_as_graphviz
        return node_as_graphviz(self)

    def get_child_nodes(self) -> dict[str, ChildTypes]:

        props = _props.get_properties(self.__class__)

        fields = _dc_fields(self)
        children = {}

        for fd in fields:
            obj = getattr(self, fd.name)
            if isinstance(obj, DFNode):
                children[fd.name] = obj

            elif _props.NodeChildren in props.get(fd.name):
                children[fd.name] = tuple(obj)

        return children

    def replace_child_nodes(self, repl: dict[str, ChildTypes]):
        return _dc_replace(self, **repl)


def node_replace_attrs(node: DFNode, **attrs):
    return _dc_replace(node, **attrs)


@custom_pprint
@dataclass(frozen=True)
class ValueNode(DFNode):
    datatype: _dt.DataType

    def __post_init__(self):
        assert isinstance(self.datatype, _dt.DataType), type(self.datatype)

    __add__ = __radd__ = partialmethod(_generic_binop, op="+")
    __sub__ = partialmethod(_generic_binop, op="-")
    __mul__ = partialmethod(_generic_binop, op="*")
    __and__ = partialmethod(_generic_binop, op="&")
    __or__ = partialmethod(_generic_binop, op="|")

    __le__ = partialmethod(_generic_binop, op="<=")
    __ge__ = partialmethod(_generic_binop, op=">=")
    __gt__ = partialmethod(_generic_binop, op=">")
    __lt__ = partialmethod(_generic_binop, op="<")

    __eq__ = partialmethod(_generic_binop, op="==")
    __ne__ = partialmethod(_generic_binop, op="!=")

    def __getattr__(self, attr: str):
        attrop = self.datatype.attribute_lookup(attr)
        return ExprNode(attrop.result_type, attrop, (self,))


@custom_pprint
@dataclass(frozen=True)
class EnterNode(ValueNode):
    region: RegionNode
    body: DFNode
    scope: Scope

    def __post_init__(self):
        assert isinstance(self.scope, Scope)

    @classmethod
    def make(cls, region, body, scope):
        _sentry_scope(scope)
        return cls(
            datatype=body.datatype, region=region, body=body, scope=scope
        )

    def __hash__(self):
        return id(self)


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
        return Scope({argnodes[k]: vn for k, vn in self.__valnodes.items()})

    def get_argnodes(self):
        return self.__argnodes

@custom_pprint
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

    def _call_region_loop(
        self, region_func, args: tuple[ValueNode], kwargs: dict[str, ValueNode]
    ) -> tuple[Scope, ValueNode | Iterable[ValueNode]]:
        ab = ArgBinder(region_func)
        ab.bind_to_valuenodes(args, kwargs)
        arguments = ab.get_argnodes()
        scope = ab.get_scope()

        res = region_func(**arguments)
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

@custom_pprint
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


@custom_pprint
@dataclass(frozen=True)
class FuncDef(DFNode):
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


@custom_pprint
@dataclass(frozen=True)
class SwitchNode(RegionNode):
    pred_node: DFNode
    region_func: Callable

    @classmethod
    def make(cls, pred, func=None):
        pred = as_node(pred)
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
        out = CaseExprNode(nodes[0].datatype, self.pred_node, tuple(nodes))
        return out

@custom_pprint
@dataclass(frozen=True)
class LoopNode(RegionNode):
    region_func: Callable

    @classmethod
    def make(cls, func):
        return cls(func)

    def __call__(self, *args: Any, **kwargs: Any):
        [args, kwargs] = self._pre_call(args, kwargs)
        scope, nodes = self._call_region_loop(self.region_func, args, kwargs)
        pred, body = nodes
        loopbody = pack(pred, body)
        return LoopExprNode(body.datatype, self, body=loopbody, scope=scope)

@custom_pprint
@dataclass(frozen=True)
class CaseExprNode(ValueNode):
    pred: DFNode
    cases: Annotated[tuple[EnterNode, ...], _props.NodeChildren]

    def __post_init__(self):
        assert isinstance(self.cases, tuple)

@custom_pprint
@dataclass(frozen=True)
class CaseNode(RegionNode):
    case_pred: DFNode
    case_func: Callable

    @classmethod
    def make(cls, case_pred: Any):
        return lambda fn: CaseNode(as_node(case_pred), fn)

@custom_pprint
@dataclass(frozen=True)
class LoopExprNode(EnterNode):

    def __hash__(self):
        return id(self)


@custom_pprint
@dataclass(frozen=True, order=False)
class UnpackNode(ValueNode):
    producer: DFNode
    index: int

@custom_pprint
@dataclass(frozen=True, order=False)
class ExpandNode(ValueNode):
    origin: ValueNode

    @classmethod
    def make(cls, node: ValueNode):
        assert isinstance(node.datatype, pyasir.Packed)
        return cls(node.datatype, origin=node)

    def __iter__(self):
        unpacked = [
            UnpackNode(ty, self.origin, i)
            for i, ty in enumerate(self.datatype.elements)
        ]
        return iter(unpacked)

@custom_pprint
@dataclass(frozen=True, order=False)
class PackNode(ValueNode):
    values: Annotated[tuple[ValueNode, ...], _props.NodeChildren]

    @classmethod
    def make(cls, *values: ValueNode):
        values = as_node_args(values)
        return cls(
            pyasir.Packed.make(*[v.datatype for v in values]), values=values
        )


func = FuncNode.make
switch = SwitchNode.make
case = CaseNode.make
loop = LoopNode.make

pack = PackNode.make
unpack = ExpandNode.make

@custom_pprint
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




@custom_pprint
@dataclass(frozen=True)
class ArgNode(ValueNode):
    name: str

    pool = weakref.WeakKeyDictionary()
    pool_naming = iter(range(2**64))

    def __post_init__(self):
        super().__post_init__()
        self.pool[self] = str(next(self.pool_naming))

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    def __repr__(self):
        return f"ArgNode({self.name!r}, uid={self._uid()})"
        # return f"ArgNode({self.name!r} at {hex(id(self))})"

    def _uid(self) -> str:
        return self.pool[self]

    def clone(self):
        return ArgNode(self.datatype, self.name)



@custom_pprint
@dataclass(frozen=True)
class ExprNode(ValueNode):
    op: _dt.OpTrait
    args: Annotated[tuple[ValueNode, ...], _props.NodeChildren]

    def __hash__(self):
        return id(self)

@custom_pprint
@dataclass(frozen=True)
class CallNode(ValueNode):
    func: FuncNode
    args: Annotated[tuple[ValueNode, ...], _props.NodeChildren]
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



def _sentry_scope(scope: Scope[ArgNode, DFNode] | None):
    if scope is not None:
        for k, v in scope.items():
            assert isinstance(k, ArgNode)
            assert isinstance(v, DFNode)


def cast(value: ValueNode, to_type: _dt.DataType) -> DFNode:
    value = as_node(value)
    to_type = _dt.ensure_type(to_type)
    if value.datatype == to_type:
        return value
    op = to_type.get_cast(value.datatype)
    return ExprNode(op.result_type, op, args=tuple([value]))


def make(__dt: _dt.DataType, *args, **kwargs) -> DFNode:
    dt = _dt.ensure_type(__dt)
    args = dt.get_make(as_node_args(args), as_node_kwargs(kwargs))
    return ExprNode(dt, _dt.MakeOp(dt), args=tuple(args.values()))


def call(__func: Callable, *args, **kwargs) -> DFNode:
    from .typedefs import Function

    assert callable(__func)
    fty = Function.lookup(__func)
    op = fty.get_call(args, kwargs)
    return CallNode(op.result_type, op, args, kwargs)


def zeroinit(ty: _dt.DataType) -> DFNode:
    op = ty.get_zero()
    return ExprNode(op.result_type, op, args=())

