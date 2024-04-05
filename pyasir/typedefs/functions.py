from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type
from functools import singledispatch

# from ctypes import c_double

# from llvmlite import ir

# from ..dispatchables.be_llvm import emit_llvm_type, emit_llvm_const, emit_llvm
# from ..dispatchables.ctypes import emit_c_type
from ..interpret import eval_op
from ..datatypes import TypeOpError, DataType, OpTrait
from .integers import Int64


_FunctionLike = Callable


registry: dict[_FunctionLike, Type[Function]] = {}


class Function(DataType):
    function: _FunctionLike

    @staticmethod
    def register(func: _FunctionLike):
        def wrap(cls: Type[Function]):
            registry[func] = cls
            return cls

        return wrap

    @staticmethod
    def lookup(func: _FunctionLike) -> Function:
        return registry[func]()


class Iterator(DataType):
    pass


class Int64Iterator(Iterator):
    element = Int64()


@dataclass(frozen=True)
class CallOp(OpTrait):
    function: _FunctionLike


@dataclass(frozen=True)
class RangeCallOp(CallOp):
    pass


@Function.register(range)
class RangeFunction(Function):
    function = range

    def get_call(self, args, kwargs):
        return RangeCallOp(result_type=Int64Iterator(), function=range)


@eval_op.register
def eval_op_RangeCallOp(op: RangeCallOp, args: tuple[tuple, dict]):
    args, kwargs = args
    assert len(args) == 1
    assert not kwargs, kwargs
    [stop] = args
    return iter(range(stop))
