from __future__ import annotations

from dataclasses import dataclass

from ..datatypes import DataType, OpTrait


class BooleanType(DataType):
    def get_binop(self, op: str, lhs: DataType, rhs: DataType) -> OpTrait:
        raise NotImplementedError

    def get_cast(self, valtype: DataType) -> OpTrait:
        raise NotImplementedError


Bool = BooleanType
