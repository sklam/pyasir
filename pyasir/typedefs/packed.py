from __future__ import annotations


from functools import cache
from ..datatypes import DataType, OpTrait

class Packed(DataType):
    elements = ()
    def __init__(self):
        assert self.elements

    @classmethod
    @cache
    def make(cls, *subtypes: DataType) -> Packed:
        ty = type(f'Packed[{", ".join(map(str, subtypes))}]',
                    (cls,), {'elements': subtypes}
        )
        return ty()


