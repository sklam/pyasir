from __future__ import annotations


from functools import cache
from ..datatypes import DataType, OpTrait, ensure_type


class Packed(DataType):
    elements = ()
    _cache = {}

    def __init__(self):
        assert self.elements

    @classmethod
    @cache
    def make(cls, *subtypes: DataType) -> Packed:
        ty = cls[subtypes]
        return ty()

    def __class_getitem__(cls, subtypes):
        subtypes = tuple([ensure_type(t) for t in subtypes])
        try:
            return cls._cache[subtypes]
        except KeyError:
            ty = type(
                f'Packed[{", ".join(map(str, subtypes))}]',
                (cls,),
                {"elements": subtypes},
            )
            cls._cache[subtypes] = ty
            return ty
