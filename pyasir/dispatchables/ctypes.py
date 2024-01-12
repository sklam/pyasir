from __future__ import annotations

from functools import singledispatch
from pyasir import datatypes as _dt


@singledispatch
def emit_c_type(datatype: _dt.DataType):
    raise NotImplementedError(datatype)
