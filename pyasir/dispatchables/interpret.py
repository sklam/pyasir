from __future__ import annotations

from typing import Any
from functools import singledispatch

from pyasir import datatypes as _dt


@singledispatch
def eval_op(op: _dt.OpTrait, *args: Any):
    raise NotImplementedError(f"no eval_op is implemented for {op}")
