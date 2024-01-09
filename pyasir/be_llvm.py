from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from llvmlite import ir

from . import nodes as _df


def generate(funcdef: _df.FuncDef, *args: Any, **kwargs: Any):
    be = LLVMBackend()
    be.emit_funcdef(funcdef, *args, **kwargs)


class LLVMBackend:
    def __init__(self):
        self._mod = ir.Module()

    def emit_funcdef(self, funcdef: _df.FuncDef, *args: Any, **kwargs: Any):
        pass
