import typing as _ty

def aot(f: _ty.Callable) -> _ty.Callable:
    return f

def struct(cl: _ty.Type) -> _ty.Type:
    return cl
