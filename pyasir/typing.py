from typing import Annotated, get_type_hints, Callable
from typing_extensions import _AnnotatedAlias
from . import datatypes as _dt


def get_annotations(func: Callable):
    hints = get_type_hints(func, include_extras=True)
    return {k: _dt.ensure_type(v) for k, v in hints.items()}
    # return {k: parse_type(v, debug=k) for k, v in hints.items()}


# def parse_type(ty, debug):
#     if isinstance(ty, _AnnotatedAlias):
#         md = ty.__metadata__
#         tys = [t for t in md if isinstance(t, _dt.DataType)]
#         if len(tys) != 1:
#             raise TypeError(f"cannot annotate type with {len(tys)} DataType. (debug: {debug})")
#         return tys[0]
#     raise TypeError(f"invalid {ty}")
