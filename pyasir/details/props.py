import sys
from typing import get_type_hints, Annotated, _AnnotatedAlias

class Property:
    pass


class NodeChildren(Property):
    pass


def get_properties(node_type):
    mod = sys.modules[node_type.__module__]
    ns = mod.__dict__
    hints = get_type_hints(node_type, ns, include_extras=True)
    props = {}
    for k, ty in hints.items():
        if isinstance(ty, _AnnotatedAlias):
            pset = frozenset(filter(lambda x: issubclass(x, Property),
                             ty.__metadata__))
        else:
            pset = frozenset()
        props[k] = pset
    return props

