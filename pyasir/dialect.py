def __getattr__(name):
    from .dialects.registry import registry

    mod = registry.get(name)
    if mod is None:
        raise AttributeError(name)
    return mod
