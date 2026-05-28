"""Stage factory and registry."""

_registry: dict[str, type] = {}


def register_stage(name: str):
    """Decorator to register a stage class."""
    def wrapper(cls):
        _registry[name] = cls
        return cls
    return wrapper


def get_stage(name: str):
    """Return stage instance by name."""
    if name not in _registry:
        raise KeyError(f"Unknown stage '{name}'. Available: {list(_registry.keys())}")
    return _registry[name]()


def list_stages():
    """Return list of registered stage names."""
    return list(_registry.keys())


# Import stage modules to trigger @register_stage decorators
from pipeline.stages import masks    # noqa: E402,F401
from pipeline.stages import hunyuangen  # noqa: E402,F401
from pipeline.stages import scale       # noqa: E402,F401
from pipeline.stages import package     # noqa: E402,F401
