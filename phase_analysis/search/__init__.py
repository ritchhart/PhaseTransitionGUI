"""Search method registry — drop in a new module and it appears in the GUI."""

from phase_analysis.search.base import SearchMethod, SearchResult, Parameter

# Registry
_METHODS: dict[str, type[SearchMethod]] = {}


def register(cls: type[SearchMethod]) -> type[SearchMethod]:
    """Decorator to register a search method."""
    _METHODS[cls.name.fget(cls)] = cls  # access property on class
    return cls


def get_method(name: str) -> type[SearchMethod]:
    return _METHODS[name]


def available_methods() -> list[type[SearchMethod]]:
    """Return all registered methods (order stable)."""
    return list(_METHODS.values())


# Auto-import implementations so they self-register
from phase_analysis.search import smcr_method      # noqa: F401
from phase_analysis.search import smcr_method_broaden      # noqa: F401
from phase_analysis.search import peak_match_method # noqa: F401
