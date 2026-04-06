"""Thread-safe strategy and indicator registries.

Registries allow users to register custom strategies and indicators by name,
enabling discovery and instantiation from configuration files.

Usage::

    from vbacktest.registry import strategy_registry, indicator_registry

    @strategy_registry.register("my_strategy")
    class MyStrategy(Strategy):
        ...

    # Or imperatively:
    strategy_registry.register("my_strategy", override=True)(MyStrategy)
    instance = strategy_registry.build("my_strategy")
"""
from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

from vbacktest.exceptions import RegistryError

_T = TypeVar("_T")


class Registry:
    """Generic thread-safe name → callable registry.

    Entries can be any callable (class, function, factory).  The ``build``
    method calls the registered callable with ``**kwargs``.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, Callable[..., Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        key: str,
        override: bool = False,
    ) -> Callable[[_T], _T]:
        """Decorator that registers *obj* under *key*.

        Args:
            key: Unique name for the entry.
            override: If ``True``, silently replace an existing entry.
                      If ``False`` (default) and the key already exists,
                      raise ``RegistryError``.

        Raises:
            RegistryError: If *key* already exists and ``override=False``.
        """
        def _decorator(obj: _T) -> _T:
            with self._lock:
                if key in self._entries and not override:
                    raise RegistryError(
                        f"{self._name} registry: key '{key}' already registered. "
                        "Use override=True to replace it.",
                        details={"key": key, "registry": self._name},
                    )
                self._entries[key] = obj  # type: ignore[assignment]
            return obj

        return _decorator

    def register_fn(
        self,
        key: str,
        obj: Callable[..., Any],
        *,
        override: bool = False,
    ) -> None:
        """Register *obj* imperatively (non-decorator form).

        Equivalent to ``register(key, override)(obj)`` but without returning
        the callable.
        """
        self.register(key, override=override)(obj)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, key: str) -> Callable[..., Any]:
        """Return the registered callable for *key*.

        Raises:
            RegistryError: If *key* is not found.
        """
        with self._lock:
            if key not in self._entries:
                available = sorted(self._entries.keys())
                raise RegistryError(
                    f"{self._name} registry: '{key}' not found. "
                    f"Available: {available}",
                    details={"key": key, "available": available},
                )
            return self._entries[key]

    def build(self, key: str, **kwargs: Any) -> Any:
        """Instantiate the registered callable for *key* with ``**kwargs``.

        Raises:
            RegistryError: If *key* is not found.
        """
        return self.get(key)(**kwargs)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def keys(self) -> list[str]:
        """Return a sorted list of registered keys."""
        with self._lock:
            return sorted(self._entries.keys())

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, entries={self.keys()})"


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

#: Global strategy registry.  Register custom strategies here.
strategy_registry: Registry = Registry("strategy")

#: Global indicator registry.  Maps names to indicator *functions*.
indicator_registry: Registry = Registry("indicator")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def register_strategy(key: str, override: bool = False) -> Any:
    """Decorator: register a strategy class in the global strategy registry."""
    return strategy_registry.register(key, override=override)


def register_indicator(key: str, override: bool = False) -> Any:
    """Decorator: register an indicator function in the global indicator registry."""
    return indicator_registry.register(key, override=override)


def get_strategy(key: str) -> Any:
    """Return strategy class registered under *key*.

    Raises :class:`~vbacktest.exceptions.RegistryError` if not found.
    """
    return strategy_registry.get(key)


def get_indicator(key: str) -> Any:
    """Return indicator function registered under *key*.

    Raises :class:`~vbacktest.exceptions.RegistryError` if not found.
    """
    return indicator_registry.get(key)


def list_strategies() -> list[str]:
    """Return sorted list of registered strategy keys."""
    return strategy_registry.keys()


def list_indicators() -> list[str]:
    """Return sorted list of registered indicator keys."""
    return indicator_registry.keys()


def _reset_registries() -> None:
    """Reset both registries to empty state. Intended for use in tests only."""
    strategy_registry._entries.clear()
    indicator_registry._entries.clear()
