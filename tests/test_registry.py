"""Tests for strategy and indicator registries."""
from __future__ import annotations

import threading

import pytest

from vbacktest.exceptions import RegistryError
from vbacktest.registry import Registry, strategy_registry, indicator_registry


# ---------------------------------------------------------------------------
# Registry unit tests (using a fresh instance per test)
# ---------------------------------------------------------------------------

@pytest.fixture()
def reg() -> Registry:
    return Registry("test")


class TestRegistryRegister:
    def test_register_and_get(self, reg: Registry) -> None:
        @reg.register("foo")
        class Foo:
            pass

        assert reg.get("foo") is Foo

    def test_register_decorator_returns_original(self, reg: Registry) -> None:
        class Bar:
            pass

        result = reg.register("bar")(Bar)
        assert result is Bar

    def test_register_fn(self, reg: Registry) -> None:
        def my_fn() -> str:
            return "hello"

        reg.register_fn("my_fn", my_fn)
        assert reg.get("my_fn") is my_fn

    def test_duplicate_key_raises(self, reg: Registry) -> None:
        @reg.register("dup")
        class A:
            pass

        with pytest.raises(RegistryError, match="already registered"):
            @reg.register("dup")
            class B:
                pass

    def test_override_replaces(self, reg: Registry) -> None:
        @reg.register("k")
        class V1:
            pass

        @reg.register("k", override=True)
        class V2:
            pass

        assert reg.get("k") is V2

    def test_contains(self, reg: Registry) -> None:
        @reg.register("x")
        class X:
            pass

        assert "x" in reg
        assert "y" not in reg

    def test_keys_sorted(self, reg: Registry) -> None:
        for k in ["c", "a", "b"]:
            reg.register_fn(k, lambda: None)
        assert reg.keys() == ["a", "b", "c"]

    def test_get_missing_raises(self, reg: Registry) -> None:
        with pytest.raises(RegistryError, match="not found"):
            reg.get("nonexistent")

    def test_error_includes_available_keys(self, reg: Registry) -> None:
        reg.register_fn("option_a", lambda: None)
        try:
            reg.get("missing")
        except RegistryError as exc:
            assert "option_a" in str(exc)


class TestRegistryBuild:
    def test_build_instantiates(self, reg: Registry) -> None:
        @reg.register("point")
        class Point:
            def __init__(self, x: int = 0, y: int = 0) -> None:
                self.x = x
                self.y = y

        p = reg.build("point", x=3, y=4)
        assert p.x == 3
        assert p.y == 4

    def test_build_calls_function(self, reg: Registry) -> None:
        calls: list[dict] = []

        def factory(**kw: object) -> dict:
            calls.append(kw)
            return kw

        reg.register_fn("fn", factory)
        result = reg.build("fn", a=1, b=2)
        assert result == {"a": 1, "b": 2}
        assert calls == [{"a": 1, "b": 2}]

    def test_build_missing_raises(self, reg: Registry) -> None:
        with pytest.raises(RegistryError):
            reg.build("nope")


class TestRegistryThreadSafety:
    def test_concurrent_registration(self) -> None:
        """Concurrent registrations should not corrupt the registry."""
        reg = Registry("thread_test")
        errors: list[Exception] = []

        def register_many(prefix: str) -> None:
            for i in range(50):
                try:
                    reg.register_fn(f"{prefix}_{i}", lambda: None)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=register_many, args=(f"t{j}",)) for j in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(reg.keys()) == 200  # 4 threads × 50 unique keys each


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

class TestModuleSingletons:
    def test_strategy_registry_is_registry(self) -> None:
        assert isinstance(strategy_registry, Registry)

    def test_indicator_registry_is_registry(self) -> None:
        assert isinstance(indicator_registry, Registry)

    def test_register_custom_strategy(self) -> None:
        from vbacktest.strategy import BarContext, ExitRule, Signal, Strategy
        from vbacktest.indicators import IndicatorSpec

        @strategy_registry.register("__test_custom__", override=True)
        class MyStrat(Strategy):
            def indicators(self) -> list[IndicatorSpec]:
                return []
            def on_bar(self, ctx: BarContext) -> list[Signal]:
                return []
            def exit_rules(self) -> list[ExitRule]:
                return []

        assert "__test_custom__" in strategy_registry
        # Build should instantiate
        instance = strategy_registry.build("__test_custom__")
        assert isinstance(instance, MyStrat)

    def test_register_custom_indicator(self) -> None:
        import pandas as pd

        def my_indicator(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
            df[f"my_ind_{period}"] = df["close"].rolling(period).mean()
            return df

        indicator_registry.register_fn("__test_ind__", my_indicator, override=True)
        assert "__test_ind__" in indicator_registry
        fn = indicator_registry.get("__test_ind__")
        assert callable(fn)
