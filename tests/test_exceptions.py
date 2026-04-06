from vbacktest.exceptions import (
    VBacktestError, ConfigError, DataError, StrategyError,
    ValidationError, RegistryError,
)


def test_exception_hierarchy():
    assert issubclass(ConfigError, VBacktestError)
    assert issubclass(DataError, VBacktestError)
    assert issubclass(StrategyError, VBacktestError)
    assert issubclass(ValidationError, VBacktestError)
    assert issubclass(RegistryError, VBacktestError)


def test_vbacktest_error_is_exception():
    assert issubclass(VBacktestError, Exception)


def test_exceptions_carry_message():
    e = ConfigError("bad config")
    assert str(e) == "bad config"
    e2 = DataError("missing column", details={"column": "close"})
    assert e2.details == {"column": "close"}


def test_details_default_empty():
    e = VBacktestError("test")
    assert e.details == {}
