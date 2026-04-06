"""CLI tests."""
from __future__ import annotations

import subprocess
import sys


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vbacktest.cli.main", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "vbacktest" in result.stdout.lower()


def test_cli_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vbacktest.cli.main", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_cli_strategies_list() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vbacktest.cli.main", "strategies"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ma_crossover" in result.stdout
    assert "rsi_mean_reversion" in result.stdout
    assert "turtle_trading" in result.stdout


def test_cli_strategies_lists_all_core() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vbacktest.cli.main", "strategies"],
        capture_output=True,
        text=True,
    )
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    assert len(lines) >= 6  # at least the 6 core strategies


def test_cli_no_command_prints_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vbacktest.cli.main"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower() or "vbacktest" in result.stdout.lower()
