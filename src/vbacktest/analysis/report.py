"""TestResult and ValidationReport data types + terminal/markdown rendering."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

# ANSI colour codes
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _colour(text: str, status: str) -> str:
    c = {"PASS": _GREEN, "WARN": _YELLOW, "FAIL": _RED}.get(status, "")
    return f"{c}{text}{_RESET}"


def _icon(status: str) -> str:
    return {"PASS": "[v]", "WARN": "[!]", "FAIL": "[x]"}.get(status, "[?]")


@dataclass
class TestResult:
    """Single test outcome within a validation category."""

    name: str
    value: str       # formatted display value
    status: str      # "PASS", "WARN", or "FAIL"
    hard_nogo: bool = False
    detail: str = ""


@dataclass
class ValidationReport:
    """Full go/no-go report for one strategy."""

    name: str
    results: dict[str, list[TestResult]] = field(default_factory=dict)
    overall: str = "PASS"
    hard_nogo_triggered: bool = False

    def verdict(self) -> str:
        """Return overall verdict string."""
        return self.overall

    def to_dict(self) -> dict[str, Any]:
        """Structured dict for programmatic access."""
        return {
            "name": self.name,
            "overall": self.overall,
            "hard_nogo": self.hard_nogo_triggered,
            "categories": {
                cat: [
                    {
                        "name": t.name,
                        "value": t.value,
                        "status": t.status,
                        "hard_nogo": t.hard_nogo,
                        "detail": t.detail,
                    }
                    for t in tests
                ]
                for cat, tests in self.results.items()
            },
        }

    def print_terminal(self) -> None:
        """Print ANSI-coloured report to stdout."""
        print(f"\n{_BOLD}Strategy: {self.name}{_RESET}")
        print("-" * 56)
        for category, tests in self.results.items():
            if not tests:
                continue
            print(f"  {_BOLD}{category}{_RESET}")
            for t in tests:
                col_icon = _colour(_icon(t.status), t.status)
                col_status = _colour(t.status, t.status)
                nogo_marker = (
                    f" {_RED}HARD NO-GO{_RESET}"
                    if t.hard_nogo and t.status == "FAIL"
                    else ""
                )
                detail = f"  {t.detail}" if t.detail else ""
                print(
                    f"  {col_icon} {t.name:<28} {t.value:<35} "
                    f"{col_status}{nogo_marker}{detail}"
                )

        warn_count = sum(
            1 for cat in self.results.values() for t in cat if t.status == "WARN"
        )
        fail_count = sum(
            1 for cat in self.results.values() for t in cat if t.status == "FAIL"
        )
        verdict_str = f"VERDICT: {self.overall}"
        print(f"\n  {_BOLD}{_colour(verdict_str, self.overall)}{_RESET}", end="")
        if warn_count:
            print(f"  ({warn_count} warning{'s' if warn_count > 1 else ''})", end="")
        if fail_count:
            print(f"  ({fail_count} failure{'s' if fail_count > 1 else ''})", end="")
        print()

    def write_markdown(self, path: str) -> None:
        """Write markdown report to *path*."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        lines.append(f"# Go/No-Go Validation Report — {date.today()}\n")
        lines.append(
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
        )
        lines.append(f"**Strategy:** {self.name}  \n")
        lines.append("## Summary\n")
        lines.append("| Verdict | Warnings | Failures | Hard NO-GO |")
        lines.append("|---|---|---|---|")
        w = sum(
            1 for cat in self.results.values() for t in cat if t.status == "WARN"
        )
        f = sum(
            1 for cat in self.results.values() for t in cat if t.status == "FAIL"
        )
        nogo = "YES" if self.hard_nogo_triggered else "No"
        lines.append(f"| {self.overall} | {w} | {f} | {nogo} |")
        lines.append("")

        lines.append("## Detailed Results\n")
        for category, tests in self.results.items():
            if not tests:
                continue
            lines.append(f"**{category}**\n")
            lines.append("| Test | Value | Status |")
            lines.append("|---|---|---|")
            for t in tests:
                status_label = t.status
                if t.hard_nogo and t.status == "FAIL":
                    status_label += " (hard NO-GO)"
                detail = f" — {t.detail}" if t.detail else ""
                lines.append(f"| {t.name} | {t.value}{detail} | {status_label} |")
            lines.append("")

        out.write_text("\n".join(lines))
