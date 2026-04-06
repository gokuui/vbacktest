#!/usr/bin/env python3
"""Port extra strategies from loser to vbacktest/extras.

Applies mechanical transformations:
- relative → absolute imports
- old on_bar signature → BarContext signature
- variable references in body
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path("/home/vinay/code/loser/src/backtest/strategies")
DST = Path("/home/vinay/code/vbacktest/src/vbacktest/strategies/extras")

CORE = {
    "bollinger_breakout.py", "ma_crossover.py", "momentum.py",
    "rsi_mean_reversion.py", "turtle_trading.py", "volume_breakout.py",
}
ML = {
    "ensemble_ml_breakout.py", "ml_breakout.py", "ml_feature_strategy.py",
}
SKIP = CORE | ML | {"__init__.py"}


def replace_on_bar_signature(text: str) -> str:
    """Replace the old multi-line on_bar signature with BarContext version.

    Finds: def on_bar(self, date, universe, ...) [-> list[Signal]]:
    Replaces with: def on_bar(self, ctx: BarContext) -> list[Signal]:
    """
    lines = text.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect start of old-style on_bar signature
        if re.match(r'\s*def on_bar\s*\(', line) and 'ctx: BarContext' not in line:
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent

            # Collect all lines until closing paren + colon
            sig_lines = [line]
            j = i + 1
            # Check if already closed on the first line
            closed = bool(re.search(r'\)\s*(?:->\s*list\[Signal\]\s*)?:\s*$', line))
            while not closed and j < len(lines):
                sig_lines.append(lines[j])
                if re.search(r'\)\s*(?:->\s*list\[Signal\]\s*)?:\s*$', lines[j]):
                    closed = True
                j += 1

            if closed:
                # Verify it has old params (date/universe/portfolio)
                full_sig = '\n'.join(sig_lines)
                if re.search(r'\bdate\b', full_sig) and re.search(r'\buniverse\b', full_sig):
                    result.append(f'{indent_str}def on_bar(self, ctx: BarContext) -> list[Signal]:')
                    i = j  # skip all sig_lines
                    continue

        result.append(line)
        i += 1

    return '\n'.join(result)


def transform(src_text: str) -> str:
    text = src_text

    # 1. Add future annotations after module docstring
    if 'from __future__ import annotations' not in text:
        if text.startswith('"""') or text.startswith("'''"):
            q = '"""' if text.startswith('"""') else "'''"
            end = text.find(q, 3)
            if end != -1:
                end += 3
                text = text[:end] + "\nfrom __future__ import annotations\n" + text[end:]
            else:
                text = 'from __future__ import annotations\n\n' + text
        else:
            text = 'from __future__ import annotations\n\n' + text

    # 2. Replace on_bar signature FIRST (before body variable replacement)
    text = replace_on_bar_signature(text)

    # 3. Fix relative imports → absolute imports
    text = re.sub(r'from \.\.(strategy) import', r'from vbacktest.\1 import', text)
    text = re.sub(r'from \.\.(indicators) import', r'from vbacktest.\1 import', text)
    text = re.sub(r'from \.\.(exit_rules) import', r'from vbacktest.\1 import', text)
    text = re.sub(r'from \.\.(portfolio) import', r'from vbacktest.\1 import', text)
    text = re.sub(r'from \.\.(config) import', r'from vbacktest.\1 import', text)
    text = re.sub(r'from \.\.([\w]+) import', r'from vbacktest.\1 import', text)
    text = re.sub(r'from \.([\w]+) import', r'from vbacktest.strategies.\1 import', text)

    # 4. Ensure BarContext is imported
    def add_bar_context(m: re.Match) -> str:
        line = m.group(0)
        if 'BarContext' not in line:
            if 'Strategy' in line:
                line = line.replace('Strategy', 'BarContext, Strategy')
            else:
                # append before closing paren or at end
                line = line.rstrip()
                line += ', BarContext'
        return line

    text = re.sub(r'from vbacktest\.strategy import [^\n]+', add_bar_context, text)

    # 5. Remove TYPE_CHECKING / Portfolio import block (no longer needed)
    text = re.sub(r'from typing import TYPE_CHECKING\n', '', text)
    text = re.sub(r'if TYPE_CHECKING:\n(?:    from [^\n]+\n)+\n?', '', text)

    # 6. Replace body variable references
    # use_fast_path detection
    text = re.sub(
        r'use_fast_path\s*=\s*current_prices\s+is\s+not\s+None\s+and\s+universe_arrays\s+is\s+not\s+None',
        'use_fast_path = ctx.universe_arrays is not None',
        text
    )
    text = re.sub(
        r'use_fast_path\s*=\s*universe_arrays\s+is\s+not\s+None',
        'use_fast_path = ctx.universe_arrays is not None',
        text
    )

    # combined fast path guard
    text = re.sub(
        r'if use_fast_path and symbol in current_prices and symbol in universe_arrays:',
        'if use_fast_path and ctx.current_prices and symbol in ctx.current_prices and symbol in ctx.universe_arrays:',
        text
    )

    # universe_arrays references (standalone var, not ctx.universe_arrays)
    text = re.sub(r'(?<!ctx\.)\buniverse_arrays\b', 'ctx.universe_arrays', text)

    # current_prices references
    text = re.sub(r'(?<!ctx\.)\bcurrent_prices\b', 'ctx.current_prices', text)

    # universe_idx references
    text = re.sub(r'(?<!ctx\.)\buniverse_idx\b', 'ctx.universe_idx', text)

    # universe.items() and universe[symbol] — careful not to hit ctx.universe_arrays
    text = re.sub(r'(?<!ctx\.)\buniverse\.items\(\)', 'ctx.universe.items()', text)
    # universe[ — but not universe_arrays[ or universe_idx[
    text = re.sub(r'(?<!ctx\.)\buniverse\[', 'ctx.universe[', text)

    # portfolio references (standalone)
    text = re.sub(r'(?<!ctx\.)\bportfolio\.has_position\b', 'ctx.portfolio.has_position', text)
    text = re.sub(r'(?<!ctx\.)\bportfolio\.get_position\b', 'ctx.portfolio.get_position', text)
    # catch remaining `portfolio` not preceded by ctx.
    text = re.sub(r'(?<!ctx\.)\bportfolio\b', 'ctx.portfolio', text)

    # date=date → date=ctx.date in Signal() calls
    text = re.sub(r'\bdate=date\b', 'date=ctx.date', text)

    # Fix renamed exit rules
    text = text.replace('MaxHoldingDaysRule', 'MaxHoldingBarsRule')

    return text


def port_file(src_path: Path) -> None:
    text = src_path.read_text()
    result = transform(text)
    dst_path = DST / src_path.name
    dst_path.write_text(result)


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in SRC.glob("*.py") if f.name not in SKIP)
    print(f"Porting {len(files)} strategy files...")
    for f in files:
        port_file(f)
        print(f"  ported: {f.name}")
    print("Done.")


if __name__ == "__main__":
    main()
