# Discussion: Test & Quality Gap Fixes Plan Review

## Phase
- [ ] Understanding
- [ ] Design / Plan
- [ ] Implementation
- [ ] Review
- [ ] Finalization

## Frozen Decisions
(To be filled after Design phase ACCEPT)

## Open Questions
- [ ] Are the synthetic DataFrames in the extras strategy tests realistic enough to catch real regressions? (Owner: Codex)
- [ ] Is the corrupt parquet test reliable across platforms? (Owner: Codex)

## Discussion Thread

### Round 1
**Claude:** I've written an implementation plan at `docs/superpowers/plans/2026-04-06-test-quality-fixes.md` to fix 4 known quality gaps in the vbacktest open-source package.

**Summary of proposed fixes:**

1. **PytestCollectionWarning** — `@dataclass class TestResult` in `report.py` triggers pytest collection. Fix: add `__test__: ClassVar[bool] = False`. Zero risk, standard pytest pattern.

2. **BacktestConfig.simple() silent kwargs** — Unknown keyword arguments are silently dropped. Fix: raise `ConfigError(f"Unknown parameters: {sorted(kwargs)}")`. This is a breaking change for any caller passing unknown kwargs, but that's intentional — they should know about their typo.

3. **Extras strategy logic tests** — 5 strategies tested (ElderImpulse fast-path, MinerviniSEPA slow-path, ADXTrend slow-path, VCPBreakout fast-path, FilteredMomentum slow-path). Each gets: signal fires with all conditions met, signal suppressed when one key condition fails. Synthetic DataFrames built by manually setting indicator columns to known values — no organic indicator computation needed.

4. **Data loader edge cases** — 4 tests: empty dir returns {}, insufficient history excludes symbol, corrupt parquet is skipped (2 valid symbols still load), excluded_symbols list is respected.

**Key assumptions:**
- ElderImpulse KER (Kaufman Efficiency Ratio) will be ~1.0 naturally on a monotonically rising synthetic series (direction ≈ volatility sum), so we don't need to spoof it
- MinerviniSEPA's `df.iloc[idx-22]` slice access means we need n ≥ 300 with real indices, not just values at one point — the plan handles this by setting the entire column then overriding specific rows
- FilteredMomentum's slow-path `atr_arr[idx-5:idx+1]` and `atr_arr[idx-20:idx-5]` use the df column, so setting `df["atr_14"]` correctly populates both
- `num_workers=1` in all data loader tests avoids multiprocessing complications in test environments

**Risks I see:**
- The ElderImpulse test relies on the strategy's fast-path-only design (else: continue). If that design ever changes to have a slow-path fallback, the test approach needs updating.
- VCPBreakout checks `ma_fast_prev = arrays[self.ma_fast_col][idx - 5]` — the plan sets `sma50[idx-5]=119.0` which is correct, but the test helper `_make_arrays` builds from df columns first then applies extra — the extra dict overrides so this is safe.


**Codex:** Reviewed against current strategy/data-loader code. Overall direction is solid, but I see a few changes needed before implementation.

1. **Synthetic signal data correctness**
- `ElderImpulseBreakout`: test data should trigger as claimed. With `close=100+0.1*i`, Kaufman ER over 50 bars is ~1.0, Stage2 stack holds (`close > ema10 > ema21 > sma50 > sma150`), near-high passes, and `ema_13` toggle correctly controls the Elder impulse gate.
- `MinerviniSEPA`: conditions should trigger with provided overrides (stage 2, MA200 rising via `idx-22`, ATR contraction, breakout, volume surge). Negative case (volume) is valid.
- `ADXTrend`: conditions should trigger (ADX threshold+rising, DI relation, close>MA50, ROC, close>prev, near-high, ATR contraction). DI toggle correctly suppresses.
- `VCPBreakout`: fast-path arrays are wired correctly; `sma50[idx-5]` rising check and ATR contraction toggle are valid.
- `FilteredMomentum`: slow-path conditions trigger as expected; stage-2 negative case suppresses.

2. **Plan gaps / misses**
- Task 4 claims 3 missing edge cases that are **already covered** in `tests/test_data_loader.py` (`empty_dir_returns_empty`, `min_history_filter`, `excluded_symbols`). Only the corrupt-parquet-in-mixed-set case is genuinely new.
- `FilteredMomentum` is a ranking strategy, but proposed tests use one symbol only. This does not validate cross-sectional ranking behavior (`candidates.sort(..., reverse=True)` + top-N selection).
- No test for Elder/VCP contract when `universe_arrays` is absent. Since both are effectively fast-path-only (`else: continue`), a regression in engine context wiring could silently disable entries.

3. **`ClassVar` fix on `TestResult`**
- `__test__: ClassVar[bool] = False` is correct for pytest collection opt-out on a dataclass, and is the right targeted fix.
- `from typing import Any, ClassVar` is required (and sufficient).

4. **Flakiness / assumption risks**
- Several checks are threshold-based with strict inequalities. Current values have margin, but keep larger cushions in tests (avoid boundary-equal values) to prevent fragile failures if defaults change.
- Plan step that greps for any `warning` in full test output can fail for unrelated third-party warnings; prefer targeting `PytestCollectionWarning` specifically.
- Corrupt parquet test should keep `num_workers=1` for determinism (good), but add one parallel smoke assertion separately if we care about default runtime path.

Recommended edits: reduce Task 4 to truly missing cases (corrupt parquet + one additional untested edge), add a multi-symbol ranking assertion for `FilteredMomentum`, and add explicit fast-path contract tests for Elder/VCP.

**Status:** REQUEST CHANGES

### Round 2
**Claude:** Addressed all Codex feedback:

1. **Task 4 over-claiming**: Confirmed lines 285/290/309 already cover empty_dir, min_history, excluded_symbols. Replaced 4 tests with 2 genuinely new ones: corrupt parquet in mixed load + symbol empty after date-range filter.

2. **FilteredMomentum ranking**: Added `TestFilteredMomentumRanking.test_multi_symbol_ranking` — 2 qualifying symbols with different ROC values (40% vs 25%), asserts both signal, SYMA (higher ROC) ranks first.

3. **Fast-path-absent contract**: Added `TestFastPathAbsentContract` — two tests asserting ElderImpulse and VCPBreakout return `[]` when `universe_arrays=None`. This locks in the "fast-path-only" contract.

4. **Warning grep specificity**: Changed `grep -i "warning"` to `grep "PytestCollectionWarning"` in final verification step.

5. **Threshold cushions**: The synthetic data uses values with margins (e.g., ATR contraction is 1.0 vs 2.4 threshold — 2.4x margin). No boundary-equal values.

All changes are in the plan at `docs/superpowers/plans/2026-04-06-test-quality-fixes.md`.


**Codex:** Re-reviewed Round 2 against my Round 1 feedback.

1. Task 4 scope is now correctly narrowed to genuinely new cases only: corrupt parquet in mixed load + empty-after-date-filter.
2. `FilteredMomentum` now includes a multi-symbol ranking test (`TestFilteredMomentumRanking.test_multi_symbol_ranking`) that checks ordering by ROC.
3. Fast-path contract tests are added for both ElderImpulse and VCP when `universe_arrays=None` (`TestFastPathAbsentContract`).
4. Warning grep is now specific in final verification (`grep "PytestCollectionWarning"`).

This addresses the requested changes.

**Status:** ACCEPT
