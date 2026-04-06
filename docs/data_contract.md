# Data Contract

vbacktest is market-agnostic. It works with any market as long as the data matches this contract.

## Required Parquet Format

Each symbol must be a separate `.parquet` file named `{symbol}.parquet` in the same directory.

### Required Columns

| Column   | Type            | Description                |
|----------|-----------------|----------------------------|
| `date`   | `datetime64[ns]`| Trading date               |
| `open`   | `float64`       | Opening price              |
| `high`   | `float64`       | Highest price              |
| `low`    | `float64`       | Lowest price               |
| `close`  | `float64`       | Closing price              |
| `volume` | `float64`       | Trading volume             |

### Optional Columns

| Column       | Type     | Description                                 |
|--------------|----------|---------------------------------------------|
| `source`     | `string` | Data source identifier (informational)      |
| `confidence` | `int`    | Data quality score 0–2 (used for filtering) |

### Constraints

- All prices must be positive (`> 0`)
- OHLC validity: `low ≤ open ≤ high`, `low ≤ close ≤ high`
- Volume must be non-negative
- No infinite or NaN values in OHLC columns
- Data must be **pre-adjusted** for splits and dividends
- Dates must be sorted ascending, no duplicates per symbol

### Calendar

vbacktest does **not** assume any trading calendar. It processes whatever dates exist in the data.
Missing dates are simply days with no trading — the engine skips them naturally.

### Market Assumptions (v1.0)

- **Daily bars only.** Intraday data is not supported.
- **Long-only.** Short selling is not supported.
- **Uniform cost model.** Commission and slippage percentages apply uniformly across all symbols.

## Data Loading

vbacktest loads all `.parquet` files in `BacktestConfig.data.validated_dir`:

```python
from vbacktest import BacktestConfig
from vbacktest.config import DataConfig

config = BacktestConfig(
    data=DataConfig(
        validated_dir="data/validated/",
        min_history_days=200,       # drop symbols with fewer bars
        min_price=0.01,             # drop penny stocks
        min_avg_volume=0,           # set > 0 to filter illiquid stocks
        start_date=None,            # clip universe to date range
        end_date=None,
    )
)
```

## Excluded Symbols

To exclude specific symbols (e.g. symbols with data quality issues):

```python
config = BacktestConfig(
    data=DataConfig(
        validated_dir="data/validated/",
        excluded_symbols=["BROKEN1", "BROKEN2"],
        excluded_symbols_file="data/excluded.txt",  # one symbol per line
    )
)
```
