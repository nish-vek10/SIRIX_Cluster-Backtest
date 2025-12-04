"""
mapping.py
----------

Map cluster signals (time-based) onto OANDA M1 candles to define:

- entry_candle_time      : candle open time
- entry_candle_index     : integer index of the candle
- entry_price            : chosen price from that candle (e.g. 'open' or 'close')
- deltas vs prop open    : trigger_open_rate - entry_price
- time offset (seconds)  : signal_time - entry_candle_time

Time is only for clustering; here we align the signal to a price path.

We assume:
- candles are M1 OHLCV, with tz-aware 'open_time' (Europe/London).
- signals have 'signal_time' and 'trigger_open_rate'.
"""

from typing import Literal, Tuple

import pandas as pd


EntryPriceSource = Literal["open", "close"]


def prepare_candles_for_mapping(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure candles are sorted and have a stable integer index 'candle_index'
    for future reference.

    Args:
        candles: DataFrame with at least 'open_time', 'open', 'close', 'high', 'low'

    Returns:
        candles_df: same data with additional 'candle_index' column.
    """
    candles = candles.sort_values("open_time").reset_index(drop=True).copy()
    candles["candle_index"] = candles.index.astype(int)
    return candles


def map_signals_to_candles(
    signals: pd.DataFrame,
    candles: pd.DataFrame,
    entry_price_source: EntryPriceSource = "open",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map each signal_time to the "current" M1 candle and define entry price.

    Mapping rule:
    - For a signal at time t_s, we find the candle whose open_time is the
      latest <= t_s (i.e. floor-to-minute).
    - This is done with pandas.merge_asof (direction="backward").
    - We allow up to 5 minutes tolerance, but will report the actual delta.

    Args:
        signals: DataFrame with at least:
            - 'signal_time'       (tz-aware datetime)
            - 'trigger_open_rate' (float, prop broker price)
        candles: DataFrame with at least:
            - 'open_time' (tz-aware datetime)
            - 'open', 'close', 'high', 'low'
        entry_price_source: which candle price to use as entry:
            - "open" (default)
            - "close"

    Returns:
        (signals_with_entries, qc_summary)

        signals_with_entries:
            original signals columns plus:
            - 'entry_candle_time'
            - 'entry_candle_index'
            - 'entry_price'
            - 'prop_vs_entry_delta'     (trigger_open_rate - entry_price)
            - 'signal_to_candle_secs'   (seconds between signal_time and candle open)

        qc_summary:
            small DataFrame with aggregate stats about deltas and time offsets.
    """
    if signals.empty:
        return signals.copy(), pd.DataFrame()

    if entry_price_source not in ("open", "close"):
        raise ValueError("entry_price_source must be 'open' or 'close'.")

    # Ensure candles are prepared
    candles = prepare_candles_for_mapping(candles)

    # We'll use a subset of candle columns for merge_asof
    candle_merge = candles[["open_time", "candle_index", "open", "close", "high", "low"]].copy()

    # Sort signals by time for merge_asof
    sig = signals.sort_values("signal_time").reset_index(drop=True).copy()

    # merge_asof: backward = find last candle whose open_time <= signal_time
    merged = pd.merge_asof(
        sig,
        candle_merge,
        left_on="signal_time",
        right_on="open_time",
        direction="backward",
        tolerance=pd.Timedelta(minutes=5),
    )

    # If some signals couldn't be matched (NaN candle_index), drop or flag them.
    # We'll drop them for backtest, but keep count in QC.
    unmatched_mask = merged["candle_index"].isna()
    n_unmatched = int(unmatched_mask.sum())

    if n_unmatched > 0:
        print(f"[WARN] {n_unmatched} signals could not be matched to candles within 5 minutes. They will be dropped.")

    # Compute entry price
    merged["entry_candle_time"] = merged["open_time"]
    merged["entry_candle_index"] = merged["candle_index"].astype("Int64")  # nullable int
    merged["entry_price"] = merged[entry_price_source].astype(float)

    # Price difference vs prop broker open rate
    merged["prop_vs_entry_delta"] = merged["trigger_open_rate"] - merged["entry_price"]

    # Time difference in seconds between signal time and candle open
    merged["signal_to_candle_secs"] = (
        (merged["signal_time"] - merged["entry_candle_time"])
        .dt.total_seconds()
    )

    # Build QC summary
    valid = merged[~unmatched_mask].copy()
    if not valid.empty:
        qc_data = {
            "n_signals_total": [len(merged)],
            "n_signals_matched": [len(valid)],
            "n_signals_unmatched": [n_unmatched],
            "delta_price_mean": [valid["prop_vs_entry_delta"].mean()],
            "delta_price_median": [valid["prop_vs_entry_delta"].median()],
            "delta_price_std": [valid["prop_vs_entry_delta"].std()],
            "time_offset_secs_mean": [valid["signal_to_candle_secs"].mean()],
            "time_offset_secs_min": [valid["signal_to_candle_secs"].min()],
            "time_offset_secs_max": [valid["signal_to_candle_secs"].max()],
        }
        qc_summary = pd.DataFrame(qc_data)
    else:
        qc_summary = pd.DataFrame()

    # Drop unmatched rows before returning (for backtest safety)
    merged = merged[~unmatched_mask].reset_index(drop=True)

    # Clean up redundant 'open_time' from candles to avoid confusion
    merged = merged.drop(columns=["open_time"])

    return merged, qc_summary
