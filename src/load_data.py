"""
load_data.py
------------

Functions to load and clean:

1) Prop XAUUSD trades (closed positions).
2) OANDA XAUUSD M1 candles.

Assumptions:
- Timestamps are already in UK time (GMT/BST).
- We'll parse them as timezone-aware datetime (Europe/London).
- We'll sort all data from oldest to newest.
"""

import pandas as pd
from pytz import timezone

from config import TRADES_FILE, CANDLES_FILE


LONDON_TZ = timezone("Europe/London")


def load_trades() -> pd.DataFrame:
    """
    Load prop trades (XAUUSD) from CSV.

    Expected columns (minimum):
    - 'User ID'
    - 'Instrument'
    - 'Amount'
    - 'Lots'
    - 'Action'  (BUY/SELL)
    - 'Open Time (UK - GMT/BST)'
    - 'Open Rate'

    Returns:
        DataFrame with:
        - parsed 'open_time' (tz-aware, Europe/London)
        - normalised column names in snake_case
        - filtered to Instrument == 'XAUUSD' (case-sensitive by default)
        - sorted ascending by 'open_time'
    """
    df = pd.read_csv(TRADES_FILE)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Rename to snake_case internals
    rename_map = {
        "User ID": "user_id",
        "Instrument": "instrument",
        "Amount": "amount",
        "Lots": "lots",
        "Action": "action",
        "Open Time (UK - GMT/BST)": "open_time",
        "Open Rate": "open_rate",
    }
    df = df.rename(columns=rename_map)

    # Filter to XAUUSD if not already
    if "instrument" in df.columns:
        df = df[df["instrument"] == "XAUUSD"].copy()

    # Parse open_time as tz-aware
    df["open_time"] = (
        pd.to_datetime(df["open_time"], errors="coerce")
          .dt.tz_localize(LONDON_TZ, ambiguous="NaT", nonexistent="NaT")
    )

    # Drop rows with bad timestamps or missing core fields
    df = df.dropna(subset=["open_time", "user_id", "action", "open_rate"])

    # Ensure correct dtypes
    df["user_id"] = df["user_id"].astype(str)  # keep as string for safety
    df["action"] = df["action"].str.upper().str.strip()
    df["open_rate"] = df["open_rate"].astype(float)

    # Sort oldest -> newest
    df = df.sort_values("open_time").reset_index(drop=True)

    return df


def load_candles() -> pd.DataFrame:
    """
    Load OANDA XAUUSD M1 OHLCV data.

    Expected columns:
    - 'Open'
    - 'High'
    - 'Low'
    - 'Close'
    - 'Volume'
    - 'Open Time (UK - GMT/BST)'

    Returns:
        DataFrame with:
        - 'open_time' as tz-aware datetime (Europe/London)
        - numeric OHLCV columns
        - sorted ascending by 'open_time'
        - an integer index from 0..N-1
    """
    df = pd.read_csv(CANDLES_FILE)

    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Open Time (UK - GMT/BST)": "open_time",
    }
    df = df.rename(columns=rename_map)

    # Parse open_time as tz-aware
    df["open_time"] = (
        pd.to_datetime(df["open_time"], errors="coerce")
          .dt.tz_localize(LONDON_TZ, ambiguous="NaT", nonexistent="NaT")
    )

    df = df.dropna(subset=["open_time"])

    # Ensure numeric OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])  # volume can be 0/nan

    # Sort oldest -> newest
    df = df.sort_values("open_time").reset_index(drop=True)

    return df


def quick_sanity_print(trades: pd.DataFrame, candles: pd.DataFrame) -> None:
    """
    Quick sanity logger to confirm basic alignment & ranges.
    """

    print("\n=== TRADES ===")
    print(f"Rows: {len(trades)}")
    if len(trades) > 0:
        print(f"Time range: {trades['open_time'].min()}  ->  {trades['open_time'].max()}")
        print(trades.head(5))
        print(trades.tail(5))

    print("\n=== CANDLES ===")
    print(f"Rows: {len(candles)}")
    if len(candles) > 0:
        print(f"Time range: {candles['open_time'].min()}  ->  {candles['open_time'].max()}")
        print(candles.head(5))
        print(candles.tail(5))
