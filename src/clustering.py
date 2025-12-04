"""
clustering.py
-------------

Builds "cluster signals" from prop trades based on:

- Time window length T_seconds (e.g. 10–50 seconds)
- Minimum number of UNIQUE traders K_unique (e.g. 3–5)
- Side (BUY / SELL)

Logic:
- Work separately per side.
- Slide a rolling T-second window over trades (sorted by open_time).
- Each user_id contributes at MOST once to the unique count in the window.
- When a NEW unique user enters the window and the unique count reaches K_unique,
  the *trade from that user* becomes a signal (entry trigger).

Important:
- Time is used only for detecting clusters (orderflow burst).
- Entry price itself will later be mapped to OANDA candles.
"""

from datetime import timedelta
from collections import deque
from typing import List, Dict, Any

import pandas as pd


def find_cluster_signals(
    trades: pd.DataFrame,
    T_seconds: int,
    K_unique: int,
) -> pd.DataFrame:
    """
    Build cluster-based entry signals from the trades DataFrame.

    Args:
        trades: DataFrame with at least:
            - 'user_id' (string or int)
            - 'action'  (BUY / SELL)
            - 'open_time' (tz-aware datetime, Europe/London)
            - 'open_rate' (float, prop broker entry price)
        T_seconds: length of the time window in seconds (e.g. 10, 15, 20, ...).
        K_unique: required number of UNIQUE users in the window to trigger a signal.

    Returns:
        DataFrame with one row per signal:
            - 'signal_time'           (time of the triggering trade)
            - 'side'                  (BUY / SELL)
            - 'trigger_user_id'       (user who triggered the K-th unique)
            - 'trigger_open_rate'     (prop open_rate at trigger)
            - 'T_seconds'
            - 'K_unique'
            - 'n_unique_in_window'    (should be == K_unique at trigger)
            - 'n_trades_in_window'    (total trades in the raw window)
            - 'window_start_time'
            - 'window_end_time'       (same as signal_time)
    """
    if T_seconds <= 0:
        raise ValueError("T_seconds must be positive.")
    if K_unique < 2:
        raise ValueError("K_unique must be at least 2 (otherwise it's trivial).")

    # Ensure expected columns
    required_cols = {"user_id", "action", "open_time", "open_rate"}
    missing = required_cols - set(trades.columns)
    if missing:
        raise ValueError(f"Trades DataFrame is missing required columns: {missing}")

    # We assume trades are already sorted oldest -> newest in load_data.py,
    # but we sort again defensively.
    trades = trades.sort_values("open_time").reset_index(drop=False)
    # We'll keep original index in 'orig_index' if needed later
    trades = trades.rename(columns={"index": "orig_index"})

    signals: List[Dict[str, Any]] = []

    # Process BUY and SELL separately
    for side in ["BUY", "SELL"]:
        side_df = trades[trades["action"] == side].copy()
        if side_df.empty:
            continue

        # Use a deque of row indices referencing side_df
        window = deque()          # holds integer positions (iloc indices) in side_df
        user_counts: Dict[str, int] = {}  # user_id -> count of trades in window

        # We'll iterate over *positional* index for speed and to access via iloc
        for pos, row in side_df.reset_index(drop=True).iterrows():
            t_current = row["open_time"]
            threshold = t_current - timedelta(seconds=T_seconds)
            user_id = str(row["user_id"])

            # 1) Slide the window: drop trades that are older than t_current - T
            while window:
                oldest_pos = window[0]
                oldest_time = side_df.iloc[oldest_pos]["open_time"]
                if oldest_time >= threshold:
                    # still inside the window
                    break
                # remove the oldest trade from the window
                window.popleft()
                oldest_user = str(side_df.iloc[oldest_pos]["user_id"])
                user_counts[oldest_user] -= 1
                if user_counts[oldest_user] <= 0:
                    del user_counts[oldest_user]

            pre_unique = len(user_counts)

            # 2) Add the current trade to the window
            window.append(pos)
            if user_id not in user_counts:
                # This is a NEW unique user in the current window
                user_counts[user_id] = 1
                post_unique = pre_unique + 1
            else:
                # Same user trading again within the window
                user_counts[user_id] += 1
                post_unique = pre_unique  # unique count unchanged

            # 3) Check if this *new* unique user pushes us to K_unique
            if pre_unique < K_unique and post_unique >= K_unique:
                # We have just hit the K-th unique trader in this window.
                # This trade becomes a "cluster signal".
                window_start_time = side_df.iloc[window[0]]["open_time"]
                signal_dict = {
                    "signal_time": t_current,
                    "side": side,
                    "trigger_user_id": user_id,
                    "trigger_open_rate": float(row["open_rate"]),
                    "T_seconds": T_seconds,
                    "K_unique": K_unique,
                    "n_unique_in_window": post_unique,
                    "n_trades_in_window": len(window),
                    "window_start_time": window_start_time,
                    "window_end_time": t_current,
                    "trigger_orig_index": int(row["orig_index"]),
                }
                signals.append(signal_dict)

    if not signals:
        return pd.DataFrame(columns=[
            "signal_time",
            "side",
            "trigger_user_id",
            "trigger_open_rate",
            "T_seconds",
            "K_unique",
            "n_unique_in_window",
            "n_trades_in_window",
            "window_start_time",
            "window_end_time",
            "trigger_orig_index",
        ])

    signals_df = pd.DataFrame(signals).sort_values("signal_time").reset_index(drop=True)
    return signals_df
