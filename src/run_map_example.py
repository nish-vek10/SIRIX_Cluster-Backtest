"""
run_map_example.py
------------------

End-to-end smoke test:

1) Load trades & candles.
2) Build cluster signals for a sample scenario (e.g. T=15s, K=3).
3) Map signals onto OANDA candles and define entry prices.
4) Print basic QC stats and a few sample rows.

This is the last step before we implement exit logic & full backtests.
"""

from config import ensure_directories
from load_data import load_trades, load_candles
from clustering import find_cluster_signals
from mapping import map_signals_to_candles


def main():
    ensure_directories()

    # 1) Load data
    print("[INFO] Loading data...")
    trades = load_trades()
    candles = load_candles()
    print(f"[INFO] Trades: {len(trades)} rows, Candles: {len(candles)} rows.")

    # 2) Example clustering scenario
    T_seconds = 15
    K_unique = 3
    print(f"[INFO] Building cluster signals (T={T_seconds}s, K={K_unique})...")
    signals = find_cluster_signals(trades, T_seconds=T_seconds, K_unique=K_unique)
    print(f"[INFO] Signals found: {len(signals)}")

    if signals.empty:
        print("[WARN] No signals for this configuration. Nothing to map.")
        return

    # 3) Map signals to candles and define entry prices
    entry_price_source = "open"   # we can later allow 'close' as a variant
    print(f"[INFO] Mapping signals to candles using entry price source: {entry_price_source} ...")
    signals_with_entries, qc_summary = map_signals_to_candles(
        signals,
        candles,
        entry_price_source=entry_price_source,
    )

    print(f"[INFO] Mapped signals: {len(signals_with_entries)}")

    # 4) Show QC summary
    if not qc_summary.empty:
        print("\n=== QC SUMMARY ===")
        print(qc_summary.to_string(index=False))
    else:
        print("[WARN] QC summary is empty (no matched signals?).")

    # 5) Show sample mapped signals
    print("\n=== SAMPLE MAPPED SIGNALS (first 10) ===")
    cols_to_show = [
        "signal_time",
        "side",
        "trigger_user_id",
        "trigger_open_rate",
        "entry_candle_time",
        "entry_candle_index",
        "entry_price",
        "prop_vs_entry_delta",
        "signal_to_candle_secs",
        "T_seconds",
        "K_unique",
    ]
    print(signals_with_entries[cols_to_show].head(10))


if __name__ == "__main__":
    main()
