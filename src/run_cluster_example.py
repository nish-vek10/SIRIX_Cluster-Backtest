"""
run_cluster_example.py
----------------------

Quick check of the clustering logic.

Example scenario:
- T = 15 seconds
- K = 3 unique traders
- Prints basic stats and a few sample signals.

Run this AFTER run_check_load.py is working.
"""

from config import ensure_directories
from load_data import load_trades
from clustering import find_cluster_signals


def main():
    ensure_directories()

    print("[INFO] Loading trades...")
    trades = load_trades()
    print(f"[INFO] Loaded {len(trades)} trades.")

    # Example scenario: T = 15s, K = 3
    T_seconds = 15
    K_unique = 3
    print(f"[INFO] Building cluster signals with T={T_seconds}s, K={K_unique}...")

    signals = find_cluster_signals(trades, T_seconds=T_seconds, K_unique=K_unique)

    print(f"[INFO] Number of signals found: {len(signals)}")

    if not signals.empty:
        print("\n=== SAMPLE SIGNALS (first 10) ===")
        print(signals.head(10))

        # Some quick breakdowns
        print("\n=== SIGNAL COUNTS BY SIDE ===")
        print(signals["side"].value_counts())

        print("\n=== SIGNAL TIME RANGE ===")
        print(signals["signal_time"].min(), "->", signals["signal_time"].max())
    else:
        print("[WARN] No signals found for this T/K combination.")


if __name__ == "__main__":
    main()
