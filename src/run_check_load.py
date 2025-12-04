"""
run_check_load.py
-----------------

Sanity check: load trades + candles and print basic info.
Run this first to confirm paths + parsing are correct.
"""

from config import ensure_directories
from load_data import load_trades, load_candles, quick_sanity_print


def main():
    ensure_directories()

    trades = load_trades()
    candles = load_candles()

    quick_sanity_print(trades, candles)


if __name__ == "__main__":
    main()
