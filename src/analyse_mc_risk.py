#!/usr/bin/env python

"""
Monte Carlo Risk Analysis (Level 3)

Takes the same trade logs you used for concurrency analysis and:

- Combines all engines into a single trade stream.
- Works in R-space using the 'pnl_R' column from your trade logs.
- Runs Monte Carlo simulations by randomly resampling trades with replacement
  (iid assumption) to build synthetic equity paths.

Outputs (under INPUT_DIR / RISK_TAG):

    <INPUT_DIR> / <RISK_TAG> /
        csv/
            mc_sim_paths_summary.csv      # one row per simulation
            mc_quantiles.csv              # quantiles for final return & max DD
        plots/
            mc_final_return_hist.png
            mc_max_drawdown_hist.png

You can safely tweak RISK_TAG without overwriting anything else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


# ============================================================
# USER CONFIG
# ============================================================

RISK_TAG = "v2_risk_mc"

# Same folder + files as your concurrency script
INPUT_DIR = Path(
    r"C:\Users\anish\PycharmProjects\ClusterOrderflow-Backtest\output\reports\scenarios\stops\v2\inverse\spread_on\tp1_time0\single"
)

TRADE_FILES = [
    INPUT_DIR / "trades_T29_K3_H5_SL2.0_TP1.0_prop_inverse_atr_static_ATR5_I4.0_spread_on_tp1_time0.csv",
    INPUT_DIR / "trades_T29_K3_H5_SL2.0_TP1.0_prop_inverse_atr_trailing_ATR5_I3.0_T2.0_spread_on_tp1_time0.csv",
    INPUT_DIR / "trades_T30_K3_H15_SL2.0_TP1.0_prop_inverse_chandelier_ATR5_I3.0_T2.0_CH30_spread_on_tp1_time0.csv",
]

# Monte Carlo settings
N_SIMULATIONS = 2000          # number of MC paths
START_EQUITY = 100_000.0      # starting account size (for %Return / %DD)
R_DOLLARS = 200.0             # 1R in dollars (used to convert R → cash)

# Output dirs (isolated from your previous script)
RISK_DIR = INPUT_DIR / RISK_TAG
CSV_DIR = RISK_DIR / "csv"
PLOTS_DIR = RISK_DIR / "plots"


# ============================================================
# HELPERS
# ============================================================

def load_all_trades(paths: List[Path]) -> pd.DataFrame:
    """
    Load all trade CSVs, parse datetime, and concatenate.

    Returns a DataFrame with at least:
        - entry_time
        - pnl_R
    """
    dfs = []
    for p in paths:
        if not p.exists():
            print(f"[WARN] File not found, skipping: {p}")
            continue

        print(f"[INFO] Loading trades from: {p}")
        df = pd.read_csv(p)

        # Parse datetime as UTC to avoid mixed tz warnings
        if "entry_time" in df.columns:
            df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)

        # Keep a label of source file / engine if needed later
        if "stop_mode" in df.columns and not df["stop_mode"].isna().all():
            engine = str(df["stop_mode"].iloc[0])
        else:
            engine = p.stem
        df["engine"] = engine

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid trade files loaded for MC analysis.")

    all_trades = pd.concat(dfs, ignore_index=True)

    # Make sure pnl_R exists
    if "pnl_R" not in all_trades.columns:
        raise RuntimeError("pnl_R column not found in trade logs — required for MC.")

    # Sort chronologically (not strictly required for iid MC, but good practice)
    all_trades = all_trades.sort_values("entry_time").reset_index(drop=True)

    print("[INFO] Total trades loaded for MC:", len(all_trades))
    print("[INFO] Trades by engine:")
    print(all_trades["engine"].value_counts())

    return all_trades


def compute_equity_path_from_R(pnl_R_array: np.ndarray,
                               start_equity: float,
                               r_dollars: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an array of pnl_R for a path, compute:

    - equity_curve: in cash
    - dd_curve: drawdown in cash (negative numbers)

    Returns (equity_curve, dd_curve).
    """
    # Convert R to dollars
    pnl_cash = pnl_R_array * r_dollars
    equity = np.empty_like(pnl_cash, dtype=float)
    dd = np.empty_like(pnl_cash, dtype=float)

    equity[0] = start_equity + pnl_cash[0]
    peak = equity[0]
    dd[0] = equity[0] - peak  # 0

    for i in range(1, len(pnl_cash)):
        equity[i] = equity[i - 1] + pnl_cash[i]
        peak = max(peak, equity[i])
        dd[i] = equity[i] - peak  # will be <= 0

    return equity, dd


def run_monte_carlo(trades: pd.DataFrame,
                    n_sims: int,
                    start_equity: float,
                    r_dollars: float,
                    rng_seed: int = 42) -> pd.DataFrame:
    """
    Run Monte Carlo paths by resampling trades (pnl_R) with replacement.

    Returns a DataFrame with one row per simulation:
        - sim_id
        - n_trades
        - final_equity
        - final_return_pct
        - total_R
        - max_drawdown_cash
        - max_drawdown_pct
        - max_drawdown_R
    """
    rng = np.random.default_rng(rng_seed)
    pnl_R = trades["pnl_R"].values
    n_trades = len(pnl_R)

    records = []

    for sim_id in range(1, n_sims + 1):
        # Sample indices with replacement
        idx = rng.integers(low=0, high=n_trades, size=n_trades)
        sampled_R = pnl_R[idx]

        equity_curve, dd_curve = compute_equity_path_from_R(
            sampled_R,
            start_equity=start_equity,
            r_dollars=r_dollars,
        )

        final_equity = equity_curve[-1]
        final_return_pct = (final_equity / start_equity - 1.0) * 100.0

        max_dd_cash = dd_curve.min()  # negative
        max_dd_pct = max_dd_cash / start_equity * 100.0

        # Total R and max DD in R terms
        total_R = sampled_R.sum()
        max_dd_R = max_dd_cash / r_dollars

        records.append(
            {
                "sim_id": sim_id,
                "n_trades": n_trades,
                "final_equity": final_equity,
                "final_return_pct": final_return_pct,
                "total_R": total_R,
                "max_drawdown_cash": max_dd_cash,
                "max_drawdown_pct": max_dd_pct,
                "max_drawdown_R": max_dd_R,
            }
        )

    return pd.DataFrame.from_records(records)


def compute_mc_quantiles(mc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute key quantiles for final_return_pct and max_drawdown_pct.

    Returns a small table with:
        metric, p05, p25, p50, p75, p95
    """
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    metrics = ["final_return_pct", "max_drawdown_pct"]

    rows = []
    for metric in metrics:
        q_vals = mc_df[metric].quantile(quantiles)
        row = {
            "metric": metric,
            "p05": q_vals.loc[0.05],
            "p25": q_vals.loc[0.25],
            "p50": q_vals.loc[0.5],
            "p75": q_vals.loc[0.75],
            "p95": q_vals.loc[0.95],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_histograms(mc_df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Plot histograms for:
        - final_return_pct
        - max_drawdown_pct
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Final return distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mc_df["final_return_pct"], bins=40, density=False)
    ax.set_title("Monte Carlo — Final Return (%) Distribution")
    ax.set_xlabel("Final Return (%)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    out1 = plots_dir / "mc_final_return_hist.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved final return histogram -> {out1}")

    # Max drawdown distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mc_df["max_drawdown_pct"], bins=40, density=False)
    ax.set_title("Monte Carlo — Max Drawdown (%) Distribution")
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    out2 = plots_dir / "mc_max_drawdown_hist.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved max drawdown histogram -> {out2}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # Create risk output folders (separate from your other script)
    RISK_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load combined trades
    trades = load_all_trades(TRADE_FILES)

    # 2) Run Monte Carlo
    print(f"[INFO] Running Monte Carlo with {N_SIMULATIONS} simulations...")
    mc_df = run_monte_carlo(
        trades=trades,
        n_sims=N_SIMULATIONS,
        start_equity=START_EQUITY,
        r_dollars=R_DOLLARS,
    )

    # 3) Save simulation-level results
    mc_path = CSV_DIR / "mc_sim_paths_summary.csv"
    mc_df.to_csv(mc_path, index=False)
    print(f"[INFO] Saved MC simulation summary -> {mc_path}")

    # 4) Aggregate quantiles
    q_df = compute_mc_quantiles(mc_df)
    q_path = CSV_DIR / "mc_quantiles.csv"
    q_df.to_csv(q_path, index=False)
    print(f"[INFO] Saved MC quantiles -> {q_path}")

    # 5) Plots
    plot_histograms(mc_df, PLOTS_DIR)

    print("\n[INFO] Monte Carlo risk analysis complete.")
    print(f"[INFO] CSVs in: {CSV_DIR}")
    print(f"[INFO] Plots in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
