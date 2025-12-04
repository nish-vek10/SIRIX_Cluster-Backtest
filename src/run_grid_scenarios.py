"""
run_grid_scenarios.py
---------------------

Run a GRID of backtest scenarios over:

- T_seconds      (cluster window)
- K_unique       (unique traders)
- hold_minutes   (time-based exit horizon)
- sl_distance    (defines 1R)
- tp_R_multiple  (TP distance in R multiples)
- entry_mode     ("prop", "oanda_open", "oanda_close")

Global toggles (you set these once at the top):

- direction_mode : "directional" or "inverse"
- use_tp_exit    : True / False
- use_time_exit  : True / False

For each scenario:

1) Build cluster signals with chosen (T_seconds, K_unique).
2) Map signals to candles for that entry_mode (for indices & QC).
3) Backtest with full SL/TP/TIME logic (backtest_time_exit).
4) Append one row to a master summary table.

Outputs:

- SUMMARY_REPORTS_DIR / "grid_scenarios_summary.csv"
    One row per scenario, with all params + metrics.

- (Optional) SCENARIO_REPORTS_DIR / "trades_<scenario_name>.csv"
    If SAVE_TRADES_PER_SCENARIO = True.

- A parameter heatmap PNG:
    - Metric vs (x_param, y_param), with fixed filters on other params.
"""

from pathlib import Path
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    ensure_directories,
    SUMMARY_REPORTS_DIR,
    SCENARIO_REPORTS_DIR,
    EQUITY_FIG_DIR,
)
from load_data import load_trades, load_candles
from clustering import find_cluster_signals
from mapping import map_signals_to_candles, prepare_candles_for_mapping
from backtest import backtest_time_exit, BacktestConfig


# =========================
# USER CONFIG — GRID
# =========================

# Ranges:
GRID_T_SECONDS = [10, 20, 30, 40, 50, 60]
GRID_K_UNIQUE = [3, 4, 5]
GRID_HOLD_MINUTES = [5, 10, 15, 20, 25, 30]
GRID_SL_DISTANCE = [2.0, 3.0, 4.0]
GRID_TP_R_MULTIPLE = [1.0, 2.0, 3.0, 4.0]  # e.g. 1R, 2R. (R:R reward)

ENTRY_MODES = ["prop"]  # subset if needed


# =========================
# USER CONFIG — GLOBAL TOGGLES
# =========================

# You manually toggle these to test directional vs inverse, and exits.
DIRECTION_MODE = "inverse"   # "directional" or "inverse"

USE_TP_EXIT = True               # if False, TP completely ignored
USE_TIME_EXIT = True             # if False, no "TIME" exit, only SL/TP/FORCED

# Spread modelling (in XAUUSD dollars)
# Example: SPREAD_DOLLARS = 0.2 => ~20 cents spread
APPLY_SPREAD = True          # True to include spread, False to ignore
SPREAD_DOLLARS = 0.2         # size of the spread in dollars

# Save per-scenario trades CSVs?
SAVE_TRADES_PER_SCENARIO = False

# Plot equity + drawdown per scenario?
PLOT_EQUITY_PER_SCENARIO = True


# =========================
# USER CONFIG — HEATMAPS
# =========================

# Metric to visualise on each heatmap
# HEATMAP_METRIC = "avg_R"   # options: "avg_R", "win_rate", "max_drawdown_R", "final_equity", etc.

# Metrics to visualise for each (K, SL, TP_R, entry_mode) combo
# These must exist as columns in grid_scenarios_summary.csv
HEATMAP_METRICS = [
    "avg_R",
    "win_rate",
    "max_drawdown_R",
    "final_equity",
    "profit_factor",
    "tp_rate",
    "sl_rate",
    "percent_return",
    "longest_win_streak",
    "longest_loss_streak",
]


# =========================
# INTERNAL HELPERS
# =========================

def get_entry_price_source(entry_mode: str) -> str:
    """
    For mapping/QC only.
    Prop uses trigger price, but we still attach candle open for QC.
    """
    if entry_mode == "oanda_open":
        return "open"
    elif entry_mode == "oanda_close":
        return "close"
    else:  # "prop"
        return "open"


def generate_all_param_heatmaps(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Generate heatmaps for EACH metric in HEATMAP_METRICS and for EACH
    combination of:
      - K_unique
      - sl_distance
      - tp_R_multiple
      - entry_mode

    For every (metric, K, SL, TP, entry_mode) combo with data, create:

      X axis: T_seconds
      Y axis: hold_minutes
      Colour: that metric (e.g. avg_R, win_rate, ...)

    Filenames look like:

      grid_heatmap_avg_R_K3_SL2.0_TP2.0_prop.png
      grid_heatmap_win_rate_K3_SL2.0_TP2.0_prop.png
      ...
    """
    if summary_df.empty:
        print("[WARN] No summary data for heatmaps.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    unique_K = sorted(summary_df["K_unique"].unique())
    unique_SL = sorted(summary_df["sl_distance"].unique())
    unique_TP = sorted(summary_df["tp_R_multiple"].unique())
    unique_entry = sorted(summary_df["entry_mode"].unique())

    x_param = "T_seconds"
    y_param = "hold_minutes"

    n_plots = 0

    for metric in HEATMAP_METRICS:
        if metric not in summary_df.columns:
            print(f"[WARN] Metric '{metric}' not found in summary_df; skipping it.")
            continue

        for K in unique_K:
            for SL in unique_SL:
                for TP in unique_TP:
                    for entry_mode in unique_entry:
                        # Filter for this combo
                        df = summary_df[
                            (summary_df["K_unique"] == K)
                            & (summary_df["sl_distance"] == SL)
                            & (summary_df["tp_R_multiple"] == TP)
                            & (summary_df["entry_mode"] == entry_mode)
                        ]

                        if df.empty:
                            # No scenario for this combo (maybe grid reduced)
                            continue

                        # Ensure columns exist
                        if x_param not in df.columns or y_param not in df.columns:
                            print(
                                f"[WARN] Missing {x_param}/{y_param} for "
                                f"K={K}, SL={SL}, TP={TP}, entry_mode={entry_mode}"
                            )
                            continue

                        # Pivot to 2D grid: Y=hold_minutes, X=T_seconds
                        pivot = df.pivot_table(
                            index=y_param,
                            columns=x_param,
                            values=metric,
                            aggfunc="mean",
                        )

                        pivot = pivot.sort_index(axis=0).sort_index(axis=1)

                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(
                            pivot.values,
                            aspect="auto",
                            origin="lower",
                            interpolation="nearest",
                            cmap="coolwarm",
                        )

                        ax.set_yticks(range(len(pivot.index)))
                        ax.set_yticklabels(pivot.index)
                        ax.set_ylabel(y_param)

                        ax.set_xticks(range(len(pivot.columns)))
                        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
                        ax.set_xlabel(x_param)

                        title = (
                            f"{metric} — K={K}, SL={SL}, TP_R={TP}, "
                            f"entry_mode={entry_mode}"
                        )
                        ax.set_title(title)

                        cbar = fig.colorbar(im, ax=ax)
                        cbar.set_label(metric)

                        fig.tight_layout()

                        filename = (
                            f"grid_heatmap_{metric}_"
                            f"K{K}_SL{SL}_TP{TP}_{entry_mode}.png"
                        )
                        out_path = out_dir / filename
                        fig.savefig(out_path, dpi=150)
                        plt.close(fig)

                        print(f"[INFO] Saved heatmap: {out_path}")
                        n_plots += 1

    if n_plots == 0:
        print("[WARN] No heatmaps were generated (no valid combinations).")
    else:
        print(f"[INFO] Generated {n_plots} heatmaps in {out_dir}")


def generate_distributions(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Generate distribution-style plots from the GRID summary:

    1) Global histograms:
       - percent_return across all scenarios
       - avg_R across all scenarios

    2) Per-SL histograms of percent_return:
       - distribution_percent_return_SL<sl>.png

    3) Per-TP histograms of percent_return:
       - distribution_percent_return_TP<tp>.png

    4) Simple scatter plots:
       - sl_distance vs percent_return
       - tp_R_multiple vs percent_return
    """
    if summary_df.empty:
        print("[WARN] No summary data for distributions.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Global histograms
    if "percent_return" in summary_df.columns:
        vals = summary_df["percent_return"].dropna()
        if not vals.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(vals, bins=50, density=True)
            ax.set_title("Distribution of Percent Return Across Scenarios")
            ax.set_xlabel("Percent Return")
            ax.set_ylabel("Density")
            fig.tight_layout()
            path = out_dir / "distribution_percent_return_all_scenarios.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[INFO] Saved: {path}")

    if "avg_R" in summary_df.columns:
        vals = summary_df["avg_R"].dropna()
        if not vals.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(vals, bins=50, density=True)
            ax.set_title("Distribution of Avg R Across Scenarios")
            ax.set_xlabel("Average R")
            ax.set_ylabel("Density")
            fig.tight_layout()
            path = out_dir / "distribution_avg_R_all_scenarios.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[INFO] Saved: {path}")

    # 2) Per-SL histograms of percent_return
    if "percent_return" in summary_df.columns and "sl_distance" in summary_df.columns:
        for sl in sorted(summary_df["sl_distance"].unique()):
            sub = summary_df[summary_df["sl_distance"] == sl]
            vals = sub["percent_return"].dropna()
            if vals.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(vals, bins=40, density=True)
            ax.set_title(f"Percent Return Distribution — SL={sl}")
            ax.set_xlabel("Percent Return")
            ax.set_ylabel("Density")
            fig.tight_layout()
            filename = f"distribution_percent_return_SL{sl}.png"
            path = out_dir / filename
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[INFO] Saved: {path}")

    # 3) Per-TP histograms of percent_return
    if "percent_return" in summary_df.columns and "tp_R_multiple" in summary_df.columns:
        for tp in sorted(summary_df["tp_R_multiple"].unique()):
            sub = summary_df[summary_df["tp_R_multiple"] == tp]
            vals = sub["percent_return"].dropna()
            if vals.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(vals, bins=40, density=True)
            ax.set_title(f"Percent Return Distribution — TP_R={tp}")
            ax.set_xlabel("Percent Return")
            ax.set_ylabel("Density")
            fig.tight_layout()
            filename = f"distribution_percent_return_TP{tp}.png"
            path = out_dir / filename
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[INFO] Saved: {path}")

    # 4) Scatter diagnostics
    if "percent_return" in summary_df.columns and "sl_distance" in summary_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(summary_df["sl_distance"], summary_df["percent_return"])
        ax.set_title("SL Distance vs Percent Return (per scenario)")
        ax.set_xlabel("SL Distance")
        ax.set_ylabel("Percent Return")
        fig.tight_layout()
        path = out_dir / "scatter_SL_vs_percent_return.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved: {path}")

    if "percent_return" in summary_df.columns and "tp_R_multiple" in summary_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(summary_df["tp_R_multiple"], summary_df["percent_return"])
        ax.set_title("TP_R Multiple vs Percent Return (per scenario)")
        ax.set_xlabel("TP_R Multiple")
        ax.set_ylabel("Percent Return")
        fig.tight_layout()
        path = out_dir / "scatter_TP_vs_percent_return.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved: {path}")


def plot_equity_and_drawdown(
    trades: pd.DataFrame,
    scenario_name: str,
    out_dir: Path,
) -> None:
    """
    Plot equity and drawdown in one figure:

    - Top: equity curve
    - Bottom: drawdown curve

    Uses:
      - trades['entry_time']
      - trades['equity']
      - trades['drawdown']
    """
    if trades.empty:
        return

    if not {"entry_time", "equity", "drawdown"}.issubset(trades.columns):
        print(f"[WARN] Missing columns for equity/drawdown plot for {scenario_name}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Sort by time just in case
    df = trades.sort_values("entry_time")

    # Equity
    ax_eq.plot(df["entry_time"], df["equity"])
    ax_eq.set_title(f"Equity & Drawdown — {scenario_name}")
    ax_eq.set_ylabel("Equity")
    ax_eq.grid(True)

    # Drawdown (usually negative)
    ax_dd.plot(df["entry_time"], df["drawdown"])
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Time")
    ax_dd.grid(True)

    fig.tight_layout()

    out_path = out_dir / f"equity_dd_{scenario_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved equity+drawdown curve to: {out_path}")


def print_top_scenarios(
    summary_df: pd.DataFrame,
    n: int = 30,
    sort_by: str = "percent_return",
) -> None:
    """
    Print the top-N scenarios sorted by a chosen metric (default: percent_return).

    Shows key parameters + a few core metrics so you can
    copy them straight into run_single_scenario.py.
    """
    if summary_df.empty:
        print("[WARN] No summary data to rank scenarios.")
        return

    if sort_by not in summary_df.columns:
        print(f"[WARN] sort_by='{sort_by}' not in summary_df columns; cannot rank.")
        print(f"       Available columns: {list(summary_df.columns)}")
        return

    df = summary_df.dropna(subset=[sort_by]).copy()
    if df.empty:
        print(f"[WARN] No non-NaN values for metric '{sort_by}'.")
        return

    df = df.sort_values(sort_by, ascending=False).head(n)

    cols_to_show = [
        "scenario_name",
        "T_seconds",
        "K_unique",
        "hold_minutes",
        "sl_distance",
        "tp_R_multiple",
        "entry_mode",
        "direction_mode",
        "use_tp_exit",
        "use_time_exit",
        # Core performance:
        "percent_return",
        "avg_R",
        "win_rate",
        "max_drawdown_R",
        "profit_factor",
        "longest_win_streak",
        "longest_loss_streak",
        "n_trades",
    ]

    cols_present = [c for c in cols_to_show if c in df.columns]

    print("\n=== TOP SCENARIOS (sorted by '{}', desc) ===".format(sort_by))
    print(df[cols_present].to_string(index=False))
    print("=== END TOP SCENARIOS ===\n")


# =========================
# MAIN GRID RUNNER
# =========================

def main():
    ensure_directories()

    mode_tag = DIRECTION_MODE.lower()
    exit_tag = f"tp{int(USE_TP_EXIT)}_time{int(USE_TIME_EXIT)}"

    spread_tag = "spread_on" if APPLY_SPREAD else "spread_off"

    # Direction-mode + exit-config specific directories
    scenario_dir = SCENARIO_REPORTS_DIR / mode_tag / spread_tag / exit_tag
    summary_dir = SUMMARY_REPORTS_DIR / mode_tag / spread_tag / exit_tag
    heatmap_dir = EQUITY_FIG_DIR.parent / "grid_heatmaps" / mode_tag / spread_tag / exit_tag
    distrib_dir = EQUITY_FIG_DIR.parent / "distributions" / mode_tag / spread_tag / exit_tag
    equity_dir = EQUITY_FIG_DIR / mode_tag / spread_tag / exit_tag

    scenario_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    distrib_dir.mkdir(parents=True, exist_ok=True)
    equity_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading trades & candles...")
    trades = load_trades()
    candles = load_candles()
    candles = prepare_candles_for_mapping(candles)
    print(f"[INFO] Trades: {len(trades)}, Candles: {len(candles)}")

    # Caches so we don't recompute the same thing unnecessarily
    signals_cache = {}         # (T_seconds, K_unique) -> signals_df
    mapped_cache = {}          # (T_seconds, K_unique, entry_mode) -> signals_mapped_df

    # For collecting all scenario summaries
    summary_rows = []

    total_scenarios = (
        len(GRID_T_SECONDS)
        * len(GRID_K_UNIQUE)
        * len(GRID_HOLD_MINUTES)
        * len(GRID_SL_DISTANCE)
        * len(GRID_TP_R_MULTIPLE)
        * len(ENTRY_MODES)
    )
    scenario_counter = 0

    for T_seconds, K_unique in product(GRID_T_SECONDS, GRID_K_UNIQUE):
        # 1) Build or retrieve signals for this (T, K)
        key_sig = (T_seconds, K_unique)
        if key_sig not in signals_cache:
            print(f"[INFO] Building signals for T={T_seconds}s, K={K_unique}...")
            sig_df = find_cluster_signals(trades, T_seconds=T_seconds, K_unique=K_unique)
            signals_cache[key_sig] = sig_df
            print(f"[INFO]   Signals found: {len(sig_df)}")
        else:
            sig_df = signals_cache[key_sig]

        if sig_df.empty:
            print(f"[WARN] No signals for T={T_seconds}, K={K_unique}; skipping all combos.")
            continue

        for entry_mode in ENTRY_MODES:
            # 2) Map to candles (cache per (T,K,entry_mode))
            key_map = (T_seconds, K_unique, entry_mode)
            if key_map not in mapped_cache:
                entry_price_source = get_entry_price_source(entry_mode)
                print(
                    f"[INFO] Mapping signals to candles for "
                    f"T={T_seconds}, K={K_unique}, entry_mode={entry_mode} "
                    f"(entry_price_source={entry_price_source})..."
                )
                sig_mapped, qc_summary = map_signals_to_candles(
                    sig_df,
                    candles,
                    entry_price_source=entry_price_source,
                )
                mapped_cache[key_map] = sig_mapped
                print(f"[INFO]   Mapped signals: {len(sig_mapped)}")

                if sig_mapped.empty:
                    print(
                        f"[WARN] No mapped signals for T={T_seconds}, "
                        f"K={K_unique}, entry_mode={entry_mode}; skipping."
                    )
                    continue
            else:
                sig_mapped = mapped_cache[key_map]

            if sig_mapped.empty:
                continue

            # 3) Loop remaining parameters: hold_minutes, sl_distance, tp_R_multiple
            for hold_minutes, sl_distance, tp_R_multiple in product(
                    GRID_HOLD_MINUTES, GRID_SL_DISTANCE, GRID_TP_R_MULTIPLE
            ):
                scenario_counter += 1
                print(
                    f"[INFO] Scenario {scenario_counter}/{total_scenarios}: "
                    f"T={T_seconds}, K={K_unique}, H={hold_minutes}, "
                    f"SL={sl_distance}, TP_R={tp_R_multiple}, "
                    f"entry_mode={entry_mode}, DIR={DIRECTION_MODE}, "
                    f"use_tp_exit={USE_TP_EXIT}, use_time_exit={USE_TIME_EXIT}"
                )

                cfg = BacktestConfig(
                    hold_minutes=hold_minutes,
                    sl_distance=sl_distance,
                    entry_mode=entry_mode,
                    direction_mode=DIRECTION_MODE,
                    tp_R_multiple=tp_R_multiple,
                    use_tp_exit=USE_TP_EXIT,
                    use_time_exit=USE_TIME_EXIT,
                    apply_spread=APPLY_SPREAD,
                    spread_dollars=SPREAD_DOLLARS,
                )

                trades_bt, summary = backtest_time_exit(sig_mapped, candles, cfg)

                # Build exit tag + scenario name (must be BEFORE plotting/saving)
                exit_tag = f"tp{int(USE_TP_EXIT)}_time{int(USE_TIME_EXIT)}"
                scenario_name = (
                    f"T{T_seconds}_K{K_unique}_H{hold_minutes}_"
                    f"SL{sl_distance}_TP{tp_R_multiple}_"
                    f"{entry_mode}_{DIRECTION_MODE}_{spread_tag}_{exit_tag}"
                )

                # Optionally plot equity+drawdown for this scenario
                if PLOT_EQUITY_PER_SCENARIO and not trades_bt.empty:
                    plot_equity_and_drawdown(trades_bt, scenario_name, equity_dir)

                # Optionally save per-scenario trades
                if SAVE_TRADES_PER_SCENARIO and not trades_bt.empty:
                    trades_path = scenario_dir / f"trades_{scenario_name}.csv"
                    trades_bt.to_csv(trades_path, index=False)
                    print(f"[INFO]   Saved trades to: {trades_path}")

                # Build a flat row with all params + metrics
                row = {
                    "scenario_name": scenario_name,
                    "T_seconds": T_seconds,
                    "K_unique": K_unique,
                    "hold_minutes": hold_minutes,
                    "sl_distance": sl_distance,
                    "tp_R_multiple": tp_R_multiple,
                    "entry_mode": entry_mode,
                    "direction_mode": DIRECTION_MODE,
                    "use_tp_exit": USE_TP_EXIT,
                    "use_time_exit": USE_TIME_EXIT,
                    "apply_spread": APPLY_SPREAD,
                    "spread_dollars": SPREAD_DOLLARS,
                }
                row.update(summary)

                summary_rows.append(row)


    # ============ SAVE SUMMARY ============
    if not summary_rows:
        print("[WARN] No scenarios produced any results. Nothing to save.")
        return

    summary_df = pd.DataFrame(summary_rows)

    summary_path = summary_dir / "grid_scenarios_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Saved grid summary to: {summary_path}")

    # ============ GRID HEATMAPS ============
    generate_all_param_heatmaps(summary_df, heatmap_dir)

    # ============ DISTRIBUTIONS ============
    generate_distributions(summary_df, distrib_dir)

    # ============ TOP SCENARIOS (console) ============
    # can change sort_by to "avg_R", "profit_factor", etc.
    print_top_scenarios(summary_df, n=30, sort_by="percent_return")

    print("[INFO] Grid run complete.")


if __name__ == "__main__":
    main()
