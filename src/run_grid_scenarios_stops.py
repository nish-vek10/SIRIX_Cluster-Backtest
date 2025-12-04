"""
run_grid_scenarios_stops.py
---------------------------

GRID backtester focused on STOP STYLES for the inverse XAU herding model.

Core ideas (aligned with the dev's write-up):

- Signals come from Sirix:
    * clusters of k unique traders in t seconds
    * this script just consumes whatever find_cluster_signals(...) returns

- We always trade INVERSE to the cluster side:
    * crowd BUY  -> we SELL
    * crowd SELL -> we BUY

- Stops can be:
    * "fixed"         : use cfg.sl_distance like before
    * "atr_static"    : ATR-based stop, fixed after entry
    * "atr_trailing"  : ATR-based stop, trailing
    * "chandelier"    : Chandelier-style trailing stop (extremes +/- ATR*trail_mult)

- Risk is still measured in R, with ABS_RISK_PER_TRADE from config.

This script:

- Sweeps over:
    * T_seconds      (cluster window; full integer range)
    * K_unique       (unique traders, integer range)
    * hold_minutes   (time horizon, integer range)
    * sl_distance    (baseline SL size in dollars, integer steps)
    * tp_R_multiple  (TP in R multiples, integer steps)
    * entry_mode     ("prop", "oanda_open", "oanda_close")
    * stop_mode      (fixed/atr_static/atr_trailing/chandelier)
    * ATR params     (period, init_mult, trail_mult, chandelier_lookback)

- AUTOMATES exit toggles:
    * use_tp_exit  in {False, True}
    * use_time_exit in {False, True}
  -> all 4 combinations are tested in one run.

- Uses spread if APPLY_SPREAD=True.

- Has toggles to enable/disable:
    * equity + drawdown plots
    * heatmaps
    * distributions
    * per-scenario trade CSVs

- Writes everything into a dedicated "stops" namespace so nothing overwrites
  the previous fixed-SL grids.
"""

from pathlib import Path
from itertools import product
import time

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
# USER CONFIG — PARAM RANGES
# =========================
# All integer ranges are inclusive and use step=1 by default.

# Cluster window (seconds)
MIN_T_SECONDS = 20
MAX_T_SECONDS = 60
STEP_T_SECONDS = 10

# Unique traders in cluster
MIN_K_UNIQUE = 3
MAX_K_UNIQUE = 3
STEP_K_UNIQUE = 1

# Time-based hold (minutes)
MIN_HOLD_MINUTES = 10
MAX_HOLD_MINUTES = 30
STEP_HOLD_MINUTES = 5

# SL distance in XAU dollars (integer steps)
MIN_SL_DISTANCE = 2.0      # 2.0
MAX_SL_DISTANCE = 4.0      # 4.0
STEP_SL_DISTANCE = 1.0

# TP in R multiples (integer steps)
MIN_TP_R_MULTIPLE = None
MAX_TP_R_MULTIPLE = None
STEP_TP_R_MULTIPLE = None

# Derived discrete lists from the ranges above
GRID_T_SECONDS = list(range(MIN_T_SECONDS, MAX_T_SECONDS + 1, STEP_T_SECONDS))
GRID_K_UNIQUE = list(range(MIN_K_UNIQUE, MAX_K_UNIQUE + 1, STEP_K_UNIQUE))
GRID_HOLD_MINUTES = list(range(MIN_HOLD_MINUTES, MAX_HOLD_MINUTES + 1, STEP_HOLD_MINUTES))

# SL: step size computation
n_sl_steps = int(round((MAX_SL_DISTANCE - MIN_SL_DISTANCE) / STEP_SL_DISTANCE))
GRID_SL_DISTANCE = [
    MIN_SL_DISTANCE + i * STEP_SL_DISTANCE
    for i in range(n_sl_steps + 1)
]

# TP grid:
# - If any of the bounds is None, we interpret this as "no TP", i.e. trailing-only.
# - In that case, we still step over a single value [None] so the rest of the grid logic works.
if (
    MIN_TP_R_MULTIPLE is None
    or MAX_TP_R_MULTIPLE is None
    or STEP_TP_R_MULTIPLE is None
):
    GRID_TP_R_MULTIPLE = [None]
else:
    GRID_TP_R_MULTIPLE = [
        float(x)
        for x in range(MIN_TP_R_MULTIPLE, MAX_TP_R_MULTIPLE + 1, STEP_TP_R_MULTIPLE)
    ]

# Entry modes
ENTRY_MODES = ["prop"]  # can add "oanda_open", "oanda_close" later if needed


# =========================
# USER CONFIG — STOP MODES
# =========================

# "fixed", "atr_static", "atr_trailing", "chandelier",
STOP_MODES = ["atr_trailing", "chandelier"]

# ATR-related ranges (min, max, step=1) so you can change them easily
MIN_ATR_PERIOD = 5
MAX_ATR_PERIOD = 5
ATR_PERIOD_STEP = 1

MIN_ATR_INIT_MULT = 2     # e.g. X×ATR initial
MAX_ATR_INIT_MULT = 4
ATR_INIT_MULT_STEP = 1

MIN_ATR_TRAIL_MULT = 2    # e.g. X×ATR trail
MAX_ATR_TRAIL_MULT = 4
ATR_TRAIL_MULT_STEP = 1

MIN_CHAN_LOOKBACK = 20    # horizon-style, metadata only for now
MAX_CHAN_LOOKBACK = 40
CHAN_LOOKBACK_STEP = 10

GRID_ATR_PERIOD = list(range(MIN_ATR_PERIOD, MAX_ATR_PERIOD + 1, ATR_PERIOD_STEP))
GRID_ATR_INIT_MULT = [float(x) for x in range(MIN_ATR_INIT_MULT, MAX_ATR_INIT_MULT + 1, ATR_INIT_MULT_STEP)]
GRID_ATR_TRAIL_MULT = [float(x) for x in range(MIN_ATR_TRAIL_MULT, MAX_ATR_TRAIL_MULT + 1, ATR_TRAIL_MULT_STEP)]
GRID_CHAN_LOOKBACK = list(range(MIN_CHAN_LOOKBACK, MAX_CHAN_LOOKBACK + 1, CHAN_LOOKBACK_STEP))


# =========================
# USER CONFIG — GLOBAL TOGGLES
# =========================

RUN_TAG = "v3"

# Direction: dev strategy is always inverse
DIRECTION_MODE = "inverse"   # keep this fixed for this model

# Exit toggles we want to AUTO-GRID across
EXIT_CONFIGS = [
    (False, False),  # trailing only, no time exit (SL/FORCED)
    # (False, True),   # trailing + time exit
    # (True,  False),
    # (True,  True),
]
# (use_tp_exit, use_time_exit)

# Spread modelling (in XAUUSD dollars)
APPLY_SPREAD = True          # True to include spread, False to ignore
SPREAD_DOLLARS = 0.2         # size of the spread in dollars

# Per-scenario outputs
SAVE_TRADES_PER_SCENARIO = False
PLOT_EQUITY_PER_SCENARIO = False   # flip to True when zooming in

# Aggregated visuals on the summary
GENERATE_HEATMAPS = False
GENERATE_DISTRIBUTIONS = False


# =========================
# USER CONFIG — HEATMAPS
# =========================

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
    "dd_per_return",  # new metric
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


def add_atr_columns(candles: pd.DataFrame, periods) -> pd.DataFrame:
    """
    Add ATR columns (e.g. ATR_5, ATR_10) to the candles DataFrame.

    Expects columns: 'open_time', 'high', 'low', 'close'.
    """
    candles = candles.sort_values("open_time").reset_index(drop=True).copy()

    high = candles["high"]
    low = candles["low"]
    close = candles["close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    candles["TR"] = tr

    for p in periods:
        col = f"ATR_{p}"
        candles[col] = candles["TR"].rolling(window=p, min_periods=p).mean()

    return candles


def iter_stop_param_combos():
    """
    Generate (stop_mode, atr_period, atr_init_mult, atr_trail_mult, chandelier_lookback) combos.

    - "fixed"         => no ATR params (all None)
    - "atr_static"    => atr_period x atr_init_mult
    - "atr_trailing"  => atr_period x atr_init_mult x atr_trail_mult
    - "chandelier"    => atr_period x atr_init_mult x atr_trail_mult x chan_lookback
    """
    for stop_mode in STOP_MODES:
        if stop_mode == "fixed":
            yield stop_mode, None, None, None, None
        elif stop_mode == "atr_static":
            for atr_period, atr_init in product(GRID_ATR_PERIOD, GRID_ATR_INIT_MULT):
                yield stop_mode, atr_period, atr_init, None, None
        elif stop_mode == "atr_trailing":
            for atr_period, atr_init, atr_trail in product(
                GRID_ATR_PERIOD, GRID_ATR_INIT_MULT, GRID_ATR_TRAIL_MULT
            ):
                yield stop_mode, atr_period, atr_init, atr_trail, None
        elif stop_mode == "chandelier":
            for atr_period, atr_init, atr_trail, chan_lb in product(
                GRID_ATR_PERIOD, GRID_ATR_INIT_MULT, GRID_ATR_TRAIL_MULT, GRID_CHAN_LOOKBACK
            ):
                yield stop_mode, atr_period, atr_init, atr_trail, chan_lb
        else:
            raise ValueError(f"Unknown stop_mode: {stop_mode}")


def generate_all_param_heatmaps(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Generate heatmaps for EACH metric in HEATMAP_METRICS and for EACH
    combination of:
      - K_unique
      - sl_distance
      - tp_R_multiple
      - entry_mode

    X axis: T_seconds
    Y axis: hold_minutes
    Colour: that metric (e.g. avg_R, win_rate, ...)
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
                        df = summary_df[
                            (summary_df["K_unique"] == K)
                            & (summary_df["sl_distance"] == SL)
                            & (summary_df["tp_R_multiple"] == TP)
                            & (summary_df["entry_mode"] == entry_mode)
                        ]

                        if df.empty:
                            continue

                        if x_param not in df.columns or y_param not in df.columns:
                            print(
                                f"[WARN] Missing {x_param}/{y_param} for "
                                f"K={K}, SL={SL}, TP={TP}, entry_mode={entry_mode}"
                            )
                            continue

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
    Distribution-style plots from the STOP-grid summary.
    """
    if summary_df.empty:
        print("[WARN] No summary data for distributions.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    if "percent_return" in summary_df.columns:
        vals = summary_df["percent_return"].dropna()
        if not vals.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(vals, bins=50, density=True)
            ax.set_title("Distribution of Percent Return Across STOP Scenarios")
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
            ax.set_title("Distribution of Avg R Across STOP Scenarios")
            ax.set_xlabel("Average R")
            ax.set_ylabel("Density")
            fig.tight_layout()
            path = out_dir / "distribution_avg_R_all_scenarios.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[INFO] Saved: {path}")

    if "percent_return" in summary_df.columns and "sl_distance" in summary_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(summary_df["sl_distance"], summary_df["percent_return"])
        ax.set_title("SL Distance vs Percent Return (STOP grid)")
        ax.set_xlabel("SL Distance")
        ax.set_ylabel("Percent Return")
        fig.tight_layout()
        path = out_dir / "scatter_SL_vs_percent_return.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved: {path}")

    if "percent_return" in summary_df.columns and "tp_R_multiple" in summary_df.columns:
        # Only plot if there are numeric TP values
        tp_vals = pd.to_numeric(summary_df["tp_R_multiple"], errors="coerce")
        mask = tp_vals.notna() & summary_df["percent_return"].notna()
        vals = summary_df[mask]
        tp_vals = tp_vals[mask]
        if not vals.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(tp_vals, vals["percent_return"])
            ax.set_title("TP_R Multiple vs Percent Return (STOP grid)")
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
    Equity + drawdown figure.
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

    df = trades.sort_values("entry_time")

    ax_eq.plot(df["entry_time"], df["equity"])
    ax_eq.set_title(f"Equity & Drawdown — {scenario_name}")
    ax_eq.set_ylabel("Equity")
    ax_eq.grid(True)

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
    Console leaderboard for this STOP grid run, favouring:
    - highest sort_by (default percent_return)
    - then lowest max_drawdown_R as secondary sort if available
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

    if "max_drawdown_R" in df.columns:
        df = df.sort_values(
            by=[sort_by, "max_drawdown_R"],
            ascending=[False, True]
        )
    else:
        df = df.sort_values(sort_by, ascending=False)

    df = df.head(n)

    cols_to_show = [
        "scenario_name",
        "stop_mode",
        "atr_period",
        "atr_init_mult",
        "atr_trail_mult",
        "chandelier_lookback",
        "T_seconds",
        "K_unique",
        "hold_minutes",
        "sl_distance",
        "tp_R_multiple",
        "entry_mode",
        "direction_mode",
        "use_tp_exit",
        "use_time_exit",
        "apply_spread",
        "spread_dollars",
        "percent_return",
        "avg_R",
        "win_rate",
        "profit_factor",
        "max_drawdown_R",
        "longest_win_streak",
        "longest_loss_streak",
        "n_trades",
        "dd_per_return",
    ]

    cols_present = [c for c in cols_to_show if c in df.columns]

    print("\n=== TOP STOP-STYLE SCENARIOS (sorted by '{}', desc; then min DD) ===".format(sort_by))
    print(df[cols_present].to_string(index=False))
    print("=== END TOP SCENARIOS ===\n")


# =========================
# MAIN GRID RUNNER
# =========================

def main():
    ensure_directories()

    # global timing + counters
    t0 = time.time()
    global_scenario_counter = 0
    global_rows_saved = 0

    spread_tag = "spread_on" if APPLY_SPREAD else "spread_off"
    mode_tag = DIRECTION_MODE.lower()

    # Pre-load data + caches (shared across all exit configs)
    print("[INFO] Loading trades & candles...")
    trades = load_trades()
    candles = load_candles()
    candles = prepare_candles_for_mapping(candles)
    candles = add_atr_columns(candles, periods=GRID_ATR_PERIOD)
    print(f"[INFO] Trades: {len(trades)}, Candles: {len(candles)}")
    print(f"[INFO] ATR columns added: {[f'ATR_{p}' for p in GRID_ATR_PERIOD]}")

    signals_cache = {}         # (T_seconds, K_unique) -> signals_df
    mapped_cache = {}          # (T_seconds, K_unique, entry_mode) -> signals_mapped_df
    stop_param_combos = list(iter_stop_param_combos())
    n_stop_combos = len(stop_param_combos)

    # For final summary table
    exit_summary_rows = []

    # Loop over all TP/TIME exit configs
    for use_tp_exit, use_time_exit in EXIT_CONFIGS:
        exit_tag = f"tp{int(use_tp_exit)}_time{int(use_time_exit)}"

        # Dedicated dirs per exit config
        scenario_dir = SCENARIO_REPORTS_DIR / "stops" / RUN_TAG / mode_tag / spread_tag / exit_tag
        summary_dir = SUMMARY_REPORTS_DIR / "stops" / RUN_TAG / mode_tag / spread_tag / exit_tag
        heatmap_dir = EQUITY_FIG_DIR.parent / "grid_heatmaps_stops" / RUN_TAG / mode_tag / spread_tag / exit_tag
        distrib_dir = EQUITY_FIG_DIR.parent / "distributions_stops" / RUN_TAG / mode_tag / spread_tag / exit_tag
        equity_dir = EQUITY_FIG_DIR / "stops" / RUN_TAG / mode_tag / spread_tag / exit_tag

        scenario_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        distrib_dir.mkdir(parents=True, exist_ok=True)
        equity_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[INFO] ===== EXIT CONFIG: use_tp_exit={use_tp_exit}, use_time_exit={use_time_exit} =====")

        summary_rows = []
        local_scenario_counter = 0

        # Compute total scenarios for this exit config
        total_scenarios_exit = (
            len(GRID_T_SECONDS)
            * len(GRID_K_UNIQUE)
            * len(GRID_HOLD_MINUTES)
            * len(GRID_SL_DISTANCE)
            * len(GRID_TP_R_MULTIPLE)
            * len(ENTRY_MODES)
            * n_stop_combos
        )

        for T_seconds, K_unique in product(GRID_T_SECONDS, GRID_K_UNIQUE):
            # 1) Build/retrieve signals for this (T, K)
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

                # 3) Loop stop param combos + hold/sl/tp
                for stop_mode, atr_period, atr_init, atr_trail, chan_lb in stop_param_combos:
                    for hold_minutes, sl_distance, tp_R_multiple in product(
                        GRID_HOLD_MINUTES, GRID_SL_DISTANCE, GRID_TP_R_MULTIPLE
                    ):
                        global_scenario_counter += 1
                        local_scenario_counter += 1

                        print(
                            f"[INFO] Scenario {local_scenario_counter}/{total_scenarios_exit} "
                            f"(global {global_scenario_counter}): "
                            f"T={T_seconds}, K={K_unique}, H={hold_minutes}, "
                            f"SL={sl_distance}, TP_R={tp_R_multiple}, "
                            f"entry_mode={entry_mode}, DIR={DIRECTION_MODE}, "
                            f"stop_mode={stop_mode}, ATR_P={atr_period}, "
                            f"ATR_I={atr_init}, ATR_T={atr_trail}, "
                            f"CH_LB={chan_lb}, use_tp_exit={use_tp_exit}, "
                            f"use_time_exit={use_time_exit}, spread={APPLY_SPREAD}"
                        )

                        cfg = BacktestConfig(
                            hold_minutes=hold_minutes,
                            sl_distance=sl_distance,
                            entry_mode=entry_mode,
                            direction_mode=DIRECTION_MODE,
                            tp_R_multiple=tp_R_multiple,
                            use_tp_exit=use_tp_exit,
                            use_time_exit=use_time_exit,
                            apply_spread=APPLY_SPREAD,
                            spread_dollars=SPREAD_DOLLARS,
                            stop_mode=stop_mode,
                            atr_period=atr_period,
                            atr_init_mult=atr_init,
                            atr_trail_mult=atr_trail,
                            chandelier_lookback=chan_lb,
                        )

                        trades_bt, summary = backtest_time_exit(sig_mapped, candles, cfg)

                        # Build stop_tag (include stop info + exit config + spread)
                        stop_tag = stop_mode
                        if atr_period is not None and atr_init is not None:
                            stop_tag += f"_ATR{atr_period}_I{atr_init}"
                        if atr_trail is not None:
                            stop_tag += f"_T{atr_trail}"
                        if chan_lb is not None:
                            stop_tag += f"_CH{chan_lb}"

                        scenario_name = (
                            f"T{T_seconds}_K{K_unique}_H{hold_minutes}_"
                            f"SL{sl_distance}_TP{tp_R_multiple}_"
                            f"{entry_mode}_{DIRECTION_MODE}_{stop_tag}_"
                            f"{spread_tag}_{exit_tag}"
                        )

                        # ----- Extra metric: drawdown per % return -----
                        max_dd_R = summary.get("max_drawdown_R")
                        pct_ret = summary.get("percent_return")
                        dd_per_return = None
                        if (max_dd_R is not None) and (pct_ret is not None):
                            try:
                                if pd.notna(max_dd_R) and pd.notna(pct_ret) and pct_ret != 0:
                                    dd_per_return = abs(max_dd_R) / pct_ret
                            except Exception:
                                dd_per_return = None

                        # Equity+DD plots
                        if PLOT_EQUITY_PER_SCENARIO and not trades_bt.empty:
                            plot_equity_and_drawdown(trades_bt, scenario_name, equity_dir)

                        # Save trades if requested
                        if SAVE_TRADES_PER_SCENARIO and not trades_bt.empty:
                            trades_path = scenario_dir / f"trades_{scenario_name}.csv"
                            trades_bt.to_csv(trades_path, index=False)
                            print(f"[INFO]   Saved trades to: {trades_path}")

                        # Build summary row
                        row = {
                            "scenario_name": scenario_name,
                            "T_seconds": T_seconds,
                            "K_unique": K_unique,
                            "hold_minutes": hold_minutes,
                            "sl_distance": sl_distance,
                            "tp_R_multiple": tp_R_multiple,
                            "entry_mode": entry_mode,
                            "direction_mode": DIRECTION_MODE,
                            "use_tp_exit": use_tp_exit,
                            "use_time_exit": use_time_exit,
                            "apply_spread": APPLY_SPREAD,
                            "spread_dollars": SPREAD_DOLLARS,
                            "stop_mode": stop_mode,
                            "atr_period": atr_period,
                            "atr_init_mult": atr_init,
                            "atr_trail_mult": atr_trail,
                            "chandelier_lookback": chan_lb,
                            "dd_per_return": dd_per_return,
                        }
                        row.update(summary)

                        summary_rows.append(row)
                        global_rows_saved += 1

        # ===== SAVE SUMMARY FOR THIS EXIT CONFIG =====
        if not summary_rows:
            print(f"[WARN] No scenarios produced any results for exit_tag={exit_tag}.")
            continue

        summary_df = pd.DataFrame(summary_rows)

        # Auto-generated name includes RUN_TAG and exit_tag
        summary_filename = f"grid_scenarios_stops_{RUN_TAG}_{exit_tag}.csv"
        summary_path = summary_dir / summary_filename
        summary_df.to_csv(summary_path, index=False)
        print(f"[INFO] Saved STOP-grid summary for {exit_tag} to: {summary_path}")

        # Optional heatmaps & distributions
        if GENERATE_HEATMAPS:
            generate_all_param_heatmaps(summary_df, heatmap_dir)

        if GENERATE_DISTRIBUTIONS:
            generate_distributions(summary_df, distrib_dir)

        print_top_scenarios(summary_df, n=30, sort_by="percent_return")

        exit_summary_rows.append({
            "exit_tag": exit_tag,
            "use_tp_exit": use_tp_exit,
            "use_time_exit": use_time_exit,
            "n_scenarios": len(summary_df),
            "summary_path": str(summary_path),
        })

    # ========= FINAL RUNTIME SUMMARY =========
    t1 = time.time()
    elapsed_sec = t1 - t0
    hrs = int(elapsed_sec // 3600)
    mins = int((elapsed_sec % 3600) // 60)
    secs = int(elapsed_sec % 60)

    print("\n================ GRID RUN COMPLETE ================")
    print(f"Total elapsed time: {hrs:02d}:{mins:02d}:{secs:02d}")
    print(f"Total scenario loops processed: {global_scenario_counter}")
    print(f"Total summary rows saved:       {global_rows_saved}")

    if exit_summary_rows:
        summary_table = pd.DataFrame(exit_summary_rows)
        print("\n=== SUMMARY BY EXIT CONFIG ===")
        print(summary_table.to_string(index=False))
        print("================================\n")


if __name__ == "__main__":
    main()
