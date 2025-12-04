"""
run_single_scenario.py
----------------------

Run ONE concrete backtest scenario end-to-end, aligned with the STOP-grid:

- Uses the SAME BacktestConfig fields as run_grid_scenarios_stops.py
- Supports stop modes:
    * "fixed"
    * "atr_static"
    * "atr_trailing"
    * "chandelier"
- Allows you to plug in exactly the same parameters as any row
  in grid_scenarios_stops_summary.csv and see the full trade log + plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from config import (
    ensure_directories,
    SUMMARY_REPORTS_DIR,
    SCENARIO_REPORTS_DIR,
    EQUITY_FIG_DIR,
)
from load_data import load_trades, load_candles
from clustering import find_cluster_signals
# same mapper as grid scripts
from mapping import map_signals_to_candles, prepare_candles_for_mapping
from backtest import backtest_time_exit, BacktestConfig


# =========================
# USER CONFIG — SCENARIO
# =========================
# Paste values from a chosen grid row here.

# Cluster params
T_seconds = 30
K_unique = 3

# Exit / risk params
hold_minutes = 15
sl_distance = 2.0    # baseline SL in dollars (defines 1R for fixed mode)

# Entry mode (same as grid)
entry_mode = "prop"  # "prop", "oanda_open", "oanda_close"

# Direction logic
# "directional" = follow the cluster signal (BUY->BUY, SELL->SELL)
# "inverse"     = fade the signal        (BUY->SELL, SELL->BUY)
direction_mode = "directional"  # "directional" or "inverse"

# TP in R-multiples of SL
# Example: sl_distance=2.0, tp_R_multiple=2.0 -> TP distance=4.0
tp_R_multiple = None  # set to None to disable TP entirely

# Exit toggles
use_tp_exit = False    # matches the grid EXIT_CONFIGS semantics
use_time_exit = False  # True to enable time-based exit at hold_minutes


# =========================
# USER CONFIG — STOP MODE
# =========================
# These fields mirror run_grid_scenarios_stops.py exactly

STOP_MODE = "chandelier"       # "fixed", "atr_static", "atr_trailing", "chandelier"

# ATR/chandelier parameters (only used if relevant for STOP_MODE)
ATR_PERIOD = 5            # same range as GRID_ATR_PERIOD in grid script
ATR_INIT_MULT = 3.0       # X * ATR initial
ATR_TRAIL_MULT = 2.0      # X * ATR trail (for trailing / chandelier)
CHANDELIER_LOOKBACK = 30  # lookback window for chandelier highs/lows

# =========================
# USER CONFIG — SPREAD
# =========================

APPLY_SPREAD = True          # True to include spread, False to ignore
SPREAD_DOLLARS = 0.2         # size of the spread in dollars


# =========================
# USER CONFIG — OUTPUT TOGGLES
# =========================

SAVE_TRADES_CSV = True           # save detailed trades log
PLOT_EQUITY = True               # plot equity + drawdown
PLOT_HEATMAP = False              # plot time-of-day heatmaps
PLOT_DISTRIBUTIONS = False        # simple per-trade R histogram

# If True, save under the same "stops" namespace as run_grid_scenarios_stops.py
USE_STOPS_NAMESPACE = True
RUN_TAG = "v3"

# =========================
# PLOTTING HELPERS
# =========================

def plot_equity_and_drawdown(
    trades: pd.DataFrame,
    scenario_name: str,
    out_dir: Path,
) -> None:
    """
    Plot equity and drawdown:

    - Smoothed equity line + light shaded area to starting equity (top)
    - Smoothed drawdown line + light shaded area (bottom)
    - Zero line added for reference on drawdown
    """
    if trades.empty:
        print(f"[WARN] No trades for equity/drawdown plot: {scenario_name}")
        return

    required_cols = {"entry_time", "equity", "drawdown"}
    if not required_cols.issubset(trades.columns):
        print(f"[WARN] Missing columns {required_cols - set(trades.columns)} "
              f"for equity/drawdown plot for {scenario_name}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    df = trades.sort_values("entry_time").copy()

    # --- simple visual smoothing helper ---
    def _smooth(series: pd.Series, window: int = 3) -> pd.Series:
        if len(series) < window:
            return series
        return (
            series.rolling(window=window, center=True)
            .mean()
            .bfill()
            .ffill()
        )

    t = df["entry_time"]
    equity = df["equity"].astype(float)
    drawdown = df["drawdown"].astype(float)

    equity_smooth = _smooth(equity, window=3)
    drawdown_smooth = _smooth(drawdown, window=3)

    # use starting equity as baseline for fill
    equity_baseline = float(equity_smooth.iloc[0])

    # --- Set up figure ---
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # -------------------------
    # Equity curve (top)
    # -------------------------
    # Shaded area from baseline to equity
    ax_eq.fill_between(
        t,
        equity_baseline,
        equity_smooth,
        where=equity_smooth >= equity_baseline,
        alpha=0.10,
        color="blue",
    )

    # Equity line
    ax_eq.plot(
        t,
        equity_smooth,
        color="blue",
        linewidth=1.8,
    )

    ax_eq.set_title(f"Equity & Drawdown — {scenario_name}")
    ax_eq.set_ylabel("Equity")
    ax_eq.grid(True, alpha=0.3)

    # -------------------------
    # Drawdown curve (bottom)
    # -------------------------
    # Shaded area (light red) below zero
    ax_dd.fill_between(
        t,
        drawdown_smooth,
        0,
        where=(drawdown_smooth < 0),
        color="red",
        alpha=0.15,
    )

    # Drawdown line
    ax_dd.plot(
        t,
        drawdown_smooth,
        color="red",
        linewidth=1.6,
    )

    # Zero line
    ax_dd.axhline(0, color="black", linewidth=0.8)

    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Time")
    ax_dd.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = out_dir / f"equity_dd_{scenario_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved equity+drawdown curve to: {out_path}")


def _plot_hour_side_metric_heatmap(
    trades: pd.DataFrame,
    scenario_name: str,
    out_dir: Path,
    value_col: str,
    metric_name: str,
    aggfunc,
    center_at_zero: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Generic helper: Hour-of-day (UK) vs trade_side heatmap for a given metric.

    - value_col: column to aggregate (e.g. 'pnl_R')
    - metric_name: label used in title / colourbar
    - aggfunc: function or string for pandas pivot_table
    - center_at_zero: if True, use a diverging norm with 0 as centre
    - vmin / vmax: optional explicit bounds (e.g. 0..1 for win rate)
    """
    if trades.empty:
        print(f"[WARN] No trades for {metric_name} heatmap.")
        return

    required_cols = {"entry_hour", "trade_side", value_col}
    if not required_cols.issubset(trades.columns):
        print(f"[WARN] Missing columns {required_cols - set(trades.columns)} "
              f"for {metric_name} heatmap in {scenario_name}")
        return

    pivot = trades.pivot_table(
        index="entry_hour",
        columns="trade_side",
        values=value_col,
        aggfunc=aggfunc,
    )

    # Ensure all 24 hours on Y-axis
    full_idx = pd.Index(range(24), name="entry_hour")
    pivot = pivot.reindex(full_idx)

    if pivot.empty or pivot.shape[1] == 0:
        print(f"[WARN] No data for {metric_name} heatmap in {scenario_name}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    data = pivot.values.astype(float)

    # Determine colour scaling / norm
    norm = None
    if center_at_zero:
        # Symmetric scale around 0 for R-type metrics
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        if np.isfinite(data_min) and np.isfinite(data_max) and data_min != data_max:
            max_abs = max(abs(data_min), abs(data_max))
            norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    else:
        # Optional explicit bounds (e.g. 0..1 for win rate)
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            vmin = vmax = None

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdYlGn",   # red = worst, green = best
        norm=norm,
        vmin=None if center_at_zero else vmin,
        vmax=None if center_at_zero else vmax,
    )

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_ylabel("Hour of day (UK)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("Trade side")

    ax.set_title(f"{metric_name} by Hour & Side — {scenario_name}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_name)

    fig.tight_layout()

    safe_metric = metric_name.lower().replace(" ", "_")
    out_path = out_dir / f"heatmap_hour_side_{safe_metric}_{scenario_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {metric_name} heatmap to: {out_path}")


def plot_time_of_day_heatmap(
    trades: pd.DataFrame,
    scenario_name: str,
    out_dir: Path,
) -> None:
    """
    Build a small "pack" of time-of-day heatmaps for this scenario:

    - Avg R (mean pnl_R, centred at 0)
    - Median R (median pnl_R, centred at 0)
    - Win rate (share of trades with pnl_R > 0)
    - Trade count
    - Avg drawdown
    - Avg price move
    - Avg PnL (cash)
    """

    if trades.empty:
        print("[WARN] No trades for time-of-day heatmaps.")
        return

    # Scenario-specific subfolder so different single runs stay separate
    scenario_dir = out_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Average R
    # -------------------------
    _plot_hour_side_metric_heatmap(
        trades=trades,
        scenario_name=scenario_name,
        out_dir=scenario_dir,
        value_col="pnl_R",
        metric_name="Avg R",
        aggfunc="mean",
        center_at_zero=True,
    )

    # -------------------------
    # 2) Median R
    # -------------------------
    _plot_hour_side_metric_heatmap(
        trades=trades,
        scenario_name=scenario_name,
        out_dir=scenario_dir,
        value_col="pnl_R",
        metric_name="Median R",
        aggfunc="median",
        center_at_zero=True,
    )

    # -------------------------
    # 3) Win rate (0..1)
    # -------------------------
    trades_win_flag = trades.copy()
    trades_win_flag["is_win"] = trades_win_flag["pnl_R"] > 0

    _plot_hour_side_metric_heatmap(
        trades=trades_win_flag,
        scenario_name=scenario_name,
        out_dir=scenario_dir,
        value_col="is_win",
        metric_name="Win rate",
        aggfunc="mean",      # mean of bools = proportion
        center_at_zero=False,
        vmin=0.0,
        vmax=1.0,
    )

    # -------------------------
    # 4) Trade count
    # -------------------------
    _plot_hour_side_metric_heatmap(
        trades=trades,
        scenario_name=scenario_name,
        out_dir=scenario_dir,
        value_col="pnl_R",   # any existing col; aggfunc='count' ignores values
        metric_name="Trade count",
        aggfunc="count",
        center_at_zero=False,
    )

    # -------------------------
    # 5) Avg drawdown
    # -------------------------
    _plot_hour_side_metric_heatmap(
        trades=trades,
        scenario_name=scenario_name,
        out_dir=scenario_dir,
        value_col="drawdown",
        metric_name="Avg drawdown",
        aggfunc="mean",
        center_at_zero=True,   # best is close to 0, worst deeply negative
    )

    # -------------------------
    # 6) Avg price move
    # -------------------------
    _plot_hour_side_metric_heatmap(
        trades=trades,
        scenario_name=scenario_name,
        out_dir=scenario_dir,
        value_col="price_move",
        metric_name="Avg price move",
        aggfunc="mean",
        center_at_zero=True,   # symmetric: positive vs negative moves
    )

    # -------------------------
    # 7) Avg PnL (cash)
    # -------------------------
    _plot_hour_side_metric_heatmap(
        trades=trades,
        scenario_name=scenario_name,
        out_dir=scenario_dir,
        value_col="pnl_cash",
        metric_name="Avg PnL (cash)",
        aggfunc="mean",
        center_at_zero=True,
    )


def plot_distributions_for_single(
    trades: pd.DataFrame,
    scenario_name: str,
    out_dir: Path,
) -> None:
    """
    Build a small pack of distribution plots for this scenario.

    For each available column in DIST_METRICS, we create a histogram:
      - pnl_R:      R multiple per trade
      - pnl_cash:   Cash PnL per trade
      - hold_minutes: Actual hold duration in minutes
      - price_move: Underlying price move (if present)

    All figures are saved under:
      <out_dir>/<scenario_name>/distribution_<col>_<scenario_name>.png
    """
    if trades.empty:
        print(f"[WARN] No trades for distributions: {scenario_name}")
        return

    # Scenario-specific subfolder so different single runs stay separate
    scenario_dir = out_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # (column name, pretty label)
    DIST_METRICS = [
        ("pnl_R", "R multiple"),
        ("pnl_cash", "PnL (cash)"),
        ("hold_minutes", "Hold minutes"),
        ("price_move", "Price move"),
        ("drawdown", "Drawdown"),
    ]

    for col, label in DIST_METRICS:
        if col not in trades.columns:
            print(f"[INFO] Column '{col}' not in trades; skipping distribution.")
            continue

        vals = trades[col].dropna()
        if vals.empty:
            print(f"[INFO] Column '{col}' has no non-NaN values; skipping distribution.")
            continue

        # Make sure it's numeric
        try:
            vals = vals.astype(float)
        except Exception:
            print(f"[WARN] Column '{col}' could not be cast to float; skipping distribution.")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        # Histogram; density=True gives you a proper distribution shape
        ax.hist(vals, bins=40, density=True)

        ax.set_title(f"Distribution of {label} — {scenario_name}")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")

        fig.tight_layout()

        out_path = scenario_dir / f"distribution_{col}_{scenario_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"[INFO] Saved distribution for '{col}' to: {out_path}")


def ensure_atr_and_chandelier_columns(
    candles: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """
    Make sure the candles DataFrame has the ATR / chandelier columns that
    backtest_time_exit() expects, based on cfg.stop_mode and parameters.

    - ATR column name:  ATR_{atr_period}
    - Chandelier high:  CH_high_{lookback}
    - Chandelier low:   CH_low_{lookback}
    """

    # We need basic OHLC columns for these calculations
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(candles.columns):
        missing = required_cols - set(candles.columns)
        raise ValueError(
            f"Candles are missing required OHLC columns for ATR/chandelier calc: {missing}"
        )

    # ------------------------------------------------------------------
    # ATR (used by atr_static, atr_trailing, chandelier)
    # ------------------------------------------------------------------
    if cfg.stop_mode in ("atr_static", "atr_trailing", "chandelier"):
        if cfg.atr_period is None:
            raise ValueError("cfg.atr_period is None but ATR-based stop_mode was requested.")

        atr_period = int(cfg.atr_period)
        atr_col = f"ATR_{atr_period}"

        if atr_col not in candles.columns:
            high = candles["high"]
            low = candles["low"]
            close = candles["close"]

            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = true_range.rolling(window=atr_period, min_periods=atr_period).mean()
            candles[atr_col] = atr

    # ------------------------------------------------------------------
    # Chandelier high/low (only needed for chandelier stop_mode)
    # ------------------------------------------------------------------
    if cfg.stop_mode == "chandelier":
        if cfg.chandelier_lookback is None:
            raise ValueError(
                "cfg.chandelier_lookback is None but chandelier stop_mode was requested."
            )

        lb = int(cfg.chandelier_lookback)
        ch_high_col = f"CH_high_{lb}"
        ch_low_col = f"CH_low_{lb}"

        if ch_high_col not in candles.columns:
            candles[ch_high_col] = candles["high"].rolling(
                window=lb, min_periods=lb
            ).max()

        if ch_low_col not in candles.columns:
            candles[ch_low_col] = candles["low"].rolling(
                window=lb, min_periods=lb
            ).min()

    return candles


# =========================
# MAIN
# =========================

def main():
    ensure_directories()

    mode_tag = direction_mode.lower()
    exit_tag = f"tp{int(use_tp_exit)}_time{int(use_time_exit)}"
    spread_tag = "spread_on" if APPLY_SPREAD else "spread_off"
    run_tag = RUN_TAG

    # Build stop_tag the SAME WAY as the grid script
    stop_tag = STOP_MODE
    if ATR_PERIOD is not None and ATR_INIT_MULT is not None:
        stop_tag += f"_ATR{int(ATR_PERIOD)}_I{float(ATR_INIT_MULT)}"
    if ATR_TRAIL_MULT is not None:
        stop_tag += f"_T{float(ATR_TRAIL_MULT)}"
    if CHANDELIER_LOOKBACK is not None and STOP_MODE == "chandelier":
        stop_tag += f"_CH{int(CHANDELIER_LOOKBACK)}"

    # Scenario name identical pattern to run_grid_scenarios_stops.py
    scenario_name = (
        f"T{T_seconds}_K{K_unique}_H{hold_minutes}_"
        f"SL{sl_distance}_TP{tp_R_multiple}_"
        f"{entry_mode}_{direction_mode}_{stop_tag}_"
        f"{spread_tag}_{exit_tag}"
    )

    print(f"[INFO] Running single scenario: {scenario_name}")

    # --------- DIRECTORY STRUCTURE (aligned with stops grid) ---------
    if USE_STOPS_NAMESPACE:
        scenario_dir = SCENARIO_REPORTS_DIR / "stops" / run_tag / mode_tag / spread_tag / exit_tag / "single"
        summary_dir = SUMMARY_REPORTS_DIR / "stops" / run_tag / mode_tag / spread_tag / exit_tag / "single"
        equity_dir = EQUITY_FIG_DIR / "stops" / run_tag / mode_tag / spread_tag / exit_tag / "single"
        heatmap_dir = EQUITY_FIG_DIR.parent / "heatmaps_stops" / run_tag / mode_tag / spread_tag / exit_tag / "single"
        distrib_dir = EQUITY_FIG_DIR.parent / "distributions_stops" / run_tag / mode_tag / spread_tag / exit_tag / "single"
    else:
        scenario_dir = SCENARIO_REPORTS_DIR / mode_tag / spread_tag / exit_tag / "single"
        summary_dir = SUMMARY_REPORTS_DIR / mode_tag / spread_tag / exit_tag / "single"
        equity_dir = EQUITY_FIG_DIR / mode_tag / spread_tag / exit_tag / "single"
        heatmap_dir = EQUITY_FIG_DIR.parent / "heatmaps" / mode_tag / spread_tag / exit_tag / "single"
        distrib_dir = EQUITY_FIG_DIR.parent / "distributions" / mode_tag / spread_tag / exit_tag / "single"

    scenario_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    equity_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    distrib_dir.mkdir(parents=True, exist_ok=True)

    # --------- ENTRY PRICE SOURCE (QC) ---------
    if entry_mode == "oanda_open":
        entry_price_source = "open"
    elif entry_mode == "oanda_close":
        entry_price_source = "close"
    else:  # "prop"
        entry_price_source = "open"

    # 1) Load data
    print("[INFO] Loading trades & candles...")
    trades = load_trades()
    candles = load_candles()
    candles = prepare_candles_for_mapping(candles)
    print(f"[INFO] Trades: {len(trades)}, Candles: {len(candles)}")

    # 2) Cluster signals
    print(f"[INFO] Building signals (T={T_seconds}s, K={K_unique})...")
    signals = find_cluster_signals(trades, T_seconds=T_seconds, K_unique=K_unique)
    print(f"[INFO] Signals found: {len(signals)}")
    if signals.empty:
        print("[WARN] No signals found for this configuration. Exiting.")
        return

    # 3) Map to candles
    print(f"[INFO] Mapping signals to candles (entry_price_source={entry_price_source})...")
    signals_with_entries, qc_summary = map_signals_to_candles(
        signals,
        candles,
        entry_price_source=entry_price_source,
    )
    print(f"[INFO] Mapped signals: {len(signals_with_entries)}")

    if qc_summary is not None and not qc_summary.empty:
        print("\n=== QC SUMMARY ===")
        print(qc_summary.to_string(index=False))

    if signals_with_entries.empty:
        print("[WARN] No mapped signals after QC. Exiting.")
        return

    # 4) Backtest with full exit logic (SL/TP/TIME/FORCED/SPREAD/STOP-MODE)
    cfg = BacktestConfig(
        hold_minutes=hold_minutes,
        sl_distance=sl_distance,
        entry_mode=entry_mode,
        direction_mode=direction_mode,
        tp_R_multiple=tp_R_multiple,
        use_tp_exit=use_tp_exit,
        use_time_exit=use_time_exit,
        apply_spread=APPLY_SPREAD,
        spread_dollars=SPREAD_DOLLARS,
        stop_mode=STOP_MODE,
        atr_period=ATR_PERIOD,
        atr_init_mult=ATR_INIT_MULT,
        atr_trail_mult=ATR_TRAIL_MULT,
        chandelier_lookback=CHANDELIER_LOOKBACK,
    )

    print(
        f"[INFO] Backtesting with: "
        f"hold={hold_minutes}m, SL={sl_distance}, TP_R={tp_R_multiple}, "
        f"entry_mode={entry_mode}, direction_mode={direction_mode}, "
        f"use_tp_exit={use_tp_exit}, use_time_exit={use_time_exit}, "
        f"spread={APPLY_SPREAD} ({SPREAD_DOLLARS}), "
        f"stop_mode={STOP_MODE}, ATR_P={ATR_PERIOD}, "
        f"ATR_I={ATR_INIT_MULT}, ATR_T={ATR_TRAIL_MULT}, "
        f"CH_LB={CHANDELIER_LOOKBACK}"
    )

    candles = ensure_atr_and_chandelier_columns(candles, cfg)

    trades_bt, summary = backtest_time_exit(signals_with_entries, candles, cfg)

    print(f"[INFO] Backtest produced {len(trades_bt)} trades.")

    # -------------------------------------------------
    # Extra summary fields (risk + single-trade extremes)
    # -------------------------------------------------
    max_risk_per_trade = np.nan
    min_risk_per_trade = np.nan
    largest_profit_cash = np.nan
    largest_loss_cash = np.nan

    if not trades_bt.empty:
        if "risk_per_trade" in trades_bt.columns:
            max_risk_per_trade = float(trades_bt["risk_per_trade"].max())
            min_risk_per_trade = float(trades_bt["risk_per_trade"].min())

        if "pnl_cash" in trades_bt.columns:
            largest_profit_cash = float(trades_bt["pnl_cash"].max())
            largest_loss_cash = float(trades_bt["pnl_cash"].min())

    # Inject into summary dict (alongside longest_win_streak, longest_loss_streak from backtest.py)
    summary.update(
        {
            "max_risk_per_trade": max_risk_per_trade,
            "min_risk_per_trade": min_risk_per_trade,
            "largest_profit_cash": largest_profit_cash,
            "largest_loss_cash": largest_loss_cash,
        }
    )

    # 5) Save detailed trades CSV
    if SAVE_TRADES_CSV:
        scenario_csv = scenario_dir / f"trades_{scenario_name}.csv"
        trades_bt.to_csv(scenario_csv, index=False)
        print(f"[INFO] Saved trades to: {scenario_csv}")

    # 6) Save summary metrics CSV (long format)
    combined_summary = {
        "scenario_name": scenario_name,
        "T_seconds": T_seconds,
        "K_unique": K_unique,
        "hold_minutes": hold_minutes,
        "sl_distance": sl_distance,
        "tp_R_multiple": tp_R_multiple,
        "entry_mode": entry_mode,
        "direction_mode": direction_mode,
        "use_tp_exit": use_tp_exit,
        "use_time_exit": use_time_exit,
        "apply_spread": APPLY_SPREAD,
        "spread_dollars": SPREAD_DOLLARS,
        "stop_mode": STOP_MODE,
        "atr_period": ATR_PERIOD,
        "atr_init_mult": ATR_INIT_MULT,
        "atr_trail_mult": ATR_TRAIL_MULT,
        "chandelier_lookback": CHANDELIER_LOOKBACK,
    }
    combined_summary.update(summary)

    rows = [{"metric": k, "value": v} for k, v in combined_summary.items()]
    summary_df = pd.DataFrame(rows)

    summary_file = summary_dir / "single_scenarios_summary.csv"
    if summary_file.exists():
        existing = pd.read_csv(summary_file)
        combined = pd.concat([existing, summary_df], ignore_index=True)
        combined.to_csv(summary_file, index=False)
    else:
        summary_df.to_csv(summary_file, index=False)

    print(f"[INFO] Updated summary file: {summary_file}")

    # 7) Plots
    if PLOT_EQUITY:
        plot_equity_and_drawdown(trades_bt, scenario_name, equity_dir)

    if PLOT_HEATMAP:
        plot_time_of_day_heatmap(trades_bt, scenario_name, heatmap_dir)

    if PLOT_DISTRIBUTIONS:
        plot_distributions_for_single(trades_bt, scenario_name, distrib_dir)

    print("[INFO] Single scenario run complete.")


if __name__ == "__main__":
    main()
