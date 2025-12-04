#!/usr/bin/env python

"""
run_three_scenarios_compare.py
------------------------------

Run THREE concrete backtest scenarios end-to-end and compare:

- Each scenario is run independently (its own trades, equity, drawdown).
- Then we build a COMBINED account equity/drawdown curve assuming
  all 3 are running on the SAME account at the same time.

Outputs:
- Two figures:
    1) Equity curves (3 scenarios + combined account)
    2) Drawdown curves (3 scenarios + combined account)
- CSVs:
    * Per-scenario + combined metrics for this run
    * Optional per-scenario trades and combined trades/equity

You configure the 3 scenarios via the USER CONFIG blocks below.

NOTE on caps:
- Each scenario has `max_open_positions`, passed into BacktestConfig.
- With cfg.max_open_positions = N, that engine will never have more than N
  concurrent open trades (cap enforced in backtest_time_exit).
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datetime import datetime

from config import (
    ensure_directories,
    EQUITY_FIG_DIR,
    SUMMARY_REPORTS_DIR,   # NEW: for metrics/trades CSV outputs
)
from load_data import load_trades, load_candles
from clustering import find_cluster_signals
from mapping import map_signals_to_candles, prepare_candles_for_mapping
from backtest import backtest_time_exit, BacktestConfig


# =========================================
# USER CONFIG — THREE SCENARIOS TO COMPARE
# =========================================

@dataclass
class ScenarioSpec:
    label: str

    # Cluster params
    T_seconds: int
    K_unique: int

    # Exit / risk params
    hold_minutes: int
    sl_distance: float

    # Entry mode
    entry_mode: str  # "prop", "oanda_open", "oanda_close"

    # Direction mode: "directional" or "inverse"
    direction_mode: str

    # TP in R multiples (None to disable)
    tp_R_multiple: Optional[float]

    # Exit toggles
    use_tp_exit: bool
    use_time_exit: bool

    # Stop mode
    stop_mode: str  # "fixed", "atr_static", "atr_trailing", "chandelier"

    # ATR / chandelier params
    atr_period: Optional[int]
    atr_init_mult: Optional[float]
    atr_trail_mult: Optional[float]
    chandelier_lookback: Optional[int]

    # Spread
    apply_spread: bool
    spread_dollars: float

    # NEW: max concurrent open positions for THIS engine
    max_open_positions: Optional[int] = None  # e.g. 1, 2, 3. None = unlimited


# ===== GLOBAL CAP (optional convenience) =====
# If you want all three engines to share the same cap per run,
# set this once and use it in the scenarios (or override per-scenario).
GLOBAL_MAX_OPEN_POSITIONS = 3  # <-- change to 1, 2, 3 between runs


# ===== SCENARIO 1 =====
SCENARIO_1 = ScenarioSpec(
    label="S1 - ATR_Trailing",  # label on the plot

    T_seconds=29,
    K_unique=3,
    hold_minutes=5,
    sl_distance=2.0,

    entry_mode="prop",        # "prop", "oanda_open", "oanda_close"
    direction_mode="inverse", # "directional" or "inverse"

    tp_R_multiple=1.0,        # or None
    use_tp_exit=True,
    use_time_exit=False,

    stop_mode="atr_trailing",   # "fixed", "atr_static", "atr_trailing", "chandelier"
    atr_period=5,
    atr_init_mult=3.0,
    atr_trail_mult=2.0,
    chandelier_lookback=None,

    apply_spread=True,
    spread_dollars=0.2,

    max_open_positions=GLOBAL_MAX_OPEN_POSITIONS,
)

# ===== SCENARIO 2 =====
SCENARIO_2 = ScenarioSpec(
    label="S2 - ATR_Static",

    T_seconds=29,
    K_unique=3,
    hold_minutes=5,
    sl_distance=2.0,

    entry_mode="prop",
    direction_mode="inverse",

    tp_R_multiple=1.0,
    use_tp_exit=True,
    use_time_exit=False,

    stop_mode="atr_static",
    atr_period=5,
    atr_init_mult=4.0,
    atr_trail_mult=None,
    chandelier_lookback=None,

    apply_spread=True,
    spread_dollars=0.2,

    max_open_positions=GLOBAL_MAX_OPEN_POSITIONS,
)

# ===== SCENARIO 3 =====
SCENARIO_3 = ScenarioSpec(
    label="S3 - Chandelier",

    T_seconds=30,
    K_unique=3,
    hold_minutes=15,
    sl_distance=2.0,

    entry_mode="prop",
    direction_mode="inverse",

    tp_R_multiple=1.0,
    use_tp_exit=True,
    use_time_exit=False,

    stop_mode="chandelier",
    atr_period=5,
    atr_init_mult=3.0,
    atr_trail_mult=2.0,
    chandelier_lookback=30,

    apply_spread=True,
    spread_dollars=0.2,

    max_open_positions=GLOBAL_MAX_OPEN_POSITIONS,
)


# =========================================
# HELPERS (STREAKS, ATR / CHANDELIER, COMBINED P&L)
# =========================================

def _longest_streak(mask: pd.Series) -> int:
    """
    Given a boolean Series (e.g. pnl_R > 0), return the longest
    consecutive True streak length.
    """
    max_streak = 0
    current = 0
    for val in mask:
        if bool(val):
            current += 1
            if current > max_streak:
                max_streak = current
        else:
            current = 0
    return max_streak


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

    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(candles.columns):
        missing = required_cols - set(candles.columns)
        raise ValueError(
            f"Candles are missing required OHLC columns for ATR/chandelier calc: {missing}"
        )

    # ATR for ATR-based and chandelier modes
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

    # Chandelier
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


def build_scenario_name(spec: ScenarioSpec) -> str:
    """
    Build a scenario_name string similar to run_grid_scenarios_stops.py
    so you can match with grid rows if needed.
    """
    # direction / entry / stop tags
    entry_mode = spec.entry_mode
    direction_mode = spec.direction_mode

    # stop_tag like in run_single_scenario
    stop_tag = spec.stop_mode
    if spec.atr_period is not None and spec.atr_init_mult is not None:
        stop_tag += f"_ATR{int(spec.atr_period)}_I{float(spec.atr_init_mult)}"
    if spec.atr_trail_mult is not None:
        stop_tag += f"_T{float(spec.atr_trail_mult)}"
    if spec.chandelier_lookback is not None and spec.stop_mode == "chandelier":
        stop_tag += f"_CH{int(spec.chandelier_lookback)}"

    spread_tag = "spread_on" if spec.apply_spread else "spread_off"
    exit_tag = f"tp{int(spec.use_tp_exit)}_time{int(spec.use_time_exit)}"

    scenario_name = (
        f"T{spec.T_seconds}_K{spec.K_unique}_H{spec.hold_minutes}_"
        f"SL{spec.sl_distance}_TP{spec.tp_R_multiple}_"
        f"{entry_mode}_{direction_mode}_{stop_tag}_"
        f"{spread_tag}_{exit_tag}"
    )
    return scenario_name


def run_single_scenario_pipeline(
    spec: ScenarioSpec,
    trades_raw: pd.DataFrame,
    candles_raw: pd.DataFrame,
) -> Tuple[str, pd.DataFrame]:
    """
    Run the full pipeline for ONE scenario:
      - cluster signals
      - map to candles
      - backtest
      - return scenario_name and trades_bt (with equity / drawdown etc.)
    """

    scenario_name = build_scenario_name(spec)
    print(f"\n[INFO] === Running scenario: {scenario_name} ({spec.label}) ===")

    # Entry price source as in run_single_scenario.py
    if spec.entry_mode == "oanda_open":
        entry_price_source = "open"
    elif spec.entry_mode == "oanda_close":
        entry_price_source = "close"
    else:  # "prop"
        entry_price_source = "open"

    # 1) Cluster signals
    print(f"[INFO] Building signals (T={spec.T_seconds}s, K={spec.K_unique})...")
    signals = find_cluster_signals(
        trades_raw,
        T_seconds=spec.T_seconds,
        K_unique=spec.K_unique,
    )
    print(f"[INFO] Signals found: {len(signals)}")
    if signals.empty:
        print("[WARN] No signals found for this configuration.")
        return scenario_name, pd.DataFrame()

    # 2) Map signals to candles
    print(f"[INFO] Mapping signals to candles (entry_price_source={entry_price_source})...")
    signals_with_entries, qc_summary = map_signals_to_candles(
        signals,
        candles_raw,
        entry_price_source=entry_price_source,
    )
    print(f"[INFO] Mapped signals: {len(signals_with_entries)}")
    if qc_summary is not None and not qc_summary.empty:
        print("\n=== QC SUMMARY ===")
        print(qc_summary.to_string(index=False))

    if signals_with_entries.empty:
        print("[WARN] No mapped signals after QC.")
        return scenario_name, pd.DataFrame()

    # 3) Backtest with full exit + stop logic (including cap)
    cfg = BacktestConfig(
        hold_minutes=spec.hold_minutes,
        sl_distance=spec.sl_distance,
        entry_mode=spec.entry_mode,
        direction_mode=spec.direction_mode,
        tp_R_multiple=spec.tp_R_multiple,
        use_tp_exit=spec.use_tp_exit,
        use_time_exit=spec.use_time_exit,
        apply_spread=spec.apply_spread,
        spread_dollars=spec.spread_dollars,
        stop_mode=spec.stop_mode,
        atr_period=spec.atr_period,
        atr_init_mult=spec.atr_init_mult,
        atr_trail_mult=spec.atr_trail_mult,
        chandelier_lookback=spec.chandelier_lookback,
        max_open_positions=spec.max_open_positions,  # NEW
    )

    print(
        f"[INFO] Backtesting with: "
        f"hold={spec.hold_minutes}m, SL={spec.sl_distance}, TP_R={spec.tp_R_multiple}, "
        f"entry_mode={spec.entry_mode}, direction_mode={spec.direction_mode}, "
        f"use_tp_exit={spec.use_tp_exit}, use_time_exit={spec.use_time_exit}, "
        f"spread={spec.apply_spread} ({spec.spread_dollars}), "
        f"stop_mode={spec.stop_mode}, ATR_P={spec.atr_period}, "
        f"ATR_I={spec.atr_init_mult}, ATR_T={spec.atr_trail_mult}, "
        f"CH_LB={spec.chandelier_lookback}, "
        f"max_open_positions={spec.max_open_positions}"
    )

    candles_bt = ensure_atr_and_chandelier_columns(candles_raw.copy(), cfg)

    trades_bt, summary = backtest_time_exit(signals_with_entries, candles_bt, cfg)
    print(f"[INFO] Backtest produced {len(trades_bt)} trades.")
    if summary:
        print("[INFO] Summary (raw from backtest):")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    return scenario_name, trades_bt


def build_combined_equity_drawdown(
    labeled_trades: List[Tuple[str, str, pd.DataFrame]],
) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Given list of (label, scenario_name, trades_df) for each scenario,
    build a combined equity/drawdown path assuming all trades are running
    on the SAME account.

    We:
    - concatenate all trades
    - sort by time (entry_time or exit_time if available)
    - use pnl_cash to build a cumulative PnL over a single base equity
    - derive equity + drawdown.

    Returns:
      combined_eq   : DataFrame with ['time', 'equity_combined', 'drawdown_combined']
      base_equity   : starting equity used
      combined_trades: the concatenated trades DataFrame (with original columns)
    """

    # Filter out empty dfs
    non_empty = [(lbl, name, df) for (lbl, name, df) in labeled_trades if not df.empty]
    if not non_empty:
        print("[WARN] All scenarios are empty. Cannot build combined equity.")
        return pd.DataFrame(), 0.0, pd.DataFrame()

    # Assume starting equity from the first non-empty scenario
    first_df = non_empty[0][2]
    if "equity" in first_df.columns:
        base_equity = float(first_df["equity"].iloc[0])
    else:
        base_equity = 100000.0  # fallback
        print(f"[WARN] 'equity' column not found; using base_equity={base_equity}")

    # Concatenate all trades
    combined = pd.concat(
        [df.assign(_scenario_label=lbl, _scenario_name=name) for (lbl, name, df) in non_empty],
        ignore_index=True,
    )

    # Choose time column
    time_col = "exit_time" if "exit_time" in combined.columns else "entry_time"
    if time_col not in combined.columns:
        raise ValueError("Neither 'exit_time' nor 'entry_time' present in combined trades.")

    if "pnl_cash" not in combined.columns:
        raise ValueError("'pnl_cash' column is required to build combined equity.")

    combined = combined.sort_values(time_col).copy()

    combined["cum_pnl_cash"] = combined["pnl_cash"].cumsum()
    combined["equity_combined"] = base_equity + combined["cum_pnl_cash"]

    # Drawdown from running max
    combined["running_max_equity"] = combined["equity_combined"].cummax()
    combined["drawdown_combined"] = combined["equity_combined"] - combined["running_max_equity"]

    combined_eq = combined[[time_col, "equity_combined", "drawdown_combined"]].rename(
        columns={time_col: "time"}
    )

    return combined_eq, base_equity, combined


def compute_scenario_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute per-scenario metrics including:
    - n_trades
    - win_rate
    - longest_win_streak / longest_loss_streak
    - max/min risk_per_trade
    - largest profit/loss
    - final_equity
    - percent_return
    - max_drawdown_cash
    - NEW:
        * max_drawdown_R
        * dd_per_return
        * return_dd_ratio
    """

    if df.empty:
        return {
            "n_trades": 0,
            "win_rate": np.nan,
            "longest_win_streak": 0,
            "longest_loss_streak": 0,
            "max_risk_per_trade": np.nan,
            "min_risk_per_trade": np.nan,
            "largest_profit_cash": np.nan,
            "largest_loss_cash": np.nan,
            "final_equity": np.nan,
            "percent_return": np.nan,
            "max_drawdown_cash": np.nan,
            "max_drawdown_R": np.nan,
            "dd_per_return": np.nan,
            "return_dd_ratio": np.nan,
        }

    df = df.sort_values("entry_time").reset_index(drop=True)

    n_trades = len(df)

    pnl_R = df["pnl_R"]
    pnl_cash = df["pnl_cash"]

    # Win/loss masks
    win_mask = pnl_R > 0
    loss_mask = pnl_R < 0

    win_rate = float(win_mask.mean())
    longest_win_streak = _longest_streak(win_mask)
    longest_loss_streak = _longest_streak(loss_mask)

    # Risk per trade
    if "risk_per_trade" in df.columns:
        max_risk = float(df["risk_per_trade"].max())
        min_risk = float(df["risk_per_trade"].min())
        avg_risk = float(df["risk_per_trade"].abs().mean())
    else:
        max_risk = min_risk = avg_risk = np.nan

    largest_profit_cash = float(pnl_cash.max())
    largest_loss_cash = float(pnl_cash.min())

    # Equity-based metrics
    start_equity = float(df["equity"].iloc[0])
    final_equity = float(df["equity"].iloc[-1])
    percent_return = (final_equity / start_equity - 1.0) * 100.0

    max_drawdown_cash = float(df["drawdown"].min())  # negative

    # ---- NEW METRICS ----

    # Max drawdown in R
    if avg_risk > 0:
        max_drawdown_R = max_drawdown_cash / avg_risk
    else:
        max_drawdown_R = np.nan

    # We will use absolute values for the ratios so they are positive & intuitive
    abs_dd = abs(max_drawdown_cash)
    abs_ret = abs(percent_return)

    # DD per unit of RETURN  (e.g. "0.08" means max DD is 8% of total return)
    if abs_ret > 0:
        dd_per_return = abs_dd / abs_ret
    else:
        dd_per_return = np.nan

    # RETURN per unit of DD  (e.g. "12.5" means 12.5% return per 1 unit of DD)
    if abs_dd > 0:
        return_dd_ratio = abs_ret / abs_dd
    else:
        return_dd_ratio = np.nan

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "longest_win_streak": longest_win_streak,
        "longest_loss_streak": longest_loss_streak,
        "max_risk_per_trade": max_risk,
        "min_risk_per_trade": min_risk,
        "largest_profit_cash": largest_profit_cash,
        "largest_loss_cash": largest_loss_cash,
        "final_equity": final_equity,
        "percent_return": percent_return,
        "max_drawdown_cash": max_drawdown_cash,
        "max_drawdown_R": max_drawdown_R,
        "dd_per_return": dd_per_return,
        "return_dd_ratio": return_dd_ratio,
    }


# =========================================
# PLOTTING
# =========================================

def plot_equity_curves(
    labeled_trades: List[Tuple[str, str, pd.DataFrame]],
    combined_eq: pd.DataFrame,
    out_dir: Path,
    run_tag: str | None = None,
) -> None:
    """
    Plot:
      - Individual equity curves (per scenario, colour-coded)
      - Combined account equity (bold dashed line)
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Base colour palette for scenarios
    base_colors = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#9467bd",  # purple (if ever needed)
        "#ff7f0e",  # orange
    ]

    # Assign colours in order of scenarios
    scenario_labels = [label for (label, _, _) in labeled_trades]
    label_to_color: Dict[str, str] = {}
    for idx, label in enumerate(scenario_labels):
        label_to_color[label] = base_colors[idx % len(base_colors)]

    combined_color = "#333333"  # dark grey for combined equity

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot individual scenario equity curves
    for label, scenario_name, df in labeled_trades:
        if df.empty or "equity" not in df.columns or "entry_time" not in df.columns:
            print(f"[WARN] Skipping equity plot for {label} (missing data).")
            continue

        df_sorted = df.sort_values("entry_time")
        color = label_to_color.get(label, "#000000")

        # Light shading from baseline equity to curve
        baseline = float(df_sorted["equity"].iloc[0])
        ax.fill_between(
            df_sorted["entry_time"],
            baseline,
            df_sorted["equity"],
            alpha=0.08,
            color=color,
        )

        # Line for equity
        ax.plot(
            df_sorted["entry_time"],
            df_sorted["equity"],
            linewidth=1.8,
            color=color,
            label=label,
        )

    # Plot combined equity
    if not combined_eq.empty:
        ax.plot(
            combined_eq["time"],
            combined_eq["equity_combined"],
            linewidth=2.4,
            linestyle="--",
            color=combined_color,
            label="Combined account equity",
        )

    ax.set_title("Equity Curves — 3 Scenarios + Combined Account")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    suffix = f"_{run_tag}" if run_tag else ""
    out_path = out_dir / f"equity_3scenarios_plus_combined{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved equity comparison figure to: {out_path}")


def plot_drawdown_curves(
    labeled_trades: List[Tuple[str, str, pd.DataFrame]],
    combined_eq: pd.DataFrame,
    out_dir: Path,
    run_tag: str | None = None,
) -> None:
    """
    Plot:
      - Individual drawdown curves (per scenario, colour-coded)
      - Combined account drawdown (bold dashed line)
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Base colour palette for scenarios (same mapping as equity)
    base_colors = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#9467bd",
        "#ff7f0e",
    ]

    scenario_labels = [label for (label, _, _) in labeled_trades]
    label_to_color: Dict[str, str] = {}
    for idx, label in enumerate(scenario_labels):
        label_to_color[label] = base_colors[idx % len(base_colors)]

    combined_color = "#333333"  # dark grey for combined drawdown

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, scenario_name, df in labeled_trades:
        if df.empty or "drawdown" not in df.columns or "entry_time" not in df.columns:
            print(f"[WARN] Skipping drawdown plot for {label} (missing data).")
            continue

        df_sorted = df.sort_values("entry_time")
        color = label_to_color.get(label, "#000000")

        # Shading for drawdown area below 0
        ax.fill_between(
            df_sorted["entry_time"],
            df_sorted["drawdown"],
            0,
            where=(df_sorted["drawdown"] < 0),
            alpha=0.10,
            color=color,
        )

        # Line for drawdown
        ax.plot(
            df_sorted["entry_time"],
            df_sorted["drawdown"],
            linewidth=1.8,
            color=color,
            label=label,
        )

    # Combined drawdown
    if not combined_eq.empty:
        ax.fill_between(
            combined_eq["time"],
            combined_eq["drawdown_combined"],
            0,
            where=(combined_eq["drawdown_combined"] < 0),
            alpha=0.15,
            color=combined_color,
        )
        ax.plot(
            combined_eq["time"],
            combined_eq["drawdown_combined"],
            linewidth=2.4,
            linestyle="--",
            color=combined_color,
            label="Combined account drawdown",
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Drawdown Curves — 3 Scenarios + Combined Account")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    suffix = f"_{run_tag}" if run_tag else ""
    out_path = out_dir / f"drawdown_3scenarios_plus_combined{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved drawdown comparison figure to: {out_path}")


# =========================================
# CSV OUTPUT HELPERS
# =========================================

def save_csv_outputs(
    run_tag: str,
    scenarios: List[ScenarioSpec],
    labeled_trades: List[Tuple[str, str, pd.DataFrame]],
    combined_eq: pd.DataFrame,
    combined_trades: pd.DataFrame,
    scenario_metric_rows: List[Dict[str, float]],
    combined_metric_row: Dict[str, float],
) -> None:
    """
    Save:
      - Per-scenario + combined metrics to a single CSV
      - Per-scenario trades to separate CSVs
      - Combined trades to a CSV
      - Combined equity curve to a CSV

    All outputs go under:
      SUMMARY_REPORTS_DIR / "multi_scenario_3way" / <run_tag> / ...
    so nothing is overwritten across runs (e.g., when you change caps).
    """
    base_dir = SUMMARY_REPORTS_DIR / "multi_scenario_3way" / run_tag
    base_dir.mkdir(parents=True, exist_ok=True)

    # ---- Metrics CSV (per scenario + combined) ----
    metrics_df = pd.DataFrame(scenario_metric_rows + [combined_metric_row])
    metrics_path = base_dir / f"three_scenarios_metrics_{run_tag}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[INFO] Saved metrics CSV to: {metrics_path}")

    # ---- Per-scenario trades CSVs ----
    trades_dir = base_dir / "trades_by_scenario"
    trades_dir.mkdir(parents=True, exist_ok=True)

    for (label, scenario_name, df) in labeled_trades:
        if df.empty:
            print(f"[INFO] Skipping trades CSV for {label} (no trades).")
            continue

        safe_name = scenario_name.replace(" ", "_")
        trades_path = trades_dir / f"trades_{safe_name}_{run_tag}.csv"
        df.to_csv(trades_path, index=False)
        print(f"[INFO] Saved trades CSV for {label} to: {trades_path}")

    # ---- Combined trades CSV ----
    if not combined_trades.empty:
        combined_trades_path = base_dir / f"combined_trades_3scenarios_{run_tag}.csv"
        combined_trades.to_csv(combined_trades_path, index=False)
        print(f"[INFO] Saved combined trades CSV to: {combined_trades_path}")

    # ---- Combined equity curve CSV ----
    if not combined_eq.empty:
        combined_eq_path = base_dir / f"combined_equity_curve_{run_tag}.csv"
        combined_eq.to_csv(combined_eq_path, index=False)
        print(f"[INFO] Saved combined equity curve CSV to: {combined_eq_path}")


# =========================================
# MAIN
# =========================================

def main():
    ensure_directories()

    # Unique tag for this 3-scenario comparison run (YYYYMMDD_HHMMSS)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Directory for multi-scenario comparison figures
    multi_dir = EQUITY_FIG_DIR / "multi_scenario_3way"
    multi_dir.mkdir(parents=True, exist_ok=True)

    # Load base data ONCE
    print("[INFO] Loading trades & candles...")
    trades_raw = load_trades()
    candles_raw = load_candles()
    candles_raw = prepare_candles_for_mapping(candles_raw)

    print(f"[INFO] Trades rows: {len(trades_raw)}, Candles rows: {len(candles_raw)}")

    # Run 3 scenarios
    scenarios = [SCENARIO_1, SCENARIO_2, SCENARIO_3]
    labeled_trades: List[Tuple[str, str, pd.DataFrame]] = []

    for spec in scenarios:
        scenario_name, trades_bt = run_single_scenario_pipeline(
            spec,
            trades_raw=trades_raw,
            candles_raw=candles_raw,
        )
        labeled_trades.append((spec.label, scenario_name, trades_bt))

    # --------- Per-scenario metrics (console + for CSV) ---------
    print("\n==============================")
    print(" PER-SCENARIO METRICS (3-way)")
    print("==============================")

    scenario_metric_rows: List[Dict[str, float]] = []
    spec_by_label = {s.label: s for s in scenarios}

    for label, scenario_name, df in labeled_trades:
        spec = spec_by_label[label]
        metrics = compute_scenario_metrics(df)

        # Console print (similar to single_scenario style)
        print(f"\n[{label}] {scenario_name}")
        print(f"  Trades               : {metrics['n_trades']}")
        print(f"  Win rate             : {metrics['win_rate'] * 100:.2f}%"
              if not np.isnan(metrics['win_rate']) else "  Win rate             : N/A")
        print(f"  Longest win streak   : {metrics['longest_win_streak']}")
        print(f"  Longest loss streak  : {metrics['longest_loss_streak']}")
        print(f"  Max risk per trade   : {metrics['max_risk_per_trade']:.2f}"
              if not np.isnan(metrics['max_risk_per_trade']) else "  Max risk per trade   : N/A")
        print(f"  Min risk per trade   : {metrics['min_risk_per_trade']:.2f}"
              if not np.isnan(metrics['min_risk_per_trade']) else "  Min risk per trade   : N/A")
        print(f"  Largest profit (cash): {metrics['largest_profit_cash']:.2f}")
        print(f"  Largest loss (cash)  : {metrics['largest_loss_cash']:.2f}")
        print(f"  Final equity         : {metrics['final_equity']:.2f}"
              if not np.isnan(metrics['final_equity']) else "  Final equity         : N/A")
        print(f"  Percent return       : {metrics['percent_return']:.2f}%"
              if not np.isnan(metrics['percent_return']) else "  Percent return       : N/A")
        print(f"  Max drawdown (cash)  : {metrics['max_drawdown_cash']:.2f}"
              if not np.isnan(metrics['max_drawdown_cash']) else "  Max drawdown (cash)  : N/A")
        print(f"  Max open positions   : {spec.max_open_positions}")

        # Row for CSV (wide format, similar content to single_scenario summary)
        row = {
            "run_tag": run_tag,
            "label": label,
            "scenario_name": scenario_name,
            "is_combined": False,
            "T_seconds": spec.T_seconds,
            "K_unique": spec.K_unique,
            "hold_minutes": spec.hold_minutes,
            "sl_distance": spec.sl_distance,
            "tp_R_multiple": spec.tp_R_multiple,
            "entry_mode": spec.entry_mode,
            "direction_mode": spec.direction_mode,
            "use_tp_exit": spec.use_tp_exit,
            "use_time_exit": spec.use_time_exit,
            "apply_spread": spec.apply_spread,
            "spread_dollars": spec.spread_dollars,
            "stop_mode": spec.stop_mode,
            "atr_period": spec.atr_period,
            "atr_init_mult": spec.atr_init_mult,
            "atr_trail_mult": spec.atr_trail_mult,
            "chandelier_lookback": spec.chandelier_lookback,
            "max_open_positions": spec.max_open_positions,
        }
        row.update(metrics)
        scenario_metric_rows.append(row)

    # Build combined account equity + drawdown
    combined_eq, base_equity, combined_trades = build_combined_equity_drawdown(labeled_trades)

    if combined_eq.empty or combined_trades.empty:
        print("[WARN] Combined equity is empty; nothing to plot or summarise.")
        return

    print(f"\n[INFO] Combined equity built using base_equity={base_equity:.2f}")

    # --------- Combined account metrics ---------
    combined_trades = combined_trades.sort_values(
        "exit_time" if "exit_time" in combined_trades.columns else "entry_time"
    ).reset_index(drop=True)

    n_total_trades = len(combined_trades)
    pnl_R_combined = combined_trades["pnl_R"] if "pnl_R" in combined_trades.columns else pd.Series(
        [np.nan] * n_total_trades
    )
    pnl_cash_combined = combined_trades["pnl_cash"]

    win_mask = pnl_R_combined > 0
    loss_mask = pnl_R_combined < 0

    n_wins = int(win_mask.sum())
    win_rate_combined = float(win_mask.mean()) if n_total_trades > 0 else np.nan

    longest_win_streak_combined = _longest_streak(win_mask)
    longest_loss_streak_combined = _longest_streak(loss_mask)

    max_risk_combined = float(combined_trades["risk_per_trade"].max()) \
        if "risk_per_trade" in combined_trades.columns else np.nan
    min_risk_combined = float(combined_trades["risk_per_trade"].min()) \
        if "risk_per_trade" in combined_trades.columns else np.nan

    largest_profit_combined = float(pnl_cash_combined.max())
    largest_loss_combined = float(pnl_cash_combined.min())

    final_equity_combined = float(combined_eq["equity_combined"].iloc[-1])
    max_dd_combined = float(combined_eq["drawdown_combined"].min())
    percent_return_combined = (final_equity_combined / base_equity - 1.0) * 100.0 if base_equity != 0 else np.nan

    # Average risk per trade for combined (if available)
    if "risk_per_trade" in combined_trades.columns:
        avg_risk_combined = float(combined_trades["risk_per_trade"].abs().mean())
    else:
        avg_risk_combined = np.nan

    # Max drawdown in R for combined
    if avg_risk_combined > 0:
        max_dd_R_combined = max_dd_combined / avg_risk_combined
    else:
        max_dd_R_combined = np.nan

    # Ratios using absolute values
    abs_dd_c = abs(max_dd_combined)
    abs_ret_c = abs(percent_return_combined)

    if abs_ret_c > 0:
        dd_per_return_combined = abs_dd_c / abs_ret_c
    else:
        dd_per_return_combined = np.nan

    if abs_dd_c > 0:
        return_dd_ratio_combined = abs_ret_c / abs_dd_c
    else:
        return_dd_ratio_combined = np.nan


    print("\n===================================")
    print(" COMBINED ACCOUNT SUMMARY (3-way) ")
    print("===================================")
    print(f"  Starting equity       : {base_equity:.2f}")
    print(f"  Final equity          : {final_equity_combined:.2f}")
    print(f"  Percent return        : {percent_return_combined:.2f}%"
          if not np.isnan(percent_return_combined) else "  Percent return        : N/A")
    print(f"  Max drawdown (cash)   : {max_dd_combined:.2f}")
    print(f"  Total trades          : {n_total_trades}")
    print(f"  Winning trades        : {n_wins}")
    print(f"  Win rate              : {win_rate_combined * 100:.2f}%"
          if not np.isnan(win_rate_combined) else "  Win rate              : N/A")
    print(f"  Longest win streak    : {longest_win_streak_combined}")
    print(f"  Longest loss streak   : {longest_loss_streak_combined}")
    print(f"  Max risk per trade    : {max_risk_combined:.2f}"
          if not np.isnan(max_risk_combined) else "  Max risk per trade    : N/A")
    print(f"  Min risk per trade    : {min_risk_combined:.2f}"
          if not np.isnan(min_risk_combined) else "  Min risk per trade    : N/A")
    print(f"  Largest profit (cash) : {largest_profit_combined:.2f}")
    print(f"  Largest loss (cash)   : {largest_loss_combined:.2f}")

    # Row for combined metrics CSV
    combined_metric_row = {
        "run_tag": run_tag,
        "label": "COMBINED",
        "scenario_name": "COMBINED",
        "is_combined": True,
        "T_seconds": np.nan,
        "K_unique": np.nan,
        "hold_minutes": np.nan,
        "sl_distance": np.nan,
        "tp_R_multiple": np.nan,
        "entry_mode": None,
        "direction_mode": None,
        "use_tp_exit": None,
        "use_time_exit": None,
        "apply_spread": None,
        "spread_dollars": None,
        "stop_mode": None,
        "atr_period": None,
        "atr_init_mult": None,
        "atr_trail_mult": None,
        "chandelier_lookback": None,
        "max_open_positions": None,
        "n_trades": n_total_trades,
        "win_rate": win_rate_combined,
        "longest_win_streak": longest_win_streak_combined,
        "longest_loss_streak": longest_loss_streak_combined,
        "max_risk_per_trade": max_risk_combined,
        "min_risk_per_trade": min_risk_combined,
        "largest_profit_cash": largest_profit_combined,
        "largest_loss_cash": largest_loss_combined,
        "final_equity": final_equity_combined,
        "percent_return": percent_return_combined,
        "max_drawdown_cash": max_dd_combined,
        "max_drawdown_R": max_dd_R_combined,
        "dd_per_return": dd_per_return_combined,
        "return_dd_ratio": return_dd_ratio_combined,
    }

    # --------- Save CSV outputs (metrics + trades) ---------
    save_csv_outputs(
        run_tag=run_tag,
        scenarios=scenarios,
        labeled_trades=labeled_trades,
        combined_eq=combined_eq,
        combined_trades=combined_trades,
        scenario_metric_rows=scenario_metric_rows,
        combined_metric_row=combined_metric_row,
    )

    # Plot equity comparison
    plot_equity_curves(labeled_trades, combined_eq, multi_dir, run_tag=run_tag)

    # Plot drawdown comparison
    plot_drawdown_curves(labeled_trades, combined_eq, multi_dir, run_tag=run_tag)

    print("\n[INFO] Three-scenario comparison run complete.")


if __name__ == "__main__":
    main()
