#!/usr/bin/env python
"""
Build param-pair heatmaps over the global leaderboard.

Design:
- Load global_grid_leaderboard_ALL.csv.
- Normalise boolean-ish columns and filter to the desired subset:
    * By default: keep ONLY spread_on (apply_spread=True or spread_tag_folder='spread_on').
    * Direction and exit_tag are NOT filtered by default (ALL included), but can be configured.
- Use a set of PARAM_COLS as axes (T_seconds, K_unique, hold_minutes, sl_distance, tp_R_multiple,
  atr_period, atr_init_mult, atr_trail_mult, chandelier_lookback).
- For EVERY UNIQUE unordered pair of PARAM_COLS, build multiple heatmaps where colour =
  one metric from HEATMAP_METRICS:
    * Positive/“good” metrics: percent_return, avg_R, profit_factor, win_rate, tp_rate,
      final_equity, total_R, gross_profit_cash, avg_win_R, median_R.
    * Risk/“bad” metrics: max_drawdown_R, max_drawdown_cash, sl_rate, gross_loss_cash,
      longest_loss_streak, std_R.
- Colour scheme:
    * Green = better, red = worse.
    * For metrics where HIGH is good → green at high values (RdYlGn).
    * For metrics where LOW is good → green at low values (RdYlGn_r).

Output:
    <BASE_OUT_DIR>/<spread_tag_summary>/heatmap_<metric>__y_<param_y>__x_<param_x>.png
"""

import os
import sys
from typing import List
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# CONFIG
# =============================================================================

# Path to the global leaderboard
INPUT_CSV = (
    r"C:\Users\anish\PycharmProjects\ClusterOrderflow-Backtest"
    r"\output\reports\summaries\global_grid_leaderboard_ALL_001.csv"
)

# Base output folder for heatmaps
BASE_OUT_DIR = (
    r"C:\Users\anish\PycharmProjects\ClusterOrderflow-Backtest"
    r"\output\figures\param_pair_heatmaps_all"
)

# ----- Filtering knobs -----
# If you want to restrict to one direction or exit tag, set these; otherwise keep as None
DIRECTION_MODE_FILTER: str | None = None      # e.g. "inverse", "directional", or None for ALL
EXIT_TAG_FILTER: str | None = None            # e.g. "tp1_time0", "tp0_time1", or None for ALL

# Spread filter: by default we keep ONLY spread_on data
SPREAD_TAG_FILTER: str | None = "spread_on"   # "spread_on", "spread_off", "none", or None for ALL

# If True, we additionally enforce apply_spread == True
ENFORCE_APPLY_SPREAD_TRUE: bool = True

# ----- Axes: param columns (these form the X/Y grid) -----
PARAM_COLS: List[str] = [
    "T_seconds",
    "K_unique",
    "hold_minutes",
    "sl_distance",
    "tp_R_multiple",
    "atr_period",
    "atr_init_mult",
    "atr_trail_mult",
    "chandelier_lookback",
]

# ----- Colour metrics: what the heatmap shows -----
POSITIVE_METRICS: List[str] = [
    "percent_return",
    "avg_R",
    "profit_factor",
    "win_rate",
    "tp_rate",
    "final_equity",
    "total_R",
    "gross_profit_cash",
    "avg_win_R",
    "median_R",
]

RISK_METRICS: List[str] = [
    "max_drawdown_R",
    "max_drawdown_cash",
    "sl_rate",
    "gross_loss_cash",
    "longest_loss_streak",
    "std_R",
]

HEATMAP_METRICS: List[str] = POSITIVE_METRICS + RISK_METRICS

# For some risk metrics, LOWER is better (we want those to be green at low values)
RISK_LOW_GOOD = {
    "sl_rate",
    "longest_loss_streak",
    "std_R",
}
# The rest in RISK_METRICS are treated as HIGH = good for colour scale purposes
RISK_HIGH_GOOD = set(RISK_METRICS) - RISK_LOW_GOOD

# Only generate heatmaps when there are at least this many grouped cells
MIN_GROUP_CELLS = 3


# =============================================================================
# UTIL: ensure directory
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =============================================================================
# LOAD + NORMALISE
# =============================================================================

def normalise_bool_column(df: pd.DataFrame, col: str) -> None:
    """Map various truthy/falsey representations into real bool dtype, in-place."""
    if col not in df.columns:
        return

    original = df[col]
    mapping = {
        "True": True,
        "true": True,
        "1": True,
        1: True,
        True: True,
        "False": False,
        "false": False,
        "0": False,
        0: False,
        False: False,
    }
    mapped = original.map(mapping)
    mapped = mapped.where(~mapped.isna(), original)

    try:
        df[col] = mapped.astype("bool")
    except Exception:
        df[col] = mapped

    print(f"[DEBUG] Normalised boolean column '{col}', unique values: {df[col].unique().tolist()}")


def load_and_filter() -> tuple[pd.DataFrame, str]:
    print(f"[INFO] Loading global leaderboard ALL from: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Input CSV not found: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Loaded {len(df):,} rows")

    # Normalise boolean-like columns if present
    for bcol in ["use_tp_exit", "use_time_exit", "apply_spread"]:
        if bcol in df.columns:
            normalise_bool_column(df, bcol)

    # Optional direction filter
    if DIRECTION_MODE_FILTER and "direction_mode_folder" in df.columns:
        before = len(df)
        df = df[df["direction_mode_folder"] == DIRECTION_MODE_FILTER]
        print(
            f"[INFO] After direction_mode_folder filter ({DIRECTION_MODE_FILTER}): "
            f"{len(df):,} / {before:,} rows"
        )
    else:
        print("[INFO] No direction_mode_folder filter applied (ALL directions kept).")

    # Optional exit tag filter
    if EXIT_TAG_FILTER and "exit_tag_folder" in df.columns:
        before = len(df)
        df = df[df["exit_tag_folder"] == EXIT_TAG_FILTER]
        print(
            f"[INFO] After exit_tag_folder filter ({EXIT_TAG_FILTER}): "
            f"{len(df):,} / {before:,} rows"
        )
    else:
        print("[INFO] No exit_tag_folder filter applied (ALL exit tags kept).")

    # Spread-tag filter (spread_on / spread_off / none)
    spread_tag_summary = "ALLspread"
    if SPREAD_TAG_FILTER and "spread_tag_folder" in df.columns:
        before = len(df)
        df = df[df["spread_tag_folder"] == SPREAD_TAG_FILTER]
        spread_tag_summary = SPREAD_TAG_FILTER
        print(
            f"[INFO] After spread_tag_folder filter ({SPREAD_TAG_FILTER}): "
            f"{len(df):,} / {before:,} rows"
        )
    else:
        print("[INFO] No spread_tag_folder filter applied (ALL spread_tag_folder kept).")

    # Enforce apply_spread == True if requested
    if ENFORCE_APPLY_SPREAD_TRUE and "apply_spread" in df.columns:
        before = len(df)
        df = df[df["apply_spread"] == True]  # noqa: E712
        print(
            f"[INFO] After apply_spread == True filter: "
            f"{len(df):,} / {before:,} rows"
        )

    print(f"[INFO] Final filtered row count: {len(df):,}")

    if len(df) == 0:
        print("[WARN] No rows left after filtering; exiting.")
        sys.exit(0)

    return df, spread_tag_summary


# =============================================================================
# HEATMAP BUILDING
# =============================================================================

def is_valid_numeric_column(df: pd.DataFrame, col: str) -> bool:
    """Check if column exists, is numeric, and has more than 1 unique non-NaN value."""
    if col not in df.columns:
        return False
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    non_na = df[col].dropna()
    return non_na.nunique() > 1


def get_cmap_for_metric(metric_col: str) -> str:
    """
    Decide which colormap to use based on metric semantics.

    - Default: 'RdYlGn' → low = red, high = green.
    - For metrics where LOWER is better (RISK_LOW_GOOD): 'RdYlGn_r' → low = green, high = red.
    """
    if metric_col in RISK_LOW_GOOD:
        return "RdYlGn_r"
    return "RdYlGn"


def build_heatmap(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    metric_col: str,
    out_path: str,
) -> bool:
    """
    Group df by (y_col, x_col) and compute mean(metric_col),
    then save a seaborn heatmap to out_path.
    Returns True if a heatmap was created, False otherwise.
    """
    # Group & pivot
    grouped = (
        df[[y_col, x_col, metric_col]]
        .dropna(subset=[y_col, x_col, metric_col])
        .groupby([y_col, x_col])[metric_col]
        .mean()
    )

    if grouped.empty or grouped.index.nlevels != 2:
        return False

    pivot = grouped.unstack(x_col)

    if pivot.size < MIN_GROUP_CELLS:
        return False

    # Sort axes for nicer structure
    pivot = pivot.sort_index(axis=0)  # sort Y
    pivot = pivot.sort_index(axis=1)  # sort X

    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return False

    cmap = get_cmap_for_metric(metric_col)

    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(
        pivot,
        annot=False,
        cmap=cmap,
        cbar=True,
        linewidths=0.1,
        linecolor="grey",
    )

    # Label the colour bar instead of title
    cbar = ax.collections[0].colorbar
    cbar.set_label(metric_col, fontsize=12)

    # Optional: cleaner title (parameter-only)
    plt.title(f"{y_col} (Y) vs {x_col} (X)", fontsize=12)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[INFO] Saved heatmap: {out_path}")
    return True


def main() -> None:
    start_time = time.time()
    df, spread_tag_summary = load_and_filter()

    # Determine valid param axes and metrics based on the actual data
    valid_params = [c for c in PARAM_COLS if is_valid_numeric_column(df, c)]
    print(f"[INFO] Valid PARAM_COLS (axes) after checks: {valid_params}")

    valid_metrics = [m for m in HEATMAP_METRICS if is_valid_numeric_column(df, m)]
    print(f"[INFO] Valid HEATMAP_METRICS (colour) after checks: {valid_metrics}")

    if len(valid_params) < 2:
        print("[WARN] Fewer than 2 valid parameter columns; no pairwise heatmaps possible.")
        sys.exit(0)

    if not valid_metrics:
        print("[WARN] No valid metric columns for heatmaps; exiting.")
        sys.exit(0)

    # Build output folder name describing the filtering
    dir_tag = []
    if DIRECTION_MODE_FILTER:
        dir_tag.append(DIRECTION_MODE_FILTER)
    else:
        dir_tag.append("ALLdir")

    dir_tag.append(spread_tag_summary)

    if EXIT_TAG_FILTER:
        dir_tag.append(EXIT_TAG_FILTER)
    else:
        dir_tag.append("ALLexit")

    out_dir = os.path.join(BASE_OUT_DIR, "_".join(dir_tag))
    ensure_dir(out_dir)
    print(f"[INFO] Saving param-pair heatmaps to: {out_dir}")

    # Iterate over all unique unordered param pairs (no repetition, no self-pair)
    total_plots = 0
    for i in range(len(valid_params)):
        for j in range(i + 1, len(valid_params)):
            y_col = valid_params[i]
            x_col = valid_params[j]

            for metric in valid_metrics:
                filename = f"heatmap_{metric}__y_{y_col}__x_{x_col}.png"
                out_path = os.path.join(out_dir, filename)

                created = build_heatmap(df, y_col, x_col, metric, out_path)
                if created:
                    total_plots += 1

    print(f"[DONE] Total heatmaps created: {total_plots}")

    elapsed = time.time() - start_time
    print(f"[TIMER] Elapsed time: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    main()
