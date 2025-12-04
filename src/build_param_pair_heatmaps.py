"""
build_param_pair_heatmaps.py
----------------------------

Purpose:
    After you have run one or more grid backtests AND
    built the global leaderboard via build_global_leaderboard.py,
    this script:

    1) Loads SUMMARY_REPORTS_DIR / "global_grid_leaderboard.csv".
    2) Filters rows by:
         - DIRECTION_MODE
         - USE_TP_EXIT
         - USE_TIME_EXIT
         - APPLY_SPREAD / SPREAD_DOLLARS (when available)
    3) For the filtered subset, generates ALL pairwise parameter heatmaps:

         Parameters:
             - T_seconds
             - K_unique
             - hold_minutes
             - sl_distance
             - tp_R_multiple

         For each metric in HEATMAP_METRICS, creates heatmaps like:
             metric vs (row_param, col_param)
             e.g. avg_R with rows=K_unique, cols=T_seconds
                  percent_return with rows=hold_minutes, cols=sl_distance
                  etc.

    4) Saves PNGs under:
         SUMMARY_REPORTS_DIR / "param_pair_heatmaps" /
             "<direction>_<spread_tag>_<exit_tag>" /

         Filenames:
             pair_heatmap_<metric>_<row_param>_vs_<col_param>.png

Notes:
    - Uses 'RdYlGn' colormap so that "good" (higher metric) tends towards GREEN
      and "bad" tends towards RED, making clusters of good regions visually obvious.
"""

from pathlib import Path
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    ensure_directories,
    SUMMARY_REPORTS_DIR,
)

# =========================
# USER CONFIG — FILTERS
# =========================

# These MUST match the grid runs you want to analyse.
# (They should be the same settings you used in run_grid_scenarios.py)
DIRECTION_MODE = "inverse"   # "directional" or "inverse"

USE_TP_EXIT = True          # True / False
USE_TIME_EXIT = False        # True / False

# Spread modelling (in XAUUSD dollars)
APPLY_SPREAD = True          # True to include only spread-on runs, False for spread-off
SPREAD_DOLLARS = 0.2         # Used only when APPLY_SPREAD=True and column present


# =========================
# USER CONFIG — HEATMAPS
# =========================

# Metrics to visualise (must exist as columns in the leaderboard)
HEATMAP_METRICS = [
    "avg_R",
    "percent_return",
    "win_rate",
    "max_drawdown_R",
    "final_equity",
    "profit_factor",
    "tp_rate",
    "sl_rate",
    "longest_win_streak",
    "longest_loss_streak",
]

# Parameters to use for row/column axes
PARAM_COLUMNS = [
    "T_seconds",
    "K_unique",
    "hold_minutes",
    "sl_distance",
    "tp_R_multiple",
]


# =========================
# INTERNAL HELPERS
# =========================

def load_filtered_data() -> pd.DataFrame:
    """
    Load global_grid_leaderboard.csv, then filter by:

        - DIRECTION_MODE
        - USE_TP_EXIT
        - USE_TIME_EXIT
        - APPLY_SPREAD / SPREAD_DOLLARS (if available)

    Returns:
        Filtered DataFrame (may be empty).
    """
    ensure_directories()

    base = Path(SUMMARY_REPORTS_DIR)
    global_path = base / "global_grid_leaderboard.csv"

    if not global_path.exists():
        raise FileNotFoundError(
            f"global_grid_leaderboard.csv not found at: {global_path}\n"
            "Run build_global_leaderboard.py first so this file is created."
        )

    df = pd.read_csv(global_path)
    print(f"[INFO] Loaded global leaderboard: {len(df)} rows from {global_path}")

    # ---- Filter by direction mode ----
    if "direction_mode" in df.columns:
        df = df[df["direction_mode"] == DIRECTION_MODE]
    else:
        # Fallback: use folder column if present
        if "direction_mode_folder" in df.columns:
            df = df[df["direction_mode_folder"] == DIRECTION_MODE]
        else:
            print("[WARN] No direction_mode column found; no direction filter applied.")

    print(f"[INFO] After direction filter ({DIRECTION_MODE}): {len(df)} rows")

    # ---- Filter by TP / Time exit toggles ----
    if "use_tp_exit" in df.columns:
        df = df[df["use_tp_exit"] == USE_TP_EXIT]
    else:
        print("[WARN] No use_tp_exit column in data; TP filter skipped.")

    print(f"[INFO] After TP filter (use_tp_exit={USE_TP_EXIT}): {len(df)} rows")

    if "use_time_exit" in df.columns:
        df = df[df["use_time_exit"] == USE_TIME_EXIT]
    else:
        print("[WARN] No use_time_exit column in data; TIME filter skipped.")

    print(f"[INFO] After TIME filter (use_time_exit={USE_TIME_EXIT}): {len(df)} rows")

    # ---- Filter by spread conditions (if columns exist) ----
    if "apply_spread" in df.columns:
        df = df[df["apply_spread"] == APPLY_SPREAD]
        print(f"[INFO] After apply_spread filter (apply_spread={APPLY_SPREAD}): {len(df)} rows")

        if APPLY_SPREAD and "spread_dollars" in df.columns:
            # Use a small tolerance in case of float rounding issues
            tol = 1e-9
            df = df[(df["spread_dollars"] - float(SPREAD_DOLLARS)).abs() <= tol]
            print(
                f"[INFO] After spread_dollars filter (spread_dollars={SPREAD_DOLLARS}): "
                f"{len(df)} rows"
            )
    else:
        print("[WARN] No apply_spread column in data; spread filters skipped.")

    return df


def ensure_param_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all PARAM_COLUMNS exist in df. If any are missing, warn and drop them
    from PARAM_COLUMNS for this run.
    """
    global PARAM_COLUMNS

    available_params = []
    for col in PARAM_COLUMNS:
        if col in df.columns:
            available_params.append(col)
        else:
            print(f"[WARN] Parameter column '{col}' not found in dataframe; "
                  f"it will be skipped for this run.")

    if not available_params:
        print("[WARN] No parameter columns available after checking; "
              "no heatmaps will be generated.")
    PARAM_COLUMNS = available_params
    return df


def generate_param_pair_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    """
    For the given DataFrame (already filtered), generate ALL pairwise
    parameter heatmaps for each metric in HEATMAP_METRICS.

    For each metric and each (row_param, col_param) pair with row_param != col_param:

        - Build pivot:
              index = row_param
              columns = col_param
              values = metric (mean)

        - Plot with RdYlGn colormap (green = better/higher metric).

        - Save as:
              out_dir / "pair_heatmap_<metric>_<row>_vs_<col>.png"
    """
    if df.empty:
        print("[WARN] Filtered dataframe is empty; no heatmaps will be generated.")
        return

    df = ensure_param_columns(df)
    if not PARAM_COLUMNS:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    n_plots = 0

    for metric in HEATMAP_METRICS:
        if metric not in df.columns:
            print(f"[WARN] Metric '{metric}' not found in data; skipping it.")
            continue

        for row_param, col_param in product(PARAM_COLUMNS, PARAM_COLUMNS):
            if row_param == col_param:
                continue

            sub = df[[row_param, col_param, metric]].dropna()

            if sub.empty:
                continue

            # Build pivot table
            pivot = sub.pivot_table(
                index=row_param,
                columns=col_param,
                values=metric,
                aggfunc="mean",
            )

            # Drop all-NaN rows/cols
            pivot = pivot.dropna(how="all", axis=0).dropna(how="all", axis=1)

            if pivot.empty:
                continue

            pivot = pivot.sort_index(axis=0).sort_index(axis=1)

            fig, ax = plt.subplots(figsize=(8, 6))

            # RdYlGn: low=red, mid=yellow, high=green
            im = ax.imshow(
                pivot.values,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="RdYlGn",
            )

            # Axis ticks & labels
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_ylabel(row_param)

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_xlabel(col_param)

            title = f"{metric} — {row_param} vs {col_param}"
            ax.set_title(title)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(metric)

            fig.tight_layout()

            filename = f"pair_heatmap_{metric}_{row_param}_vs_{col_param}.png"
            out_path = out_dir / filename
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"[INFO] Saved heatmap: {out_path}")
            n_plots += 1

    if n_plots == 0:
        print("[WARN] No param-pair heatmaps were generated.")
    else:
        print(f"[INFO] Generated {n_plots} param-pair heatmaps in {out_dir}")


def main():
    ensure_directories()

    # 1) Load and filter global leaderboard
    df = load_filtered_data()

    if df.empty:
        print("[WARN] No rows left after filtering; aborting.")
        return

    # 2) Build output directory tag
    direction_tag = DIRECTION_MODE
    spread_tag = "spread_on" if APPLY_SPREAD else "spread_off"
    exit_tag = f"tp{int(USE_TP_EXIT)}_time{int(USE_TIME_EXIT)}"

    base = Path(SUMMARY_REPORTS_DIR)
    out_dir = base / "param_pair_heatmaps" / f"{direction_tag}_{spread_tag}_{exit_tag}"

    print(f"[INFO] Saving param-pair heatmaps to: {out_dir}")

    # 3) Generate all pairwise parameter heatmaps
    generate_param_pair_heatmaps(df, out_dir)

    print("[INFO] Param-pair heatmap build complete.")


if __name__ == "__main__":
    main()
