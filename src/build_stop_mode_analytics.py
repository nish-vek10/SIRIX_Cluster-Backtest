#!/usr/bin/env python

"""
build_stopmode_analytics.py
---------------------------

Takes global_grid_leaderboard_ALL.csv and builds:

1) A global enriched CSV with:
   - dd_return_ratio
   - return_per_dd_R
   - score_simple
   - score_blended

2) Per-stop-mode CSVs (atr_static, atr_trailing, chandelier, fixed, nan)
   under:  <BASE_DIR>/output/reports/summaries/stopmode_analytics/by_mode/

3) A summary by stop_mode_clean:
   - n_rows, percent_return_min / max / mean
   - dd_return_ratio_mean
   - return_per_dd_R_mean
   - score_simple_mean
   - score_blended_mean

   Saved as:
   - summary_by_stop_mode.csv
   - summary_by_stop_mode.xlsx  (with red→green colour scale on score columns)
"""

import os
import numpy as np
import pandas as pd

from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule


# =====================================================================================
# PATHS – EDIT BASE_DIR IF NEEDED
# =====================================================================================

# =====================================================================================
# PATHS – EDIT BASE_DIR IF NEEDED
# =====================================================================================

BASE_DIR = r"C:\Users\anish\PycharmProjects\ClusterOrderflow-Backtest"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "output",
    "reports",
    "summaries",
    "global_grid_leaderboard_ALL_001.csv",
)

# ----- NEW: auto-incrementing run tag folder (v1, v2, v3...) -----
STOPMODE_ROOT = os.path.join(
    BASE_DIR,
    "output",
    "reports",
    "summaries",
    "stopmode_analytics",
)

# Ensure main folder exists
os.makedirs(STOPMODE_ROOT, exist_ok=True)

# Detect existing vX folders
existing = [
    d for d in os.listdir(STOPMODE_ROOT)
    if os.path.isdir(os.path.join(STOPMODE_ROOT, d)) and d.startswith("v")
]

# Extract numbers
numbers = []
for folder in existing:
    try:
        numbers.append(int(folder[1:]))  # remove 'v'
    except:
        pass

next_ver = max(numbers) + 1 if numbers else 1
RUN_TAG = f"v{next_ver}"

OUT_BASE_DIR = os.path.join(STOPMODE_ROOT, RUN_TAG)
print(f"[INFO] Using run output folder: {OUT_BASE_DIR}")

# Output paths
GLOBAL_ENRICHED_CSV = os.path.join(OUT_BASE_DIR, "global_with_ratios_and_scores.csv")
BY_MODE_DIR = os.path.join(OUT_BASE_DIR, "by_mode")
SUMMARY_CSV = os.path.join(OUT_BASE_DIR, "summary_by_stop_mode.csv")
SUMMARY_XLSX = os.path.join(OUT_BASE_DIR, "summary_by_stop_mode.xlsx")


# =====================================================================================
# HELPERS
# =====================================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ensure_dir(OUT_BASE_DIR)
    ensure_dir(BY_MODE_DIR)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    print(f"[INFO] Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    required_cols = {"percent_return", "max_drawdown_R", "win_rate", "stop_mode"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    # -------------------------------------------------------------------------
    # 1) Clean stop_mode into stop_mode_clean
    # -------------------------------------------------------------------------
    df["stop_mode_clean"] = df["stop_mode"].fillna("nan").astype(str)

    # -------------------------------------------------------------------------
    # 2) Ratios
    #    dd_return_ratio    = abs(max_drawdown_R) / percent_return
    #    return_per_dd_R    = percent_return / abs(max_drawdown_R)
    #
    #    We guard against division by 0 or NaNs.
    # -------------------------------------------------------------------------
    percent = df["percent_return"].astype(float)
    dd_R = df["max_drawdown_R"].astype(float)

    df["dd_return_ratio"] = np.where(
        percent != 0,
        dd_R.abs() / percent,
        np.nan,
    )

    df["return_per_dd_R"] = np.where(
        dd_R != 0,
        percent / dd_R.abs(),
        np.nan,
    )

    # -------------------------------------------------------------------------
    # 3) Scores
    #    score_simple  = return_per_dd_R - dd_return_ratio
    #        -> high = better balance (more return, less DD)
    #
    #    score_blended = (return_per_dd_R * win_rate) / dd_return_ratio
    #        -> only defined when dd_return_ratio > 0 (i.e. "well-behaved"
    #           ratio with positive percent_return)
    #
    #    Negative or NaN dd_return_ratio implies losing / weird configs;
    #    we set score_blended = NaN for those (they're obviously not
    #    "good" candidates anyway).
    # -------------------------------------------------------------------------
    win_rate = df["win_rate"].astype(float)

    df["score_simple"] = df["return_per_dd_R"] - df["dd_return_ratio"]

    df["score_blended"] = np.where(
        (df["dd_return_ratio"] > 0) & win_rate.notna(),
        (df["return_per_dd_R"] * win_rate) / df["dd_return_ratio"],
        np.nan,
    )

    # -------------------------------------------------------------------------
    # 4) Save global enriched file
    # -------------------------------------------------------------------------
    df.to_csv(GLOBAL_ENRICHED_CSV, index=False)
    print(f"[INFO] Saved global enriched file: {GLOBAL_ENRICHED_CSV}")

    # -------------------------------------------------------------------------
    # 5) Per-stop-mode CSVs
    # -------------------------------------------------------------------------
    modes = sorted(df["stop_mode_clean"].unique())
    print(f"[INFO] Found stop_mode_clean values: {modes}")

    for mode in modes:
        sub = df[df["stop_mode_clean"] == mode].copy()
        out_path = os.path.join(BY_MODE_DIR, f"grid_{mode}.csv")
        sub.to_csv(out_path, index=False)
        print(f"[INFO] Saved mode file: {out_path}  (rows={len(sub)})")

    # -------------------------------------------------------------------------
    # 6) Summary by stop_mode_clean
    # -------------------------------------------------------------------------
    group = df.groupby("stop_mode_clean", dropna=False)

    summary = group.agg(
        n_rows=("scenario_name", "size"),
        percent_return_min=("percent_return", "min"),
        percent_return_max=("percent_return", "max"),
        percent_return_mean=("percent_return", "mean"),
        dd_return_ratio_mean=("dd_return_ratio", "mean"),
        return_per_dd_R_mean=("return_per_dd_R", "mean"),
        win_rate_mean=("win_rate", "mean"),
        score_simple_mean=("score_simple", "mean"),
        score_blended_mean=("score_blended", "mean"),
    ).reset_index()

    # 3 decimal places for scores & means (nice + consistent)
    for col in [
        "percent_return_min",
        "percent_return_max",
        "percent_return_mean",
        "dd_return_ratio_mean",
        "return_per_dd_R_mean",
        "win_rate_mean",
        "score_simple_mean",
        "score_blended_mean",
    ]:
        summary[col] = summary[col].round(3)

    summary.rename(columns={"stop_mode_clean": "stop_mode_clean"}, inplace=True)

    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[INFO] Saved summary CSV: {SUMMARY_CSV}")

    # -------------------------------------------------------------------------
    # 7) Styled Excel with red to green scale on score columns
    # -------------------------------------------------------------------------
    with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="summary")
        ws = writer.sheets["summary"]

        # Colour scale for these columns:
        score_cols = ["score_simple_mean", "score_blended_mean"]
        from_row = 2
        to_row = len(summary) + 1

        for col_name in score_cols:
            if col_name not in summary.columns:
                continue
            col_idx = summary.columns.get_loc(col_name) + 1  # 1-based
            col_letter = get_column_letter(col_idx)
            cell_range = f"{col_letter}{from_row}:{col_letter}{to_row}"

            # Red (min) → Yellow (mid) → Green (max)
            color_scale = ColorScaleRule(
                start_type="min", start_color="FF0000",
                mid_type="percentile", mid_value=50, mid_color="FFFFEB84",
                end_type="max", end_color="008000",
            )
            ws.conditional_formatting.add(cell_range, color_scale)

    print(f"[INFO] Saved styled summary Excel: {SUMMARY_XLSX}")
    print("[DONE] stop-mode analytics complete.")


if __name__ == "__main__":
    main()
