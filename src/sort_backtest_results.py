#!/usr/bin/env python

"""
Sort backtest grid results by:
1) percent_return (largest → smallest)
2) smallest absolute max_drawdown_R (i.e. least painful drawdown)

Edit INPUT_PATH and OUTPUT_PATH below.
"""

import os
import pandas as pd

# ======== EDIT THESE TWO PATHS ========
INPUT_PATH  = r"C:\Users\anish\PycharmProjects\ClusterOrderflow-Backtest\output\reports\summaries\stops\inverse\spread_on\tp0_time1\grid_scenarios_stops_summary.csv"
OUTPUT_PATH = r"C:\Users\anish\PycharmProjects\ClusterOrderflow-Backtest\output\reports\x_sorted_summaries\sorted_results_tp0_tp1.csv"
# ======================================


def sort_backtest_file(
    input_path: str,
    output_path: str,
    percent_col: str = "percent_return",
    dd_col: str = "max_drawdown_R",
) -> None:
    """
    Reads a CSV/TSV-like file, sorts by:
      - percent_return (DESC: best first)
      - abs(max_drawdown_R) (ASC: smallest absolute DD first)
    and writes the result as CSV to output_path.
    """

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("[INFO] Loading file:", input_path)

    # Auto-detect delimiter (comma or tab)
    df = pd.read_csv(input_path, sep=None, engine="python")

    # Convert sort columns to numeric
    for col in [percent_col, dd_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found. Available columns: {list(df.columns)}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Helper column: absolute drawdown
    abs_dd_col = f"{dd_col}_abs"
    df[abs_dd_col] = df[dd_col].abs()

    print("[INFO] Performing multi-sort:",
          f"{percent_col} (DESC), |{dd_col}| (ASC)")

    df_sorted = df.sort_values(
        by=[percent_col, abs_dd_col],
        ascending=[False, True],   # best % first, then smallest absolute DD
        kind="mergesort",
    )

    # Drop helper column from final output to keep CSV clean
    df_sorted = df_sorted.drop(columns=[abs_dd_col])

    # Create output directory if needed
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save CSV
    df_sorted.to_csv(output_path, index=False)
    print("[OK] Sorted file saved to:", output_path)


if __name__ == "__main__":
    sort_backtest_file(INPUT_PATH, OUTPUT_PATH)
