"""
build_global_leaderboard.py
---------------------------

Aggregate ALL grid_scenarios*_summary.csv files across:

- stop modes: no-stops legacy, stops, stops_chandelier, stops_atr, ...
- direction modes: directional / inverse
- spread tags: spread_on / spread_off / legacy without explicit spread level
- exit configs: tp0_time0, tp0_time1, tp1_time0, tp1_time1, ...

and build a single combined leaderboard:

- Saves: SUMMARY_REPORTS_DIR / "global_grid_leaderboard.csv"
- Prints top-N scenarios, default ranking:
    * primary: highest percent_return
    * secondary: lowest max_drawdown_R (if available)

Run this AFTER you've run one or more grid backtests.
"""

from pathlib import Path

import pandas as pd

from config import (
    ensure_directories,
    SUMMARY_REPORTS_DIR,
)


def print_top_global(
    df: pd.DataFrame,
    n: int = 50,
    sort_by: str = "percent_return",
) -> None:
    """
    Print the top-N scenarios globally across all grid runs.

    Default ranking:
      - primary: sort_by (default: percent_return), descending
      - secondary: max_drawdown_R ascending (lower DD is better), if available
    """
    if df.empty:
        print("[WARN] Global leaderboard is empty.")
        return

    if sort_by not in df.columns:
        print(f"[WARN] sort_by='{sort_by}' not in leaderboard columns; cannot rank.")
        print(f"       Available columns: {list(df.columns)}")
        return

    ranked = df.dropna(subset=[sort_by]).copy()
    if ranked.empty:
        print(f"[WARN] No non-NaN values for metric '{sort_by}' in global leaderboard.")
        return

    # Build sort keys:
    # - main: sort_by (desc)
    # - secondary: max_drawdown_R (asc) when available
    sort_cols = [sort_by]
    ascending = [False]  # main metric: higher is better

    if "max_drawdown_R" in ranked.columns:
        sort_cols.append("max_drawdown_R")
        ascending.append(True)  # lower drawdown is better

    ranked = ranked.sort_values(by=sort_cols, ascending=ascending).head(n)

    cols_to_show = [
        "scenario_name",
        "stop_tag_folder",       # stops vs no-stops / chandelier / atr, etc.
        "run_tag_folder",        # run tag (v1, v2, 2025-12-02, etc.)
        "direction_mode",        # from grid summary (or derived)
        "direction_mode_folder", # from path
        "spread_tag_folder",     # from path
        "exit_tag_folder",       # from path
        "summary_file_name",     # grid_scenarios_summary vs grid_scenarios_stops_summary
        "apply_spread",          # from grid summary (bool, new runs)
        "spread_dollars",        # from grid summary (new runs)
        "T_seconds",
        "K_unique",
        "hold_minutes",
        "sl_distance",
        "tp_R_multiple",
        "entry_mode",
        "use_tp_exit",
        "use_time_exit",

        # Performance metrics (if present):
        "percent_return",
        "avg_R",
        "win_rate",
        "profit_factor",
        "max_drawdown_R",
        "longest_win_streak",
        "longest_loss_streak",
        "n_trades",
    ]

    cols_present = [c for c in cols_to_show if c in ranked.columns]

    print(
        "\n=== GLOBAL TOP SCENARIOS "
        "(sorted by '{}', desc; then max_drawdown_R asc) ===".format(sort_by)
    )
    print(ranked[cols_present].to_string(index=False))
    print("=== END GLOBAL TOP SCENARIOS ===\n")


def parse_path_metadata(base: Path, csv_path: Path) -> dict:
    """
    Infer stop_tag_folder, run_tag_folder, direction_mode_folder,
    spread_tag_folder, exit_tag_folder from the relative path
    under SUMMARY_REPORTS_DIR.

    Handles examples like:

    Legacy / no-stops:
      summaries/directional/tp1_time0/grid_scenarios_summary.csv
      summaries/inverse/spread_on/tp0_time1/grid_scenarios_summary.csv

    Stops (old style, no RUN_TAG):
      summaries/stops/inverse/spread_on/tp0_time1/grid_scenarios_stops_summary.csv
      summaries/stops_chandelier/inverse/spread_on/tp1_time1/grid_scenarios_stops_summary.csv

    Stops (new style, with RUN_TAG):
      summaries/stops/v2/inverse/spread_on/tp0_time1/grid_scenarios_stops_summary.csv
      summaries/stops_chandelier/2025-12-02/inverse/spread_on/tp1_time0/grid_scenarios_stops_summary.csv
    """
    rel = csv_path.relative_to(base)
    parts = rel.parts  # tuple of path components

    meta = {
        "stop_tag_folder": "no_stops_legacy",   # top-level folder before direction
        "run_tag_folder": "no_run_tag_legacy",  # optional extra folder between stop_tag and direction
        "direction_mode_folder": "unknown",
        "spread_tag_folder": "spread_unknown",
        "exit_tag_folder": "exit_unknown",
        "summary_file_name": csv_path.name,
    }

    if len(parts) < 2:
        print(f"[WARN] Skipping unexpected summary path (too short): {csv_path}")
        return meta

    # -------------------------------------------------
    # 1) Locate direction folder ("directional"/"inverse")
    # -------------------------------------------------
    direction_idx = None
    for i, p in enumerate(parts[:-1]):  # exclude the CSV file itself
        if p in ("directional", "inverse"):
            direction_idx = i
            break

    if direction_idx is None:
        # We didn't find a clean direction folder; fallback
        print(f"[WARN] Could not find direction folder in path: {csv_path}")
        return meta

    direction_mode_folder = parts[direction_idx]

    # -------------------------------------------------
    # 2) Determine stop_tag_folder and optional run_tag_folder
    # -------------------------------------------------
    if direction_idx == 0:
        # Path starts with "directional"/"inverse" → pure legacy no-stops layout
        stop_tag_folder = "no_stops_legacy"
        run_tag_folder = "no_run_tag_legacy"
    elif direction_idx == 1:
        # [stop_tag]/[direction]/...
        stop_tag_folder = parts[0]
        run_tag_folder = "no_run_tag_legacy"
    else:
        # [stop_tag]/[run_tag]/[direction]/...
        stop_tag_folder = parts[0]
        run_tag_folder = parts[1]

    meta["stop_tag_folder"] = stop_tag_folder
    meta["run_tag_folder"] = run_tag_folder
    meta["direction_mode_folder"] = direction_mode_folder

    # -------------------------------------------------
    # 3) Everything between direction and filename:
    #    either [exit_tag] or [spread_tag]/[exit_tag]/...
    # -------------------------------------------------
    remaining = parts[direction_idx + 1 : -1]  # exclude direction + CSV filename

    if len(remaining) == 1:
        # direction/exit_tag/file  (old legacy)
        meta["exit_tag_folder"] = remaining[0]
        meta["spread_tag_folder"] = "spread_off_legacy"
    elif len(remaining) >= 2:
        # direction/spread_tag/exit_tag/file (newer style)
        meta["spread_tag_folder"] = remaining[0]
        meta["exit_tag_folder"] = remaining[1]
    # else: keep defaults

    return meta


def main():
    ensure_directories()

    base = Path(SUMMARY_REPORTS_DIR)

    print(f"[INFO] Scanning for grid_scenarios*_summary.csv under: {base}")

    all_dfs = []

    # Recursively find all variants:
    # - grid_scenarios_summary.csv
    # - grid_scenarios_stops_summary.csv
    for csv_path in base.rglob("grid_scenarios*_summary.csv"):
        meta = parse_path_metadata(base, csv_path)

        print(
            f"[INFO] Loading summary from: {csv_path} "
            f"(stop_tag_folder={meta['stop_tag_folder']}, "
            f"run_tag_folder={meta['run_tag_folder']}, "
            f"direction_mode_folder={meta['direction_mode_folder']}, "
            f"spread_tag_folder={meta['spread_tag_folder']}, "
            f"exit_tag_folder={meta['exit_tag_folder']})"
        )

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Failed to read {csv_path}: {e}")
            continue

        # Annotate with folder-based metadata
        for key, value in meta.items():
            df[key] = value

        all_dfs.append(df)

    if not all_dfs:
        print(
            "[WARN] No grid_scenarios*_summary.csv files found. "
            "Have you run run_grid_scenarios.py (with/without stops) yet?"
        )
        return

    global_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure direction_mode column exists (grid script should include it),
    # but if not, fall back to folder-based info
    if "direction_mode" not in global_df.columns:
        global_df["direction_mode"] = global_df["direction_mode_folder"]

    # ---- Save combined leaderboard ----
    base_name = "global_grid_leaderboard_ALL"
    ext = ".csv"

    out_path = base / f"{base_name}{ext}"

    # If file already exists, add _001, _002, _003 ...
    if out_path.exists():
        counter = 1
        while True:
            candidate = base / f"{base_name}_{counter:03d}{ext}"
            if not candidate.exists():
                out_path = candidate
                break
            counter += 1

    global_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved global leaderboard to: {out_path}")

    # Print global top scenarios
    print_top_global(global_df, n=50, sort_by="percent_return")


if __name__ == "__main__":
    main()
