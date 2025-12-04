#!/usr/bin/env python

"""
Analyse concurrency (number of open positions) across multiple strategies,
and analyse winning/losing streak distributions.

- Input: 3 (or more) trade logs in CSV format, each with columns:
    * entry_time
    * exit_time
    * stop_mode (e.g. 'atr_static', 'atr_trailing', 'chandelier')
  plus all your usual fields (signal_time, pnl_R, pnl_cash, etc.).

- Output folder structure (under INPUT_DIR / RUN_TAG):

    <INPUT_DIR> / <RUN_TAG> /
        csv /
            positions_concurrency_summary.csv        (if DO_CONCURRENCY_ANALYSIS)
            positions_concurrency_overview.csv       (if DO_CONCURRENCY_ANALYSIS)
            streaks_summary.csv
            streaks_detail_all.csv                   (if DO_STREAK_DISTRIBUTIONS)
            streaks_detail_<engine>.csv              (per engine, if DO_STREAK_DISTRIBUTIONS)
            streaks_detail_combined.csv              (if DO_STREAK_DISTRIBUTIONS)
            intervals /
                positions_concurrency_intervals_<engine>.csv     (if DO_CONCURRENCY_ANALYSIS)
                positions_concurrency_intervals_combined.csv     (if DO_CONCURRENCY_ANALYSIS)
        plots /
            positions_concurrency_<engine>.png       (if DO_CONCURRENCY_ANALYSIS)
            streaks_win_dist_<engine>.png            (if DO_STREAK_DISTRIBUTIONS)
            streaks_loss_dist_<engine>.png           (if DO_STREAK_DISTRIBUTIONS)
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any


# ============================================================
# USER CONFIG
# ============================================================

RUN_TAG = "v2"

# Master toggles
DO_CONCURRENCY_ANALYSIS = True       # set False to skip concurrency CSVs/plots
DO_STREAK_DISTRIBUTIONS = True       # set False to skip new streak distribution outputs

# Folder where your trade logs live
INPUT_DIR = Path(
    r"C:\Users\anish\PycharmProjects\ClusterOrderflow-Backtest\output\reports\scenarios\stops\v2\inverse\spread_on\tp1_time0\single"
)

# Explicit file paths for each engine
TRADE_FILES = [
    INPUT_DIR / "trades_T29_K3_H5_SL2.0_TP1.0_prop_inverse_atr_static_ATR5_I4.0_spread_on_tp1_time0.csv",
    INPUT_DIR / "trades_T29_K3_H5_SL2.0_TP1.0_prop_inverse_atr_trailing_ATR5_I3.0_T2.0_spread_on_tp1_time0.csv",
    INPUT_DIR / "trades_T30_K3_H15_SL2.0_TP1.0_prop_inverse_chandelier_ATR5_I3.0_T2.0_CH30_spread_on_tp1_time0.csv",
]

# Root output directory (versioned by RUN_TAG)
OUTPUT_DIR = INPUT_DIR / RUN_TAG
CSV_DIR = OUTPUT_DIR / "csv"
INTERVALS_DIR = CSV_DIR / "intervals"
PLOTS_DIR = OUTPUT_DIR / "plots"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_and_tag_trades(paths: List[Path]) -> pd.DataFrame:
    """
    Load all trade CSVs, parse datetime columns, and tag them with an 'engine'
    name derived from the CSV content or filename.

    Returns a single concatenated DataFrame with a new 'engine' column.
    """
    dfs = []

    for p in paths:
        if not p.exists():
            print(f"[WARN] File not found, skipping: {p}")
            continue

        print(f"[INFO] Loading trades from: {p}")
        df = pd.read_csv(p)

        # Parse datetimes (force everything to UTC to avoid mixed-tz warnings)
        for col in ["entry_time", "exit_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)

        # Decide engine label: use 'stop_mode' if present, otherwise filename stem
        if "stop_mode" in df.columns and not df["stop_mode"].isna().all():
            engine = str(df["stop_mode"].iloc[0])
        else:
            engine = p.stem

        df["engine"] = engine
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid trade files loaded. Check TRADE_FILES paths.")

    all_trades = pd.concat(dfs, ignore_index=True)
    print("[INFO] Loaded total trades by engine:")
    print(all_trades["engine"].value_counts())
    return all_trades


def build_intervals_for_subset(df_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Build a list of intervals with constant concurrency for a subset of trades
    (e.g. for a single engine or all engines combined).

    Returns DataFrame with:
        - start_time
        - end_time
        - duration_minutes
        - concurrency
    """
    events: List[tuple] = []

    for _, row in df_subset.iterrows():
        start = row.get("entry_time")
        end = row.get("exit_time")

        # Skip malformed rows
        if pd.isna(start) or pd.isna(end):
            continue
        if end <= start:
            continue

        events.append((start, +1))
        events.append((end, -1))

    if not events:
        print("[WARN] No valid start/end times for this subset; returning empty intervals.")
        return pd.DataFrame(columns=["start_time", "end_time", "duration_minutes", "concurrency"])

    # Sort events: exits (-1) before entries (+1) at the same timestamp
    events.sort(key=lambda x: (x[0], x[1]))

    intervals: List[Dict[str, Any]] = []
    current_conc = 0
    last_time = None

    for time, delta in events:
        # Record interval [last_time, time) with current_conc
        if last_time is not None and time > last_time:
            if current_conc > 0:
                duration_min = (time - last_time).total_seconds() / 60.0
                intervals.append(
                    {
                        "start_time": last_time,
                        "end_time": time,
                        "duration_minutes": duration_min,
                        "concurrency": current_conc,
                    }
                )

        # Apply event at this timestamp
        current_conc += delta
        last_time = time

    return pd.DataFrame(intervals)


def summarize_concurrency(intervals_df: pd.DataFrame, engine_name: str) -> pd.DataFrame:
    """
    From intervals with concurrency, build a summary per concurrency level.

    Output columns:
        - engine
        - concurrency
        - n_episodes               (distinct contiguous runs at that concurrency)
        - total_minutes            (total time with that concurrency)
        - pct_of_trading_time      (relative to total minutes with any open position)
        - avg_minutes_per_episode  (total_minutes / n_episodes)
    """
    if intervals_df.empty:
        print(f"[WARN] No intervals for engine '{engine_name}'.")
        return pd.DataFrame(
            columns=[
                "engine",
                "concurrency",
                "n_episodes",
                "total_minutes",
                "pct_of_trading_time",
                "avg_minutes_per_episode",
            ]
        )

    total_active_minutes = intervals_df["duration_minutes"].sum()

    # Total minutes per concurrency
    grp = intervals_df.groupby("concurrency")["duration_minutes"].sum().reset_index()
    grp.rename(columns={"duration_minutes": "total_minutes"}, inplace=True)

    # Count episodes per concurrency
    conc_series = intervals_df["concurrency"].values
    episodes: Dict[int, int] = {}
    prev_c = None

    for c in conc_series:
        if c <= 0:
            prev_c = c
            continue
        # Start of a new episode if we just switched into this concurrency
        if c != prev_c:
            episodes[c] = episodes.get(c, 0) + 1
        prev_c = c

    grp["n_episodes"] = grp["concurrency"].map(lambda k: episodes.get(int(k), 0))

    # Probability (percentage) of being at each concurrency level (time-weighted)
    grp["pct_of_trading_time"] = grp["total_minutes"] / total_active_minutes * 100.0

    # ✅ NEW: average minutes per episode at this concurrency
    grp["avg_minutes_per_episode"] = grp.apply(
        lambda r: r["total_minutes"] / r["n_episodes"] if r["n_episodes"] > 0 else 0.0,
        axis=1,
    )

    grp["engine"] = engine_name

    # Reorder columns
    grp = grp[
        [
            "engine",
            "concurrency",
            "n_episodes",
            "total_minutes",
            "pct_of_trading_time",
            "avg_minutes_per_episode",
        ]
    ]
    return grp


def compute_pnl_per_concurrency(df_trades: pd.DataFrame, intervals_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each interval (constant concurrency), compute which trades were active
    and sum their pnl_cash contributions.

    Returns dataframe with columns:
        concurrency
        interval_pnl
        trade_count
        mean_pnl_per_trade
    """

    results = []

    for _, row in intervals_df.iterrows():
        start, end, conc = row["start_time"], row["end_time"], row["concurrency"]

        # Trades actively open during this interval
        active = df_trades[
            (df_trades["entry_time"] <= end) &
            (df_trades["exit_time"] >= start)
        ]

        interval_pnl = active["pnl_cash"].sum() if not active.empty else 0.0
        trade_count = len(active)

        results.append({
            "concurrency": conc,
            "interval_pnl": interval_pnl,
            "trade_count": trade_count,
            "mean_pnl_per_trade": interval_pnl / trade_count if trade_count > 0 else 0.0
        })

    return pd.DataFrame(results)


def plot_concurrency_distribution(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    """
    For each engine in summary_df, plot a bar chart of:
        concurrency vs pct_of_trading_time

    Saves figures as positions_concurrency_<engine>.png in plots_dir.
    """
    engines = summary_df["engine"].unique()

    for eng in engines:
        sub = summary_df[summary_df["engine"] == eng].sort_values("concurrency")
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(sub["concurrency"].astype(str), sub["pct_of_trading_time"])

        ax.set_title(f"Concurrency distribution (% of time) — {eng}")
        ax.set_xlabel("Number of open positions")
        ax.set_ylabel("% of trading time at this concurrency")

        fig.tight_layout()

        safe_eng = str(eng).replace(" ", "_")
        out_path = plots_dir / f"positions_concurrency_{safe_eng}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"[INFO] Saved concurrency plot for {eng} -> {out_path}")


def compute_streaks(df: pd.DataFrame, engine_name: str) -> dict:
    """
    Compute *aggregate* longest win streak & longest loss streak for a given engine.

    A win = pnl_cash > 0
    A loss = pnl_cash < 0
    """

    if df.empty or "pnl_cash" not in df.columns:
        return {
            "engine": engine_name,
            "longest_win_streak": 0,
            "win_streak_total_pnl": 0.0,
            "longest_loss_streak": 0,
            "loss_streak_total_pnl": 0.0,
        }

    df_sorted = df.sort_values("entry_time")

    longest_win = 0
    longest_loss = 0
    win_total = 0.0
    loss_total = 0.0

    current_win = 0
    current_loss = 0
    current_win_total = 0.0
    current_loss_total = 0.0

    for pnl in df_sorted["pnl_cash"]:
        if pnl > 0:
            current_win += 1
            current_win_total += pnl
            current_loss = 0
            current_loss_total = 0.0
        elif pnl < 0:
            current_loss += 1
            current_loss_total += pnl
            current_win = 0
            current_win_total = 0.0
        else:
            current_win = 0
            current_loss = 0

        longest_win = max(longest_win, current_win)
        win_total = max(win_total, current_win_total)

        longest_loss = max(longest_loss, current_loss)
        loss_total = min(loss_total, current_loss_total)

    return {
        "engine": engine_name,
        "longest_win_streak": longest_win,
        "win_streak_total_pnl": win_total,
        "longest_loss_streak": longest_loss,
        "loss_streak_total_pnl": loss_total,
    }


def compute_streak_sequences(df: pd.DataFrame, engine_name: str) -> pd.DataFrame:
    """
    Build a detailed streak table for a given engine:

    Returns DataFrame with columns:
        - engine
        - streak_type  ('win' or 'loss')
        - streak_len
        - streak_total_pnl
    """
    if df.empty or "pnl_cash" not in df.columns:
        return pd.DataFrame(
            columns=["engine", "streak_type", "streak_len", "streak_total_pnl"]
        )

    df_sorted = df.sort_values("entry_time").reset_index(drop=True)

    streaks = []
    current_type = None  # 'win' or 'loss'
    current_len = 0
    current_total = 0.0

    for _, row in df_sorted.iterrows():
        pnl = row["pnl_cash"]
        if pnl > 0:
            t = "win"
        elif pnl < 0:
            t = "loss"
        else:
            # flat -> break current streak, but don't start a new one
            if current_type is not None and current_len > 0:
                streaks.append(
                    {
                        "engine": engine_name,
                        "streak_type": current_type,
                        "streak_len": current_len,
                        "streak_total_pnl": current_total,
                    }
                )
            current_type = None
            current_len = 0
            current_total = 0.0
            continue

        # Extend or start streak
        if current_type is None:
            current_type = t
            current_len = 1
            current_total = pnl
        elif current_type == t:
            current_len += 1
            current_total += pnl
        else:
            # streak type changed -> close old streak, start new
            streaks.append(
                {
                    "engine": engine_name,
                    "streak_type": current_type,
                    "streak_len": current_len,
                    "streak_total_pnl": current_total,
                }
            )
            current_type = t
            current_len = 1
            current_total = pnl

    # Flush last streak
    if current_type is not None and current_len > 0:
        streaks.append(
            {
                "engine": engine_name,
                "streak_type": current_type,
                "streak_len": current_len,
                "streak_total_pnl": current_total,
            }
        )

    return pd.DataFrame(streaks)


def plot_streak_distributions(streaks_all: pd.DataFrame, plots_dir: Path) -> None:
    """
    For each engine (including 'combined'), build distributions of streak lengths:

    - one bar chart for win streaks
    - one bar chart for loss streaks

    Saves:
        streaks_win_dist_<engine>.png
        streaks_loss_dist_<engine>.png
    """
    if streaks_all.empty:
        print("[WARN] No streaks to plot.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    for engine in streaks_all["engine"].unique():
        sub = streaks_all[streaks_all["engine"] == engine]

        for s_type in ["win", "loss"]:
            sub_type = sub[sub["streak_type"] == s_type]
            if sub_type.empty:
                continue

            counts = sub_type.groupby("streak_len").size().reset_index(name="count")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(counts["streak_len"].astype(str), counts["count"])

            title_type = "Winning" if s_type == "win" else "Losing"
            ax.set_title(f"{title_type} streak length distribution — {engine}")
            ax.set_xlabel("Streak length (number of trades)")
            ax.set_ylabel("Number of streaks")

            fig.tight_layout()

            safe_eng = str(engine).replace(" ", "_")
            out_path = plots_dir / f"streaks_{s_type}_dist_{safe_eng}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"[INFO] Saved {title_type.lower()} streak distribution for {engine} -> {out_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # Create folder structure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    INTERVALS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load all trades and tag with engine
    all_trades = load_and_tag_trades(TRADE_FILES)

    summaries = []
    overview_rows = []
    streaks_list = []
    streaks_detail_frames = []

    # 2) Per-engine analysis
    for engine in sorted(all_trades["engine"].unique()):
        df_e = all_trades[all_trades["engine"] == engine].copy()
        print(f"\n[INFO] Processing engine: {engine}")

        # ----- Concurrency analysis per engine -----
        if DO_CONCURRENCY_ANALYSIS:
            print(f"[INFO] Building concurrency intervals for engine: {engine}")
            intervals_e = build_intervals_for_subset(df_e)
            print(f"[INFO] Intervals for {engine}: {len(intervals_e)} rows")

            summary_e = summarize_concurrency(intervals_e, engine_name=engine)

            # High-level overview metrics
            if not intervals_e.empty:
                max_conc = intervals_e["concurrency"].max()
                avg_conc = (
                    (intervals_e["concurrency"] * intervals_e["duration_minutes"]).sum()
                    / intervals_e["duration_minutes"].sum()
                )
                total_minutes = intervals_e["duration_minutes"].sum()
            else:
                max_conc = 0
                avg_conc = 0.0
                total_minutes = 0.0

            overview_rows.append(
                {
                    "engine": engine,
                    "total_trades": len(df_e),
                    "max_concurrency": max_conc,
                    "avg_concurrency_time_weighted": avg_conc,
                    "total_trading_minutes_with_open_positions": total_minutes,
                }
            )

            # PnL per concurrency level
            pnl_stats = compute_pnl_per_concurrency(df_e, intervals_e)
            pnl_grouped = pnl_stats.groupby("concurrency").agg({
                "interval_pnl": "sum",
                "trade_count": "sum",
                "mean_pnl_per_trade": "mean"
            }).reset_index()
            pnl_grouped["engine"] = engine

            summary_e = summary_e.merge(pnl_grouped, on=["engine", "concurrency"], how="left")

            # Save per-engine intervals
            intervals_path = INTERVALS_DIR / f"positions_concurrency_intervals_{engine}.csv"
            intervals_e.to_csv(intervals_path, index=False)
            print(f"[INFO] Saved per-interval concurrency for {engine} -> {intervals_path}")

            summaries.append(summary_e)

        # ----- Aggregate streak metrics -----
        streak_stats = compute_streaks(df_e, engine)
        streaks_list.append(streak_stats)

        # ----- Detailed streak sequences (for distributions) -----
        if DO_STREAK_DISTRIBUTIONS:
            streaks_detail_e = compute_streak_sequences(df_e, engine)
            streaks_detail_frames.append(streaks_detail_e)

            detail_path = CSV_DIR / f"streaks_detail_{engine}.csv"
            streaks_detail_e.to_csv(detail_path, index=False)
            print(f"[INFO] Saved detailed streaks for {engine} -> {detail_path}")

    # 3) Combined concurrency (all engines together)
    if DO_CONCURRENCY_ANALYSIS:
        print("\n[INFO] Building combined concurrency intervals (all engines together)...")
        intervals_combined = build_intervals_for_subset(all_trades)
        summary_combined = summarize_concurrency(intervals_combined, engine_name="combined")

        # PnL per concurrency for combined
        pnl_stats_combined = compute_pnl_per_concurrency(all_trades, intervals_combined)
        pnl_grouped_combined = pnl_stats_combined.groupby("concurrency").agg({
            "interval_pnl": "sum",
            "trade_count": "sum",
            "mean_pnl_per_trade": "mean"
        }).reset_index()
        pnl_grouped_combined["engine"] = "combined"
        summary_combined = summary_combined.merge(
            pnl_grouped_combined, on=["engine", "concurrency"], how="left"
        )

        summaries.append(summary_combined)

        if not intervals_combined.empty:
            max_conc_c = intervals_combined["concurrency"].max()
            avg_conc_c = (
                (intervals_combined["concurrency"] * intervals_combined["duration_minutes"]).sum()
                / intervals_combined["duration_minutes"].sum()
            )
            total_minutes_c = intervals_combined["duration_minutes"].sum()
        else:
            max_conc_c = 0
            avg_conc_c = 0.0
            total_minutes_c = 0.0

        overview_rows.append(
            {
                "engine": "combined",
                "total_trades": len(all_trades),
                "max_concurrency": max_conc_c,
                "avg_concurrency_time_weighted": avg_conc_c,
                "total_trading_minutes_with_open_positions": total_minutes_c,
            }
        )

        # Save combined intervals
        combined_intervals_path = INTERVALS_DIR / "positions_concurrency_intervals_combined.csv"
        intervals_combined.to_csv(combined_intervals_path, index=False)
        print(f"[INFO] Saved combined intervals -> {combined_intervals_path}")

    # Combined streak analysis
    streak_combined = compute_streaks(all_trades, "combined")
    streaks_list.append(streak_combined)

    if DO_STREAK_DISTRIBUTIONS:
        streaks_detail_combined = compute_streak_sequences(all_trades, "combined")
        streaks_detail_frames.append(streaks_detail_combined)

        combined_detail_path = CSV_DIR / "streaks_detail_combined.csv"
        streaks_detail_combined.to_csv(combined_detail_path, index=False)
        print(f"[INFO] Saved detailed streaks for combined -> {combined_detail_path}")

    # 4) Save summary + overview CSVs (if concurrency is enabled)
    if DO_CONCURRENCY_ANALYSIS and summaries:
        summary_df = pd.concat(summaries, ignore_index=True)
        overview_df = pd.DataFrame(overview_rows)

        summary_path = CSV_DIR / "positions_concurrency_summary.csv"
        overview_path = CSV_DIR / "positions_concurrency_overview.csv"

        summary_df.to_csv(summary_path, index=False)
        overview_df.to_csv(overview_path, index=False)

        print(f"\n[INFO] Saved concurrency summary -> {summary_path}")
        print(f"[INFO] Saved concurrency overview -> {overview_path}")
    else:
        summary_df = pd.DataFrame()  # empty placeholder if needed

    # 5) Save streak analytics (aggregate)
    streaks_df = pd.DataFrame(streaks_list)
    streaks_path = CSV_DIR / "streaks_summary.csv"
    streaks_df.to_csv(streaks_path, index=False)
    print(f"[INFO] Saved streak analytics -> {streaks_path}")

    # 6) Streak distribution plots + detailed CSV (if enabled)
    if DO_STREAK_DISTRIBUTIONS and streaks_detail_frames:
        streaks_detail_all = pd.concat(streaks_detail_frames, ignore_index=True)
        all_detail_path = CSV_DIR / "streaks_detail_all.csv"
        streaks_detail_all.to_csv(all_detail_path, index=False)
        print(f"[INFO] Saved all detailed streaks -> {all_detail_path}")

        plot_streak_distributions(streaks_detail_all, PLOTS_DIR)

    # 7) Plot bar charts for concurrency (if enabled)
    if DO_CONCURRENCY_ANALYSIS and not summary_df.empty:
        plot_concurrency_distribution(summary_df, PLOTS_DIR)

    print("\n[INFO] Done. You can now inspect the CSVs (in csv/) and charts (in plots/).")


if __name__ == "__main__":
    main()
