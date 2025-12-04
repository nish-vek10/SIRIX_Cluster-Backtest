"""
config.py
---------

Central configuration for the XAU_ClusterOrderflow_Backtest project.

- File paths for input/output.
- Global risk/equity settings.
- Default parameter grids for T (sec), K (cluster size), exit times, SL ranges, etc.
"""

from pathlib import Path

# ====== BASE PATHS ======

# Root directory of the project (this file lives in src/, so parent.parent is project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
SIGNALS_DIR = INTERMEDIATE_DIR / "signals"
BACKTESTS_DIR = INTERMEDIATE_DIR / "backtests"

# Output folders
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
SUMMARY_REPORTS_DIR = REPORTS_DIR / "summaries"
SCENARIO_REPORTS_DIR = REPORTS_DIR / "scenarios"

FIGURES_DIR = OUTPUT_DIR / "figures"
EQUITY_FIG_DIR = FIGURES_DIR / "equity_curves"
HEATMAP_FIG_DIR = FIGURES_DIR / "heatmaps"
DIST_FIG_DIR = FIGURES_DIR / "distributions"

# ====== INPUT FILE NAMES ======

# You can change these if your filenames differ
TRADES_FILE = INPUT_DIR / "xauusd_trades.csv"
CANDLES_FILE = INPUT_DIR / "xauusd_oanda_m1.csv"

# ====== GLOBAL RISK SETTINGS ======

INITIAL_EQUITY = 10_000.0       # starting equity

# --- RISK MODE TOGGLE ---
# "static"         -> use ABS_RISK_PER_TRADE (e.g. always $200 risk)
# "percent_equity" -> risk a fixed percentage of CURRENT equity (e.g. 2% per trade)
RISK_MODE = "percent_equity"            # change to "percent_equity" for dynamic 2% risk

# Static risk (used when RISK_MODE == "static")
ABS_RISK_PER_TRADE = 200.0      # $ risk per trade (legacy behaviour)

# Dynamic risk (used when RISK_MODE == "percent_equity")
RISK_PERCENT_PER_TRADE = 0.02   # 2% of current equity per trade


# SL ranges in price (XAUUSD dollars)
SL_DISTANCE_GRID = [1.0, 2.0, 3.0, 4.0, 5.0]   # we can modify later

# ====== PARAMETER GRIDS (DEFAULTS) ======

# Entry clustering parameters
T_SECONDS_GRID = [10, 15, 20, 30, 40, 50]      # time window in seconds
K_CLUSTER_GRID = [3, 4, 5]                     # required unique traders in the window

# Time-based exit grid (in minutes)
EXIT_HOLD_MINUTES_GRID = [5, 10, 15, 20, 25]

# Price-based TP grid (XAUUSD dollars)
TP_DISTANCE_GRID = [1.0, 2.0, 3.0, 4.0, 5.0]

# ====== ENSURE DIRECTORIES EXIST ======

def ensure_directories() -> None:
    """
    Create all required directories if they don't exist.
    Call this once at the start of any main script.
    """
    for path in [
        DATA_DIR,
        INPUT_DIR,
        INTERMEDIATE_DIR,
        SIGNALS_DIR,
        BACKTESTS_DIR,
        OUTPUT_DIR,
        REPORTS_DIR,
        SUMMARY_REPORTS_DIR,
        SCENARIO_REPORTS_DIR,
        FIGURES_DIR,
        EQUITY_FIG_DIR,
        HEATMAP_FIG_DIR,
        DIST_FIG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
