"""
backtest.py
-----------

Backtest engine for cluster-based XAU orderflow signals.

Entry modes:
    - "prop"        : entry at prop trigger price/time (signal_time, trigger_open_rate)
    - "oanda_open"  : entry at OANDA candle open for the mapped candle
    - "oanda_close" : entry at OANDA candle close for the mapped candle

Risk framework:
    * Initial equity from config.INITIAL_EQUITY
    * Two risk modes (from config):
        - RISK_MODE = "static":
              Fixed absolute risk per trade from config.ABS_RISK_PER_TRADE (e.g. 200)
        - RISK_MODE = "percent_equity":
              risk_per_trade = RISK_PERCENT_PER_TRADE * CURRENT equity
    * sl_distance (in XAUUSD dollars) defines 1R:
          1R move = sl_distance in price.
      => PnL_R    = price_move / sl_distance
         PnL_cash = PnL_R * risk_per_trade   (per trade)

Stop styles:
    - "fixed"         : classic fixed SL at entry_mid +/- sl_distance
    - "atr_static"    : ATR-based SL, fixed after entry (init_mult * ATR_period)
    - "atr_trailing"  : ATR-based SL that trails via close +/- trail_mult * ATR
    - "chandelier"    : Chandelier-style SL (high/low extremes +/- trail_mult * ATR)

    For ATR modes we expect candles to have columns like "ATR_5", "ATR_14", etc.

Exit logic:
    - REAL SL (based on mid price, possibly ATR-based)
    - Optional TP (tp_R_multiple, based on sl_distance)
    - Optional time-based exit (hold_minutes):
        * target_time = entry_time + hold_minutes
        * if no SL/TP, exit at last candle in window (TIME or FORCED)

Signals input:
    - Must contain:
        * 'signal_time'
        * 'side' (BUY/SELL)          -> signal side
        * 'trigger_user_id'
        * 'trigger_open_rate'
        * 'entry_candle_index'       -> mapped candle index
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from config import (
    INITIAL_EQUITY,
    ABS_RISK_PER_TRADE,
    RISK_MODE,
    RISK_PERCENT_PER_TRADE,
)


@dataclass
class BacktestConfig:
    hold_minutes: int
    sl_distance: float                 # in price units (XAUUSD dollars) — 1R definition
    entry_mode: str = "prop"           # "prop", "oanda_open", "oanda_close"

    # Direction: "directional" = follow signal, "inverse" = fade
    direction_mode: str = "directional"

    # TP distance in R-multiples of sl_distance (e.g. 2.0 => TP at 2R, RR=1:2)
    tp_R_multiple: Optional[float] = None

    # Toggles
    use_tp_exit: bool = True           # if False or tp_R_multiple is None => no TP
    use_time_exit: bool = True         # if False => no "TIME" exit label, only SL/TP/FORCED

    # Spread modelling
    apply_spread: bool = False         # whether to model spread
    spread_dollars: float = 0.0        # spread size in dollars when apply_spread=True

    # Stop style
    stop_mode: str = "fixed"           # "fixed", "atr_static", "atr_trailing", "chandelier"

    # ATR params (for non-fixed stops)
    atr_period: Optional[int] = None          # e.g. 5, 14 -> uses column "ATR_5", "ATR_14"
    atr_init_mult: Optional[float] = None     # initial stop distance = atr_init_mult * ATR
    atr_trail_mult: Optional[float] = None    # trailing distance  = atr_trail_mult * ATR
    chandelier_lookback: Optional[int] = None # currently for metadata only (logic uses 1-bar extremes)

    # per-engine cap on concurrent open positions
    max_open_positions: Optional[int] = None  # e.g. 1, 2, 3; None = unlimited


def _compute_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with 'pnl_R' (and optionally 'equity'/'drawdown'),
    compute:

    - cumulative_R (always, from pnl_R)
    - equity, drawdown:

        * If 'equity'/'drawdown' already exist and are populated, we KEEP them
          (this is the case for dynamic risk per trade).
        * Otherwise, we reconstruct equity/drawdown assuming STATIC risk:
              equity = INITIAL_EQUITY + cumulative_R * ABS_RISK_PER_TRADE
    """
    if trades.empty:
        trades = trades.copy()
        trades["cumulative_R"] = []
        trades["equity"] = []
        trades["drawdown"] = []
        return trades

    trades = trades.sort_values("entry_time").reset_index(drop=True).copy()
    trades["cumulative_R"] = trades["pnl_R"].cumsum()

    # If equity already present and not all NaN, we assume it was
    # computed inside the main loop (dynamic or static) and we keep it.
    if "equity" in trades.columns and trades["equity"].notna().any():
        # Ensure drawdown exists; if missing, compute from equity.
        if "drawdown" not in trades.columns or trades["drawdown"].isna().all():
            roll_max = trades["equity"].cummax()
            trades["drawdown"] = trades["equity"] - roll_max
        return trades

    # Fallback: static-risk reconstruction from R
    trades["equity"] = INITIAL_EQUITY + trades["cumulative_R"] * ABS_RISK_PER_TRADE
    roll_max = trades["equity"].cummax()
    trades["drawdown"] = trades["equity"] - roll_max

    return trades


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


def _compute_summary_metrics(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute core performance metrics from a trades DataFrame containing:
    - 'pnl_R'
    - 'pnl_cash'
    - 'equity'
    - 'drawdown'
    - 'risk_per_trade' (cash)
    """
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate": np.nan,
            "avg_R": np.nan,
            "avg_win_R": np.nan,
            "avg_loss_R": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_cash": np.nan,
            "max_drawdown_R": np.nan,
            "final_equity": INITIAL_EQUITY,
            "total_R": 0.0,
            "tp_rate": np.nan,
            "sl_rate": np.nan,
            "percent_return": 0.0,
            "gross_profit_cash": 0.0,
            "gross_loss_cash": 0.0,
            "longest_win_streak": 0,
            "longest_loss_streak": 0,
            "std_R": np.nan,
            "median_R": np.nan,
        }

    pnl_R = trades["pnl_R"]
    pnl_cash = trades["pnl_cash"]

    wins = pnl_R[pnl_R > 0]
    losses = pnl_R[pnl_R < 0]

    n_trades = len(pnl_R)
    n_wins = len(wins)
    n_losses = len(losses)

    win_rate = n_wins / n_trades if n_trades > 0 else np.nan
    avg_R = pnl_R.mean()
    avg_win_R = wins.mean() if n_wins > 0 else np.nan
    avg_loss_R = losses.mean() if n_losses > 0 else np.nan

    gross_profit_cash = pnl_cash[pnl_cash > 0].sum()
    gross_loss_cash = pnl_cash[pnl_cash < 0].sum()  # negative number
    gross_loss_abs = -gross_loss_cash

    profit_factor = (gross_profit_cash / gross_loss_abs) if gross_loss_abs > 0 else np.nan

    final_equity = trades["equity"].iloc[-1]
    total_R = pnl_R.sum()

    max_dd_cash = trades["drawdown"].min()  # <= 0

    # For static mode, R = cash / ABS_RISK_PER_TRADE (old behaviour).
    # For percent_equity mode, approximate R using average risk_per_trade.
    if RISK_MODE == "percent_equity":
        if "risk_per_trade" in trades.columns and trades["risk_per_trade"].notna().any():
            avg_risk_cash = trades["risk_per_trade"].abs().mean()
            max_dd_R = max_dd_cash / avg_risk_cash if avg_risk_cash > 0 else np.nan
        else:
            max_dd_R = np.nan
    else:
        max_dd_R = max_dd_cash / ABS_RISK_PER_TRADE if ABS_RISK_PER_TRADE > 0 else np.nan

    tp_rate = float((trades["exit_reason"] == "TP").mean()) if n_trades > 0 else np.nan
    sl_rate = float((trades["exit_reason"] == "SL").mean()) if n_trades > 0 else np.nan

    percent_return = (final_equity / INITIAL_EQUITY - 1.0) * 100.0

    win_mask = pnl_R > 0
    loss_mask = pnl_R < 0
    longest_win_streak = _longest_streak(win_mask)
    longest_loss_streak = _longest_streak(loss_mask)

    std_R = pnl_R.std(ddof=0)
    median_R = pnl_R.median()

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_R": avg_R,
        "avg_win_R": avg_win_R,
        "avg_loss_R": avg_loss_R,
        "profit_factor": profit_factor,
        "max_drawdown_cash": max_dd_cash,
        "max_drawdown_R": max_dd_R,
        "final_equity": final_equity,
        "total_R": total_R,
        "tp_rate": tp_rate,
        "sl_rate": sl_rate,
        "percent_return": percent_return,
        "gross_profit_cash": gross_profit_cash,
        "gross_loss_cash": gross_loss_cash,
        "longest_win_streak": longest_win_streak,
        "longest_loss_streak": longest_loss_streak,
        "std_R": std_R,
        "median_R": median_R,
    }


def backtest_time_exit(
    signals_with_entries: pd.DataFrame,
    candles: pd.DataFrame,
    cfg: BacktestConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Backtest with:
        - REAL SL (fixed or ATR/chandelier based)
        - optional TP (tp_R_multiple, based on sl_distance)
        - optional time-based exit (hold_minutes)

    Execution modelling:
        - Candles / trigger prices are treated as MID.
        - If cfg.apply_spread == True:
            * BUY:  entry at mid + spread/2, exit at mid - spread/2
            * SELL: entry at mid - spread/2, exit at mid + spread/2
        - If cfg.apply_spread == False:
            * entry_price == entry_mid
            * exit_price  == exit_mid
    """
    if signals_with_entries.empty:
        return pd.DataFrame(), _compute_summary_metrics(pd.DataFrame())

    if cfg.hold_minutes <= 0:
        raise ValueError("hold_minutes must be positive.")
    if cfg.sl_distance <= 0:
        raise ValueError("sl_distance must be positive (defines 1R).")
    if cfg.entry_mode not in ("prop", "oanda_open", "oanda_close"):
        raise ValueError("entry_mode must be one of: 'prop', 'oanda_open', 'oanda_close'.")
    if cfg.direction_mode not in ("directional", "inverse"):
        raise ValueError("direction_mode must be 'directional' or 'inverse'.")
    if cfg.apply_spread and cfg.spread_dollars <= 0:
        raise ValueError("spread_dollars must be positive when apply_spread=True.")
    if cfg.stop_mode not in ("fixed", "atr_static", "atr_trailing", "chandelier"):
        raise ValueError("stop_mode must be one of: 'fixed', 'atr_static', 'atr_trailing', 'chandelier'.")

        # sanity check for max_open_positions
        if cfg.max_open_positions is not None and cfg.max_open_positions <= 0:
            raise ValueError("max_open_positions must be a positive integer or None.")

    # Effective TP usage
    use_tp_exit = cfg.use_tp_exit and (cfg.tp_R_multiple is not None) and (cfg.tp_R_multiple > 0)
    tp_R_multiple = cfg.tp_R_multiple if use_tp_exit else None
    tp_distance = cfg.sl_distance * tp_R_multiple if use_tp_exit else None

    # Spread (half-spread used for exec modelling)
    half_spread = cfg.spread_dollars / 2.0 if cfg.apply_spread else 0.0

    # Sort candles by time and pull arrays for fast scanning
    candles = candles.sort_values("open_time").reset_index(drop=True).copy()
    times = candles["open_time"].to_numpy()
    opens = candles["open"].to_numpy()
    highs = candles["high"].to_numpy()
    lows = candles["low"].to_numpy()
    closes = candles["close"].to_numpy()
    idxes = candles["candle_index"].to_numpy()

    # ATR array (if needed)
    atrs = None
    if cfg.stop_mode in ("atr_static", "atr_trailing", "chandelier"):
        if cfg.atr_period is None or cfg.atr_init_mult is None:
            raise ValueError("ATR-based stop_mode requires atr_period and atr_init_mult.")
        atr_col = f"ATR_{cfg.atr_period}"
        if atr_col not in candles.columns:
            raise ValueError(f"ATR column '{atr_col}' not found in candles.")
        atrs = candles[atr_col].to_numpy()

    # Map candle_index -> position in arrays
    pos_by_candle_index = {int(idxes[i]): i for i in range(len(idxes))}

    rows = []

    # Equity + drawdown tracking for risk sizing
    equity = INITIAL_EQUITY
    running_max_equity = INITIAL_EQUITY

    # track currently open trades by their exit_time
    max_open = cfg.max_open_positions
    open_trades_exit_times = []  # list of datetimes

    for _, sig in signals_with_entries.iterrows():
        signal_side_raw = str(sig["side"]).upper()  # original signal side
        signal_time = sig["signal_time"]
        trigger_price = float(sig["trigger_open_rate"])
        entry_candle_index = int(sig["entry_candle_index"])

        if entry_candle_index not in pos_by_candle_index:
            # Shouldn't happen, but guard anyway
            continue

        candle_pos = pos_by_candle_index[entry_candle_index]
        candle_open_time = times[candle_pos]
        candle_open = opens[candle_pos]
        candle_close = closes[candle_pos]

        # ----- TRADE SIDE: directional vs inverse -----
        if cfg.direction_mode == "directional":
            trade_side = signal_side_raw
        else:  # "inverse"
            if signal_side_raw == "BUY":
                trade_side = "SELL"
            elif signal_side_raw == "SELL":
                trade_side = "BUY"
            else:
                # unexpected value; skip
                continue

        # ----- ENTRY LOGIC BY MODE (MID price) -----
        if cfg.entry_mode == "prop":
            entry_time = signal_time
            entry_mid = trigger_price            # treat trigger price as MID for risk logic
        elif cfg.entry_mode == "oanda_open":
            entry_time = candle_open_time
            entry_mid = float(candle_open)
        else:  # "oanda_close"
            # Enter at the end of the mapped candle
            entry_time = candle_open_time + timedelta(minutes=1)
            entry_mid = float(candle_close)

        if trade_side == "BUY":
            direction = 1.0
        else:
            direction = -1.0

        # ----- per-engine cap on concurrent open positions -----
        if max_open is not None:
            # prune any trades that have already closed by this entry_time
            open_trades_exit_times = [et for et in open_trades_exit_times if et > entry_time]

            if len(open_trades_exit_times) >= max_open:
                # Cap reached at the moment of this trade's entry_time -> skip this signal
                # (No risk, no PnL, doesn't affect equity)
                continue


        # ----- INITIAL SL (MID) BY STOP MODE -----
        if cfg.stop_mode == "fixed":
            if trade_side == "BUY":
                sl_mid = entry_mid - cfg.sl_distance
            else:
                sl_mid = entry_mid + cfg.sl_distance
        else:
            # ATR-based variants
            # Find entry bar index (floor)
            entry_bar_pos = np.searchsorted(times, entry_time, side="right") - 1
            if entry_bar_pos < 0:
                entry_bar_pos = 0
            if atrs is None:
                continue
            atr_entry = atrs[entry_bar_pos]
            if np.isnan(atr_entry) or atr_entry <= 0:
                # Skip trades where ATR is not yet defined
                continue
            init_stop_dist = cfg.atr_init_mult * atr_entry
            if trade_side == "BUY":
                sl_mid = entry_mid - init_stop_dist
            else:
                sl_mid = entry_mid + init_stop_dist

        # ----- TP (MID) -----
        if use_tp_exit:
            if trade_side == "BUY":
                tp_mid = entry_mid + tp_distance
            else:
                tp_mid = entry_mid - tp_distance
        else:
            tp_mid = np.nan

        # ----- Holding window based on entry_time -----
        target_time = entry_time + timedelta(minutes=cfg.hold_minutes)
        truncated = False

        # Find first candle that starts at or before entry_time (floor)
        start_pos = np.searchsorted(times, entry_time, side="right") - 1
        if start_pos < 0:
            start_pos = 0
            truncated = True

        # Find last candle whose open_time <= target_time (floor)
        last_pos = np.searchsorted(times, target_time, side="right") - 1
        if last_pos >= len(times):
            last_pos = len(times) - 1
            truncated = True

        if last_pos < start_pos:
            last_pos = start_pos
            truncated = True

        # Default exit (if nothing else hits): TIME or FORCED, at MID close
        exit_pos = last_pos
        exit_mid = closes[last_pos]
        exit_reason = "TIME" if cfg.use_time_exit else "FORCED"

        # ----- Scan candles from entry to end-of-hold for SL/TP hits (MID-based) -----
        for pos in range(start_pos, last_pos + 1):
            hi = highs[pos]
            lo = lows[pos]

            # 1) SL check (risk first, on MID levels)
            if trade_side == "BUY":
                if lo <= sl_mid:
                    exit_reason = "SL"
                    exit_pos = pos
                    exit_mid = sl_mid
                    break
            else:  # SELL
                if hi >= sl_mid:
                    exit_reason = "SL"
                    exit_pos = pos
                    exit_mid = sl_mid
                    break

            # 2) TP check (MID)
            if use_tp_exit and not np.isnan(tp_mid):
                if trade_side == "BUY":
                    if hi >= tp_mid:
                        exit_reason = "TP"
                        exit_pos = pos
                        exit_mid = tp_mid
                        break
                else:  # SELL
                    if lo <= tp_mid:
                        exit_reason = "TP"
                        exit_pos = pos
                        exit_mid = tp_mid
                        break

            # 3) Trailing logic AFTER SL/TP checks (for next bar)
            if cfg.stop_mode in ("atr_trailing", "chandelier") and atrs is not None:
                atr_now = atrs[pos]
                if np.isnan(atr_now) or atr_now <= 0:
                    continue

                trail_mult = cfg.atr_trail_mult if cfg.atr_trail_mult is not None else cfg.atr_init_mult

                if trade_side == "BUY":
                    if cfg.stop_mode == "atr_trailing":
                        candidate = closes[pos] - trail_mult * atr_now
                    else:  # chandelier: use bar high
                        candidate = hi - trail_mult * atr_now
                    sl_mid = max(sl_mid, candidate)
                else:
                    if cfg.stop_mode == "atr_trailing":
                        candidate = closes[pos] + trail_mult * atr_now
                    else:  # chandelier: use bar low
                        candidate = lo + trail_mult * atr_now
                    sl_mid = min(sl_mid, candidate)

        exit_time = times[exit_pos]
        exit_candle_index = int(idxes[exit_pos])

        # register this trade as open from entry_time until exit_time
        if max_open is not None:
            open_trades_exit_times.append(exit_time)


        # ----- Map MID prices to EXEC prices (including spread if enabled) -----
        if trade_side == "BUY":
            entry_price = entry_mid + half_spread
            exit_price = exit_mid - half_spread
        else:  # SELL
            entry_price = entry_mid - half_spread
            exit_price = exit_mid + half_spread

        # ----- PnL in R -----
        price_move = (exit_price - entry_price) * direction
        pnl_R = price_move / cfg.sl_distance

        # ----- RISK PER TRADE (cash) & PnL_cash / Equity / Drawdown -----
        if RISK_MODE == "percent_equity":
            risk_cash = equity * RISK_PERCENT_PER_TRADE
        else:
            risk_cash = ABS_RISK_PER_TRADE

        pnl_cash = pnl_R * risk_cash
        equity = equity + pnl_cash
        running_max_equity = max(running_max_equity, equity)
        drawdown = equity - running_max_equity  # <= 0

        # Human-readable note about what was used
        if cfg.apply_spread:
            sl_note = f"{cfg.stop_mode} stop + TP/time + spread"
        else:
            sl_note = f"{cfg.stop_mode} stop + TP/time"

        row = {
            "signal_time": signal_time,
            "signal_side": signal_side_raw,          # original cluster side
            "trade_side": trade_side,                # actual executed side
            "side": trade_side,                      # for backwards compatibility
            "trigger_user_id": sig["trigger_user_id"],
            "trigger_open_rate": trigger_price,
            "entry_mode": cfg.entry_mode,
            "direction_mode": cfg.direction_mode,
            "tp_R_multiple": tp_R_multiple,
            "use_tp_exit": use_tp_exit,
            "use_time_exit": cfg.use_time_exit,
            "apply_spread": cfg.apply_spread,
            "spread_dollars": cfg.spread_dollars,
            "stop_mode": cfg.stop_mode,
            "atr_period": cfg.atr_period,
            "atr_init_mult": cfg.atr_init_mult,
            "atr_trail_mult": cfg.atr_trail_mult,
            "chandelier_lookback": cfg.chandelier_lookback,
            # Times & prices
            "entry_time": entry_time,
            "entry_hour": entry_time.hour,
            "entry_mid": entry_mid,
            "entry_price": entry_price,
            "sl_price": sl_mid,          # final SL level used (MID)
            "tp_price": tp_mid,          # MID TP level (or NaN)
            "exit_time": exit_time,
            "exit_hour": exit_time.hour,
            "exit_mid": exit_mid,
            "exit_price": exit_price,
            # Indices & config
            "entry_candle_index": entry_candle_index,
            "exit_candle_index": exit_candle_index,
            "hold_minutes": cfg.hold_minutes,
            "sl_distance": cfg.sl_distance,
            "truncated_exit": truncated,
            "exit_reason": exit_reason,  # "SL", "TP", "TIME", "FORCED"
            # Risk & PnL
            "price_move": price_move,
            "pnl_R": pnl_R,
            "risk_per_trade": risk_cash,
            "pnl_cash": pnl_cash,
            "equity": equity,
            "drawdown": drawdown,
            "stop_loss_note": sl_note,
        }
        rows.append(row)

    trades = pd.DataFrame(rows)

    # Equity curve & drawdown (will keep existing equity/drawdown, only add cumulative_R)
    trades = _compute_equity_curve(trades)

    # Summary
    summary = _compute_summary_metrics(trades)

    return trades, summary
