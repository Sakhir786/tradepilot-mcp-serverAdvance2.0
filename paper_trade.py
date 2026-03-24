"""
TradePilot Paper Trading Backtester
=====================================
Tests the 18-layer engine against historical stock data.

How it works:
1. Fetches 2 years of daily OHLCV data for a stock
2. Slides a window through the data (needs 200+ bars minimum)
3. At each checkpoint, runs the 18-layer engine on available data
4. When a signal fires, checks forward bars to see if target or stop was hit
5. Tracks all trades: wins, losses, P&L, accuracy

Usage:
    python paper_trade.py AAPL              # Default swing mode, 10 checkpoints
    python paper_trade.py SPY --mode swing --checkpoints 20
    python paper_trade.py TSLA --mode scalp --checkpoints 5
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tradepilot_integration'))

from polygon_client import get_candles_for_mode, get_market_context
from tradepilot_integration.engine_18layer_core import (
    TradePilotEngine18Layer,
    TradeMode,
    SignalStrength,
    FullAnalysisResult
)


@dataclass
class PaperTrade:
    """A single simulated trade"""
    trade_id: int
    symbol: str
    signal_date: str
    direction: str          # BULLISH / BEARISH
    action: str             # BUY_CALL / BUY_PUT / FLAT
    confidence: str         # SUPREME / EXCELLENT / STRONG / MODERATE / WEAK
    win_probability: float
    entry_price: float
    target_price: float
    stop_price: float
    risk_reward: float

    # Outcome
    outcome: str = "PENDING"  # WIN / LOSS / TIMEOUT / SKIPPED
    exit_price: float = 0.0
    exit_date: str = ""
    bars_held: int = 0
    pnl_pct: float = 0.0
    hit_target: bool = False
    hit_stop: bool = False


@dataclass
class BacktestResult:
    """Overall backtest summary"""
    symbol: str
    mode: str
    total_signals: int
    trades_taken: int
    trades_skipped: int     # NO_TRADE / WEAK / FLAT signals

    wins: int
    losses: int
    timeouts: int           # Neither target nor stop hit
    win_rate: float
    avg_pnl_pct: float
    total_pnl_pct: float

    best_trade_pnl: float
    worst_trade_pnl: float
    avg_bars_held: float
    avg_risk_reward: float

    # Breakdown by confidence
    by_confidence: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    trades: List[Dict] = field(default_factory=list)
    run_time_seconds: float = 0.0


def simulate_trade_outcome(
    trade: PaperTrade,
    forward_bars: List[Dict],
    max_hold_bars: int = 30
) -> PaperTrade:
    """
    Check if a trade would have hit target or stop using forward price bars.

    Args:
        trade: The paper trade with entry/target/stop
        forward_bars: OHLCV bars AFTER the signal date
        max_hold_bars: Max bars to hold before timeout
    """
    if not forward_bars or trade.action == "FLAT":
        trade.outcome = "SKIPPED"
        return trade

    is_long = trade.direction == "BULLISH"

    for i, bar in enumerate(forward_bars[:max_hold_bars]):
        high = bar.get("h", bar.get("high", 0))
        low = bar.get("l", bar.get("low", 0))
        close = bar.get("c", bar.get("close", 0))

        if is_long:
            # Long trade: target is above entry, stop is below
            if high >= trade.target_price:
                trade.outcome = "WIN"
                trade.hit_target = True
                trade.exit_price = trade.target_price
                trade.pnl_pct = ((trade.target_price - trade.entry_price) / trade.entry_price) * 100
                trade.bars_held = i + 1
                trade.exit_date = _bar_date(bar)
                return trade
            if low <= trade.stop_price:
                trade.outcome = "LOSS"
                trade.hit_stop = True
                trade.exit_price = trade.stop_price
                trade.pnl_pct = ((trade.stop_price - trade.entry_price) / trade.entry_price) * 100
                trade.bars_held = i + 1
                trade.exit_date = _bar_date(bar)
                return trade
        else:
            # Short/PUT trade: target is below entry, stop is above
            if low <= trade.target_price:
                trade.outcome = "WIN"
                trade.hit_target = True
                trade.exit_price = trade.target_price
                trade.pnl_pct = ((trade.entry_price - trade.target_price) / trade.entry_price) * 100
                trade.bars_held = i + 1
                trade.exit_date = _bar_date(bar)
                return trade
            if high >= trade.stop_price:
                trade.outcome = "LOSS"
                trade.hit_stop = True
                trade.exit_price = trade.stop_price
                trade.pnl_pct = ((trade.entry_price - trade.stop_price) / trade.entry_price) * 100 * -1
                trade.bars_held = i + 1
                trade.exit_date = _bar_date(bar)
                return trade

    # Timeout - close at last available bar's close
    if forward_bars:
        last_bar = forward_bars[min(max_hold_bars - 1, len(forward_bars) - 1)]
        last_close = last_bar.get("c", last_bar.get("close", trade.entry_price))
        trade.outcome = "TIMEOUT"
        trade.exit_price = last_close
        trade.bars_held = min(max_hold_bars, len(forward_bars))
        trade.exit_date = _bar_date(last_bar)
        if is_long:
            trade.pnl_pct = ((last_close - trade.entry_price) / trade.entry_price) * 100
        else:
            trade.pnl_pct = ((trade.entry_price - last_close) / trade.entry_price) * 100

    return trade


def _bar_date(bar: Dict) -> str:
    """Extract date string from a bar"""
    ts = bar.get("t", bar.get("timestamp", 0))
    if ts:
        try:
            return datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        except Exception:
            pass
    return ""


def run_backtest(
    symbol: str,
    mode: str = "swing",
    checkpoints: int = 10,
    max_hold_bars: int = 30,
    min_confidence: str = "MODERATE"
) -> BacktestResult:
    """
    Run paper trading backtest on a stock.

    Args:
        symbol: Stock ticker (e.g. AAPL, SPY)
        mode: Trading mode (swing, scalp, intraday, leaps)
        checkpoints: Number of historical points to test
        max_hold_bars: Max bars to hold a trade before timeout
        min_confidence: Minimum confidence to take a trade

    Returns:
        BacktestResult with all trades and statistics
    """
    start_time = time.time()
    symbol = symbol.upper()

    print(f"\n{'='*60}")
    print(f"  TRADEPILOT PAPER TRADING BACKTEST")
    print(f"  Symbol: {symbol} | Mode: {mode.upper()} | Checkpoints: {checkpoints}")
    print(f"{'='*60}\n")

    # 1. Fetch full historical data
    print(f"[1/4] Fetching historical data for {symbol}...")
    candles_data = get_candles_for_mode(symbol, mode=mode)

    if not candles_data or "results" not in candles_data:
        print(f"ERROR: No data returned for {symbol}")
        return BacktestResult(
            symbol=symbol, mode=mode, total_signals=0, trades_taken=0,
            trades_skipped=0, wins=0, losses=0, timeouts=0, win_rate=0,
            avg_pnl_pct=0, total_pnl_pct=0, best_trade_pnl=0,
            worst_trade_pnl=0, avg_bars_held=0, avg_risk_reward=0
        )

    all_bars = candles_data["results"]
    total_bars = len(all_bars)
    print(f"  Got {total_bars} bars")

    if total_bars < 250:
        print(f"ERROR: Need at least 250 bars, got {total_bars}")
        return BacktestResult(
            symbol=symbol, mode=mode, total_signals=0, trades_taken=0,
            trades_skipped=0, wins=0, losses=0, timeouts=0, win_rate=0,
            avg_pnl_pct=0, total_pnl_pct=0, best_trade_pnl=0,
            worst_trade_pnl=0, avg_bars_held=0, avg_risk_reward=0
        )

    # 2. Fetch market context once
    print(f"[2/4] Fetching market context (SPY + VIX)...")
    market_ctx = {}
    try:
        market_ctx = get_market_context(mode=mode)
        print(f"  Market bias: {market_ctx.get('market_bias', 'unknown')}")
    except Exception as e:
        print(f"  Warning: {e}")

    # 3. Initialize engine
    print(f"[3/4] Initializing 18-layer engine...")
    engine = TradePilotEngine18Layer()

    trade_mode = {
        "scalp": TradeMode.SCALP,
        "swing": TradeMode.SWING,
        "intraday": TradeMode.INTRADAY,
        "leaps": TradeMode.LEAPS
    }.get(mode, TradeMode.SWING)

    # Confidence hierarchy for filtering
    confidence_levels = ["SUPREME", "EXCELLENT", "STRONG", "MODERATE", "WEAK", "NO_TRADE"]
    min_conf_idx = confidence_levels.index(min_confidence) if min_confidence in confidence_levels else 3

    # 4. Slide through historical data
    print(f"[4/4] Running backtest across {checkpoints} checkpoints...\n")

    # Reserve last max_hold_bars for forward testing
    usable_bars = total_bars - max_hold_bars
    min_window = 200  # Engine needs at least 200 bars

    if usable_bars <= min_window:
        print(f"ERROR: Not enough bars for backtesting. Need {min_window + max_hold_bars}, have {total_bars}")
        return BacktestResult(
            symbol=symbol, mode=mode, total_signals=0, trades_taken=0,
            trades_skipped=0, wins=0, losses=0, timeouts=0, win_rate=0,
            avg_pnl_pct=0, total_pnl_pct=0, best_trade_pnl=0,
            worst_trade_pnl=0, avg_bars_held=0, avg_risk_reward=0
        )

    # Calculate checkpoint positions (evenly spaced from min_window to usable_bars)
    step = max(1, (usable_bars - min_window) // max(1, checkpoints - 1))
    checkpoint_positions = list(range(min_window, usable_bars, step))[:checkpoints]

    trades = []
    trade_id = 0

    for cp_num, bar_idx in enumerate(checkpoint_positions, 1):
        # Create a "snapshot" of data up to this point
        window_data = {
            "results": all_bars[:bar_idx],
            "resultsCount": bar_idx,
            "_mode_config": candles_data.get("_mode_config", {})
        }

        # Forward bars for outcome testing
        forward_bars = all_bars[bar_idx:bar_idx + max_hold_bars]

        current_bar = all_bars[bar_idx - 1]
        current_price = current_bar.get("c", current_bar.get("close", 0))
        signal_date = _bar_date(current_bar)

        print(f"  Checkpoint {cp_num}/{len(checkpoint_positions)} | Bar {bar_idx}/{total_bars} | Date: {signal_date} | Price: ${current_price:.2f}")

        try:
            result = engine.analyze(
                ticker=symbol,
                candles_data=window_data,
                options_data=None,
                mode=trade_mode,
                market_context=market_ctx
            )

            conf_str = result.confidence.value if hasattr(result.confidence, 'value') else str(result.confidence)
            conf_idx = confidence_levels.index(conf_str) if conf_str in confidence_levels else 5

            trade_id += 1
            trade = PaperTrade(
                trade_id=trade_id,
                symbol=symbol,
                signal_date=signal_date,
                direction=result.direction,
                action=result.action,
                confidence=conf_str,
                win_probability=result.win_probability,
                entry_price=result.entry_price if result.entry_price > 0 else current_price,
                target_price=result.target_price,
                stop_price=result.stop_price,
                risk_reward=result.risk_reward
            )

            # Skip weak signals
            if conf_idx > min_conf_idx or result.action == "FLAT" or not result.trade_valid:
                trade.outcome = "SKIPPED"
                trades.append(trade)
                print(f"    -> SKIP: {result.action} | {conf_str} | Not enough confidence")
                continue

            # Validate entry/target/stop
            if trade.target_price <= 0 or trade.stop_price <= 0:
                trade.outcome = "SKIPPED"
                trades.append(trade)
                print(f"    -> SKIP: Invalid target/stop levels")
                continue

            # Simulate the trade
            trade = simulate_trade_outcome(trade, forward_bars, max_hold_bars)
            trades.append(trade)

            icon = "WIN" if trade.outcome == "WIN" else "LOSS" if trade.outcome == "LOSS" else "TIMEOUT"
            print(f"    -> {result.action} | {conf_str} ({result.win_probability:.0f}%)")
            print(f"       Entry: ${trade.entry_price:.2f} | Target: ${trade.target_price:.2f} | "
                  f"Stop: ${trade.stop_price:.2f} | R:R {trade.risk_reward:.1f}")
            print(f"       Result: {icon} | Exit: ${trade.exit_price:.2f} | "
                  f"P&L: {trade.pnl_pct:+.2f}% | Held: {trade.bars_held} bars")

        except Exception as e:
            print(f"    -> ERROR: {str(e)[:80]}")
            continue

    # Calculate statistics
    taken_trades = [t for t in trades if t.outcome != "SKIPPED"]
    skipped_trades = [t for t in trades if t.outcome == "SKIPPED"]
    wins = [t for t in taken_trades if t.outcome == "WIN"]
    losses = [t for t in taken_trades if t.outcome == "LOSS"]
    timeouts = [t for t in taken_trades if t.outcome == "TIMEOUT"]

    win_rate = (len(wins) / len(taken_trades) * 100) if taken_trades else 0
    pnls = [t.pnl_pct for t in taken_trades]
    avg_pnl = sum(pnls) / len(pnls) if pnls else 0
    total_pnl = sum(pnls)

    # By confidence breakdown
    by_conf = {}
    for level in confidence_levels[:5]:
        level_trades = [t for t in taken_trades if t.confidence == level]
        level_wins = [t for t in level_trades if t.outcome == "WIN"]
        if level_trades:
            by_conf[level] = {
                "trades": len(level_trades),
                "wins": len(level_wins),
                "win_rate": round(len(level_wins) / len(level_trades) * 100, 1),
                "avg_pnl": round(sum(t.pnl_pct for t in level_trades) / len(level_trades), 2)
            }

    elapsed = time.time() - start_time

    result = BacktestResult(
        symbol=symbol,
        mode=mode,
        total_signals=len(trades),
        trades_taken=len(taken_trades),
        trades_skipped=len(skipped_trades),
        wins=len(wins),
        losses=len(losses),
        timeouts=len(timeouts),
        win_rate=round(win_rate, 1),
        avg_pnl_pct=round(avg_pnl, 2),
        total_pnl_pct=round(total_pnl, 2),
        best_trade_pnl=round(max(pnls), 2) if pnls else 0,
        worst_trade_pnl=round(min(pnls), 2) if pnls else 0,
        avg_bars_held=round(sum(t.bars_held for t in taken_trades) / len(taken_trades), 1) if taken_trades else 0,
        avg_risk_reward=round(sum(t.risk_reward for t in taken_trades) / len(taken_trades), 2) if taken_trades else 0,
        by_confidence=by_conf,
        trades=[asdict(t) for t in trades],
        run_time_seconds=round(elapsed, 1)
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS: {symbol} ({mode.upper()})")
    print(f"{'='*60}")
    print(f"  Total Signals:    {result.total_signals}")
    print(f"  Trades Taken:     {result.trades_taken}")
    print(f"  Trades Skipped:   {result.trades_skipped}")
    print(f"  ---")
    print(f"  Wins:             {result.wins}")
    print(f"  Losses:           {result.losses}")
    print(f"  Timeouts:         {result.timeouts}")
    print(f"  WIN RATE:         {result.win_rate}%")
    print(f"  ---")
    print(f"  Avg P&L/Trade:    {result.avg_pnl_pct:+.2f}%")
    print(f"  Total P&L:        {result.total_pnl_pct:+.2f}%")
    print(f"  Best Trade:       {result.best_trade_pnl:+.2f}%")
    print(f"  Worst Trade:      {result.worst_trade_pnl:+.2f}%")
    print(f"  Avg Bars Held:    {result.avg_bars_held}")
    print(f"  Avg Risk/Reward:  {result.avg_risk_reward}")
    print(f"  ---")
    print(f"  Run Time:         {result.run_time_seconds}s")

    if by_conf:
        print(f"\n  BY CONFIDENCE LEVEL:")
        for level, stats in by_conf.items():
            print(f"    {level:12s}: {stats['trades']} trades | "
                  f"{stats['win_rate']}% win rate | {stats['avg_pnl']:+.2f}% avg P&L")

    print(f"{'='*60}\n")

    # Save results to file
    output_file = f"backtest_{symbol}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)
    print(f"  Results saved to: {output_file}")

    return result


def main():
    parser = argparse.ArgumentParser(description="TradePilot Paper Trading Backtester")
    parser.add_argument("symbol", help="Stock ticker (e.g. AAPL, SPY, TSLA)")
    parser.add_argument("--mode", default="swing", choices=["scalp", "swing", "intraday", "leaps"],
                        help="Trading mode (default: swing)")
    parser.add_argument("--checkpoints", type=int, default=10,
                        help="Number of historical points to test (default: 10)")
    parser.add_argument("--max-hold", type=int, default=30,
                        help="Max bars to hold a trade (default: 30)")
    parser.add_argument("--min-confidence", default="MODERATE",
                        choices=["SUPREME", "EXCELLENT", "STRONG", "MODERATE", "WEAK"],
                        help="Minimum confidence to take trade (default: MODERATE)")

    args = parser.parse_args()
    run_backtest(
        symbol=args.symbol,
        mode=args.mode,
        checkpoints=args.checkpoints,
        max_hold_bars=args.max_hold,
        min_confidence=args.min_confidence
    )


if __name__ == "__main__":
    main()
