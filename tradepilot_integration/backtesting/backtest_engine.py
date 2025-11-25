"""
TradePilot Backtesting Engine
==============================
Backtest playbooks and strategies against historical data.

Features:
- Individual playbook backtesting
- Walk-forward analysis
- Performance metrics (win rate, Sharpe, max DD)
- Equity curve generation
- Trade journal export

Author: TradePilot Integration
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import pandas as pd
import numpy as np


class BacktestMode(Enum):
    """Backtesting modes"""
    FULL = "FULL"              # Full historical period
    WALK_FORWARD = "WALK_FORWARD"  # Rolling window
    MONTE_CARLO = "MONTE_CARLO"    # Randomized simulation


@dataclass
class BacktestTrade:
    """Individual backtest trade"""
    trade_id: int
    ticker: str
    entry_date: str
    entry_price: float
    direction: str
    action: str
    playbook: str
    playbook_id: int
    confidence: str
    win_probability: float
    
    # Option details
    strike: float
    delta: float
    expiry_dte: int
    option_entry_price: float
    
    # Exit
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    option_exit_price: Optional[float] = None
    exit_reason: str = ""
    
    # Results
    pnl: float = 0.0
    pnl_pct: float = 0.0
    is_winner: bool = False
    
    # Risk metrics
    risk_amount: float = 0.0
    target_price: float = 0.0
    stop_price: float = 0.0
    risk_reward: float = 0.0
    
    # Duration
    hold_days: int = 0


@dataclass
class BacktestResults:
    """Complete backtest results"""
    # Configuration
    ticker: str
    start_date: str
    end_date: str
    mode: BacktestMode
    initial_capital: float
    
    # Summary
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # PnL
    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade stats
    avg_hold_days: float
    avg_risk_reward: float
    expectancy: float
    
    # Equity curve
    equity_curve: List[Dict[str, Any]]
    
    # Individual trades
    trades: List[BacktestTrade]
    
    # By playbook
    playbook_performance: Dict[str, Dict[str, Any]]


class TradePilotBacktester:
    """
    Backtest TradePilot playbooks and strategies
    
    Usage:
        backtester = TradePilotBacktester(initial_capital=100000)
        results = backtester.backtest(
            ticker="SPY",
            candles_data=historical_data,
            start_date="2023-01-01",
            end_date="2024-01-01"
        )
        print(backtester.get_summary(results))
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 position_size_pct: float = 0.10,
                 commission_per_contract: float = 0.65):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            position_size_pct: Position size as % of capital
            commission_per_contract: Commission per option contract
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.commission = commission_per_contract
        
        # Engine reference (lazy loaded)
        self._engine = None
        
        # Results storage
        self._trades: List[BacktestTrade] = []
        self._equity_curve: List[Dict] = []
    
    @property
    def engine(self):
        """Lazy load engine"""
        if self._engine is None:
            try:
                from integrations.engine_18layer_core import TradePilotEngine18Layer
                self._engine = TradePilotEngine18Layer()
            except ImportError:
                raise RuntimeError("Engine not available for backtesting")
        return self._engine
    
    def backtest(self,
                 ticker: str,
                 candles_data: Dict,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 mode: BacktestMode = BacktestMode.FULL,
                 playbook_filter: Optional[List[int]] = None) -> BacktestResults:
        """
        Run backtest on historical data
        
        Args:
            ticker: Stock symbol
            candles_data: Historical OHLCV data from Polygon.io
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            mode: Backtesting mode
            playbook_filter: Only test specific playbook IDs
            
        Returns:
            BacktestResults with full analysis
        """
        self._trades = []
        self._equity_curve = []
        
        # Prepare data
        df = self._prepare_dataframe(candles_data)
        if df is None:
            raise ValueError("Invalid candle data")
        
        # Filter date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        if len(df) < 200:
            raise ValueError("Insufficient data for backtesting")
        
        # Run backtest
        if mode == BacktestMode.FULL:
            self._run_full_backtest(ticker, df, playbook_filter)
        elif mode == BacktestMode.WALK_FORWARD:
            self._run_walk_forward(ticker, df, playbook_filter)
        else:
            self._run_full_backtest(ticker, df, playbook_filter)
        
        # Calculate results
        return self._calculate_results(ticker, df, mode)
    
    def _prepare_dataframe(self, candles_data: Dict) -> Optional[pd.DataFrame]:
        """Convert candles to DataFrame"""
        try:
            if not candles_data or "results" not in candles_data:
                return None
            
            df = pd.DataFrame(candles_data["results"])
            
            column_mapping = {
                "o": "open", "h": "high", "l": "low",
                "c": "close", "v": "volume", "t": "timestamp"
            }
            df = df.rename(columns=column_mapping)
            
            if "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("datetime")
            
            return df.sort_index()
            
        except Exception as e:
            print(f"[Backtest] DataFrame error: {e}")
            return None
    
    def _run_full_backtest(self, ticker: str, df: pd.DataFrame, 
                          playbook_filter: Optional[List[int]] = None):
        """Run full period backtest"""
        capital = self.initial_capital
        trade_id = 0
        
        # Need at least 200 bars for analysis
        lookback = 200
        
        for i in range(lookback, len(df)):
            # Get historical window for analysis
            window_data = {
                "results": df.iloc[i-lookback:i].reset_index().to_dict('records')
            }
            
            # Convert to Polygon format
            for r in window_data["results"]:
                r["t"] = int(r.get("datetime", datetime.now()).timestamp() * 1000) if isinstance(r.get("datetime"), datetime) else r.get("timestamp", 0)
                r["o"] = r.get("open", 0)
                r["h"] = r.get("high", 0)
                r["l"] = r.get("low", 0)
                r["c"] = r.get("close", 0)
                r["v"] = r.get("volume", 0)
            
            try:
                # Run analysis
                from integrations.engine_18layer_core import TradeMode
                result = self.engine.analyze(
                    ticker=ticker,
                    candles_data=window_data,
                    mode=TradeMode.SWING
                )
                
                # Check for valid trade
                if not result.trade_valid:
                    continue
                
                # Filter playbooks if specified
                if playbook_filter and result.playbook_id not in playbook_filter:
                    continue
                
                # Get current bar data
                current_bar = df.iloc[i]
                entry_date = str(current_bar.name)
                entry_price = float(current_bar['close'])
                
                # Simulate option price (simplified - use actual delta)
                option_entry = self._estimate_option_price(
                    entry_price, result.strike, result.delta, result.expiry_dte
                )
                
                # Calculate position size
                contracts = int((capital * self.position_size_pct) / (option_entry * 100))
                if contracts < 1:
                    continue
                
                # Create trade
                trade = BacktestTrade(
                    trade_id=trade_id,
                    ticker=ticker,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    direction=result.direction,
                    action=result.action,
                    playbook=result.matched_playbook or "Unknown",
                    playbook_id=result.playbook_id or 0,
                    confidence=result.confidence.value,
                    win_probability=result.win_probability,
                    strike=result.strike,
                    delta=result.delta,
                    expiry_dte=result.expiry_dte,
                    option_entry_price=option_entry,
                    target_price=result.target_price,
                    stop_price=result.stop_price,
                    risk_reward=result.risk_reward,
                    risk_amount=contracts * option_entry * 100
                )
                
                # Simulate trade outcome
                self._simulate_trade(trade, df.iloc[i:min(i+result.expiry_dte+1, len(df))])
                
                # Update capital
                capital += trade.pnl - (self.commission * contracts * 2)  # Entry + exit
                
                self._trades.append(trade)
                self._equity_curve.append({
                    "date": entry_date,
                    "capital": capital,
                    "trade_id": trade_id
                })
                
                trade_id += 1
                
            except Exception as e:
                print(f"[Backtest] Error at {i}: {e}")
                continue
    
    def _run_walk_forward(self, ticker: str, df: pd.DataFrame,
                         playbook_filter: Optional[List[int]] = None,
                         train_pct: float = 0.7):
        """Run walk-forward optimization"""
        # Split into train/test windows
        window_size = int(len(df) * 0.2)  # 20% windows
        step_size = int(window_size * 0.5)  # 50% overlap
        
        for start in range(0, len(df) - window_size, step_size):
            window_df = df.iloc[start:start + window_size]
            window_data = {
                "results": window_df.reset_index().to_dict('records')
            }
            
            # Process this window
            # (Simplified - full implementation would optimize parameters)
            pass
    
    def _simulate_trade(self, trade: BacktestTrade, future_bars: pd.DataFrame):
        """Simulate trade outcome using future price data"""
        if len(future_bars) == 0:
            trade.exit_reason = "No data"
            return
        
        is_bullish = trade.direction == "BULLISH"
        
        for i, (date, bar) in enumerate(future_bars.iterrows()):
            if i == 0:
                continue  # Skip entry bar
            
            high = float(bar['high'])
            low = float(bar['low'])
            close = float(bar['close'])
            
            # Check target hit
            if is_bullish:
                if high >= trade.target_price:
                    trade.exit_date = str(date)
                    trade.exit_price = trade.target_price
                    trade.exit_reason = "Target hit"
                    trade.is_winner = True
                    break
                elif low <= trade.stop_price:
                    trade.exit_date = str(date)
                    trade.exit_price = trade.stop_price
                    trade.exit_reason = "Stop hit"
                    trade.is_winner = False
                    break
            else:
                if low <= trade.target_price:
                    trade.exit_date = str(date)
                    trade.exit_price = trade.target_price
                    trade.exit_reason = "Target hit"
                    trade.is_winner = True
                    break
                elif high >= trade.stop_price:
                    trade.exit_date = str(date)
                    trade.exit_price = trade.stop_price
                    trade.exit_reason = "Stop hit"
                    trade.is_winner = False
                    break
        
        # If no exit, close at expiry
        if trade.exit_date is None:
            trade.exit_date = str(future_bars.index[-1])
            trade.exit_price = float(future_bars.iloc[-1]['close'])
            trade.exit_reason = "Expiry"
            trade.is_winner = (trade.exit_price > trade.entry_price) if is_bullish else (trade.exit_price < trade.entry_price)
        
        # Calculate PnL
        trade.option_exit_price = self._estimate_option_price(
            trade.exit_price, trade.strike, trade.delta, 
            max(0, trade.expiry_dte - len(future_bars))
        )
        
        contracts = int(trade.risk_amount / (trade.option_entry_price * 100))
        trade.pnl = (trade.option_exit_price - trade.option_entry_price) * 100 * contracts
        trade.pnl_pct = trade.pnl / trade.risk_amount if trade.risk_amount > 0 else 0
        trade.hold_days = len(future_bars) - 1
    
    def _estimate_option_price(self, stock_price: float, strike: float, 
                              delta: float, dte: int) -> float:
        """Simplified option price estimation"""
        # Very simplified - real implementation would use Black-Scholes
        intrinsic = max(0, stock_price - strike) if delta > 0 else max(0, strike - stock_price)
        time_value = stock_price * 0.02 * (dte / 30) * abs(delta)  # Rough estimate
        return max(0.10, intrinsic + time_value)
    
    def _calculate_results(self, ticker: str, df: pd.DataFrame, 
                          mode: BacktestMode) -> BacktestResults:
        """Calculate backtest metrics"""
        trades = self._trades
        
        if not trades:
            return BacktestResults(
                ticker=ticker,
                start_date=str(df.index[0]),
                end_date=str(df.index[-1]),
                mode=mode,
                initial_capital=self.initial_capital,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_pnl=0,
                total_pnl_pct=0,
                avg_win=0,
                avg_loss=0,
                largest_win=0,
                largest_loss=0,
                profit_factor=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                calmar_ratio=0,
                avg_hold_days=0,
                avg_risk_reward=0,
                expectancy=0,
                equity_curve=[],
                trades=[],
                playbook_performance={}
            )
        
        # Basic stats
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        
        win_pnls = [t.pnl for t in winners]
        loss_pnls = [t.pnl for t in losers]
        
        total_pnl = sum(t.pnl for t in trades)
        
        # Drawdown calculation
        equity = [self.initial_capital]
        for t in trades:
            equity.append(equity[-1] + t.pnl)
        
        peak = equity[0]
        max_dd = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd
        
        # Returns for Sharpe
        returns = [t.pnl_pct for t in trades]
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if returns else 1
        
        # Playbook performance
        playbook_perf = {}
        for t in trades:
            pb = t.playbook
            if pb not in playbook_perf:
                playbook_perf[pb] = {"trades": 0, "wins": 0, "pnl": 0}
            playbook_perf[pb]["trades"] += 1
            playbook_perf[pb]["wins"] += 1 if t.is_winner else 0
            playbook_perf[pb]["pnl"] += t.pnl
        
        for pb in playbook_perf:
            playbook_perf[pb]["win_rate"] = playbook_perf[pb]["wins"] / playbook_perf[pb]["trades"] * 100
        
        return BacktestResults(
            ticker=ticker,
            start_date=str(df.index[0]),
            end_date=str(df.index[-1]),
            mode=mode,
            initial_capital=self.initial_capital,
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=len(winners) / len(trades) * 100 if trades else 0,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl / self.initial_capital * 100,
            avg_win=np.mean(win_pnls) if win_pnls else 0,
            avg_loss=np.mean(loss_pnls) if loss_pnls else 0,
            largest_win=max(win_pnls) if win_pnls else 0,
            largest_loss=min(loss_pnls) if loss_pnls else 0,
            profit_factor=abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else 0,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd / self.initial_capital * 100,
            sharpe_ratio=avg_return / std_return * np.sqrt(252) if std_return > 0 else 0,
            sortino_ratio=0,  # Would need downside deviation
            calmar_ratio=total_pnl / max_dd if max_dd > 0 else 0,
            avg_hold_days=np.mean([t.hold_days for t in trades]),
            avg_risk_reward=np.mean([t.risk_reward for t in trades if t.risk_reward > 0]),
            expectancy=total_pnl / len(trades) if trades else 0,
            equity_curve=self._equity_curve,
            trades=trades,
            playbook_performance=playbook_perf
        )
    
    def get_summary(self, results: BacktestResults) -> str:
        """Generate human-readable summary"""
        lines = []
        lines.append("=" * 70)
        lines.append("📊 TRADEPILOT BACKTEST RESULTS")
        lines.append("=" * 70)
        lines.append(f"Ticker: {results.ticker}")
        lines.append(f"Period: {results.start_date} to {results.end_date}")
        lines.append(f"Initial Capital: ${results.initial_capital:,.2f}")
        lines.append("")
        
        lines.append("📈 PERFORMANCE:")
        lines.append(f"   Total Trades: {results.total_trades}")
        lines.append(f"   Win Rate: {results.win_rate:.1f}%")
        lines.append(f"   Total PnL: ${results.total_pnl:,.2f} ({results.total_pnl_pct:.1f}%)")
        lines.append(f"   Profit Factor: {results.profit_factor:.2f}")
        lines.append("")
        
        lines.append("💰 TRADE STATS:")
        lines.append(f"   Average Win: ${results.avg_win:,.2f}")
        lines.append(f"   Average Loss: ${results.avg_loss:,.2f}")
        lines.append(f"   Largest Win: ${results.largest_win:,.2f}")
        lines.append(f"   Largest Loss: ${results.largest_loss:,.2f}")
        lines.append(f"   Expectancy: ${results.expectancy:,.2f} per trade")
        lines.append("")
        
        lines.append("📉 RISK METRICS:")
        lines.append(f"   Max Drawdown: ${results.max_drawdown:,.2f} ({results.max_drawdown_pct:.1f}%)")
        lines.append(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
        lines.append(f"   Calmar Ratio: {results.calmar_ratio:.2f}")
        lines.append("")
        
        lines.append("📖 PLAYBOOK PERFORMANCE:")
        for pb, stats in sorted(results.playbook_performance.items(), 
                               key=lambda x: x[1]['win_rate'], reverse=True):
            lines.append(f"   {pb}: {stats['win_rate']:.0f}% ({stats['wins']}/{stats['trades']}) | ${stats['pnl']:,.0f}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def export_trades(self, results: BacktestResults, format: str = "json") -> str:
        """Export trade journal"""
        if format == "json":
            trades_data = [
                {
                    "id": t.trade_id,
                    "ticker": t.ticker,
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "direction": t.direction,
                    "action": t.action,
                    "playbook": t.playbook,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "is_winner": t.is_winner,
                    "exit_reason": t.exit_reason
                }
                for t in results.trades
            ]
            return json.dumps(trades_data, indent=2)
        elif format == "csv":
            lines = ["id,ticker,entry_date,exit_date,direction,action,playbook,pnl,is_winner"]
            for t in results.trades:
                lines.append(f"{t.trade_id},{t.ticker},{t.entry_date},{t.exit_date},{t.direction},{t.action},{t.playbook},{t.pnl:.2f},{t.is_winner}")
            return "\n".join(lines)
        else:
            return str(results.trades)


# Convenience function
def create_backtester(initial_capital: float = 100000) -> TradePilotBacktester:
    """Create a new backtester instance"""
    return TradePilotBacktester(initial_capital=initial_capital)


if __name__ == "__main__":
    print("TradePilot Backtester initialized")
    backtester = TradePilotBacktester(initial_capital=50000)
    print(f"Initial capital: ${backtester.initial_capital:,.2f}")
