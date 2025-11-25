"""
TradePilot Advanced Multi-Ticker Scanner
=========================================
High-performance scanner for finding high-probability setups across multiple tickers.

Features:
- Parallel scanning with rate limiting
- Multiple filter criteria
- Ranking by probability
- Watchlist management
- Scheduled scanning
- Export to various formats

Author: TradePilot Integration
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class ScannerMode(Enum):
    """Scanner operation modes"""
    QUICK = "QUICK"        # Fast scan, essential layers only
    STANDARD = "STANDARD"  # Normal 18-layer scan
    DEEP = "DEEP"          # Full analysis with options


class FilterOperator(Enum):
    """Filter comparison operators"""
    EQ = "=="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


@dataclass
class ScanFilter:
    """Individual scan filter"""
    field: str
    operator: FilterOperator
    value: Any
    description: str = ""


@dataclass
class ScanResult:
    """Individual ticker scan result"""
    ticker: str
    scan_time: str
    success: bool
    
    # Core signals
    direction: str
    action: str
    confidence: str
    win_probability: float
    trade_valid: bool
    
    # Playbook
    matched_playbook: Optional[str]
    playbook_id: Optional[int]
    
    # Option details
    strike: float
    delta: float
    expiry_dte: int
    
    # Execution
    entry_price: float
    target_price: float
    stop_price: float
    risk_reward: float
    
    # Raw data
    layer_summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ScanSummary:
    """Complete scan summary"""
    scan_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    
    mode: ScannerMode
    tickers_scanned: int
    successful_scans: int
    failed_scans: int
    
    # Filtered results
    setups_found: int
    bullish_setups: int
    bearish_setups: int
    
    # Ranked results
    results: List[ScanResult]
    top_bullish: List[ScanResult]
    top_bearish: List[ScanResult]
    
    # Errors
    errors: List[Dict[str, str]]
    
    # Metadata
    filters_applied: List[str]


class TradePilotScanner:
    """
    Advanced multi-ticker scanner with filtering and ranking.
    
    Usage:
        scanner = TradePilotScanner()
        scanner.add_filter("win_probability", ">=", 80)
        scanner.add_filter("confidence", "in", ["SUPREME", "EXCELLENT", "STRONG"])
        results = scanner.scan(["SPY", "QQQ", "AAPL", "TSLA", "NVDA"])
    """
    
    def __init__(self, engine=None, max_workers: int = 5, rate_limit: float = 0.5):
        """
        Initialize scanner
        
        Args:
            engine: TradePilotEngine18Layer instance (optional, will create if not provided)
            max_workers: Maximum parallel scans
            rate_limit: Minimum seconds between API calls
        """
        self._engine = engine
        self._max_workers = max_workers
        self._rate_limit = rate_limit
        self._last_api_call = 0
        self._lock = threading.Lock()
        
        # Filters
        self._filters: List[ScanFilter] = []
        
        # Watchlists
        self._watchlists: Dict[str, List[str]] = {
            "mega_caps": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
            "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM"],
            "etfs": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP"],
            "high_beta": ["TSLA", "NVDA", "AMD", "COIN", "MARA", "RIOT", "PLTR", "SOFI"],
            "indices": ["SPY", "QQQ", "IWM", "DIA"],
            "financials": ["JPM", "BAC", "GS", "MS", "C", "WFC", "BLK", "SCHW"],
            "healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN"],
            "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "VLO", "MPC"]
        }
        
        # Default filters
        self._default_filters = [
            ScanFilter("trade_valid", FilterOperator.EQ, True, "Trade must be valid"),
            ScanFilter("win_probability", FilterOperator.GTE, 75, "Minimum 75% probability")
        ]
    
    @property
    def engine(self):
        """Lazy load engine"""
        if self._engine is None:
            from engine_18layer_core import TradePilotEngine18Layer
            self._engine = TradePilotEngine18Layer()
        return self._engine
    
    def add_filter(self, field: str, operator: str, value: Any, description: str = "") -> 'TradePilotScanner':
        """
        Add a scan filter
        
        Args:
            field: Field to filter (e.g., "win_probability", "confidence", "direction")
            operator: Comparison operator ("==", ">=", "in", etc.)
            value: Value to compare against
            description: Optional description
            
        Returns:
            self for chaining
        """
        op_map = {
            "==": FilterOperator.EQ,
            "!=": FilterOperator.NE,
            ">": FilterOperator.GT,
            ">=": FilterOperator.GTE,
            "<": FilterOperator.LT,
            "<=": FilterOperator.LTE,
            "in": FilterOperator.IN,
            "not_in": FilterOperator.NOT_IN,
            "contains": FilterOperator.CONTAINS
        }
        
        op = op_map.get(operator, FilterOperator.EQ)
        self._filters.append(ScanFilter(field, op, value, description))
        return self
    
    def clear_filters(self) -> 'TradePilotScanner':
        """Clear all custom filters"""
        self._filters = []
        return self
    
    def get_watchlist(self, name: str) -> List[str]:
        """Get a predefined watchlist"""
        return self._watchlists.get(name.lower(), [])
    
    def add_watchlist(self, name: str, tickers: List[str]) -> 'TradePilotScanner':
        """Add a custom watchlist"""
        self._watchlists[name.lower()] = [t.upper() for t in tickers]
        return self
    
    def _rate_limit_wait(self):
        """Enforce rate limiting"""
        with self._lock:
            elapsed = time.time() - self._last_api_call
            if elapsed < self._rate_limit:
                time.sleep(self._rate_limit - elapsed)
            self._last_api_call = time.time()
    
    def _scan_single(self, ticker: str, mode: ScannerMode, 
                    trade_mode: str = "SWING") -> ScanResult:
        """Scan a single ticker"""
        try:
            self._rate_limit_wait()
            
            # Import dependencies
            from engine_18layer_core import TradeMode
            
            tm = TradeMode.SCALP if trade_mode.upper() == "SCALP" else TradeMode.SWING
            
            # Get candle data
            try:
                from polygon_client import get_candles, get_option_chain_snapshot
                candles_data = get_candles(ticker.upper(), tf="day", limit=300)
            except ImportError:
                return ScanResult(
                    ticker=ticker,
                    scan_time=datetime.now().isoformat(),
                    success=False,
                    direction="NEUTRAL",
                    action="FLAT",
                    confidence="WEAK",
                    win_probability=0,
                    trade_valid=False,
                    matched_playbook=None,
                    playbook_id=None,
                    strike=0,
                    delta=0,
                    expiry_dte=0,
                    entry_price=0,
                    target_price=0,
                    stop_price=0,
                    risk_reward=0,
                    error="Polygon client not available"
                )
            
            if not candles_data or "results" not in candles_data:
                return ScanResult(
                    ticker=ticker,
                    scan_time=datetime.now().isoformat(),
                    success=False,
                    direction="NEUTRAL",
                    action="FLAT",
                    confidence="WEAK",
                    win_probability=0,
                    trade_valid=False,
                    matched_playbook=None,
                    playbook_id=None,
                    strike=0,
                    delta=0,
                    expiry_dte=0,
                    entry_price=0,
                    target_price=0,
                    stop_price=0,
                    risk_reward=0,
                    error="No candle data"
                )
            
            # Get options data for DEEP mode
            options_data = None
            if mode == ScannerMode.DEEP:
                try:
                    options_data = get_option_chain_snapshot(ticker.upper(), limit=50)
                except:
                    pass
            
            # Run analysis
            result = self.engine.analyze(
                ticker=ticker.upper(),
                candles_data=candles_data,
                options_data=options_data,
                mode=tm
            )
            
            return ScanResult(
                ticker=ticker.upper(),
                scan_time=datetime.now().isoformat(),
                success=True,
                direction=result.direction,
                action=result.action,
                confidence=result.confidence.value,
                win_probability=result.win_probability,
                trade_valid=result.trade_valid,
                matched_playbook=result.matched_playbook,
                playbook_id=result.playbook_id,
                strike=result.strike,
                delta=result.delta,
                expiry_dte=result.expiry_dte,
                entry_price=result.entry_price,
                target_price=result.target_price,
                stop_price=result.stop_price,
                risk_reward=result.risk_reward,
                layer_summary={
                    "momentum": result.technical_layers.get("layer_1", {}).get("momentum_score"),
                    "trend": result.technical_layers.get("layer_5", {}).get("trend_direction"),
                    "structure_bias": result.technical_layers.get("layer_6", {}).get("bias"),
                    "volatility": result.technical_layers.get("layer_8", {}).get("regime")
                }
            )
            
        except Exception as e:
            return ScanResult(
                ticker=ticker,
                scan_time=datetime.now().isoformat(),
                success=False,
                direction="NEUTRAL",
                action="FLAT",
                confidence="WEAK",
                win_probability=0,
                trade_valid=False,
                matched_playbook=None,
                playbook_id=None,
                strike=0,
                delta=0,
                expiry_dte=0,
                entry_price=0,
                target_price=0,
                stop_price=0,
                risk_reward=0,
                error=str(e)
            )
    
    def _apply_filter(self, result: ScanResult, filter: ScanFilter) -> bool:
        """Apply a single filter to a result"""
        try:
            value = getattr(result, filter.field, None)
            if value is None:
                return False
            
            if filter.operator == FilterOperator.EQ:
                return value == filter.value
            elif filter.operator == FilterOperator.NE:
                return value != filter.value
            elif filter.operator == FilterOperator.GT:
                return value > filter.value
            elif filter.operator == FilterOperator.GTE:
                return value >= filter.value
            elif filter.operator == FilterOperator.LT:
                return value < filter.value
            elif filter.operator == FilterOperator.LTE:
                return value <= filter.value
            elif filter.operator == FilterOperator.IN:
                return value in filter.value
            elif filter.operator == FilterOperator.NOT_IN:
                return value not in filter.value
            elif filter.operator == FilterOperator.CONTAINS:
                return filter.value in str(value)
            
            return True
            
        except Exception:
            return False
    
    def _passes_filters(self, result: ScanResult) -> bool:
        """Check if result passes all filters"""
        all_filters = self._default_filters + self._filters
        
        for f in all_filters:
            if not self._apply_filter(result, f):
                return False
        
        return True
    
    def scan(self, 
             tickers: List[str],
             mode: ScannerMode = ScannerMode.STANDARD,
             trade_mode: str = "SWING") -> ScanSummary:
        """
        Scan multiple tickers
        
        Args:
            tickers: List of ticker symbols
            mode: Scanner mode (QUICK, STANDARD, DEEP)
            trade_mode: Trading mode (SCALP, SWING)
            
        Returns:
            ScanSummary with all results
        """
        import uuid
        
        scan_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        
        results: List[ScanResult] = []
        errors: List[Dict[str, str]] = []
        
        # Parallel scanning
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self._scan_single, ticker, mode, trade_mode): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.error:
                        errors.append({"ticker": ticker, "error": result.error})
                        
                except Exception as e:
                    errors.append({"ticker": ticker, "error": str(e)})
        
        # Apply filters
        successful = [r for r in results if r.success]
        filtered = [r for r in successful if self._passes_filters(r)]
        
        # Separate bullish and bearish
        bullish = [r for r in filtered if r.direction == "BULLISH"]
        bearish = [r for r in filtered if r.direction == "BEARISH"]
        
        # Sort by win probability
        bullish.sort(key=lambda x: x.win_probability, reverse=True)
        bearish.sort(key=lambda x: x.win_probability, reverse=True)
        filtered.sort(key=lambda x: x.win_probability, reverse=True)
        
        end_time = datetime.now()
        
        return ScanSummary(
            scan_id=scan_id,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=(end_time - start_time).total_seconds(),
            mode=mode,
            tickers_scanned=len(tickers),
            successful_scans=len(successful),
            failed_scans=len(tickers) - len(successful),
            setups_found=len(filtered),
            bullish_setups=len(bullish),
            bearish_setups=len(bearish),
            results=filtered,
            top_bullish=bullish[:5],
            top_bearish=bearish[:5],
            errors=errors,
            filters_applied=[f"{f.field} {f.operator.value} {f.value}" for f in self._default_filters + self._filters]
        )
    
    def scan_watchlist(self, 
                       watchlist_name: str,
                       mode: ScannerMode = ScannerMode.STANDARD,
                       trade_mode: str = "SWING") -> ScanSummary:
        """Scan a predefined watchlist"""
        tickers = self.get_watchlist(watchlist_name)
        if not tickers:
            raise ValueError(f"Watchlist '{watchlist_name}' not found")
        return self.scan(tickers, mode, trade_mode)
    
    def quick_scan(self, 
                   tickers: List[str],
                   trade_mode: str = "SWING") -> ScanSummary:
        """Quick scan with minimal analysis"""
        return self.scan(tickers, ScannerMode.QUICK, trade_mode)
    
    def deep_scan(self,
                  tickers: List[str],
                  trade_mode: str = "SWING") -> ScanSummary:
        """Deep scan with full options analysis"""
        return self.scan(tickers, ScannerMode.DEEP, trade_mode)
    
    def find_best_setup(self, tickers: List[str], trade_mode: str = "SWING") -> Optional[ScanResult]:
        """Find the single best setup from a list"""
        summary = self.scan(tickers, ScannerMode.STANDARD, trade_mode)
        return summary.results[0] if summary.results else None
    
    def to_dict(self, summary: ScanSummary) -> Dict:
        """Convert summary to dictionary"""
        return {
            "scan_id": summary.scan_id,
            "timing": {
                "start": summary.start_time,
                "end": summary.end_time,
                "duration_seconds": summary.duration_seconds
            },
            "summary": {
                "mode": summary.mode.value,
                "tickers_scanned": summary.tickers_scanned,
                "successful_scans": summary.successful_scans,
                "failed_scans": summary.failed_scans,
                "setups_found": summary.setups_found,
                "bullish_setups": summary.bullish_setups,
                "bearish_setups": summary.bearish_setups
            },
            "top_bullish": [
                {
                    "ticker": r.ticker,
                    "action": r.action,
                    "confidence": r.confidence,
                    "win_probability": r.win_probability,
                    "playbook": r.matched_playbook,
                    "strike": r.strike,
                    "expiry_dte": r.expiry_dte
                }
                for r in summary.top_bullish
            ],
            "top_bearish": [
                {
                    "ticker": r.ticker,
                    "action": r.action,
                    "confidence": r.confidence,
                    "win_probability": r.win_probability,
                    "playbook": r.matched_playbook,
                    "strike": r.strike,
                    "expiry_dte": r.expiry_dte
                }
                for r in summary.top_bearish
            ],
            "all_results": [
                {
                    "ticker": r.ticker,
                    "direction": r.direction,
                    "action": r.action,
                    "confidence": r.confidence,
                    "win_probability": r.win_probability,
                    "trade_valid": r.trade_valid,
                    "playbook": r.matched_playbook,
                    "option": {
                        "strike": r.strike,
                        "delta": r.delta,
                        "expiry_dte": r.expiry_dte
                    },
                    "execution": {
                        "entry": r.entry_price,
                        "target": r.target_price,
                        "stop": r.stop_price,
                        "risk_reward": r.risk_reward
                    }
                }
                for r in summary.results
            ],
            "errors": summary.errors,
            "filters_applied": summary.filters_applied
        }
    
    def to_json(self, summary: ScanSummary) -> str:
        """Convert summary to JSON string"""
        return json.dumps(self.to_dict(summary), indent=2)
    
    def get_human_readable(self, summary: ScanSummary) -> str:
        """Generate human-readable summary"""
        lines = []
        lines.append("=" * 80)
        lines.append("🔍 TRADEPILOT SCANNER RESULTS")
        lines.append("=" * 80)
        lines.append(f"Scan ID: {summary.scan_id}")
        lines.append(f"Duration: {summary.duration_seconds:.1f}s")
        lines.append(f"Mode: {summary.mode.value}")
        lines.append("")
        lines.append(f"📊 SUMMARY:")
        lines.append(f"   Tickers Scanned: {summary.tickers_scanned}")
        lines.append(f"   Successful: {summary.successful_scans}")
        lines.append(f"   Failed: {summary.failed_scans}")
        lines.append(f"   Setups Found: {summary.setups_found}")
        lines.append(f"   🟢 Bullish: {summary.bullish_setups}")
        lines.append(f"   🔴 Bearish: {summary.bearish_setups}")
        lines.append("")
        
        if summary.top_bullish:
            lines.append("🟢 TOP BULLISH SETUPS:")
            for r in summary.top_bullish[:3]:
                lines.append(f"   {r.ticker}: {r.action} | {r.confidence} ({r.win_probability:.0f}%)")
                lines.append(f"      Playbook: {r.matched_playbook or 'N/A'}")
                lines.append(f"      Strike: ${r.strike:.2f} | {r.expiry_dte} DTE")
            lines.append("")
        
        if summary.top_bearish:
            lines.append("🔴 TOP BEARISH SETUPS:")
            for r in summary.top_bearish[:3]:
                lines.append(f"   {r.ticker}: {r.action} | {r.confidence} ({r.win_probability:.0f}%)")
                lines.append(f"      Playbook: {r.matched_playbook or 'N/A'}")
                lines.append(f"      Strike: ${r.strike:.2f} | {r.expiry_dte} DTE")
            lines.append("")
        
        if summary.errors:
            lines.append(f"⚠️ ERRORS ({len(summary.errors)}):")
            for e in summary.errors[:5]:
                lines.append(f"   {e['ticker']}: {e['error']}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


# Convenience functions
def create_scanner(max_workers: int = 5) -> TradePilotScanner:
    """Create a new scanner instance"""
    return TradePilotScanner(max_workers=max_workers)


def scan_market(watchlist: str = "etfs", min_probability: float = 80) -> ScanSummary:
    """Quick market scan"""
    scanner = TradePilotScanner()
    scanner.add_filter("win_probability", ">=", min_probability)
    return scanner.scan_watchlist(watchlist)


if __name__ == "__main__":
    # Example usage
    scanner = TradePilotScanner()
    scanner.add_filter("confidence", "in", ["SUPREME", "EXCELLENT", "STRONG"])
    scanner.add_filter("win_probability", ">=", 80)
    
    print("TradePilot Scanner initialized")
    print(f"Available watchlists: {list(scanner._watchlists.keys())}")
