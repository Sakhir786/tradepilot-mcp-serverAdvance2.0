"""
TradePilot Risk Management Module
==================================
Comprehensive risk management for options trading.

Features:
- Dynamic position sizing based on confidence
- Portfolio-level risk management
- Maximum drawdown protection
- Correlation-aware position limits
- Kelly Criterion position sizing
- Risk-adjusted returns tracking

Author: TradePilot Integration
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math


class RiskProfile(Enum):
    """Risk profile presets"""
    CONSERVATIVE = "CONSERVATIVE"   # Max 15% per trade, 30% total
    MODERATE = "MODERATE"           # Max 25% per trade, 50% total
    AGGRESSIVE = "AGGRESSIVE"       # Max 40% per trade, 75% total
    CUSTOM = "CUSTOM"


@dataclass
class Position:
    """Active position tracking"""
    position_id: str
    ticker: str
    direction: str  # BULLISH or BEARISH
    action: str  # BUY_CALL, BUY_PUT
    
    # Entry details
    entry_time: str
    entry_price: float
    contracts: int
    contract_cost: float
    total_cost: float
    
    # Target/Stop
    target_price: float
    stop_price: float
    breakeven_price: float
    
    # Option details
    strike: float
    expiry_date: str
    delta: float
    
    # Playbook
    playbook: Optional[str] = None
    win_probability: float = 0.0
    confidence: str = ""
    
    # Current status
    current_price: Optional[float] = None
    current_pnl: float = 0.0
    current_pnl_pct: float = 0.0
    
    # Invalidation
    invalidation_above: Optional[float] = None
    invalidation_below: Optional[float] = None
    is_invalidated: bool = False
    invalidation_reason: str = ""
    
    # Risk metrics
    risk_amount: float = 0.0
    max_loss: float = 0.0
    risk_reward: float = 0.0
    
    # Status
    is_open: bool = True
    close_time: Optional[str] = None
    close_price: Optional[float] = None
    final_pnl: float = 0.0
    close_reason: str = ""


@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    # Portfolio value
    total_portfolio_value: float
    cash_available: float
    positions_value: float
    
    # Exposure
    total_exposure_pct: float
    bullish_exposure_pct: float
    bearish_exposure_pct: float
    net_exposure_pct: float
    
    # Position counts
    total_positions: int
    bullish_positions: int
    bearish_positions: int
    
    # Risk limits
    max_position_size_pct: float
    max_total_exposure_pct: float
    remaining_capacity_pct: float
    
    # PnL
    unrealized_pnl: float
    unrealized_pnl_pct: float
    daily_pnl: float
    
    # Drawdown
    current_drawdown_pct: float
    max_drawdown_pct: float
    high_water_mark: float
    
    # Risk scores
    overall_risk_score: float  # 0-100
    concentration_risk: float
    correlation_risk: float


@dataclass
class PositionSizeResult:
    """Position sizing calculation result"""
    recommended_contracts: int
    recommended_cost: float
    recommended_pct_of_portfolio: float
    
    max_contracts_allowed: int
    max_cost_allowed: float
    
    risk_per_contract: float
    total_risk: float
    
    kelly_fraction: float
    kelly_contracts: int
    
    reasoning: List[str]
    warnings: List[str]


class TradePilotRiskManager:
    """
    Comprehensive risk management for options trading.
    
    Usage:
        risk_mgr = TradePilotRiskManager(portfolio_value=100000)
        size = risk_mgr.calculate_position_size(analysis_result, option_price=2.50)
        risk_mgr.add_position(position)
        metrics = risk_mgr.get_risk_metrics()
    """
    
    def __init__(self, 
                 portfolio_value: float = 100000,
                 risk_profile: RiskProfile = RiskProfile.MODERATE):
        """
        Initialize risk manager
        
        Args:
            portfolio_value: Total portfolio value
            risk_profile: Risk profile preset
        """
        self.portfolio_value = portfolio_value
        self.risk_profile = risk_profile
        
        # Active positions
        self._positions: Dict[str, Position] = {}
        
        # Historical tracking
        self._trade_history: List[Position] = []
        self._daily_pnl: Dict[str, float] = {}
        self._high_water_mark = portfolio_value
        
        # Risk limits based on profile
        self._set_risk_limits(risk_profile)
        
        # Confidence-based sizing multipliers
        self.confidence_multipliers = {
            "SUPREME": 1.0,
            "EXCELLENT": 0.75,
            "STRONG": 0.50,
            "MODERATE": 0.30,
            "WEAK": 0.0
        }
        
        # Kelly Criterion settings
        self.kelly_fraction_cap = 0.25  # Max 25% Kelly
        self.kelly_safety_factor = 0.5  # Half-Kelly
    
    def _set_risk_limits(self, profile: RiskProfile):
        """Set risk limits based on profile"""
        limits = {
            RiskProfile.CONSERVATIVE: {
                "max_position_pct": 0.15,
                "max_total_exposure_pct": 0.30,
                "max_single_ticker_pct": 0.20,
                "max_positions": 5,
                "max_drawdown_pct": 0.10,
                "min_risk_reward": 2.0
            },
            RiskProfile.MODERATE: {
                "max_position_pct": 0.25,
                "max_total_exposure_pct": 0.50,
                "max_single_ticker_pct": 0.30,
                "max_positions": 10,
                "max_drawdown_pct": 0.15,
                "min_risk_reward": 1.5
            },
            RiskProfile.AGGRESSIVE: {
                "max_position_pct": 0.40,
                "max_total_exposure_pct": 0.75,
                "max_single_ticker_pct": 0.50,
                "max_positions": 20,
                "max_drawdown_pct": 0.25,
                "min_risk_reward": 1.0
            }
        }
        
        self.limits = limits.get(profile, limits[RiskProfile.MODERATE])
    
    def calculate_position_size(self,
                               analysis_result: Any,
                               option_price: float,
                               contract_multiplier: int = 100) -> PositionSizeResult:
        """
        Calculate optimal position size based on analysis and risk parameters
        
        Args:
            analysis_result: FullAnalysisResult from engine
            option_price: Current option contract price
            contract_multiplier: Options contract multiplier (usually 100)
            
        Returns:
            PositionSizeResult with recommended sizing
        """
        warnings = []
        reasoning = []
        
        # Get confidence multiplier
        confidence = getattr(analysis_result, 'confidence', None)
        confidence_str = confidence.value if hasattr(confidence, 'value') else str(confidence)
        conf_mult = self.confidence_multipliers.get(confidence_str, 0.0)
        
        if conf_mult == 0:
            return PositionSizeResult(
                recommended_contracts=0,
                recommended_cost=0,
                recommended_pct_of_portfolio=0,
                max_contracts_allowed=0,
                max_cost_allowed=0,
                risk_per_contract=option_price * contract_multiplier,
                total_risk=0,
                kelly_fraction=0,
                kelly_contracts=0,
                reasoning=["Confidence too low for position"],
                warnings=["Signal not strong enough to trade"]
            )
        
        reasoning.append(f"Confidence: {confidence_str} (multiplier: {conf_mult})")
        
        # Calculate available capacity
        current_exposure = self._get_current_exposure()
        remaining_capacity = self.limits["max_total_exposure_pct"] - current_exposure
        
        if remaining_capacity <= 0:
            warnings.append("Portfolio at maximum exposure")
            return PositionSizeResult(
                recommended_contracts=0,
                recommended_cost=0,
                recommended_pct_of_portfolio=0,
                max_contracts_allowed=0,
                max_cost_allowed=0,
                risk_per_contract=option_price * contract_multiplier,
                total_risk=0,
                kelly_fraction=0,
                kelly_contracts=0,
                reasoning=reasoning,
                warnings=warnings
            )
        
        reasoning.append(f"Remaining capacity: {remaining_capacity*100:.1f}%")
        
        # Check ticker concentration
        ticker = getattr(analysis_result, 'ticker', '')
        ticker_exposure = self._get_ticker_exposure(ticker)
        ticker_remaining = self.limits["max_single_ticker_pct"] - ticker_exposure
        
        if ticker_remaining <= 0:
            warnings.append(f"Maximum exposure to {ticker} reached")
            ticker_remaining = 0
        
        # Calculate max position size
        max_position_pct = min(
            self.limits["max_position_pct"] * conf_mult,
            remaining_capacity,
            ticker_remaining
        )
        
        # Apply drawdown protection
        current_drawdown = self._get_current_drawdown()
        if current_drawdown > self.limits["max_drawdown_pct"] * 0.5:
            drawdown_factor = 1 - (current_drawdown / self.limits["max_drawdown_pct"])
            max_position_pct *= max(0.25, drawdown_factor)
            warnings.append(f"Position reduced due to {current_drawdown*100:.1f}% drawdown")
        
        reasoning.append(f"Max position size: {max_position_pct*100:.1f}%")
        
        # Calculate Kelly Criterion
        win_prob = getattr(analysis_result, 'win_probability', 50) / 100
        risk_reward = getattr(analysis_result, 'risk_reward', 2.0)
        
        # Kelly formula: f* = (p * b - q) / b
        # where p = win probability, q = 1-p, b = risk-reward ratio
        if risk_reward > 0:
            kelly_fraction = (win_prob * risk_reward - (1 - win_prob)) / risk_reward
            kelly_fraction = max(0, min(kelly_fraction, self.kelly_fraction_cap))
            kelly_fraction *= self.kelly_safety_factor  # Half-Kelly for safety
        else:
            kelly_fraction = 0
        
        reasoning.append(f"Kelly fraction: {kelly_fraction*100:.1f}%")
        
        # Final position size (take smaller of max and Kelly)
        final_position_pct = min(max_position_pct, kelly_fraction) if kelly_fraction > 0 else max_position_pct
        
        # Convert to contracts
        max_cost = self.portfolio_value * final_position_pct
        cost_per_contract = option_price * contract_multiplier
        
        if cost_per_contract <= 0:
            warnings.append("Invalid option price")
            cost_per_contract = 1
        
        recommended_contracts = max(0, int(max_cost / cost_per_contract))
        recommended_cost = recommended_contracts * cost_per_contract
        
        # Risk calculation
        risk_per_contract = cost_per_contract  # Max loss is premium paid
        total_risk = recommended_contracts * risk_per_contract
        
        reasoning.append(f"Recommended: {recommended_contracts} contracts @ ${option_price:.2f}")
        
        return PositionSizeResult(
            recommended_contracts=recommended_contracts,
            recommended_cost=recommended_cost,
            recommended_pct_of_portfolio=recommended_cost / self.portfolio_value if self.portfolio_value > 0 else 0,
            max_contracts_allowed=int(self.portfolio_value * self.limits["max_position_pct"] / cost_per_contract),
            max_cost_allowed=self.portfolio_value * self.limits["max_position_pct"],
            risk_per_contract=risk_per_contract,
            total_risk=total_risk,
            kelly_fraction=kelly_fraction,
            kelly_contracts=int(self.portfolio_value * kelly_fraction / cost_per_contract) if cost_per_contract > 0 else 0,
            reasoning=reasoning,
            warnings=warnings
        )
    
    def add_position(self, position: Position) -> bool:
        """
        Add a new position to tracking
        
        Returns:
            True if position was added, False if rejected
        """
        # Check position limits
        if len(self._positions) >= self.limits["max_positions"]:
            return False
        
        # Check total exposure
        position_pct = position.total_cost / self.portfolio_value
        current_exposure = self._get_current_exposure()
        
        if current_exposure + position_pct > self.limits["max_total_exposure_pct"]:
            return False
        
        # Check ticker concentration
        ticker_exposure = self._get_ticker_exposure(position.ticker)
        if ticker_exposure + position_pct > self.limits["max_single_ticker_pct"]:
            return False
        
        # Calculate risk metrics
        position.risk_amount = position.total_cost  # Premium at risk
        position.max_loss = position.total_cost
        
        self._positions[position.position_id] = position
        return True
    
    def update_position(self, position_id: str, current_price: float):
        """Update position with current option price"""
        if position_id not in self._positions:
            return
        
        position = self._positions[position_id]
        position.current_price = current_price
        
        # Calculate PnL
        current_value = current_price * position.contracts * 100
        position.current_pnl = current_value - position.total_cost
        position.current_pnl_pct = position.current_pnl / position.total_cost if position.total_cost > 0 else 0
    
    def close_position(self, position_id: str, close_price: float, reason: str = "Manual close"):
        """Close a position"""
        if position_id not in self._positions:
            return
        
        position = self._positions[position_id]
        position.is_open = False
        position.close_time = datetime.now().isoformat()
        position.close_price = close_price
        position.close_reason = reason
        
        # Calculate final PnL
        close_value = close_price * position.contracts * 100
        position.final_pnl = close_value - position.total_cost
        
        # Move to history
        self._trade_history.append(position)
        del self._positions[position_id]
        
        # Update daily PnL
        today = datetime.now().strftime("%Y-%m-%d")
        self._daily_pnl[today] = self._daily_pnl.get(today, 0) + position.final_pnl
    
    def check_invalidations(self, current_prices: Dict[str, float]) -> List[Position]:
        """
        Check for position invalidations
        
        Args:
            current_prices: Dict of ticker -> current stock price
            
        Returns:
            List of invalidated positions
        """
        invalidated = []
        
        for pos_id, position in self._positions.items():
            if position.is_invalidated:
                continue
            
            ticker_price = current_prices.get(position.ticker)
            if ticker_price is None:
                continue
            
            # Check invalidation levels
            if position.invalidation_above and ticker_price > position.invalidation_above:
                position.is_invalidated = True
                position.invalidation_reason = f"Price {ticker_price:.2f} above invalidation {position.invalidation_above:.2f}"
                invalidated.append(position)
            
            elif position.invalidation_below and ticker_price < position.invalidation_below:
                position.is_invalidated = True
                position.invalidation_reason = f"Price {ticker_price:.2f} below invalidation {position.invalidation_below:.2f}"
                invalidated.append(position)
        
        return invalidated
    
    def _get_current_exposure(self) -> float:
        """Calculate current portfolio exposure percentage"""
        total_cost = sum(p.total_cost for p in self._positions.values() if p.is_open)
        return total_cost / self.portfolio_value if self.portfolio_value > 0 else 0
    
    def _get_ticker_exposure(self, ticker: str) -> float:
        """Calculate exposure to a specific ticker"""
        ticker_cost = sum(
            p.total_cost for p in self._positions.values() 
            if p.is_open and p.ticker == ticker
        )
        return ticker_cost / self.portfolio_value if self.portfolio_value > 0 else 0
    
    def _get_current_drawdown(self) -> float:
        """Calculate current drawdown from high water mark"""
        current_value = self.portfolio_value + sum(
            p.current_pnl for p in self._positions.values() if p.is_open
        )
        
        if current_value > self._high_water_mark:
            self._high_water_mark = current_value
            return 0
        
        return (self._high_water_mark - current_value) / self._high_water_mark
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current portfolio risk metrics"""
        # Calculate values
        positions_value = sum(p.total_cost for p in self._positions.values() if p.is_open)
        unrealized_pnl = sum(p.current_pnl for p in self._positions.values() if p.is_open)
        
        bullish_value = sum(
            p.total_cost for p in self._positions.values() 
            if p.is_open and p.direction == "BULLISH"
        )
        bearish_value = sum(
            p.total_cost for p in self._positions.values() 
            if p.is_open and p.direction == "BEARISH"
        )
        
        total_exposure = positions_value / self.portfolio_value if self.portfolio_value > 0 else 0
        remaining = self.limits["max_total_exposure_pct"] - total_exposure
        
        # Concentration risk (Herfindahl index)
        if positions_value > 0:
            weights = [(p.total_cost / positions_value) ** 2 for p in self._positions.values() if p.is_open]
            concentration = sum(weights) if weights else 0
        else:
            concentration = 0
        
        # Today's PnL
        today = datetime.now().strftime("%Y-%m-%d")
        daily_pnl = self._daily_pnl.get(today, 0)
        
        # Drawdown
        current_drawdown = self._get_current_drawdown()
        max_dd = max(current_drawdown, getattr(self, '_max_drawdown', 0))
        self._max_drawdown = max_dd
        
        # Risk score (0-100, higher = riskier)
        risk_score = (
            total_exposure / self.limits["max_total_exposure_pct"] * 40 +
            concentration * 30 +
            current_drawdown / self.limits["max_drawdown_pct"] * 30
        ) * 100
        
        return RiskMetrics(
            total_portfolio_value=self.portfolio_value,
            cash_available=self.portfolio_value - positions_value,
            positions_value=positions_value,
            total_exposure_pct=total_exposure,
            bullish_exposure_pct=bullish_value / self.portfolio_value if self.portfolio_value > 0 else 0,
            bearish_exposure_pct=bearish_value / self.portfolio_value if self.portfolio_value > 0 else 0,
            net_exposure_pct=(bullish_value - bearish_value) / self.portfolio_value if self.portfolio_value > 0 else 0,
            total_positions=len([p for p in self._positions.values() if p.is_open]),
            bullish_positions=len([p for p in self._positions.values() if p.is_open and p.direction == "BULLISH"]),
            bearish_positions=len([p for p in self._positions.values() if p.is_open and p.direction == "BEARISH"]),
            max_position_size_pct=self.limits["max_position_pct"],
            max_total_exposure_pct=self.limits["max_total_exposure_pct"],
            remaining_capacity_pct=remaining,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl / self.portfolio_value if self.portfolio_value > 0 else 0,
            daily_pnl=daily_pnl,
            current_drawdown_pct=current_drawdown,
            max_drawdown_pct=max_dd,
            high_water_mark=self._high_water_mark,
            overall_risk_score=min(100, risk_score),
            concentration_risk=concentration,
            correlation_risk=0  # Would need position correlation data
        )
    
    def get_positions(self, open_only: bool = True) -> List[Position]:
        """Get all positions"""
        if open_only:
            return [p for p in self._positions.values() if p.is_open]
        return list(self._positions.values())
    
    def get_trade_history(self, limit: int = 100) -> List[Position]:
        """Get trade history"""
        return self._trade_history[-limit:]
    
    def update_portfolio_value(self, new_value: float):
        """Update portfolio value (e.g., after deposit/withdrawal)"""
        self.portfolio_value = new_value
        if new_value > self._high_water_mark:
            self._high_water_mark = new_value
    
    def to_dict(self) -> Dict:
        """Export risk manager state to dictionary"""
        return {
            "portfolio_value": self.portfolio_value,
            "risk_profile": self.risk_profile.value,
            "limits": self.limits,
            "positions": [
                {
                    "id": p.position_id,
                    "ticker": p.ticker,
                    "direction": p.direction,
                    "action": p.action,
                    "contracts": p.contracts,
                    "total_cost": p.total_cost,
                    "current_pnl": p.current_pnl,
                    "is_open": p.is_open
                }
                for p in self._positions.values()
            ],
            "metrics": {
                "total_exposure": self._get_current_exposure(),
                "current_drawdown": self._get_current_drawdown(),
                "open_positions": len([p for p in self._positions.values() if p.is_open])
            }
        }


# Convenience function
def create_risk_manager(portfolio_value: float = 100000,
                       profile: str = "MODERATE") -> TradePilotRiskManager:
    """Create a new risk manager instance"""
    profile_map = {
        "CONSERVATIVE": RiskProfile.CONSERVATIVE,
        "MODERATE": RiskProfile.MODERATE,
        "AGGRESSIVE": RiskProfile.AGGRESSIVE
    }
    return TradePilotRiskManager(
        portfolio_value=portfolio_value,
        risk_profile=profile_map.get(profile.upper(), RiskProfile.MODERATE)
    )


if __name__ == "__main__":
    # Example usage
    risk_mgr = TradePilotRiskManager(portfolio_value=50000, risk_profile=RiskProfile.MODERATE)
    
    print("TradePilot Risk Manager initialized")
    print(f"Portfolio: ${risk_mgr.portfolio_value:,.2f}")
    print(f"Profile: {risk_mgr.risk_profile.value}")
    print(f"Limits: {risk_mgr.limits}")
