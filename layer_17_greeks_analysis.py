"""
LAYER 17: GREEKS ANALYSIS
========================
Analyzes option Greeks (Delta, Gamma, Theta, Vega) to optimize strike selection
and timing for maximum profit potential.

Weight: 5% of total score
Priority: MEDIUM

Features:
- Delta Analysis (optimal sensitivity to stock moves)
- Gamma Analysis (acceleration/risk assessment)
- Theta Decay (time decay cost optimization)
- Vega/IV Sensitivity (IV expansion profit potential)
- Composite Greeks Score (0-100)
- Strike Selection Optimizer
- AI Strategy Recommendations

Data Source: Polygon.io Options API
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import statistics


class Layer17GreeksAnalysis:
    """
    Layer 17: Greeks Analysis Engine
    
    Scoring Breakdown:
    - Delta Optimization: 35 points (directional edge)
    - Gamma Favorability: 20 points (acceleration potential)
    - Theta Cost: 30 points (decay management)
    - Vega/IV Setup: 15 points (IV expansion edge)
    """
    
    def __init__(self, polygon_client, config: Optional[Dict] = None):
        """
        Initialize Greeks Analysis Layer
        
        Args:
            polygon_client: Polygon.io API client instance
            config: Optional configuration overrides
        """
        self.polygon = polygon_client
        self.config = config or {}
        
        # Layer metadata
        self.layer_name = "Layer 17: Greeks Analysis"
        self.weight = 0.05  # 5% of total score
        self.priority = "MEDIUM"
        
        # Scoring weights
        self.weights = {
            'delta': 0.35,      # 35 points - directional edge
            'gamma': 0.20,      # 20 points - acceleration
            'theta': 0.30,      # 30 points - decay cost
            'vega_iv': 0.15     # 15 points - IV expansion edge
        }
        
        # Delta targets by strategy
        self.delta_targets = {
            'directional': (0.50, 0.70),    # ATM to slightly ITM
            'conservative': (0.70, 0.85),    # Deeper ITM
            'speculative': (0.30, 0.50),     # OTM
            'scalp': (0.55, 0.65),           # Near ATM for quick moves
            'swing': (0.60, 0.75)            # Moderate ITM
        }
        
        # Greeks thresholds for quality assessment
        self.thresholds = {
            'gamma': {
                'high': 0.02,       # High gamma (rapid acceleration)
                'medium': 0.01,     # Moderate gamma
                'low': 0.005        # Low gamma
            },
            'theta': {
                'low': -20,         # Low decay (<$20/day)
                'medium': -50,      # Medium decay
                'high': -100        # High decay (>$100/day)
            },
            'vega': {
                'high': 50,         # High vega (>$50 per 1% IV)
                'medium': 25,       # Medium vega
                'low': 10           # Low vega
            }
        }
    
    async def analyze(
        self,
        symbol: str,
        strategy: str = 'directional',
        expiration_date: Optional[str] = None,
        strike_range: Optional[Tuple[float, float]] = None,
        current_iv_rank: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze Greeks for options chain and provide recommendations
        
        Args:
            symbol: Stock ticker
            strategy: Trading strategy type
            expiration_date: Target expiration (YYYY-MM-DD)
            strike_range: Optional (min_strike, max_strike)
            current_iv_rank: Current IV Rank from Layer 16
            
        Returns:
            Complete Greeks analysis with scores and recommendations
        """
        try:
            # Get options chain with Greeks
            options_data = await self._fetch_options_chain(
                symbol, expiration_date, strike_range
            )
            
            if not options_data:
                return self._create_error_response("No options data available")
            
            # Get current stock price
            stock_price = await self._fetch_current_price(symbol)
            
            # Analyze Greeks for each strike
            strike_analyses = []
            for option in options_data:
                analysis = self._analyze_strike_greeks(
                    option,
                    stock_price,
                    strategy,
                    current_iv_rank
                )
                strike_analyses.append(analysis)
            
            # Find optimal strikes
            optimal_strikes = self._find_optimal_strikes(
                strike_analyses, strategy
            )
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                optimal_strikes['best'] if optimal_strikes['best'] else {}
            )
            
            # Generate AI recommendations
            recommendations = self._generate_recommendations(
                optimal_strikes,
                composite_score,
                strategy,
                current_iv_rank
            )
            
            # Build complete response
            return {
                'layer': 'Layer 17: Greeks Analysis',
                'symbol': symbol,
                'strategy': strategy,
                'timestamp': datetime.now().isoformat(),
                'stock_price': stock_price,
                'composite_score': composite_score,
                'score_breakdown': self._get_score_breakdown(
                    optimal_strikes['best'] if optimal_strikes['best'] else {}
                ),
                'optimal_strikes': optimal_strikes,
                'all_strikes_analysis': strike_analyses[:10],  # Top 10
                'recommendations': recommendations,
                'greeks_summary': self._create_greeks_summary(strike_analyses),
                'risk_assessment': self._assess_risk(optimal_strikes['best']),
                'layer_weight': self.weight,
                'weighted_score': composite_score * self.weight
            }
            
        except Exception as e:
            return self._create_error_response(str(e))
    
    async def _fetch_options_chain(
        self,
        symbol: str,
        expiration_date: Optional[str],
        strike_range: Optional[Tuple[float, float]]
    ) -> List[Dict]:
        """Fetch options chain with Greeks from Polygon.io"""
        try:
            # If no expiration specified, use next monthly expiration
            if not expiration_date:
                expiration_date = self._get_next_monthly_expiration()
            
            # Fetch options snapshot
            # Note: This is pseudo-code - adjust to your actual Polygon.io client
            options = await self.polygon.options_snapshot_chain(
                underlying_ticker=symbol,
                expiration_date=expiration_date
            )
            
            # Filter by strike range if provided
            if strike_range:
                min_strike, max_strike = strike_range
                options = [
                    opt for opt in options
                    if min_strike <= opt.get('strike_price', 0) <= max_strike
                ]
            
            return options
            
        except Exception as e:
            print(f"Error fetching options chain: {e}")
            return []
    
    async def _fetch_current_price(self, symbol: str) -> float:
        """Fetch current stock price"""
        try:
            # Fetch current quote
            quote = await self.polygon.get_last_quote(symbol)
            return (quote.get('bid', 0) + quote.get('ask', 0)) / 2
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return 0.0
    
    def _analyze_strike_greeks(
        self,
        option: Dict,
        stock_price: float,
        strategy: str,
        current_iv_rank: Optional[float]
    ) -> Dict[str, Any]:
        """Analyze Greeks for a single strike"""
        
        # Extract Greeks
        greeks = option.get('greeks', {})
        delta = abs(greeks.get('delta', 0))  # Use absolute for calls/puts
        gamma = greeks.get('gamma', 0)
        theta = greeks.get('theta', 0)
        vega = greeks.get('vega', 0)
        
        # Extract option details
        strike = option.get('strike_price', 0)
        iv = greeks.get('implied_volatility', 0) * 100  # Convert to percentage
        
        # Calculate scores
        delta_score = self._score_delta(delta, strategy)
        gamma_score = self._score_gamma(gamma, delta)
        theta_score = self._score_theta(theta, option.get('days_to_expiration', 0))
        vega_iv_score = self._score_vega_iv(vega, iv, current_iv_rank)
        
        # Calculate total score
        total_score = (
            delta_score * self.weights['delta'] +
            gamma_score * self.weights['gamma'] +
            theta_score * self.weights['theta'] +
            vega_iv_score * self.weights['vega_iv']
        )
        
        return {
            'strike': strike,
            'option_type': option.get('option_type', 'call'),
            'greeks': {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            },
            'iv': iv,
            'iv_rank': current_iv_rank,
            'scores': {
                'delta': delta_score,
                'gamma': gamma_score,
                'theta': theta_score,
                'vega_iv': vega_iv_score,
                'total': total_score
            },
            'distance_from_spot': abs(strike - stock_price),
            'moneyness': self._calculate_moneyness(strike, stock_price),
            'days_to_expiration': option.get('days_to_expiration', 0),
            'bid_ask_spread': option.get('ask', 0) - option.get('bid', 0),
            'volume': option.get('volume', 0),
            'open_interest': option.get('open_interest', 0)
        }
    
    def _score_delta(self, delta: float, strategy: str) -> float:
        """
        Score delta based on strategy
        
        Returns: 0-100 score
        """
        target_range = self.delta_targets.get(strategy, self.delta_targets['directional'])
        min_delta, max_delta = target_range
        
        # Perfect score if within target range
        if min_delta <= delta <= max_delta:
            # Score higher for center of range
            center = (min_delta + max_delta) / 2
            distance_from_center = abs(delta - center)
            range_width = max_delta - min_delta
            
            # Linear scoring from center
            score = 100 - (distance_from_center / range_width * 20)
            return max(80, score)  # Minimum 80 in range
        
        # Score decreases outside range
        if delta < min_delta:
            # Too far OTM
            distance = min_delta - delta
            score = max(0, 80 - (distance / min_delta * 100))
        else:
            # Too far ITM
            distance = delta - max_delta
            score = max(0, 80 - (distance / (1 - max_delta) * 100))
        
        return score
    
    def _score_gamma(self, gamma: float, delta: float) -> float:
        """
        Score gamma (acceleration potential)
        
        Higher gamma = faster profit acceleration, but also faster losses
        
        Returns: 0-100 score
        """
        # Gamma typically peaks at ATM (delta ~0.50)
        # High gamma is good for long options if near ATM
        
        if gamma >= self.thresholds['gamma']['high']:
            # High gamma
            if 0.45 <= delta <= 0.55:
                # Perfect: High gamma at ATM
                return 100
            elif 0.40 <= delta <= 0.60:
                # Good: High gamma near ATM
                return 85
            else:
                # Risky: High gamma far from ATM
                return 60
        
        elif gamma >= self.thresholds['gamma']['medium']:
            # Medium gamma
            if 0.40 <= delta <= 0.60:
                return 70
            else:
                return 50
        
        else:
            # Low gamma
            if delta > 0.70:
                # Expected for deep ITM
                return 60
            else:
                # Poor: Low gamma at ATM/OTM
                return 30
    
    def _score_theta(self, theta: float, dte: int) -> float:
        """
        Score theta decay
        
        Lower (less negative) theta is better for option buyers
        
        Returns: 0-100 score
        """
        # Theta should be negative for long options
        if theta > 0:
            return 0  # Something's wrong
        
        # Days to expiration matters
        if dte <= 7:
            # Very short-term: Theta accelerates
            if theta >= self.thresholds['theta']['low']:
                return 70  # Low decay despite short DTE
            elif theta >= self.thresholds['theta']['medium']:
                return 40  # Expected for short DTE
            else:
                return 10  # Very high decay, risky
        
        elif dte <= 30:
            # Short-term: Moderate theta expected
            if theta >= self.thresholds['theta']['low']:
                return 90  # Excellent: Low decay
            elif theta >= self.thresholds['theta']['medium']:
                return 70  # Good: Moderate decay
            else:
                return 30  # High decay
        
        else:
            # Longer-term: Lower theta expected
            if theta >= self.thresholds['theta']['low']:
                return 100  # Perfect: Minimal decay
            elif theta >= self.thresholds['theta']['medium']:
                return 85  # Good decay rate
            else:
                return 50  # Higher than expected
    
    def _score_vega_iv(
        self,
        vega: float,
        iv: float,
        iv_rank: Optional[float]
    ) -> float:
        """
        Score vega/IV setup
        
        Best setup: High vega + Low IV (profit from IV expansion)
        Worst setup: High vega + High IV (IV crush risk)
        
        Returns: 0-100 score
        """
        # Determine IV environment
        if iv_rank is not None:
            if iv_rank < 30:
                iv_state = 'low'
            elif iv_rank < 70:
                iv_state = 'medium'
            else:
                iv_state = 'high'
        else:
            # Fallback: Use IV directly
            if iv < 25:
                iv_state = 'low'
            elif iv < 50:
                iv_state = 'medium'
            else:
                iv_state = 'high'
        
        # Determine vega level
        if vega >= self.thresholds['vega']['high']:
            vega_level = 'high'
        elif vega >= self.thresholds['vega']['medium']:
            vega_level = 'medium'
        else:
            vega_level = 'low'
        
        # Scoring matrix
        score_matrix = {
            'high': {
                'low': 100,     # Perfect: High vega + Low IV
                'medium': 70,   # Good: High vega + Medium IV
                'high': 20      # Poor: High vega + High IV (crush risk)
            },
            'medium': {
                'low': 85,      # Good: Medium vega + Low IV
                'medium': 60,   # Neutral
                'high': 40      # Risky: Medium vega + High IV
            },
            'low': {
                'low': 60,      # Neutral: Low vega + Low IV
                'medium': 50,   # Neutral
                'high': 60      # Neutral: Low vega + High IV (less risk)
            }
        }
        
        return score_matrix[vega_level][iv_state]
    
    def _calculate_composite_score(self, best_strike: Dict) -> float:
        """Calculate composite Greeks score (0-100)"""
        if not best_strike:
            return 0.0
        
        scores = best_strike.get('scores', {})
        return scores.get('total', 0.0)
    
    def _get_score_breakdown(self, best_strike: Dict) -> Dict[str, Any]:
        """Get detailed score breakdown"""
        if not best_strike:
            return {}
        
        scores = best_strike.get('scores', {})
        greeks = best_strike.get('greeks', {})
        
        return {
            'delta_score': {
                'score': scores.get('delta', 0),
                'weight': self.weights['delta'],
                'weighted': scores.get('delta', 0) * self.weights['delta'],
                'actual_delta': greeks.get('delta', 0),
                'assessment': self._assess_delta(greeks.get('delta', 0))
            },
            'gamma_score': {
                'score': scores.get('gamma', 0),
                'weight': self.weights['gamma'],
                'weighted': scores.get('gamma', 0) * self.weights['gamma'],
                'actual_gamma': greeks.get('gamma', 0),
                'assessment': self._assess_gamma(greeks.get('gamma', 0))
            },
            'theta_score': {
                'score': scores.get('theta', 0),
                'weight': self.weights['theta'],
                'weighted': scores.get('theta', 0) * self.weights['theta'],
                'actual_theta': greeks.get('theta', 0),
                'daily_decay_dollars': greeks.get('theta', 0),
                'assessment': self._assess_theta(greeks.get('theta', 0))
            },
            'vega_iv_score': {
                'score': scores.get('vega_iv', 0),
                'weight': self.weights['vega_iv'],
                'weighted': scores.get('vega_iv', 0) * self.weights['vega_iv'],
                'actual_vega': greeks.get('vega', 0),
                'iv': best_strike.get('iv', 0),
                'iv_rank': best_strike.get('iv_rank'),
                'assessment': self._assess_vega_iv(
                    greeks.get('vega', 0),
                    best_strike.get('iv_rank')
                )
            }
        }
    
    def _find_optimal_strikes(
        self,
        strike_analyses: List[Dict],
        strategy: str
    ) -> Dict[str, Any]:
        """Find optimal strikes across different criteria"""
        
        if not strike_analyses:
            return {'best': None, 'alternatives': []}
        
        # Sort by total score
        sorted_strikes = sorted(
            strike_analyses,
            key=lambda x: x['scores']['total'],
            reverse=True
        )
        
        # Best overall
        best = sorted_strikes[0]
        
        # Find alternatives with different characteristics
        alternatives = []
        
        # Best gamma (for quick moves)
        best_gamma = max(
            strike_analyses,
            key=lambda x: x['scores']['gamma']
        )
        if best_gamma != best:
            alternatives.append({
                'type': 'Best Gamma',
                'reason': 'Maximum acceleration for quick moves',
                **best_gamma
            })
        
        # Best theta (for longer holds)
        best_theta = max(
            strike_analyses,
            key=lambda x: x['scores']['theta']
        )
        if best_theta != best and best_theta != best_gamma:
            alternatives.append({
                'type': 'Best Theta',
                'reason': 'Minimal decay for longer holds',
                **best_theta
            })
        
        # Best vega/IV (for IV expansion plays)
        best_vega = max(
            strike_analyses,
            key=lambda x: x['scores']['vega_iv']
        )
        if best_vega != best and best_vega not in [best_gamma, best_theta]:
            alternatives.append({
                'type': 'Best Vega/IV',
                'reason': 'Maximum profit from IV expansion',
                **best_vega
            })
        
        return {
            'best': best,
            'alternatives': alternatives[:3],  # Top 3 alternatives
            'total_analyzed': len(strike_analyses)
        }
    
    def _generate_recommendations(
        self,
        optimal_strikes: Dict,
        composite_score: float,
        strategy: str,
        iv_rank: Optional[float]
    ) -> Dict[str, Any]:
        """Generate AI-driven recommendations"""
        
        best = optimal_strikes.get('best')
        if not best:
            return {
                'action': 'NO_TRADE',
                'confidence': 0,
                'reasoning': 'No suitable strikes found',
                'warnings': ['Insufficient options data']
            }
        
        greeks = best.get('greeks', {})
        scores = best.get('scores', {})
        
        # Determine action
        if composite_score >= 85:
            action = 'STRONG_BUY'
            confidence = 90 + (composite_score - 85) * 2
        elif composite_score >= 70:
            action = 'BUY'
            confidence = 70 + (composite_score - 70)
        elif composite_score >= 50:
            action = 'CONSIDER'
            confidence = 50 + (composite_score - 50)
        else:
            action = 'AVOID'
            confidence = composite_score
        
        # Build reasoning
        reasoning_parts = []
        
        # Delta reasoning
        delta = greeks.get('delta', 0)
        if scores.get('delta', 0) >= 80:
            reasoning_parts.append(
                f"✅ Optimal delta ({delta:.2f}) for {strategy} strategy"
            )
        elif scores.get('delta', 0) < 50:
            reasoning_parts.append(
                f"⚠️ Delta ({delta:.2f}) suboptimal for {strategy}"
            )
        
        # Gamma reasoning
        gamma = greeks.get('gamma', 0)
        if scores.get('gamma', 0) >= 85:
            reasoning_parts.append(
                f"✅ Excellent gamma ({gamma:.4f}) for acceleration"
            )
        elif scores.get('gamma', 0) < 50:
            reasoning_parts.append(
                f"⚠️ Low gamma ({gamma:.4f}) limits profit acceleration"
            )
        
        # Theta reasoning
        theta = greeks.get('theta', 0)
        dte = best.get('days_to_expiration', 0)
        if scores.get('theta', 0) >= 80:
            reasoning_parts.append(
                f"✅ Low theta decay (${theta:.2f}/day with {dte} DTE)"
            )
        elif scores.get('theta', 0) < 40:
            reasoning_parts.append(
                f"⚠️ High theta decay (${theta:.2f}/day) - short-term trade only"
            )
        
        # Vega/IV reasoning
        vega = greeks.get('vega', 0)
        if scores.get('vega_iv', 0) >= 85:
            reasoning_parts.append(
                f"✅ Perfect vega/IV setup (Vega: ${vega:.2f}, IV Rank: {iv_rank or 'N/A'})"
            )
        elif scores.get('vega_iv', 0) < 40:
            reasoning_parts.append(
                f"⚠️ IV crush risk (Vega: ${vega:.2f}, High IV environment)"
            )
        
        # Generate warnings
        warnings = []
        
        if dte <= 7:
            warnings.append("⏰ Very short DTE - theta decay accelerating")
        
        if scores.get('vega_iv', 0) < 30:
            warnings.append("⚠️ High IV environment - consider IV crush risk")
        
        if best.get('bid_ask_spread', 0) > 0.5:
            warnings.append("💰 Wide bid-ask spread - use limit orders")
        
        if best.get('volume', 0) < 100:
            warnings.append("📊 Low volume - liquidity concerns")
        
        # Entry/exit suggestions
        entry_exit = self._generate_entry_exit_plan(best, strategy)
        
        return {
            'action': action,
            'confidence': min(100, confidence),
            'reasoning': reasoning_parts,
            'warnings': warnings,
            'recommended_strike': best.get('strike'),
            'option_type': best.get('option_type'),
            'entry_exit_plan': entry_exit,
            'alternatives': [
                {
                    'type': alt.get('type'),
                    'strike': alt.get('strike'),
                    'reason': alt.get('reason')
                }
                for alt in optimal_strikes.get('alternatives', [])
            ]
        }
    
    def _generate_entry_exit_plan(
        self,
        strike_analysis: Dict,
        strategy: str
    ) -> Dict[str, Any]:
        """Generate entry and exit plan based on Greeks"""
        
        greeks = strike_analysis.get('greeks', {})
        delta = greeks.get('delta', 0)
        gamma = greeks.get('gamma', 0)
        theta = greeks.get('theta', 0)
        dte = strike_analysis.get('days_to_expiration', 0)
        
        plan = {}
        
        # Entry timing
        if greeks.get('vega', 0) > 50 and strike_analysis.get('iv_rank', 50) < 30:
            plan['entry_timing'] = "Enter now - IV likely to expand"
        elif theta < -50 and dte <= 7:
            plan['entry_timing'] = "Enter immediately if taking trade - theta accelerating"
        else:
            plan['entry_timing'] = "Can scale in - no urgency"
        
        # Hold duration
        if dte <= 7:
            plan['recommended_hold'] = "1-3 days maximum (high theta)"
        elif dte <= 30:
            plan['recommended_hold'] = "3-10 days (moderate theta)"
        else:
            plan['recommended_hold'] = "Can hold 2+ weeks"
        
        # Profit target
        if delta >= 0.60:
            plan['profit_target'] = "40-60% (ITM has less % gain potential)"
        elif 0.45 <= delta <= 0.60:
            plan['profit_target'] = "60-100% (ATM sweet spot)"
        else:
            plan['profit_target'] = "100-200% (OTM high risk/reward)"
        
        # Stop loss
        if gamma > 0.02:
            plan['stop_loss'] = "30-40% max loss (high gamma cuts both ways)"
        else:
            plan['stop_loss'] = "40-50% max loss (lower volatility)"
        
        # Greeks-specific exits
        plan['greeks_exit_signals'] = []
        
        if delta < 0.30:
            plan['greeks_exit_signals'].append(
                "Exit if delta drops below 0.20 (too far OTM)"
            )
        
        if dte <= 7:
            plan['greeks_exit_signals'].append(
                f"Exit by day {max(1, dte-2)} to avoid final theta crush"
            )
        
        if strike_analysis.get('iv_rank', 50) > 70:
            plan['greeks_exit_signals'].append(
                "Exit on any IV contraction (IV crush risk)"
            )
        
        return plan
    
    def _create_greeks_summary(self, strike_analyses: List[Dict]) -> Dict[str, Any]:
        """Create summary statistics across all strikes"""
        
        if not strike_analyses:
            return {}
        
        deltas = [s['greeks']['delta'] for s in strike_analyses]
        gammas = [s['greeks']['gamma'] for s in strike_analyses]
        thetas = [s['greeks']['theta'] for s in strike_analyses]
        vegas = [s['greeks']['vega'] for s in strike_analyses]
        
        return {
            'delta': {
                'min': min(deltas),
                'max': max(deltas),
                'avg': statistics.mean(deltas),
                'median': statistics.median(deltas)
            },
            'gamma': {
                'min': min(gammas),
                'max': max(gammas),
                'avg': statistics.mean(gammas),
                'peak_gamma_strike': max(strike_analyses, key=lambda x: x['greeks']['gamma'])['strike']
            },
            'theta': {
                'min': min(thetas),
                'max': max(thetas),
                'avg': statistics.mean(thetas),
                'total_daily_decay': sum(thetas)
            },
            'vega': {
                'min': min(vegas),
                'max': max(vegas),
                'avg': statistics.mean(vegas),
                'median': statistics.median(vegas)
            }
        }
    
    def _assess_risk(self, best_strike: Optional[Dict]) -> Dict[str, Any]:
        """Assess overall risk based on Greeks"""
        
        if not best_strike:
            return {'risk_level': 'UNKNOWN', 'factors': []}
        
        greeks = best_strike.get('greeks', {})
        dte = best_strike.get('days_to_expiration', 0)
        
        risk_factors = []
        risk_score = 0  # 0 = low risk, 100 = high risk
        
        # Gamma risk
        gamma = greeks.get('gamma', 0)
        if gamma > 0.03:
            risk_factors.append("High gamma - rapid profit/loss swings")
            risk_score += 25
        
        # Theta risk
        theta = greeks.get('theta', 0)
        if theta < -75:
            risk_factors.append("High theta decay - losing $75+/day")
            risk_score += 30
        
        # DTE risk
        if dte <= 7:
            risk_factors.append("Very short DTE - theta acceleration")
            risk_score += 20
        elif dte <= 14:
            risk_factors.append("Short DTE - limited time for thesis")
            risk_score += 10
        
        # Vega/IV risk
        iv_rank = best_strike.get('iv_rank', 50)
        vega = greeks.get('vega', 0)
        if iv_rank > 70 and vega > 50:
            risk_factors.append("IV crush risk - high IV with high vega")
            risk_score += 25
        
        # Delta risk
        delta = greeks.get('delta', 0)
        if delta < 0.30:
            risk_factors.append("Far OTM - low probability of profit")
            risk_score += 20
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "VERY_HIGH"
        elif risk_score >= 50:
            risk_level = "HIGH"
        elif risk_score >= 30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'factors': risk_factors,
            'mitigation': self._suggest_risk_mitigation(risk_factors, greeks, dte)
        }
    
    def _suggest_risk_mitigation(
        self,
        risk_factors: List[str],
        greeks: Dict,
        dte: int
    ) -> List[str]:
        """Suggest ways to mitigate identified risks"""
        
        mitigations = []
        
        # Gamma mitigation
        if greeks.get('gamma', 0) > 0.03:
            mitigations.append("Use tighter stop losses due to high gamma")
            mitigations.append("Consider smaller position size")
        
        # Theta mitigation
        if greeks.get('theta', 0) < -75:
            mitigations.append("Exit quickly if trade doesn't work")
            mitigations.append("Consider longer DTE options")
        
        # DTE mitigation
        if dte <= 7:
            mitigations.append("Have clear profit target and exit plan")
            mitigations.append("Don't hold through final 2 days")
        
        # IV mitigation
        if any('IV crush' in factor for factor in risk_factors):
            mitigations.append("Exit before earnings or major events")
            mitigations.append("Consider selling premium instead")
        
        return mitigations
    
    # Helper assessment methods
    def _assess_delta(self, delta: float) -> str:
        """Assess delta quality"""
        if 0.50 <= delta <= 0.70:
            return "OPTIMAL - Great directional edge"
        elif 0.40 <= delta <= 0.50:
            return "GOOD - Slightly OTM"
        elif 0.70 <= delta <= 0.85:
            return "GOOD - In the money"
        elif 0.30 <= delta <= 0.40:
            return "SPECULATIVE - Further OTM"
        else:
            return "POOR - Too far from optimal range"
    
    def _assess_gamma(self, gamma: float) -> str:
        """Assess gamma quality"""
        if gamma >= 0.02:
            return "HIGH - Rapid acceleration potential"
        elif gamma >= 0.01:
            return "MEDIUM - Moderate acceleration"
        else:
            return "LOW - Slower profit acceleration"
    
    def _assess_theta(self, theta: float) -> str:
        """Assess theta quality"""
        if theta >= -20:
            return "EXCELLENT - Minimal decay"
        elif theta >= -50:
            return "GOOD - Manageable decay"
        elif theta >= -100:
            return "MODERATE - Significant daily cost"
        else:
            return "POOR - High decay risk"
    
    def _assess_vega_iv(self, vega: float, iv_rank: Optional[float]) -> str:
        """Assess vega/IV setup"""
        if iv_rank is None:
            return "UNKNOWN - No IV Rank data"
        
        if vega >= 50 and iv_rank < 30:
            return "PERFECT - High vega + Low IV (expansion edge)"
        elif vega >= 50 and iv_rank >= 70:
            return "RISKY - High vega + High IV (crush risk)"
        elif vega >= 25 and iv_rank < 50:
            return "GOOD - Moderate setup"
        else:
            return "NEUTRAL - No strong edge"
    
    # Utility methods
    def _calculate_moneyness(self, strike: float, stock_price: float) -> str:
        """Calculate if option is ITM, ATM, or OTM"""
        pct_diff = abs((strike - stock_price) / stock_price) * 100
        
        if pct_diff <= 1:
            return "ATM"
        elif strike < stock_price:
            return "ITM"
        else:
            return "OTM"
    
    def _get_next_monthly_expiration(self) -> str:
        """Get next monthly options expiration (3rd Friday)"""
        today = datetime.now()
        
        # Find next month
        if today.month == 12:
            year = today.year + 1
            month = 1
        else:
            year = today.year
            month = today.month + 1
        
        # Find 3rd Friday
        first_day = datetime(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        
        return third_friday.strftime('%Y-%m-%d')
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'layer': 'Layer 17: Greeks Analysis',
            'error': error_message,
            'composite_score': 0.0,
            'score_breakdown': {},
            'optimal_strikes': {'best': None, 'alternatives': []},
            'recommendations': {
                'action': 'ERROR',
                'confidence': 0,
                'reasoning': [f"Error: {error_message}"],
                'warnings': []
            },
            'layer_weight': self.weight,
            'weighted_score': 0.0
        }


# Example usage and testing
async def test_greeks_layer():
    """Test Layer 17 Greeks Analysis"""
    
    # This would normally be your Polygon.io client
    class MockPolygonClient:
        async def options_snapshot_chain(self, underlying_ticker, expiration_date):
            # Mock options chain data
            return [
                {
                    'strike_price': 150.0,
                    'option_type': 'call',
                    'days_to_expiration': 30,
                    'bid': 5.20,
                    'ask': 5.40,
                    'volume': 1500,
                    'open_interest': 5000,
                    'greeks': {
                        'delta': 0.55,
                        'gamma': 0.018,
                        'theta': -35.50,
                        'vega': 58.20,
                        'implied_volatility': 0.28
                    }
                },
                {
                    'strike_price': 155.0,
                    'option_type': 'call',
                    'days_to_expiration': 30,
                    'bid': 2.80,
                    'ask': 2.95,
                    'volume': 2000,
                    'open_interest': 7500,
                    'greeks': {
                        'delta': 0.42,
                        'gamma': 0.022,
                        'theta': -28.75,
                        'vega': 62.50,
                        'implied_volatility': 0.30
                    }
                }
            ]
        
        async def get_last_quote(self, symbol):
            return {'bid': 152.30, 'ask': 152.50}
    
    # Initialize layer
    polygon_client = MockPolygonClient()
    layer = Layer17GreeksAnalysis(polygon_client)
    
    # Run analysis
    result = await layer.analyze(
        symbol='AAPL',
        strategy='directional',
        current_iv_rank=25.0  # Low IV environment
    )
    
    print("\n" + "="*60)
    print("LAYER 17: GREEKS ANALYSIS TEST")
    print("="*60)
    print(f"\nComposite Score: {result['composite_score']:.2f}/100")
    print(f"Weighted Score: {result['weighted_score']:.2f}")
    
    print("\n--- Optimal Strike ---")
    best = result['optimal_strikes']['best']
    if best:
        print(f"Strike: ${best['strike']}")
        print(f"Delta: {best['greeks']['delta']:.3f}")
        print(f"Gamma: {best['greeks']['gamma']:.4f}")
        print(f"Theta: ${best['greeks']['theta']:.2f}/day")
        print(f"Vega: ${best['greeks']['vega']:.2f}")
    
    print("\n--- Recommendations ---")
    recs = result['recommendations']
    print(f"Action: {recs['action']}")
    print(f"Confidence: {recs['confidence']:.1f}%")
    print("\nReasoning:")
    for reason in recs['reasoning']:
        print(f"  {reason}")
    
    if recs['warnings']:
        print("\nWarnings:")
        for warning in recs['warnings']:
            print(f"  {warning}")
    
    print("\n--- Risk Assessment ---")
    risk = result['risk_assessment']
    print(f"Risk Level: {risk['risk_level']}")
    print(f"Risk Score: {risk['risk_score']}/100")
    
    return result


if __name__ == "__main__":
    # Run test
    asyncio.run(test_greeks_layer())
