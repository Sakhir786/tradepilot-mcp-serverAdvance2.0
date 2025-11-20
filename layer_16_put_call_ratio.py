"""
Layer 16: Put/Call Ratio Analysis with AI Strategy Selector
Pine Script logic preserved + AI-driven contrarian trading intelligence

Calculates:
- Put/Call Ratio from options volume
- SMA(200) and STDEV(200) bands
- Contrarian buy/sell signals
- AI-driven strategy selection
- Cross-layer integration with L14 (IV) and L15 (Max Pain)

Theory:
- High P/C Ratio (>1.5) = Extreme fear → Contrarian BUY
- Low P/C Ratio (<0.7) = Extreme greed → Contrarian SELL
- Works best when combined with cheap IV and safe Max Pain
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math


class Layer16PutCallRatio:
    """
    Professional Put/Call Ratio analyzer with AI strategy selection
    
    Core Logic (Pine Script exact):
    - SMA(200) of P/C Ratio
    - STDEV(200) of P/C Ratio
    - Upper Band = MA + (1.5 × SD)
    - Lower Band = MA - (1.5 × SD)
    - BUY when PCR > Upper Band (extreme fear)
    - SELL when PCR < Lower Band (extreme greed)
    
    AI Enhancement:
    - Contrarian strategy selector
    - Cross-layer analysis (L14 + L15 + L16)
    - Win rate estimation
    - Position sizing
    - Supreme setup detection
    """
    
    def __init__(self):
        """Initialize Layer 16 Put/Call Ratio analyzer"""
        
        # ==================== CORE SETTINGS (Pine Script) ====================
        self.lookback = 200  # SMA/STDEV period
        self.std_multiplier = 1.5  # Standard deviation multiple
        
        # ==================== SENTIMENT THRESHOLDS ====================
        self.extreme_fear_level = 1.3  # PCR > 1.3 = extreme fear
        self.fear_level = 1.1  # PCR > 1.1 = fear
        self.neutral_upper = 1.0  # PCR = 1.0 = balanced
        self.neutral_lower = 0.9  # PCR = 0.9 = slightly bullish
        self.greed_level = 0.8  # PCR < 0.8 = greed
        self.extreme_greed_level = 0.7  # PCR < 0.7 = extreme greed
        
        # ==================== AI STRATEGY SETTINGS ====================
        self.strategies = {
            'SCALP': {'min_dte': 0, 'max_dte': 2, 'risk_level': 'HIGH'},
            'DAY': {'min_dte': 0, 'max_dte': 1, 'risk_level': 'HIGH'},
            'SWING': {'min_dte': 7, 'max_dte': 14, 'risk_level': 'MEDIUM'},
            'WEEK': {'min_dte': 14, 'max_dte': 30, 'risk_level': 'MEDIUM'},
            'MONTH': {'min_dte': 30, 'max_dte': 60, 'risk_level': 'LOW'},
            'LEAPS': {'min_dte': 90, 'max_dte': 365, 'risk_level': 'LOW'}
        }
        
    def analyze(self,
                options_data: Dict,
                pcr_history: Optional[List[float]] = None,
                iv_rank: Optional[float] = None,
                max_pain_score: Optional[float] = None) -> Dict:
        """
        Complete Put/Call Ratio analysis with AI strategy selection
        
        Args:
            options_data: Polygon.io options chain data (to calculate P/C)
            pcr_history: Optional pre-calculated P/C ratio history (200+ values)
            iv_rank: Optional IV Rank from Layer 14 (0-100)
            max_pain_score: Optional Max Pain score from Layer 15 (0-100)
            
        Returns:
            Complete analysis with signals, AI strategy, cross-layer intelligence
        """
        
        # Calculate current P/C Ratio
        current_pcr = self._calculate_pcr_from_options(options_data)
        
        if current_pcr is None:
            return self._empty_result("Unable to calculate P/C Ratio")
        
        # Get or calculate historical P/C
        if pcr_history is None:
            # If no history provided, can't calculate bands
            return self._simple_analysis(current_pcr)
        
        # Validate history
        if len(pcr_history) < self.lookback:
            return self._simple_analysis(current_pcr)
        
        # Add current to history
        pcr_series = pcr_history + [current_pcr]
        
        # ==================== CORE CALCULATIONS (EXACT PINE SCRIPT) ====================
        core_analysis = self._calculate_bands(pcr_series)
        
        # ==================== SENTIMENT CLASSIFICATION ====================
        sentiment_analysis = self._classify_sentiment(current_pcr, core_analysis)
        
        # ==================== AI STRATEGY SELECTION ====================
        ai_strategy = self._select_contrarian_strategy(
            pcr_current=current_pcr,
            pcr_signal=core_analysis['signal'],
            sentiment=sentiment_analysis,
            iv_rank=iv_rank,
            max_pain_score=max_pain_score
        )
        
        # ==================== CROSS-LAYER INTEGRATION ====================
        cross_layer = self._analyze_cross_layer(
            pcr_signal=core_analysis['signal'],
            sentiment=sentiment_analysis,
            iv_rank=iv_rank,
            max_pain_score=max_pain_score
        )
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "core": core_analysis,
            "sentiment": sentiment_analysis,
            "ai_strategy": ai_strategy,
            "cross_layer": cross_layer,
            "recommendation": self._generate_recommendation(
                core_analysis,
                sentiment_analysis,
                ai_strategy,
                cross_layer
            )
        }
    
    # ==================== P/C RATIO CALCULATION ====================
    
    def _calculate_pcr_from_options(self, options_data: Dict) -> Optional[float]:
        """
        Calculate Put/Call Ratio from Polygon.io options data
        
        P/C Ratio = Total Put Volume / Total Call Volume
        
        Args:
            options_data: Polygon.io options chain response
            
        Returns:
            P/C Ratio or None if insufficient data
        """
        if not options_data or "results" not in options_data:
            return None
        
        results = options_data["results"]
        if not results or len(results) == 0:
            return None
        
        total_call_volume = 0
        total_put_volume = 0
        
        for contract in results:
            # Extract contract type
            contract_type = self._extract_contract_type(contract)
            if not contract_type:
                continue
            
            # Extract volume
            volume = self._extract_volume(contract)
            if volume is None or volume <= 0:
                continue
            
            # Accumulate
            if contract_type == "call":
                total_call_volume += volume
            else:  # put
                total_put_volume += volume
        
        # Calculate P/C Ratio
        if total_call_volume == 0:
            return None  # Avoid division by zero
        
        pcr = total_put_volume / total_call_volume
        
        return pcr
    
    def _extract_contract_type(self, contract: Dict) -> Optional[str]:
        """Extract contract type (call/put)"""
        if "details" in contract and "contract_type" in contract["details"]:
            return contract["details"]["contract_type"].lower()
        elif "contract_type" in contract:
            return contract["contract_type"].lower()
        return None
    
    def _extract_volume(self, contract: Dict) -> Optional[int]:
        """Extract volume from contract"""
        return contract.get("volume", contract.get("day", {}).get("volume", 0))
    
    # ==================== CORE CALCULATIONS (EXACT PINE SCRIPT) ====================
    
    def _calculate_bands(self, pcr_series: List[float]) -> Dict:
        """
        Calculate bands exactly as Pine Script
        
        ma = SMA(pcr, 200)
        sd = STDEV(pcr, 200)
        upper_band = ma + (1.5 × sd)
        lower_band = ma - (1.5 × sd)
        """
        # Take last 200 values
        data = pcr_series[-self.lookback:] if len(pcr_series) >= self.lookback else pcr_series
        
        if len(data) == 0:
            return self._empty_core_result()
        
        current_pcr = pcr_series[-1]
        
        # Calculate SMA (exact)
        ma = sum(data) / len(data)
        
        # Calculate STDEV (exact)
        variance = sum((x - ma) ** 2 for x in data) / len(data)
        sd = math.sqrt(variance)
        
        # Calculate bands (exact)
        upper_band = ma + (self.std_multiplier * sd)
        lower_band = ma - (self.std_multiplier * sd)
        
        # Generate signal (exact Pine Script logic)
        if current_pcr > upper_band:
            signal = "BUY"  # Extreme fear → Contrarian buy
        elif current_pcr < lower_band:
            signal = "SELL"  # Extreme greed → Contrarian sell
        else:
            signal = "NEUTRAL"
        
        # Calculate distance from bands (for extremity)
        if current_pcr > upper_band:
            extremity = ((current_pcr - upper_band) / upper_band) * 100
        elif current_pcr < lower_band:
            extremity = ((lower_band - current_pcr) / lower_band) * 100
        else:
            extremity = 0
        
        return {
            "pcr_current": current_pcr,
            "ma": ma,
            "stdev": sd,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "signal": signal,
            "extremity": extremity,
            "distance_from_ma": abs(current_pcr - ma),
            "z_score": (current_pcr - ma) / sd if sd > 0 else 0
        }
    
    def _empty_core_result(self) -> Dict:
        """Return empty core result"""
        return {
            "pcr_current": None,
            "ma": None,
            "stdev": None,
            "upper_band": None,
            "lower_band": None,
            "signal": "NEUTRAL",
            "extremity": 0
        }
    
    # ==================== SENTIMENT CLASSIFICATION ====================
    
    def _classify_sentiment(self, pcr: float, core: Dict) -> Dict:
        """
        Classify market sentiment based on P/C Ratio
        
        PCR > 1.3: EXTREME_FEAR (way too many puts)
        PCR 1.1-1.3: FEAR (more puts than normal)
        PCR 0.9-1.1: NEUTRAL (balanced)
        PCR 0.7-0.9: GREED (more calls than normal)
        PCR < 0.7: EXTREME_GREED (way too many calls)
        """
        if pcr >= self.extreme_fear_level:
            sentiment = "EXTREME_FEAR"
            description = "Way too many puts - extreme pessimism"
            color = "RED"
        elif pcr >= self.fear_level:
            sentiment = "FEAR"
            description = "More puts than normal - bearish positioning"
            color = "ORANGE"
        elif pcr >= self.neutral_lower:
            sentiment = "NEUTRAL"
            description = "Balanced put/call positioning"
            color = "YELLOW"
        elif pcr >= self.greed_level:
            sentiment = "GREED"
            description = "More calls than normal - bullish positioning"
            color = "LIME"
        else:
            sentiment = "EXTREME_GREED"
            description = "Way too many calls - extreme optimism"
            color = "GREEN"
        
        # Contrarian interpretation
        if sentiment in ["EXTREME_FEAR", "FEAR"]:
            contrarian_view = "Contrarian BUY opportunity - market too bearish"
        elif sentiment in ["EXTREME_GREED", "GREED"]:
            contrarian_view = "Contrarian SELL opportunity - market too bullish"
        else:
            contrarian_view = "No extreme sentiment - wait for better setup"
        
        return {
            "sentiment": sentiment,
            "description": description,
            "color": color,
            "contrarian_view": contrarian_view,
            "is_extreme": sentiment in ["EXTREME_FEAR", "EXTREME_GREED"],
            "puts_vs_calls": "MORE_PUTS" if pcr > 1.0 else "MORE_CALLS" if pcr < 1.0 else "BALANCED"
        }
    
    # ==================== AI STRATEGY SELECTION ====================
    
    def _select_contrarian_strategy(self,
                                    pcr_current: float,
                                    pcr_signal: str,
                                    sentiment: Dict,
                                    iv_rank: Optional[float],
                                    max_pain_score: Optional[float]) -> Dict:
        """
        AI-driven contrarian strategy selector
        
        Analyzes P/C extremes + IV + Max Pain to recommend optimal trade
        """
        
        if pcr_signal == "NEUTRAL":
            return {
                "strategy": "WAIT",
                "trade_type": "NO_TRADE",
                "reason": "No extreme sentiment - no contrarian opportunity",
                "confidence": 0,
                "win_rate_estimate": 50,
                "position_size_pct": 0
            }
        
        # Score the contrarian opportunity
        opportunity_score = self._score_contrarian_opportunity(
            pcr_signal=pcr_signal,
            sentiment=sentiment,
            iv_rank=iv_rank,
            max_pain_score=max_pain_score
        )
        
        # Select best strategy
        if pcr_signal == "BUY":  # Extreme fear
            trade_type = "BUY_CALLS"
            best_strategy = self._select_best_strategy_for_long(
                opportunity_score,
                iv_rank
            )
        else:  # SELL - Extreme greed
            trade_type = "BUY_PUTS"
            best_strategy = self._select_best_strategy_for_short(
                opportunity_score,
                iv_rank
            )
        
        # Calculate optimal DTE
        optimal_dte = self._calculate_optimal_dte(
            best_strategy,
            sentiment['sentiment'],
            iv_rank
        )
        
        # Estimate win rate
        win_rate = self._estimate_contrarian_win_rate(
            opportunity_score,
            iv_rank,
            max_pain_score
        )
        
        # Calculate position size
        position_size = self._calculate_contrarian_position_size(
            opportunity_score,
            best_strategy,
            iv_rank
        )
        
        return {
            "strategy": best_strategy,
            "trade_type": trade_type,
            "optimal_dte": optimal_dte,
            "confidence": opportunity_score['total_score'],
            "win_rate_estimate": win_rate,
            "position_size_pct": position_size,
            "reason": opportunity_score['primary_reason'],
            "risk_level": self.strategies[best_strategy]['risk_level']
        }
    
    def _score_contrarian_opportunity(self,
                                      pcr_signal: str,
                                      sentiment: Dict,
                                      iv_rank: Optional[float],
                                      max_pain_score: Optional[float]) -> Dict:
        """Score the contrarian opportunity"""
        
        score = 0
        reasons = []
        
        # Factor 1: Sentiment Extremity (0-40 points)
        if sentiment['is_extreme']:
            score += 40
            reasons.append(f"Extreme sentiment ({sentiment['sentiment']})")
        else:
            score += 20
            reasons.append(f"Moderate sentiment ({sentiment['sentiment']})")
        
        # Factor 2: IV Rank (0-30 points)
        if iv_rank is not None:
            if pcr_signal == "BUY" and iv_rank < 30:
                score += 30
                reasons.append("Cheap IV - good for buying options")
            elif pcr_signal == "BUY" and iv_rank < 50:
                score += 20
                reasons.append("Moderate IV")
            elif pcr_signal == "SELL" and iv_rank > 60:
                score += 25
                reasons.append("High IV - good for selling premium")
            else:
                score += 10
        else:
            score += 10  # Neutral if no IV data
        
        # Factor 3: Max Pain Safety (0-30 points)
        if max_pain_score is not None:
            if max_pain_score > 70:
                score += 30
                reasons.append("Safe distance from Max Pain")
            elif max_pain_score > 50:
                score += 20
                reasons.append("Moderate Max Pain risk")
            else:
                score += 5
                reasons.append("High Max Pain risk - caution")
        else:
            score += 15  # Neutral if no Max Pain data
        
        return {
            'total_score': min(score, 100),
            'reasons': reasons,
            'primary_reason': reasons[0] if reasons else "Contrarian signal"
        }
    
    def _select_best_strategy_for_long(self,
                                       opportunity_score: Dict,
                                       iv_rank: Optional[float]) -> str:
        """Select best strategy for contrarian long (calls)"""
        
        score = opportunity_score['total_score']
        
        # High confidence + low IV = longer DTE
        if score > 80 and (iv_rank is None or iv_rank < 30):
            return "WEEK"  # 14-30 days
        elif score > 60:
            return "SWING"  # 7-14 days
        elif score > 40:
            return "WEEK"  # 14-30 days (safer)
        else:
            return "MONTH"  # 30-60 days (very safe)
    
    def _select_best_strategy_for_short(self,
                                        opportunity_score: Dict,
                                        iv_rank: Optional[float]) -> str:
        """Select best strategy for contrarian short (puts)"""
        
        score = opportunity_score['total_score']
        
        # High confidence = medium DTE
        if score > 80:
            return "SWING"  # 7-14 days
        elif score > 60:
            return "WEEK"  # 14-30 days
        else:
            return "MONTH"  # 30-60 days
    
    def _calculate_optimal_dte(self,
                               strategy: str,
                               sentiment: str,
                               iv_rank: Optional[float]) -> int:
        """Calculate optimal DTE for strategy"""
        
        base_dte = {
            'SCALP': 1,
            'DAY': 1,
            'SWING': 10,
            'WEEK': 21,
            'MONTH': 45,
            'LEAPS': 180
        }.get(strategy, 21)
        
        # Adjust for IV
        if iv_rank is not None:
            if iv_rank < 25:
                adjustment = 1.2  # Can go longer with cheap IV
            elif iv_rank > 70:
                adjustment = 0.8  # Go shorter with expensive IV
            else:
                adjustment = 1.0
        else:
            adjustment = 1.0
        
        return int(base_dte * adjustment)
    
    def _estimate_contrarian_win_rate(self,
                                      opportunity_score: int,
                                      iv_rank: Optional[float],
                                      max_pain_score: Optional[float]) -> int:
        """Estimate win rate for contrarian trade"""
        
        # Base win rates for contrarian trades
        base_rate = 68  # Contrarian trades historically work ~68% of time
        
        score = opportunity_score['total_score']
        
        # Adjust for opportunity strength
        if score > 80:
            rate = base_rate + 15
        elif score > 60:
            rate = base_rate + 10
        elif score > 40:
            rate = base_rate + 5
        else:
            rate = base_rate
        
        # Adjust for IV
        if iv_rank is not None and iv_rank < 30:
            rate += 5  # Cheap options boost
        
        # Adjust for Max Pain safety
        if max_pain_score is not None and max_pain_score > 70:
            rate += 5  # Safe setup boost
        
        return min(rate, 90)  # Cap at 90%
    
    def _calculate_contrarian_position_size(self,
                                            opportunity_score: Dict,
                                            strategy: str,
                                            iv_rank: Optional[float]) -> int:
        """Calculate position size as % of allocation"""
        
        base_size = {
            'SCALP': 15,
            'DAY': 20,
            'SWING': 40,
            'WEEK': 40,
            'MONTH': 35,
            'LEAPS': 30
        }.get(strategy, 30)
        
        score = opportunity_score['total_score']
        
        # Adjust for confidence
        if score > 80:
            multiplier = 1.5
        elif score > 60:
            multiplier = 1.2
        else:
            multiplier = 1.0
        
        # Adjust for IV
        if iv_rank is not None and iv_rank < 30:
            multiplier *= 1.2  # Size up with cheap IV
        
        final_size = base_size * multiplier
        
        return int(max(10, min(60, final_size)))  # Clamp 10-60%
    
    # ==================== CROSS-LAYER INTEGRATION ====================
    
    def _analyze_cross_layer(self,
                             pcr_signal: str,
                             sentiment: Dict,
                             iv_rank: Optional[float],
                             max_pain_score: Optional[float]) -> Dict:
        """
        Cross-layer analysis with L14 (IV) and L15 (Max Pain)
        
        Detects SUPREME SETUPS when all 3 layers align
        """
        
        if iv_rank is None or max_pain_score is None:
            return {
                "enabled": False,
                "reason": "Missing IV or Max Pain data"
            }
        
        # Check for supreme alignment
        is_supreme = False
        combined_signal = "NEUTRAL"
        combined_confidence = 0
        combined_win_rate = 0
        
        # SUPREME BUY SETUP
        if (pcr_signal == "BUY" and  # Extreme fear
            sentiment['is_extreme'] and  # Very extreme
            iv_rank < 30 and  # Cheap IV
            max_pain_score > 70):  # Safe from Max Pain
            
            is_supreme = True
            combined_signal = "SUPREME_BUY"
            combined_confidence = 95
            combined_win_rate = 87
        
        # STRONG BUY SETUP
        elif (pcr_signal == "BUY" and
              iv_rank < 40 and
              max_pain_score > 60):
            
            is_supreme = False
            combined_signal = "STRONG_BUY"
            combined_confidence = 85
            combined_win_rate = 78
        
        # SUPREME SELL SETUP
        elif (pcr_signal == "SELL" and  # Extreme greed
              sentiment['is_extreme'] and
              iv_rank > 60):  # Expensive IV
            
            is_supreme = True
            combined_signal = "SUPREME_SELL"
            combined_confidence = 90
            combined_win_rate = 80
        
        # STRONG SELL SETUP
        elif (pcr_signal == "SELL" and
              iv_rank > 50):
            
            is_supreme = False
            combined_signal = "STRONG_SELL"
            combined_confidence = 80
            combined_win_rate = 72
        
        return {
            "enabled": True,
            "iv_rank": iv_rank,
            "max_pain_score": max_pain_score,
            "pcr_signal": pcr_signal,
            "is_supreme": is_supreme,
            "combined_signal": combined_signal,
            "combined_confidence": combined_confidence,
            "combined_win_rate": combined_win_rate,
            "all_layers_aligned": is_supreme,
            "interpretation": self._interpret_cross_layer(
                combined_signal,
                is_supreme
            )
        }
    
    def _interpret_cross_layer(self,
                               combined_signal: str,
                               is_supreme: bool) -> str:
        """Interpret cross-layer signal"""
        
        interpretations = {
            "SUPREME_BUY": "🔥 SUPREME SETUP: Extreme fear + cheap IV + safe Max Pain = STRONG BUY CALLS",
            "STRONG_BUY": "✅ STRONG SETUP: Fear + cheap IV + reasonable Max Pain = BUY CALLS",
            "SUPREME_SELL": "🔥 SUPREME SETUP: Extreme greed + expensive IV = STRONG BUY PUTS or SELL PREMIUM",
            "STRONG_SELL": "✅ STRONG SETUP: Greed + high IV = BUY PUTS or SELL PREMIUM",
            "NEUTRAL": "⚠️ No strong alignment across layers - wait for better setup"
        }
        
        return interpretations.get(combined_signal, "No clear signal")
    
    # ==================== RECOMMENDATION ====================
    
    def _generate_recommendation(self,
                                core: Dict,
                                sentiment: Dict,
                                ai_strategy: Dict,
                                cross_layer: Dict) -> str:
        """Generate final trading recommendation"""
        
        # Check for supreme setup first
        if cross_layer.get('is_supreme'):
            signal = cross_layer['combined_signal']
            confidence = cross_layer['combined_confidence']
            win_rate = cross_layer['combined_win_rate']
            
            if signal == "SUPREME_BUY":
                return (f"🔥 SUPREME BUY SETUP (Confidence: {confidence}%, Win Rate: {win_rate}%)\n"
                        f"→ {ai_strategy['trade_type']} with {ai_strategy['optimal_dte']} DTE\n"
                        f"→ Position Size: {ai_strategy['position_size_pct']}%\n"
                        f"→ All 3 layers aligned for contrarian long")
            else:  # SUPREME_SELL
                return (f"🔥 SUPREME SELL SETUP (Confidence: {confidence}%, Win Rate: {win_rate}%)\n"
                        f"→ {ai_strategy['trade_type']} with {ai_strategy['optimal_dte']} DTE\n"
                        f"→ Position Size: {ai_strategy['position_size_pct']}%\n"
                        f"→ All 3 layers aligned for contrarian short")
        
        # Strong setup (not supreme)
        elif cross_layer.get('combined_confidence', 0) >= 80:
            return (f"✅ STRONG SETUP (Confidence: {cross_layer['combined_confidence']}%)\n"
                    f"→ {ai_strategy['trade_type']} with {ai_strategy['optimal_dte']} DTE\n"
                    f"→ Position Size: {ai_strategy['position_size_pct']}%")
        
        # Regular contrarian signal
        elif core['signal'] != "NEUTRAL":
            return (f"⚠️ CONTRARIAN {core['signal']} SIGNAL\n"
                    f"→ {sentiment['contrarian_view']}\n"
                    f"→ Strategy: {ai_strategy['strategy']}\n"
                    f"→ Confidence: {ai_strategy['confidence']}%")
        
        # No signal
        else:
            return "⏸️ WAIT - No extreme sentiment detected. No contrarian opportunity."
    
    # ==================== SIMPLE ANALYSIS (NO HISTORY) ====================
    
    def _simple_analysis(self, pcr: float) -> Dict:
        """Simple analysis when no history available"""
        
        sentiment = self._classify_sentiment(pcr, {})
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "core": {
                "pcr_current": pcr,
                "ma": None,
                "upper_band": None,
                "lower_band": None,
                "signal": "INSUFFICIENT_DATA"
            },
            "sentiment": sentiment,
            "ai_strategy": {
                "strategy": "WAIT",
                "reason": "Need 200+ days of P/C history for bands calculation"
            },
            "cross_layer": {
                "enabled": False,
                "reason": "Insufficient data"
            },
            "recommendation": f"Current P/C Ratio: {pcr:.2f} - {sentiment['description']}"
        }
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "success": False,
            "error": reason,
            "core": self._empty_core_result(),
            "sentiment": None,
            "ai_strategy": None,
            "cross_layer": None,
            "recommendation": f"Unable to analyze: {reason}"
        }
