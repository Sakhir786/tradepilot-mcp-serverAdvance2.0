"""
Layer 15: Gamma & Max Pain Analysis
Standalone module for options trading intelligence

Calculates:
- Max Pain: Strike where most options expire worthless
- GEX (Gamma Exposure): Market maker hedging pressure
- Pin Risk: Probability of price pinning at Max Pain
- Proximity Score: Distance-based risk assessment

Works with Polygon.io options chain data
Can be used standalone or integrated into trading systems
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math


class Layer15GammaMaxPain:
    """
    Professional Max Pain and Gamma Exposure calculator
    
    Max Pain Theory:
    - Options sellers (MMs) want maximum options to expire worthless
    - Strike with minimum total loss = Max Pain
    - Stock price tends to gravitate toward Max Pain near expiration
    
    Gamma Exposure (GEX):
    - Positive GEX: MMs sell when price rises (suppresses volatility)
    - Negative GEX: MMs buy when price rises (amplifies moves)
    
    Pin Risk:
    - High probability stock "pins" at Max Pain on expiry day
    - Closer to Max Pain + closer to expiry = higher pin risk
    """
    
    def __init__(self):
        """Initialize Layer 15 Gamma & Max Pain analyzer"""
        
        # Scoring thresholds (% distance from Max Pain)
        self.safe_distance = 5.0       # >5% away = safe
        self.caution_distance = 2.0    # 2-5% away = caution
        self.danger_distance = 1.0     # 1-2% away = danger
        self.extreme_danger = 0.5      # <0.5% away = extreme pin risk
        
        # Days to expiration thresholds
        self.expiry_week = 7           # Last week before expiry
        self.expiry_day_hours = 24     # Last 24 hours
        
        # GEX regime thresholds
        self.gex_threshold_high = 1000000    # Strong positive/negative
        self.gex_threshold_medium = 100000   # Medium
        
    def analyze(self, 
                options_data: Dict, 
                current_price: float,
                expiration_filter: Optional[str] = None) -> Dict:
        """
        Complete Max Pain and GEX analysis
        
        Args:
            options_data: Polygon.io options chain response
            current_price: Current stock price
            expiration_filter: Optional specific expiration date (YYYY-MM-DD)
            
        Returns:
            Complete analysis with Max Pain, GEX, Pin Risk, Score
        """
        
        # Validate input
        if not options_data or "results" not in options_data:
            return self._empty_result("No options data provided", current_price)
        
        results = options_data["results"]
        if not results or len(results) == 0:
            return self._empty_result("Empty options chain", current_price)
        
        # Parse options data
        parsed_data = self._parse_polygon_data(results, expiration_filter)
        
        if not parsed_data:
            return self._empty_result("No valid options data after parsing", current_price)
        
        # Get nearest expiration
        nearest_expiry = self._get_nearest_expiration(parsed_data)
        
        if not nearest_expiry:
            return self._empty_result("No valid expiration found", current_price)
        
        # Filter for nearest expiration
        expiry_data = parsed_data[nearest_expiry]
        
        # Calculate Max Pain
        max_pain_result = self._calculate_max_pain(expiry_data, current_price)
        
        # Calculate GEX
        gex_result = self._calculate_gex(expiry_data, current_price)
        
        # Calculate Pin Risk
        pin_risk_result = self._calculate_pin_risk(
            current_price,
            max_pain_result['max_pain'],
            nearest_expiry
        )
        
        # Calculate Score
        score_result = self._calculate_score(
            pin_risk_result['distance_pct'],
            pin_risk_result['days_to_expiry'],
            gex_result['gex_regime']
        )
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "expiration": nearest_expiry,
            "current_price": current_price,
            "max_pain": max_pain_result,
            "gex": gex_result,
            "pin_risk": pin_risk_result,
            "score": score_result,
            "strikes_analyzed": len(expiry_data),
            "recommendation": self._generate_recommendation(score_result, pin_risk_result)
        }
    
    def analyze_all_expirations(self,
                                options_data: Dict,
                                current_price: float) -> Dict:
        """
        Analyze Max Pain for all available expirations
        
        Useful for seeing Max Pain term structure
        """
        if not options_data or "results" not in options_data:
            return {"error": "No options data"}
        
        parsed_data = self._parse_polygon_data(options_data["results"], None)
        
        if not parsed_data:
            return {"error": "No valid options data"}
        
        all_expirations = {}
        
        for expiry_date, expiry_data in parsed_data.items():
            max_pain = self._calculate_max_pain(expiry_data, current_price)
            gex = self._calculate_gex(expiry_data, current_price)
            pin_risk = self._calculate_pin_risk(current_price, max_pain['max_pain'], expiry_date)
            
            all_expirations[expiry_date] = {
                "max_pain": max_pain['max_pain'],
                "distance_pct": pin_risk['distance_pct'],
                "days_to_expiry": pin_risk['days_to_expiry'],
                "gex_total": gex['gex_total'],
                "gex_regime": gex['gex_regime']
            }
        
        return {
            "current_price": current_price,
            "expirations": all_expirations,
            "nearest_expiry": self._get_nearest_expiration(parsed_data)
        }
    
    # ==================== DATA PARSING ====================
    
    def _parse_polygon_data(self, 
                           results: List[Dict],
                           expiration_filter: Optional[str]) -> Dict:
        """
        Parse Polygon.io options chain data
        
        Returns:
            Dictionary keyed by expiration date:
            {
                "2025-11-28": {
                    580: {
                        "call_oi": 5000,
                        "put_oi": 3000,
                        "call_gamma": 0.02,
                        "put_gamma": 0.02,
                        "call_delta": 0.55,
                        "put_delta": -0.45
                    }
                }
            }
        """
        parsed = {}
        
        for contract in results:
            # Extract expiration date
            expiry = self._extract_expiration(contract)
            if not expiry:
                continue
            
            # Filter by expiration if specified
            if expiration_filter and expiry != expiration_filter:
                continue
            
            # Filter out expired contracts
            if not self._is_valid_expiration(expiry):
                continue
            
            # Extract strike price
            strike = self._extract_strike(contract)
            if strike is None:
                continue
            
            # Extract contract type
            contract_type = self._extract_contract_type(contract)
            if not contract_type:
                continue
            
            # Extract data
            oi = self._extract_open_interest(contract)
            gamma = self._extract_gamma(contract)
            delta = self._extract_delta(contract)
            
            # Initialize expiration dict
            if expiry not in parsed:
                parsed[expiry] = {}
            
            # Initialize strike dict
            if strike not in parsed[expiry]:
                parsed[expiry][strike] = {
                    "call_oi": 0,
                    "put_oi": 0,
                    "call_gamma": 0,
                    "put_gamma": 0,
                    "call_delta": 0,
                    "put_delta": 0
                }
            
            # Populate data
            if contract_type == "call":
                parsed[expiry][strike]["call_oi"] = oi
                parsed[expiry][strike]["call_gamma"] = gamma
                parsed[expiry][strike]["call_delta"] = delta
            else:  # put
                parsed[expiry][strike]["put_oi"] = oi
                parsed[expiry][strike]["put_gamma"] = gamma
                parsed[expiry][strike]["put_delta"] = delta
        
        return parsed
    
    def _extract_expiration(self, contract: Dict) -> Optional[str]:
        """Extract expiration date from contract"""
        # Try different possible fields
        if "details" in contract and "expiration_date" in contract["details"]:
            return contract["details"]["expiration_date"]
        elif "expiration_date" in contract:
            return contract["expiration_date"]
        return None
    
    def _extract_strike(self, contract: Dict) -> Optional[float]:
        """Extract strike price from contract"""
        if "details" in contract and "strike_price" in contract["details"]:
            return float(contract["details"]["strike_price"])
        elif "strike_price" in contract:
            return float(contract["strike_price"])
        return None
    
    def _extract_contract_type(self, contract: Dict) -> Optional[str]:
        """Extract contract type (call/put)"""
        if "details" in contract and "contract_type" in contract["details"]:
            return contract["details"]["contract_type"].lower()
        elif "contract_type" in contract:
            return contract["contract_type"].lower()
        return None
    
    def _extract_open_interest(self, contract: Dict) -> int:
        """Extract open interest"""
        return contract.get("open_interest", 0)
    
    def _extract_gamma(self, contract: Dict) -> float:
        """Extract gamma from Greeks"""
        if "greeks" in contract and "gamma" in contract["greeks"]:
            return float(contract["greeks"]["gamma"])
        return 0.0
    
    def _extract_delta(self, contract: Dict) -> float:
        """Extract delta from Greeks"""
        if "greeks" in contract and "delta" in contract["greeks"]:
            return float(contract["greeks"]["delta"])
        return 0.0
    
    def _is_valid_expiration(self, expiry_str: str) -> bool:
        """Check if expiration is valid (not expired, within 2 years)"""
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            today = datetime.now().date()
            max_date = today + timedelta(days=730)  # 2 years
            
            return today <= expiry_date <= max_date
        except:
            return False
    
    def _get_nearest_expiration(self, parsed_data: Dict) -> Optional[str]:
        """Get nearest valid expiration date"""
        if not parsed_data:
            return None
        
        today = datetime.now().date()
        nearest = None
        min_days = float('inf')
        
        for expiry_str in parsed_data.keys():
            try:
                expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                days = (expiry_date - today).days
                if days >= 0 and days < min_days:
                    min_days = days
                    nearest = expiry_str
            except:
                continue
        
        return nearest
    
    # ==================== MAX PAIN CALCULATION ====================
    
    def _calculate_max_pain(self, 
                           expiry_data: Dict[float, Dict],
                           current_price: float) -> Dict:
        """
        Calculate Max Pain strike
        
        Max Pain = Strike where total loss for option holders is maximum
        
        Algorithm:
        1. For each strike, calculate:
           - Call loss = sum of (strike - X) * call_OI for all X < strike
           - Put loss = sum of (X - strike) * put_OI for all X > strike
           - Total loss = call_loss + put_loss
        2. Strike with MINIMUM total loss = Max Pain
        """
        if not expiry_data:
            return {
                "max_pain": current_price,
                "total_loss_at_max_pain": 0,
                "confidence": "LOW",
                "reason": "No data"
            }
        
        strikes = sorted(expiry_data.keys())
        
        if len(strikes) == 0:
            return {
                "max_pain": current_price,
                "total_loss_at_max_pain": 0,
                "confidence": "LOW",
                "reason": "No strikes"
            }
        
        max_pain_strike = None
        min_total_loss = float('inf')
        loss_by_strike = {}
        
        # Calculate total loss at each strike
        for test_strike in strikes:
            call_loss = 0.0
            put_loss = 0.0
            
            for strike, data in expiry_data.items():
                # Call holders lose if strike > test_strike
                if strike < test_strike:
                    call_loss += (test_strike - strike) * data["call_oi"] * 100
                
                # Put holders lose if strike < test_strike
                if strike > test_strike:
                    put_loss += (strike - test_strike) * data["put_oi"] * 100
            
            total_loss = call_loss + put_loss
            loss_by_strike[test_strike] = total_loss
            
            if total_loss < min_total_loss:
                min_total_loss = total_loss
                max_pain_strike = test_strike
        
        # Determine confidence
        confidence = self._calculate_max_pain_confidence(
            expiry_data,
            max_pain_strike,
            current_price
        )
        
        return {
            "max_pain": max_pain_strike,
            "total_loss_at_max_pain": min_total_loss,
            "loss_by_strike": loss_by_strike,
            "confidence": confidence,
            "strike_range": {"min": min(strikes), "max": max(strikes)},
            "total_call_oi": sum(d["call_oi"] for d in expiry_data.values()),
            "total_put_oi": sum(d["put_oi"] for d in expiry_data.values())
        }
    
    def _calculate_max_pain_confidence(self,
                                       expiry_data: Dict,
                                       max_pain_strike: float,
                                       current_price: float) -> str:
        """Calculate confidence in Max Pain calculation"""
        
        # Check open interest levels
        total_oi = sum(d["call_oi"] + d["put_oi"] for d in expiry_data.values())
        
        if total_oi < 1000:
            return "LOW"
        elif total_oi < 10000:
            return "MEDIUM"
        else:
            return "HIGH"
    
    # ==================== GEX CALCULATION ====================
    
    def _calculate_gex(self,
                      expiry_data: Dict[float, Dict],
                      current_price: float) -> Dict:
        """
        Calculate Gamma Exposure (GEX)
        
        GEX = Sum of (Gamma * OI * 100 * Strike^2)
        
        Positive GEX: MMs are net short gamma
        - When price rises, MMs sell stock to hedge (suppresses volatility)
        - Price tends to be range-bound
        
        Negative GEX: MMs are net long gamma  
        - When price rises, MMs buy stock to hedge (amplifies moves)
        - Price tends to be more volatile
        """
        if not expiry_data:
            return {
                "gex_total": 0,
                "gex_by_strike": {},
                "gex_regime": "NEUTRAL",
                "interpretation": "No data"
            }
        
        gex_by_strike = {}
        total_call_gex = 0
        total_put_gex = 0
        
        for strike, data in expiry_data.items():
            # Call GEX (positive - MMs short gamma)
            call_gex = data["call_gamma"] * data["call_oi"] * 100 * (strike ** 2)
            
            # Put GEX (negative - MMs long gamma)  
            put_gex = -1 * data["put_gamma"] * data["put_oi"] * 100 * (strike ** 2)
            
            strike_gex = call_gex + put_gex
            gex_by_strike[strike] = strike_gex
            
            total_call_gex += call_gex
            total_put_gex += put_gex
        
        total_gex = total_call_gex + total_put_gex
        
        # Determine GEX regime
        gex_regime = self._classify_gex_regime(total_gex)
        
        # Find largest GEX strike (gamma wall)
        max_gex_strike = max(gex_by_strike.items(), key=lambda x: abs(x[1]))[0] if gex_by_strike else None
        
        return {
            "gex_total": total_gex,
            "gex_call": total_call_gex,
            "gex_put": total_put_gex,
            "gex_by_strike": gex_by_strike,
            "gex_regime": gex_regime,
            "gamma_wall": max_gex_strike,
            "interpretation": self._interpret_gex(gex_regime, current_price, max_gex_strike)
        }
    
    def _classify_gex_regime(self, total_gex: float) -> str:
        """Classify GEX regime"""
        if total_gex > self.gex_threshold_high:
            return "STRONG_POSITIVE"
        elif total_gex > self.gex_threshold_medium:
            return "POSITIVE"
        elif total_gex < -self.gex_threshold_high:
            return "STRONG_NEGATIVE"
        elif total_gex < -self.gex_threshold_medium:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    def _interpret_gex(self, 
                      gex_regime: str,
                      current_price: float,
                      gamma_wall: Optional[float]) -> str:
        """Interpret GEX regime for trading"""
        interpretations = {
            "STRONG_POSITIVE": "Strong suppression of volatility. Price likely to stay range-bound. MMs will sell rallies and buy dips.",
            "POSITIVE": "Moderate suppression. Some resistance to large moves.",
            "NEUTRAL": "Balanced. Price can move freely based on fundamentals.",
            "NEGATIVE": "Moderate amplification. Expect larger-than-normal moves.",
            "STRONG_NEGATIVE": "Strong amplification. Risk of explosive moves in either direction. MMs will chase price."
        }
        
        base = interpretations.get(gex_regime, "Unknown regime")
        
        if gamma_wall:
            wall_text = f" Gamma wall at ${gamma_wall:.2f} acts as magnetic level."
            return base + wall_text
        
        return base
    
    # ==================== PIN RISK CALCULATION ====================
    
    def _calculate_pin_risk(self,
                           current_price: float,
                           max_pain: float,
                           expiration_date: str) -> Dict:
        """
        Calculate Pin Risk
        
        Pin Risk = Probability price will "pin" at Max Pain on expiry
        
        Factors:
        1. Distance from Max Pain (closer = higher risk)
        2. Days to expiration (closer = higher risk)
        3. Open interest concentration (higher = higher risk)
        """
        
        # Calculate distance
        distance = abs(current_price - max_pain)
        distance_pct = (distance / current_price) * 100
        
        # Calculate days to expiration
        try:
            expiry_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
            today = datetime.now().date()
            days_to_expiry = (expiry_date - today).days
        except:
            days_to_expiry = 999  # Unknown
        
        # Classify pin risk
        pin_risk_level = self._classify_pin_risk(distance_pct, days_to_expiry)
        
        # Calculate probability
        pin_probability = self._calculate_pin_probability(distance_pct, days_to_expiry)
        
        return {
            "distance": distance,
            "distance_pct": distance_pct,
            "days_to_expiry": days_to_expiry,
            "pin_risk_level": pin_risk_level,
            "pin_probability": pin_probability,
            "warning": self._generate_pin_warning(pin_risk_level, days_to_expiry)
        }
    
    def _classify_pin_risk(self, distance_pct: float, days_to_expiry: int) -> str:
        """Classify pin risk level"""
        
        # Expiry day special case
        if days_to_expiry == 0:
            if distance_pct < self.extreme_danger:
                return "EXTREME"
            elif distance_pct < self.danger_distance:
                return "HIGH"
            else:
                return "MODERATE"
        
        # Last week before expiry
        elif days_to_expiry <= self.expiry_week:
            if distance_pct < self.danger_distance:
                return "HIGH"
            elif distance_pct < self.caution_distance:
                return "MODERATE"
            else:
                return "LOW"
        
        # More than a week out
        else:
            if distance_pct < self.caution_distance:
                return "MODERATE"
            else:
                return "LOW"
    
    def _calculate_pin_probability(self, distance_pct: float, days_to_expiry: int) -> int:
        """Calculate pin probability (0-100%)"""
        
        # Base probability from distance
        if distance_pct < self.extreme_danger:
            base_prob = 90
        elif distance_pct < self.danger_distance:
            base_prob = 70
        elif distance_pct < self.caution_distance:
            base_prob = 50
        elif distance_pct < self.safe_distance:
            base_prob = 30
        else:
            base_prob = 10
        
        # Adjust for days to expiry
        if days_to_expiry == 0:
            time_multiplier = 1.5
        elif days_to_expiry <= 3:
            time_multiplier = 1.2
        elif days_to_expiry <= 7:
            time_multiplier = 1.0
        else:
            time_multiplier = 0.7
        
        final_prob = min(100, int(base_prob * time_multiplier))
        
        return final_prob
    
    def _generate_pin_warning(self, pin_risk_level: str, days_to_expiry: int) -> str:
        """Generate pin risk warning"""
        if pin_risk_level == "EXTREME":
            return f"⚠️ EXTREME PIN RISK - Price very close to Max Pain with {days_to_expiry} days left. Avoid weeklies!"
        elif pin_risk_level == "HIGH":
            return f"⚠️ HIGH PIN RISK - Price approaching Max Pain. {days_to_expiry} days to expiry."
        elif pin_risk_level == "MODERATE":
            return f"⚠️ MODERATE PIN RISK - Monitor Max Pain proximity. {days_to_expiry} days to expiry."
        else:
            return f"✅ LOW PIN RISK - Safe distance from Max Pain. {days_to_expiry} days to expiry."
    
    # ==================== SCORING ====================
    
    def _calculate_score(self,
                        distance_pct: float,
                        days_to_expiry: int,
                        gex_regime: str) -> Dict:
        """
        Calculate 0-100 score based on Max Pain proximity and GEX
        
        Scoring:
        - Far from Max Pain (>5%): 90-100 (safe)
        - Moderate distance (2-5%): 60-80 (caution)
        - Close to Max Pain (1-2%): 30-50 (danger)
        - At Max Pain (<1%): 0-20 (extreme danger)
        
        Adjustments:
        - Last week: -10 points
        - Expiry day: -20 points
        - Negative GEX: -10 points (more volatile)
        """
        
        # Base score from distance
        if distance_pct > self.safe_distance:
            base_score = 95
        elif distance_pct > self.caution_distance:
            base_score = 70
        elif distance_pct > self.danger_distance:
            base_score = 40
        elif distance_pct > self.extreme_danger:
            base_score = 20
        else:
            base_score = 5
        
        # Time decay penalty
        if days_to_expiry == 0:
            time_penalty = -20
        elif days_to_expiry <= 3:
            time_penalty = -15
        elif days_to_expiry <= 7:
            time_penalty = -10
        else:
            time_penalty = 0
        
        # GEX adjustment
        if "NEGATIVE" in gex_regime:
            gex_penalty = -10
        else:
            gex_penalty = 0
        
        final_score = max(0, min(100, base_score + time_penalty + gex_penalty))
        
        return {
            "score": final_score,
            "base_score": base_score,
            "time_penalty": time_penalty,
            "gex_penalty": gex_penalty,
            "rating": self._score_to_rating(final_score)
        }
    
    def _score_to_rating(self, score: int) -> str:
        """Convert score to rating"""
        if score >= 80:
            return "EXCELLENT"
        elif score >= 60:
            return "GOOD"
        elif score >= 40:
            return "FAIR"
        elif score >= 20:
            return "POOR"
        else:
            return "DANGER"
    
    # ==================== RECOMMENDATION ====================
    
    def _generate_recommendation(self, 
                                score_result: Dict,
                                pin_risk_result: Dict) -> str:
        """Generate trading recommendation"""
        
        score = score_result['score']
        pin_risk = pin_risk_result['pin_risk_level']
        days = pin_risk_result['days_to_expiry']
        
        if score >= 80:
            return f"✅ SAFE - Far from Max Pain. Good for weekly options. {days} days to expiry."
        
        elif score >= 60:
            return f"⚠️ CAUTION - Moderate distance from Max Pain. Consider longer DTE. {days} days to expiry."
        
        elif score >= 40:
            return f"⚠️ WARNING - Close to Max Pain. Avoid weeklies, use monthlies. {days} days to expiry."
        
        elif score >= 20:
            return f"🚨 DANGER - Very close to Max Pain with {days} days left. High pin risk!"
        
        else:
            return f"🚨 EXTREME DANGER - At Max Pain on/near expiry! Avoid all directional options!"
    
    def _empty_result(self, reason: str, current_price: float) -> Dict:
        """Return empty result structure"""
        return {
            "success": False,
            "error": reason,
            "current_price": current_price,
            "max_pain": None,
            "gex": None,
            "pin_risk": None,
            "score": None,
            "recommendation": f"Unable to calculate: {reason}"
        }
