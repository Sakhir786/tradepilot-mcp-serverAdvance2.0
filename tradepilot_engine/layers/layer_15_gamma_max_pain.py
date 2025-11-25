"""
Layer 15: Gamma & Max Pain Analysis (Raw Data Output)
Standalone module for options trading intelligence
Outputs RAW gamma/max pain data only - no signals or recommendations
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
    """
    
    def __init__(self):
        """Initialize Layer 15 Gamma & Max Pain analyzer"""
        
        # Distance thresholds (% from Max Pain)
        self.safe_distance = 5.0
        self.caution_distance = 2.0
        self.danger_distance = 1.0
        self.extreme_danger = 0.5
        
        # Days to expiration thresholds
        self.expiry_week = 7
        self.expiry_day_hours = 24
        
        # GEX thresholds
        self.gex_threshold_high = 1000000
        self.gex_threshold_medium = 100000
        
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
            Dict with RAW Max Pain, GEX, Pin Risk data
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
        
        # Calculate Pin Risk metrics
        pin_risk_result = self._calculate_pin_risk(
            current_price,
            max_pain_result['max_pain'],
            nearest_expiry
        )
        
        # Return RAW DATA ONLY - no signals or recommendations
        return {
            "success": True,
            
            # Max Pain Data
            "max_pain": max_pain_result['max_pain'],
            "max_pain_total_loss": max_pain_result['total_loss_at_max_pain'],
            "max_pain_confidence": max_pain_result['confidence'],
            "strike_min": max_pain_result['strike_range']['min'] if max_pain_result.get('strike_range') else None,
            "strike_max": max_pain_result['strike_range']['max'] if max_pain_result.get('strike_range') else None,
            "total_call_oi": max_pain_result.get('total_call_oi', 0),
            "total_put_oi": max_pain_result.get('total_put_oi', 0),
            "total_oi": max_pain_result.get('total_call_oi', 0) + max_pain_result.get('total_put_oi', 0),
            "put_call_oi_ratio": round(max_pain_result.get('total_put_oi', 0) / max_pain_result.get('total_call_oi', 1), 2) if max_pain_result.get('total_call_oi', 0) > 0 else None,
            
            # Price vs Max Pain
            "price_vs_max_pain": round(current_price - max_pain_result['max_pain'], 2) if max_pain_result['max_pain'] else None,
            "price_above_max_pain": current_price > max_pain_result['max_pain'] if max_pain_result['max_pain'] else None,
            "price_below_max_pain": current_price < max_pain_result['max_pain'] if max_pain_result['max_pain'] else None,
            "distance_to_max_pain": round(pin_risk_result['distance'], 2),
            "distance_to_max_pain_pct": round(pin_risk_result['distance_pct'], 2),
            
            # Distance Threshold Comparisons (raw booleans)
            "within_extreme_danger": pin_risk_result['distance_pct'] < self.extreme_danger,
            "within_danger": pin_risk_result['distance_pct'] < self.danger_distance,
            "within_caution": pin_risk_result['distance_pct'] < self.caution_distance,
            "within_safe": pin_risk_result['distance_pct'] < self.safe_distance,
            "beyond_safe": pin_risk_result['distance_pct'] >= self.safe_distance,
            
            # GEX Data
            "gex_total": round(gex_result['gex_total'], 0),
            "gex_call": round(gex_result['gex_call'], 0),
            "gex_put": round(gex_result['gex_put'], 0),
            "gex_regime": gex_result['gex_regime'],
            "gamma_wall": gex_result['gamma_wall'],
            
            # GEX Regime Comparisons (raw booleans)
            "gex_is_positive": gex_result['gex_total'] > 0,
            "gex_is_negative": gex_result['gex_total'] < 0,
            "gex_above_high_threshold": gex_result['gex_total'] > self.gex_threshold_high,
            "gex_below_neg_high_threshold": gex_result['gex_total'] < -self.gex_threshold_high,
            "gex_above_medium_threshold": gex_result['gex_total'] > self.gex_threshold_medium,
            "gex_below_neg_medium_threshold": gex_result['gex_total'] < -self.gex_threshold_medium,
            
            # Expiration Data
            "expiration": nearest_expiry,
            "days_to_expiry": pin_risk_result['days_to_expiry'],
            "is_expiry_day": pin_risk_result['days_to_expiry'] == 0,
            "is_expiry_week": pin_risk_result['days_to_expiry'] <= self.expiry_week,
            "is_monthly": pin_risk_result['days_to_expiry'] > self.expiry_week,
            
            # Pin Probability (raw calculation)
            "pin_probability_pct": pin_risk_result['pin_probability'],
            
            # Thresholds (for reference)
            "threshold_extreme_danger_pct": self.extreme_danger,
            "threshold_danger_pct": self.danger_distance,
            "threshold_caution_pct": self.caution_distance,
            "threshold_safe_pct": self.safe_distance,
            "threshold_gex_high": self.gex_threshold_high,
            "threshold_gex_medium": self.gex_threshold_medium,
            
            # Strikes Analyzed
            "strikes_analyzed": len(expiry_data),
            
            # Price Context
            "current_price": round(current_price, 2),
            
            # Timestamp
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_all_expirations(self,
                                options_data: Dict,
                                current_price: float) -> Dict:
        """Analyze Max Pain for all available expirations"""
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
                "distance_pct": round(pin_risk['distance_pct'], 2),
                "days_to_expiry": pin_risk['days_to_expiry'],
                "gex_total": round(gex['gex_total'], 0),
                "gex_regime": gex['gex_regime'],
                "total_oi": max_pain.get('total_call_oi', 0) + max_pain.get('total_put_oi', 0)
            }
        
        return {
            "current_price": current_price,
            "expirations": all_expirations,
            "nearest_expiry": self._get_nearest_expiration(parsed_data),
            "expiration_count": len(all_expirations)
        }
    
    # ==================== DATA PARSING ====================
    
    def _parse_polygon_data(self, 
                           results: List[Dict],
                           expiration_filter: Optional[str]) -> Dict:
        """Parse Polygon.io options chain data"""
        parsed = {}
        
        for contract in results:
            expiry = self._extract_expiration(contract)
            if not expiry:
                continue
            
            if expiration_filter and expiry != expiration_filter:
                continue
            
            if not self._is_valid_expiration(expiry):
                continue
            
            strike = self._extract_strike(contract)
            if strike is None:
                continue
            
            contract_type = self._extract_contract_type(contract)
            if not contract_type:
                continue
            
            oi = self._extract_open_interest(contract)
            gamma = self._extract_gamma(contract)
            delta = self._extract_delta(contract)
            
            if expiry not in parsed:
                parsed[expiry] = {}
            
            if strike not in parsed[expiry]:
                parsed[expiry][strike] = {
                    "call_oi": 0,
                    "put_oi": 0,
                    "call_gamma": 0,
                    "put_gamma": 0,
                    "call_delta": 0,
                    "put_delta": 0
                }
            
            if contract_type == "call":
                parsed[expiry][strike]["call_oi"] = oi
                parsed[expiry][strike]["call_gamma"] = gamma
                parsed[expiry][strike]["call_delta"] = delta
            else:
                parsed[expiry][strike]["put_oi"] = oi
                parsed[expiry][strike]["put_gamma"] = gamma
                parsed[expiry][strike]["put_delta"] = delta
        
        return parsed
    
    def _extract_expiration(self, contract: Dict) -> Optional[str]:
        """Extract expiration date from contract"""
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
        """Check if expiration is valid"""
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            today = datetime.now().date()
            max_date = today + timedelta(days=730)
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
        """Calculate Max Pain strike"""
        if not expiry_data:
            return {
                "max_pain": current_price,
                "total_loss_at_max_pain": 0,
                "confidence": "LOW",
                "strike_range": None,
                "total_call_oi": 0,
                "total_put_oi": 0
            }
        
        strikes = sorted(expiry_data.keys())
        
        if len(strikes) == 0:
            return {
                "max_pain": current_price,
                "total_loss_at_max_pain": 0,
                "confidence": "LOW",
                "strike_range": None,
                "total_call_oi": 0,
                "total_put_oi": 0
            }
        
        max_pain_strike = None
        min_total_loss = float('inf')
        loss_by_strike = {}
        
        for test_strike in strikes:
            call_loss = 0.0
            put_loss = 0.0
            
            for strike, data in expiry_data.items():
                if strike < test_strike:
                    call_loss += (test_strike - strike) * data["call_oi"] * 100
                
                if strike > test_strike:
                    put_loss += (strike - test_strike) * data["put_oi"] * 100
            
            total_loss = call_loss + put_loss
            loss_by_strike[test_strike] = total_loss
            
            if total_loss < min_total_loss:
                min_total_loss = total_loss
                max_pain_strike = test_strike
        
        total_oi = sum(d["call_oi"] + d["put_oi"] for d in expiry_data.values())
        
        if total_oi < 1000:
            confidence = "LOW"
        elif total_oi < 10000:
            confidence = "MEDIUM"
        else:
            confidence = "HIGH"
        
        return {
            "max_pain": max_pain_strike,
            "total_loss_at_max_pain": min_total_loss,
            "loss_by_strike": loss_by_strike,
            "confidence": confidence,
            "strike_range": {"min": min(strikes), "max": max(strikes)},
            "total_call_oi": sum(d["call_oi"] for d in expiry_data.values()),
            "total_put_oi": sum(d["put_oi"] for d in expiry_data.values())
        }
    
    # ==================== GEX CALCULATION ====================
    
    def _calculate_gex(self,
                      expiry_data: Dict[float, Dict],
                      current_price: float) -> Dict:
        """Calculate Gamma Exposure (GEX)"""
        if not expiry_data:
            return {
                "gex_total": 0,
                "gex_call": 0,
                "gex_put": 0,
                "gex_by_strike": {},
                "gex_regime": "NEUTRAL",
                "gamma_wall": None
            }
        
        gex_by_strike = {}
        total_call_gex = 0
        total_put_gex = 0
        
        for strike, data in expiry_data.items():
            call_gex = data["call_gamma"] * data["call_oi"] * 100 * (strike ** 2)
            put_gex = -1 * data["put_gamma"] * data["put_oi"] * 100 * (strike ** 2)
            
            strike_gex = call_gex + put_gex
            gex_by_strike[strike] = strike_gex
            
            total_call_gex += call_gex
            total_put_gex += put_gex
        
        total_gex = total_call_gex + total_put_gex
        
        # Classify GEX regime
        if total_gex > self.gex_threshold_high:
            gex_regime = "STRONG_POSITIVE"
        elif total_gex > self.gex_threshold_medium:
            gex_regime = "POSITIVE"
        elif total_gex < -self.gex_threshold_high:
            gex_regime = "STRONG_NEGATIVE"
        elif total_gex < -self.gex_threshold_medium:
            gex_regime = "NEGATIVE"
        else:
            gex_regime = "NEUTRAL"
        
        max_gex_strike = max(gex_by_strike.items(), key=lambda x: abs(x[1]))[0] if gex_by_strike else None
        
        return {
            "gex_total": total_gex,
            "gex_call": total_call_gex,
            "gex_put": total_put_gex,
            "gex_by_strike": gex_by_strike,
            "gex_regime": gex_regime,
            "gamma_wall": max_gex_strike
        }
    
    # ==================== PIN RISK CALCULATION ====================
    
    def _calculate_pin_risk(self,
                           current_price: float,
                           max_pain: float,
                           expiration_date: str) -> Dict:
        """Calculate Pin Risk metrics"""
        
        distance = abs(current_price - max_pain) if max_pain else 0
        distance_pct = (distance / current_price) * 100 if current_price else 0
        
        try:
            expiry_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
            today = datetime.now().date()
            days_to_expiry = (expiry_date - today).days
        except:
            days_to_expiry = 999
        
        # Calculate pin probability (raw calculation)
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
        
        if days_to_expiry == 0:
            time_multiplier = 1.5
        elif days_to_expiry <= 3:
            time_multiplier = 1.2
        elif days_to_expiry <= 7:
            time_multiplier = 1.0
        else:
            time_multiplier = 0.7
        
        pin_probability = min(100, int(base_prob * time_multiplier))
        
        return {
            "distance": distance,
            "distance_pct": distance_pct,
            "days_to_expiry": days_to_expiry,
            "pin_probability": pin_probability
        }
    
    def _empty_result(self, reason: str, current_price: float) -> Dict:
        """Return empty result structure"""
        return {
            "success": False,
            "error": reason,
            "max_pain": None,
            "max_pain_total_loss": None,
            "max_pain_confidence": None,
            "strike_min": None,
            "strike_max": None,
            "total_call_oi": 0,
            "total_put_oi": 0,
            "total_oi": 0,
            "put_call_oi_ratio": None,
            "price_vs_max_pain": None,
            "price_above_max_pain": None,
            "price_below_max_pain": None,
            "distance_to_max_pain": None,
            "distance_to_max_pain_pct": None,
            "within_extreme_danger": None,
            "within_danger": None,
            "within_caution": None,
            "within_safe": None,
            "beyond_safe": None,
            "gex_total": 0,
            "gex_call": 0,
            "gex_put": 0,
            "gex_regime": "UNKNOWN",
            "gamma_wall": None,
            "gex_is_positive": None,
            "gex_is_negative": None,
            "gex_above_high_threshold": None,
            "gex_below_neg_high_threshold": None,
            "gex_above_medium_threshold": None,
            "gex_below_neg_medium_threshold": None,
            "expiration": None,
            "days_to_expiry": None,
            "is_expiry_day": None,
            "is_expiry_week": None,
            "is_monthly": None,
            "pin_probability_pct": None,
            "threshold_extreme_danger_pct": self.extreme_danger,
            "threshold_danger_pct": self.danger_distance,
            "threshold_caution_pct": self.caution_distance,
            "threshold_safe_pct": self.safe_distance,
            "threshold_gex_high": self.gex_threshold_high,
            "threshold_gex_medium": self.gex_threshold_medium,
            "strikes_analyzed": 0,
            "current_price": current_price
        }
