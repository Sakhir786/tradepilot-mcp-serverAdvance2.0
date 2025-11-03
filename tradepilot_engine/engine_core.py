"""
TradePilot Engine Core - Orchestrates all 10 analysis layers
"""

import pandas as pd
from typing import Dict, List, Optional
from .data_processor import DataProcessor
from .json_utils import clean_for_json
from .layers import (
    Layer1Momentum,
    Layer2Volume,
    Layer3Divergence,
    Layer4VolumeStrength,
    Layer5Trend,
    Layer6Structure,
    Layer7Liquidity,
    Layer8VolatilityRegime,
    Layer9Confirmation,
    Layer10CandleIntelligence
)


class TradePilotEngine:
    """Main engine that runs all 10 layers of technical analysis"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.layers = {
            "layer_1_momentum": Layer1Momentum(),
            "layer_2_volume": Layer2Volume(),
            "layer_3_divergence": Layer3Divergence(),
            "layer_4_volume_strength": Layer4VolumeStrength(),
            "layer_5_trend": Layer5Trend(),
            "layer_6_structure": Layer6Structure(),
            "layer_7_liquidity": Layer7Liquidity(),
            "layer_8_volatility_regime": Layer8VolatilityRegime(),
            "layer_9_confirmation": Layer9Confirmation(),
            "layer_10_candle_intelligence": Layer10CandleIntelligence()
        }

    def analyze(self, candles_data: Dict, symbol: str, timeframe: str = "day") -> Dict:
        """Run full analysis through all 10 layers"""
        df = self.data_processor.polygon_to_dataframe(candles_data)

        if df is None or not self.data_processor.validate_data(df):
            return {
                "error": "Insufficient or invalid data",
                "symbol": symbol,
                "timeframe": timeframe,
                "bars_received": len(df) if df is not None else 0
            }

        df = self.data_processor.calculate_basic_features(df)

        results = {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars_analyzed": len(df),
            "latest_price": float(df["close"].iloc[-1]),
            "latest_datetime": str(df.index[-1]),
            "layers": {}
        }

        # Run all analysis layers
        results["layers"]["layer_1_momentum"] = self.layers["layer_1_momentum"].analyze(df)
        results["layers"]["layer_2_volume"] = self.layers["layer_2_volume"].analyze(df)
        results["layers"]["layer_3_divergence"] = self.layers["layer_3_divergence"].analyze(df)
        results["layers"]["layer_4_volume_strength"] = self.layers["layer_4_volume_strength"].analyze(df)
        results["layers"]["layer_5_trend"] = self.layers["layer_5_trend"].analyze(df)
        results["layers"]["layer_6_structure"] = self.layers["layer_6_structure"].analyze(df)
        results["layers"]["layer_7_liquidity"] = self.layers["layer_7_liquidity"].analyze(df)
        results["layers"]["layer_8_volatility_regime"] = self.layers["layer_8_volatility_regime"].analyze(df)
        results["layers"]["layer_9_confirmation"] = self.layers["layer_9_confirmation"].analyze(df, results["layers"])
        results["layers"]["layer_10_candle_intelligence"] = self.layers["layer_10_candle_intelligence"].analyze(df)

        results["overall_signal"] = self._generate_overall_signal(results["layers"])
        return clean_for_json(results)

    def get_signal_summary(self, candles_data: Dict, symbol: str) -> Dict:
        """Get condensed signal summary for quick decision making"""
        full_analysis = self.analyze(candles_data, symbol)

        if "error" in full_analysis:
            return full_analysis

        layers = full_analysis["layers"]

        summary = {
            "symbol": symbol,
            "latest_price": full_analysis["latest_price"],
            "momentum_bias": layers["layer_1_momentum"]["signal"],
            "momentum_score": layers["layer_1_momentum"]["momentum_score"],
            "trend_direction": layers["layer_5_trend"]["trend_direction"],
            "trend_strength": layers["layer_5_trend"]["adx"],
            "volume_flow": layers["layer_2_volume"]["volume_flow_score"],
            "volatility_regime": layers["layer_8_volatility_regime"]["regime"],
            "structure_bias": layers["layer_6_structure"]["bias"],
            "confirmation_signal": layers["layer_9_confirmation"]["signal"],
            "overall_signal": full_analysis["overall_signal"]["direction"],
            "overall_confidence": full_analysis["overall_signal"]["confidence"],
            "recommendation": full_analysis["overall_signal"]["recommendation"]
        }

        return clean_for_json(summary)

    def _generate_overall_signal(self, layers: Dict) -> Dict:
        """Generate overall signal based on all layer outputs"""
        signals = []
        weights = []

        # Momentum layer (25%)
        momentum_score = layers["layer_1_momentum"]["momentum_score"]
        signals.append(1 if momentum_score > 0 else -1 if momentum_score < 0 else 0)
        weights.append(0.25)

        # Volume layer (20%)
        volume_score = layers["layer_2_volume"]["volume_flow_score"]
        signals.append(1 if volume_score > 0 else -1 if volume_score < 0 else 0)
        weights.append(0.20)

        # Trend layer (20%)
        trend_score = layers["layer_5_trend"]["trend_score"]
        signals.append(1 if trend_score > 0 else -1 if trend_score < 0 else 0)
        weights.append(0.20)

        # Volatility (10%)
        vol_regime = layers["layer_8_volatility_regime"]["regime"]
        vol_signal = 1 if vol_regime in ["LOW", "NORMAL"] else -1 if vol_regime == "EXTREME" else 0
        signals.append(vol_signal)
        weights.append(0.10)

        # Confirmation (25%)
        confirmation = layers["layer_9_confirmation"]["confirmation_signal"]
        signals.append(confirmation)
        weights.append(0.25)

        weighted_signal = sum(s * w for s, w in zip(signals, weights))
        confidence = abs(weighted_signal) * 100

        if weighted_signal > 0.3:
            direction = "BULLISH"
            if confidence > 70:
                recommendation = "STRONG BUY"
            elif confidence > 50:
                recommendation = "BUY"
            else:
                recommendation = "WEAK BUY"
        elif weighted_signal < -0.3:
            direction = "BEARISH"
            if confidence > 70:
                recommendation = "STRONG SELL"
            elif confidence > 50:
                recommendation = "SELL"
            else:
                recommendation = "WEAK SELL"
        else:
            direction = "NEUTRAL"
            recommendation = "HOLD"

        return {
            "direction": direction,
            "confidence": round(confidence, 2),
            "weighted_signal": round(weighted_signal, 3),
            "recommendation": recommendation,
            "contributing_signals": {
                "momentum": signals[0],
                "volume": signals[1],
                "trend": signals[2],
                "volatility": signals[3],
                "confirmation": signals[4],
            },
        }
