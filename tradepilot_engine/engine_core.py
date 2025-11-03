def analyze(self, candles_data: Dict, symbol: str, timeframe: str = "day") -> Dict:
    """
    Run full analysis through all 10 layers
    """
    # --- Normalize and prepare data ---
    try:
        df = self.data_processor.polygon_to_dataframe(candles_data)
    except Exception as e:
        return {"error": f"Data conversion failed: {str(e)}", "symbol": symbol, "timeframe": timeframe}

    if df is None or not self.data_processor.validate_data(df):
        return {
            "error": "Insufficient or invalid data",
            "symbol": symbol,
            "timeframe": timeframe,
            "bars_received": len(df) if df is not None else 0
        }

    # --- Handle scalp / minute-level data ---
    if timeframe in ["minute", "min", "scalp"]:
        # Low volatility fix
        if df["close"].std() < 0.02:
            df["close"] = df["close"] + (df["close"].mean() * 0.0001)

        # Adaptive lookback settings for short-term analysis
        self.short_window = 5
        self.medium_window = 9
        self.long_window = 21
    else:
        self.short_window = 14
        self.medium_window = 26
        self.long_window = 50

    # --- Calculate features ---
    try:
        df = self.data_processor.calculate_basic_features(df)
    except Exception as e:
        return {"error": f"Feature calculation failed: {str(e)}"}

    # --- Run each analysis layer ---
    results = {
        "symbol": symbol,
        "timeframe": timeframe,
        "bars_analyzed": len(df),
        "latest_price": float(df["close"].iloc[-1]),
        "latest_datetime": str(df.index[-1]),
        "layers": {}
    }

    try:
        results["layers"]["layer_1_momentum"] = self.layers["layer_1_momentum"].analyze(df)
        results["layers"]["layer_2_volume"] = self.layers["layer_2_volume"].analyze(df)
        results["layers"]["layer_3_divergence"] = self.layers["layer_3_divergence"].analyze(df)
        results["layers"]["layer_4_volume_strength"] = self.layers["layer_4_volume_strength"].analyze(df)
        results["layers"]["layer_5_trend"] = self.layers["layer_5_trend"].analyze(df)
        results["layers"]["layer_6_structure"] = self.layers["layer_6_structure"].analyze(df)
        results["layers"]["layer_7_liquidity"] = self.layers["layer_7_liquidity"].analyze(df)
        results["layers"]["layer_8_volatility_regime"] = self.layers["layer_8_volatility_regime"].analyze(df)
        results["layers"]["layer_9_confirmation"] = self.layers["layer_9_confirmation"].analyze(
            df, results["layers"]
        )
        results["layers"]["layer_10_candle_intelligence"] = self.layers["layer_10_candle_intelligence"].analyze(df)
    except Exception as e:
        return {"error": f"Layer analysis failed: {str(e)}"}

    # --- Combine overall signal ---
    try:
        results["overall_signal"] = self._generate_overall_signal(results["layers"])
    except Exception as e:
        return {"error": f"Signal aggregation failed: {str(e)}"}

    return clean_for_json(results)
