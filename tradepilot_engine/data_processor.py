"""
Data Processor - Converts Polygon.io data to engine-ready format
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

class DataProcessor:
    """Process and prepare data from Polygon.io for indicator calculations"""
    
    @staticmethod
    def polygon_to_dataframe(candles_data: Dict) -> Optional[pd.DataFrame]:
        """
        Convert Polygon.io candles JSON to pandas DataFrame
        
        Args:
            candles_data: Raw JSON response from Polygon.io /candles endpoint
            
        Returns:
            DataFrame with OHLCV data or None if invalid
        """
        if not candles_data or "results" not in candles_data:
            return None
            
        results = candles_data["results"]
        if not results or len(results) == 0:
            return None
        
        # Extract data
        df = pd.DataFrame(results)
        
        # Rename columns to standard format
        column_mapping = {
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "t": "timestamp",
            "vw": "vwap",
            "n": "trades"
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert timestamp to datetime
        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("datetime")
        
        # Ensure required columns exist
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Sort by datetime
        df = df.sort_index()
        
        return df
    
    @staticmethod
    def validate_data(df: pd.DataFrame, min_bars: int = 200) -> bool:
        """
        Validate that DataFrame has sufficient data for analysis
        
        Args:
            df: DataFrame to validate
            min_bars: Minimum number of bars required
            
        Returns:
            True if valid, False otherwise
        """
        if df is None or df.empty:
            return False
            
        if len(df) < min_bars:
            return False
            
        # Check for required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_columns):
            return False
            
        # Check for NaN values
        if df[required_columns].isnull().any().any():
            return False
            
        return True
    
    @staticmethod
    def calculate_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic derived features needed by multiple layers
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Typical Price
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        
        # True Range
        df["true_range"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        
        # Price Change
        df["price_change"] = df["close"].diff()
        df["price_change_pct"] = df["close"].pct_change() * 100
        
        # Body Size (for candle analysis)
        df["body_size"] = abs(df["close"] - df["open"])
        df["upper_wick"] = df["high"] - np.maximum(df["close"], df["open"])
        df["lower_wick"] = np.minimum(df["close"], df["open"]) - df["low"]
        
        # Candle Range
        df["candle_range"] = df["high"] - df["low"]
        df["body_percent"] = np.where(
            df["candle_range"] > 0,
            df["body_size"] / df["candle_range"],
            0
        )
        
        # Bullish/Bearish
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)
        df["is_bearish"] = (df["close"] < df["open"]).astype(int)
        
        return df
    
    @staticmethod
    def get_latest_values(df: pd.DataFrame, columns: List[str]) -> Dict:
        """
        Get latest values for specified columns
        
        Args:
            df: DataFrame
            columns: List of column names
            
        Returns:
            Dictionary of column: value pairs
        """
        result = {}
        for col in columns:
            if col in df.columns:
                value = df[col].iloc[-1]
                # Convert numpy types to native Python types
                if isinstance(value, (np.integer, np.floating)):
                    value = float(value)
                result[col] = value
            else:
                result[col] = None
        return result
    
    @staticmethod
    def calculate_rolling_stats(df: pd.DataFrame, column: str, window: int) -> Dict:
        """
        Calculate rolling statistics for a column
        
        Args:
            df: DataFrame
            column: Column name
            window: Rolling window size
            
        Returns:
            Dictionary with mean, std, min, max
        """
        if column not in df.columns:
            return {}
            
        series = df[column]
        return {
            f"{column}_mean_{window}": float(series.rolling(window).mean().iloc[-1]),
            f"{column}_std_{window}": float(series.rolling(window).std().iloc[-1]),
            f"{column}_min_{window}": float(series.rolling(window).min().iloc[-1]),
            f"{column}_max_{window}": float(series.rolling(window).max().iloc[-1])
        }
