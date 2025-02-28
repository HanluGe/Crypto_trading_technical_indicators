import pandas as pd
import numpy as np

class TechnicalIndicators:
    def __init__(self, data):
        """Initialize with market data."""
        self.data = data.copy()
    
    def get_tech_indicators(self):
        """Compute all technical indicators and return a DataFrame."""
        df = self.data.copy()
        
        df["SMA_20"] = self.calculate_sma(df["close"], window=20)
        df["SMA_50"] = self.calculate_sma(df["close"], window=500)
        df["SMA_200"] = self.calculate_sma(df["close"], window=2000)
        df["EMA_14"] = self.calculate_ema(df["close"], span=14)
        df["MACD"], df["Signal_Line"] = self.calculate_macd(df["close"])
        df["Upper_BB"], df["Lower_BB"] = self.calculate_bollinger_bands(df["close"])
        df["%K"], df["%D"] = self.calculate_stochastic_oscillator(df["high"], df["low"], df["close"])
        df["RSI"] = self.calculate_rsi(df["close"], period=14)
        
        return df

    @staticmethod
    def calculate_sma(data, window=14):
        """Calculate Simple Moving Average (SMA)."""
        return data.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(data, span=14):
        """Calculate Exponential Moving Average (EMA)."""
        return data.ewm(span=span, adjust=False).mean()

    @staticmethod
    def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
        """Calculate MACD (Moving Average Convergence Divergence)."""
        short_ema = TechnicalIndicators.calculate_ema(data, short_window)
        long_ema = TechnicalIndicators.calculate_ema(data, long_window)
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def calculate_bollinger_bands(data, window=20, num_std=2):
        """Calculate Bollinger Bands (Upper and Lower)."""
        sma = TechnicalIndicators.calculate_sma(data, window)
        rolling_std = data.rolling(window=window).std()
        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)
        return upper_band, lower_band

    @staticmethod
    def calculate_stochastic_oscillator(data_high, data_low, data_close, period=14):
        """Calculate Stochastic Oscillator %K and %D."""
        lowest_low = data_low.rolling(window=period).min()
        highest_high = data_high.rolling(window=period).max()
        k_percent = 100 * (data_close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=3).mean()  # 3-day moving average of %K
        return k_percent, d_percent

    @staticmethod
    def calculate_rsi(data, period=14):
        """Compute the Relative Strength Index (RSI) using EMA."""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi