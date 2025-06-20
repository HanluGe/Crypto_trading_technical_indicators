import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Basic Long/Short Trading Strategy
def long_short_strategy(signal_series):
    position = pd.Series(0, index=signal_series.index)
    position[signal_series == 1] = 1  
    position[signal_series == 0] = 0  
    position[signal_series == -1] = -1  
    return position

# Long-Only Trading Strategy
def long_only_strategy(signal_series):
    position = pd.Series(0, index=signal_series.index)
    
    for i in range(1, len(signal_series)): 
        if signal_series.iloc[i] == 1:
            position.iloc[i] = 1 
        elif signal_series.iloc[i] == -1:
            position.iloc[i] = 0 
        else: 
            position.iloc[i] = position.iloc[i - 1] 

    return position

# Short-Only Trading Strategy
def short_only_strategy(signal_series):
    position = pd.Series(0, index=signal_series.index)
    
    for i in range(1, len(signal_series)): 
        if signal_series.iloc[i] == 1:
            position.iloc[i] = 0 
        elif signal_series.iloc[i] == -1:
            position.iloc[i] = -1 
        else: 
            position.iloc[i] = position.iloc[i - 1] 

    return position

# Volatility-adjusted long-short trading strategy
def vol_long_short_strategy(signal_series, df_vol, vol_thresholds=None):
    """
    Volatility-adjusted long-short strategy:
    - When volatility is high, positions can be opened even with low signal strength (±1)
    - When volatility is low, only strong signals (±5) can trigger positions
    - Exit rule: If a position is held, close immediately when volatility exceeds the volatility at entry time

    Parameters:
    - signal_series: Series of trading signals (range: -5 to +5)
    - df_vol: Series of volatility values (same index as signal_series)
    - vol_thresholds: Dictionary of minimum volatility required for different signal strengths (optional)

    Returns:
    - position: Series of resulting positions after applying the strategy
    """
    if vol_thresholds is None:
        # Default volatility thresholds based on absolute signal strength (stronger signal → lower volatility requirement)
        vol_thresholds = {
            6: 0.0,
            5: 0.1,
            4: 0.15,
            3: 0.2,
            2: 0.25,
            1: 0.3
        }
    
    position = pd.Series(0, index=signal_series.index)

    for i in range(1, len(signal_series)):
        signal = signal_series.iloc[i]
        vol = df_vol.iloc[i]

        # Determine the required volatility threshold based on the signal strength
        abs_signal = abs(signal)
        
        if abs_signal == 0:
            continue
        else:
            if vol > vol_thresholds[min(int(abs_signal) + 1, 6)]:
                position.iloc[i] = np.sign(signal)
            
    return position

