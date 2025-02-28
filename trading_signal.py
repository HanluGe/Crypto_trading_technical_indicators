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
