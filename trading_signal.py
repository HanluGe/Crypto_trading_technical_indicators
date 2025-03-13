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

# # Volitility-adjusted Long/Short Trading Strategy
# def vol_long_short_strategy(signal_series, df_vol, high_vol_threshold=0.02):
#     position = pd.Series(0, index=signal_series.index)

#     for i in range(1, len(signal_series)):
#         if position.iloc[i - 1] == 0:  # 只有在空仓时才能开仓
#             if signal_series.iloc[i] == 2:
#                 position.iloc[i] = 1  # 开多仓
#             elif signal_series.iloc[i] == -2:
#                 position.iloc[i] = -1  # 开空仓
#             else:
#                 position.iloc[i] = 0  # 继续空仓
#         else:  # 已有持仓时，考虑平仓
#             if df_vol.iloc[i] > high_vol_threshold:  # 高波动率，立即平仓
#                 position.iloc[i] = 0
#             else:  # 低波动率，基于信号决定是否平仓
#                 if position.iloc[i - 1] == 1 and signal_series.iloc[i] <= 0:  # 平掉多仓
#                     position.iloc[i] = 0
#                 elif position.iloc[i - 1] == -1 and signal_series.iloc[i] >= 0:  # 平掉空仓
#                     position.iloc[i] = 0
#                 else:  # 继续持有
#                     position.iloc[i] = position.iloc[i - 1]

#     return position


# 修改后的波动率调整多空交易策略
def vol_long_short_strategy(signal_series, df_vol, vol_thresholds=None):
    """
    根据波动率调整的多空交易策略：
    - 波动率较高时，可以在较低信号强度（±1）时开仓
    - 波动率较低时，只有信号强度较高（±5）时才能开仓
    - 平仓规则：当已有持仓时，任何时刻波动率超过当前持仓时的值，立即平仓
    
    参数：
    - signal_series: 交易信号序列（-5 ~ 5）
    - df_vol: 波动率序列
    - vol_thresholds: 波动率阈值列表，决定不同信号级别所需的最小波动率
    
    返回：
    - position: 调整后的仓位
    """
    if vol_thresholds is None:
        # 设定不同信号强度的波动率阈值（从低到高）
        vol_thresholds = {
            6: 0,
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

        # 计算当前信号级别需要的最小波动率
        abs_signal = abs(signal)
        
        if abs_signal == 0:
            continue
        else:
            if vol > vol_thresholds[min(int(abs_signal)+1,6)]:
                position.iloc[i] = np.sign(signal) 
            
    return position
