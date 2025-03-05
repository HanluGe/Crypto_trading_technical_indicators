import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_cumulative_asset(series_close, series_position, initial_cash=10000, trade_size=1,fee_rate=0.0002):
    
    cash = initial_cash  # Available cash
    position = 0  # Crypto holding (in units of crypto)
    asset_values = []  # Total portfolio value
    cash_values = []  # Available cash balance
    holdings = []  # Crypto holdings

    # Merge price and position into a DataFrame
    df = pd.concat([series_close, series_position], axis=1)
    df.columns = ["close", "Position"]

    for i in range(len(df)):
        price = df["close"].iloc[i]
        position_signal = df["Position"].iloc[i]

        # Open Long position (Buy crypto)
        if position_signal == 1 and position == 0:
            fee = cash * fee_rate 
            position = ((cash - fee) / price) * trade_size  # Buy asset
            cash = 0

        # Open Short position (Sell crypto)
        elif position_signal == -1 and position == 0:
            fee = cash * fee_rate
            position = -((cash - fee) / price) * trade_size  # Short sell asset
            cash = 2 * cash # Assume margin account usage for shorting
            
        # Close Position (Sell or Cover Short)
        elif position_signal == 0 and position != 0:
            revenue = position * price
            fee = abs(revenue) * fee_rate
            cash += revenue - fee  # Sell crypto or cover short
            position = 0  # Reset position

        # Store values
        asset_value = cash + position * price  # Total asset value
        asset_values.append(asset_value)
        cash_values.append(cash)
        holdings.append(position * price)

    # Create DataFrame with the computed values
    df["Cumulative_Asset"] = asset_values
    df["Cash_Balance"] = cash_values
    df["Crypto_Holding"] = holdings

    return df[["close", "Position", "Cumulative_Asset", "Cash_Balance", "Crypto_Holding"]]


def evaluate_strategy(df, risk_free_rate=0.0):

    df = df.copy()
    df["Returns"] = df["Cumulative_Asset"].pct_change().fillna(0)

    # Sharpe Ratio
    daily_return_mean = df["Returns"].mean() * 365 * 24 * 12
    daily_return_std = df["Returns"].std() * np.sqrt(365 * 24 * 12)
    sharpe_ratio = (daily_return_mean - risk_free_rate) / daily_return_std if daily_return_std != 0 else np.nan

    # Maximum Drawdown (MDD)
    cumulative_max = df["Cumulative_Asset"].cummax()
    drawdown = (df["Cumulative_Asset"] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    # Win Rate
    win_trades = (df["Returns"] > 0).sum()
    total_trades = len(df["Returns"])
    win_rate = win_trades / total_trades if total_trades > 0 else np.nan

    # Final Cumulative Return
    final_return = df["Cumulative_Asset"].iloc[-1] / df["Cumulative_Asset"].iloc[0] - 1

    # Store results in a dictionary
    results = {
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown * 100,
        "Holding Win Rate (%)": win_rate * 100,
        "Trading Win Rate (%)": calculate_win_rate(df),
        "Final Cumulative Return (%)": final_return * 100
    }

    return results

def calculate_win_rate(df):
    df = df.copy()
    
    trades = []  # Store returns of completed trades
    in_trade = False
    entry_asset = 0  # Asset value at entry
    
    for i in range(len(df)):
        current_position = df["Position"].iloc[i]
        current_asset = df["Cumulative_Asset"].iloc[i]
        
        if current_position != 0 and not in_trade:
            # Entering a new trade
            in_trade = True
            entry_asset = current_asset
        
        elif current_position == 0 and in_trade:
            # Closing a trade
            trade_return = (current_asset - entry_asset) / entry_asset  # Trade Return
            trades.append(trade_return)
            in_trade = False  # Reset trade status

    # Count winning trades
    winning_trades = sum(1 for r in trades if r > 0)
    total_trades = len(trades)

    # Compute win rate
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else np.nan

    return win_rate
