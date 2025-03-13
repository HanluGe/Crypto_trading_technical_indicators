import matplotlib.pyplot as plt

def plot_btc_price_and_volume(df):
    """
    Plots the historical BTC close price and trading volume with a custom color palette.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'close' and 'volume' columns with a datetime index.
    """
    
    # Drop missing values
    df = df.dropna(subset=['close', 'volume'])

    # Define custom colors
    colors = {
        "line": "#05676D",  # Dark teal for close price
        "volume": "#F78C6B",  # Warm tone for volume
        "grid": "#A1D2CD",  # Soft blue-green for grid lines
    }

    # Create the figure and axis
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot the BTC close price
    ax1.plot(df.index, df["close"], label="BTC Close Price", color=colors["line"], linewidth=1.8)
    ax1.set_ylabel("Close Price (USD)", fontsize=12, color=colors["line"])
    ax1.set_xlabel("Date", fontsize=12, color="black")
    ax1.set_title("BTC Historical Close Price and Trading Volume", fontsize=14, color="black")
    ax1.tick_params(axis="y", labelsize=10, colors=colors["line"])
    ax1.tick_params(axis="x", labelsize=10, rotation=25, colors="black")
    ax1.grid(True, linestyle="--", alpha=0.5, color=colors["grid"])

    # Create a secondary y-axis for volume
    ax2 = ax1.twinx()
    ax2.fill_between(df.index, df["volume"], color=colors["volume"], alpha=0.4, label="Trading Volume")
    ax2.set_ylabel("Trading Volume", fontsize=12, color=colors["volume"])
    ax2.tick_params(axis="y", labelsize=10, colors=colors["volume"])

    # Add legends
    ax1.legend(loc="upper left", fontsize=10)
    ax2.legend(loc="upper right", fontsize=10)

    # Show plot
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_violin_by_category(df_cleaned):
    """
    Generates violin plots for numerical columns in a given DataFrame,
    grouped by their top-level category.

    Parameters:
    df_cleaned (pd.DataFrame): The cleaned DataFrame with a multi-index column structure.
    """
    threshold = 10  # If a column has fewer than this number of unique values, treat it as categorical
    categorical_columns = [col for col in df_cleaned.columns if df_cleaned[col].nunique() < threshold]
    numerical_columns = [col for col in df_cleaned.columns if col not in categorical_columns]

    # Extract factor categories (first-level index)
    categories = df_cleaned[numerical_columns].columns.get_level_values(0).unique()

    # Create violin plots for each category separately
    for category in categories:
        # Select numerical columns for the current category
        df = df_cleaned[numerical_columns][category]
        if df.empty:
            continue

        # Create figure
        plt.figure(figsize=(14, 6))
        sns.violinplot(data=df, inner="quartile", palette="coolwarm")

        # Formatting
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.xlabel(category, fontsize=12)
        plt.ylabel("Values", fontsize=12)
        plt.title(f"Distribution of {category} Indicators (Showing 25% and 75% Quantiles)", fontsize=14)
        plt.grid(axis='y', linestyle="--", alpha=0.5)

        # Show plot
        plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_category_distribution(categorical_df):
    """
    Plots a heatmap showing the distribution of categories in each factor.
    Zeros are masked and displayed as blank (white).
    
    Parameters:
        categorical_df (pd.DataFrame): A dataframe where each column contains categorical values.
    """
    # Count occurrences of each unique value in each column
    category_counts = pd.DataFrame({col: categorical_df[col].value_counts(normalize=True) for col in categorical_df.columns})

    # Transpose for easier visualization
    category_counts = category_counts.T.fillna(0)

    # Set up the figure size with increased height
    fig, ax = plt.subplots(figsize=(14, len(category_counts) * 0.3))

    # Define a masked array where zeros are replaced with NaN for white color
    masked_data = category_counts.replace(0, np.nan)

    # Create the heatmap with white areas for zeros
    sns.heatmap(masked_data, cmap="coolwarm", annot=True, fmt=".2%", linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Category Proportion'}, annot_kws={"size": 8}, 
                vmin=masked_data.min().min(), vmax=masked_data.max().max(), 
                mask=category_counts == 0)  # Mask zeros to keep them white

    # Formatting
    ax.set_title("Enhanced Distribution of Categories in Each Factor", fontsize=14)
    ax.set_xlabel("Category Values", fontsize=12)
    ax.set_ylabel("Factors", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Show the plot
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_factor_correlation_heatmaps(factor_correlations, cell_height=0.3, cell_width=1.3):
    """
    Plots separate heatmaps for each factor category in the provided correlation dataframe.

    Parameters:
    - factor_correlations: Pandas DataFrame with a multi-index where the first level is the category.
    - cell_height: float, height of each row in the heatmap.
    - cell_width: float, width of each column in the heatmap.
    """
    # Group data by category
    category_groups = factor_correlations.groupby(level=0)

    # Find the maximum number of revenue factors across all categories
    max_revenue_factors = max(len(data.columns) for _, data in category_groups)

    for category, data in category_groups:
        fig_height = len(data.index) * cell_height  # Ensure uniform row height
        fig_width = max_revenue_factors * cell_width  # Ensure uniform column width

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Create heatmap
        im = ax.imshow(data, cmap="coolwarm", aspect="auto", interpolation="nearest", vmin=-0.02, vmax=0.02)
        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)  # Adjust colorbar size
        cbar.set_label("Correlation")

        # Set axis labels
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_yticklabels([col[1] for col in data.index], fontsize=8)
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels([col[1] for col in data.columns], rotation=45, ha="right", fontsize=8)

        ax.set_xlabel("Revenue Factors")
        ax.set_title(f"Correlation of {category} with Revenue")

        # Show plot
        plt.show()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_top_factors_correlation(df_extended, top_20_factors_index_categorical, top_20_factors_index_numerical):
    """
    Plots a heatmap of the correlation matrix for the top 30 factors.

    Parameters:
    - df_extended: DataFrame containing all factor values.
    - top_20_factors_index_categorical: MultiIndex of top 20 categorical factors.
    - top_20_factors_index_numerical: MultiIndex of top 20 numerical factors.
    """

    # Convert MultiIndex to list of tuples
    top_20_factors_index_categorical_list = top_20_factors_index_categorical.tolist()
    top_20_factors_index_numerical_list = top_20_factors_index_numerical.tolist()

    # Concatenate both lists
    top_30_factors_index = top_20_factors_index_categorical_list + top_20_factors_index_numerical_list

    # Compute correlation matrix
    top_factors_corr = df_extended[top_30_factors_index].corr()

    # Create an upper triangle mask
    mask = np.triu(np.ones_like(top_factors_corr, dtype=bool))

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        top_factors_corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        linewidths=0.75,
        cbar_kws={"shrink": 0.7, "aspect": 30},
        annot=False,
        square=True
    )

    # Customize plot appearance
    ax.set_title("Heatmap of Correlation for Top Factors", fontsize=16, fontweight='bold', pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, fontweight='bold')

    # Adjust colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("Correlation", fontsize=12, fontweight='bold')

    plt.show()
    
    return top_factors_corr

import matplotlib.pyplot as plt

def plot_quantile_vs_sharpe(quantile_results):
    """
    Plots the relationship between upper quantiles and average Sharpe ratio
    with an evenly spaced x-axis.

    Parameters:
    - quantile_results: Dictionary where keys are upper quantiles and values are Avg Sharpe Ratios.

    Returns:
    - Displays the plot.
    """
    # Ensure x-axis is evenly spaced
    x_labels = list(quantile_results.keys())
    y_values = list(quantile_results.values())
    x_positions = range(len(x_labels))  # Evenly spaced x positions

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(x_positions, y_values, marker='o', linestyle='-', color='b', label="Avg Sharpe Ratio")

    # Set x-axis labels and spacing
    plt.xticks(ticks=x_positions, labels=x_labels)
    plt.xlabel("Upper Quantile")
    plt.ylabel("Avg Sharpe Ratio")
    plt.title("Sharpe Ratio vs. Upper Quantile (Evenly Spaced X-axis)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

# Example usage:
# plot_quantile_vs_sharpe(quantile_plot_results)

import matplotlib.pyplot as plt

def plot_trading_performance(df_result):
    """
    Plots the total asset value over time to visualize trading strategy performance.

    Parameters:
    - df_result: DataFrame containing the trading strategy results, including the "Cumulative_Asset" column.

    Returns:
    - Displays the plot.
    """
    plt.figure(figsize=(12, 6))

    # Plot Total Asset Value with improved styling
    plt.plot(df_result.index, df_result["Cumulative_Asset"], label="Total Asset Value", color='b', linewidth=2.0, linestyle='-')

    # Add grid, labels, and title
    plt.xlabel("Date", fontsize=12, fontweight='bold')
    plt.ylabel("Value", fontsize=12, fontweight='bold')
    plt.title("Trading Strategy Performance", fontsize=14, fontweight='bold')

    # Beautify x-axis and y-axis ticks
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)

    # Add grid for better readability
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    # Add legend with better positioning and styling
    plt.legend(loc="upper left", fontsize=12, frameon=True, shadow=True)

    # Show the plot
    plt.show()

# Example usage:
# plot_trading_performance(df_result)


def plot_buy_sell_signals(BTC, series_position_long_short):
    """
    Plots buy and sell signals on the BTC close price chart.

    Parameters:
    - BTC: DataFrame containing BTC price data with a "close" column.
    - series_position_long_short: Series containing buy (1), sell (-1), and hold (0) signals.

    Returns:
    - Displays the buy/sell signal plot.
    """
    import matplotlib.pyplot as plt

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot close price
    ax.plot(BTC.index, BTC["close"], label="Close Price", color="black", alpha=0.7)

    # Reindex signals to match BTC index
    signals = series_position_long_short.reindex(BTC.index, fill_value=0)

    # Identify buy and sell points
    buy_signals = BTC.index[signals == 1]
    sell_signals = BTC.index[signals == -1]

    # Plot buy (green) and sell (red) signals
    ax.scatter(buy_signals, BTC.loc[buy_signals, "close"], marker="o", color="green", label="Buy", alpha=0.6)
    ax.scatter(sell_signals, BTC.loc[sell_signals, "close"], marker="o", color="red", label="Sell", alpha=0.6)

    # Set labels and title
    ax.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax.set_ylabel("Close Price", fontsize=12, fontweight='bold')
    ax.set_title("Buy/Sell Signals on Close Price", fontsize=14, fontweight='bold')

    # Improve legend to remove duplicates
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper left", fontsize=10, frameon=True, shadow=True)

    # Improve grid readability
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    # Display the plot
    plt.show()

# Example usage:
# plot_buy_sell_signals(BTC, series_position_long_short)

import matplotlib.pyplot as plt

def plot_realized_volatility(df_with_indicators):
    """
    Plots the realized volatility over time with enhanced visualization.

    Parameters:
    - df_with_indicators: DataFrame containing a "volatility" column and a datetime index.

    Returns:
    - Displays the volatility time series chart.
    """
    plt.figure(figsize=(12, 6))

    # Plot volatility curve
    plt.plot(df_with_indicators.index, df_with_indicators["volatility"], 
             label="Realized Volatility", color="blue", linewidth=1.5, alpha=0.8)

    # Customize labels and title
    plt.xlabel("Date", fontsize=12, fontweight='bold')
    plt.ylabel("Volatility", fontsize=12, fontweight='bold')
    plt.title("Realized Volatility Over Time", fontsize=14, fontweight='bold')

    # Improve x-axis readability
    plt.xticks(rotation=30, fontsize=10)
    plt.yticks(fontsize=10)

    # Add grid and legend
    plt.legend(loc="upper right", fontsize=12, frameon=True, shadow=True)
    plt.grid(linestyle="--", alpha=0.5)

    # Show the plot
    plt.show()

# Example usage:
# plot_realized_volatility(df_with_indicators)