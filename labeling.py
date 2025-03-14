import pandas as pd
import numpy as np

def generate_labels(df, price_col='close'):
    """
    Generate return labels based on the input DataFrame, divided as follows:
      - If next_return is in [-∞, -0.1), label is -2
      - If next_return is in [-0.1, -0.02), label is -1
      - If next_return is in [-0.02, 0.02), label is 0
      - If next_return is in [0.02, 0.1), label is 1
      - If next_return is in [0.1, +∞), label is 2

    Calculation:
      For each row (assuming the data is sorted in ascending order by date), calculate the next day's return:
         next_return = (next_day_price / current_price - 1) * 100
         
    Parameters:
       df: A DataFrame containing price data, which must be sorted in ascending order by date.
       price_col: The name of the price column used to calculate returns, default is 'close'.
       
    Returns:
       A Series with the same number of rows as df representing the corresponding labels;
       the last row has a label of NaN because of the absence of next-day data.
    """
    df = df.copy()
    # Calculate the next day's return and convert it to a percentage
    df['next_return'] = df[price_col].shift(-1) / df[price_col] - 1
    df['next_return'] *= 100

    # Define the new interval boundaries
    bins = [-np.inf, -0.1, -0.02, 0.02, 0.1, np.inf]
    labels = [-2, -1, 0, 1, 2]

    df['label'] = pd.cut(df['next_return'], bins=bins, labels=labels, right=False)
    return df['label']
