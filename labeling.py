import pandas as pd
import numpy as np

def generate_labels(df, price_col='close'):
    """
    根据输入的 DataFrame 生成收益标签，按照如下划分：
      - next_return 在 [-∞, -0.1) 内 → 标签 -2
      - next_return 在 [-0.1, -0.02) 内 → 标签 -1
      - next_return 在 [-0.02, 0.02) 内 → 标签 0
      - next_return 在 [0.02, 0.1) 内 → 标签 1
      - next_return 在 [0.1, +∞) 内 → 标签 2

    计算方法：
      对于每一行（假设数据已按日期升序排列），计算下一日收益率：
         next_return = (next_day_price / current_price - 1) * 100
         
    参数:
       df: 包含价格数据的 DataFrame，必须按日期升序排列
       price_col: 用于计算收益率的价格列名称，默认为 'close'
       
    返回:
       一个与 df 行数相同的 Series，表示对应的标签；最后一行由于缺少下一日数据，标签为 NaN。
    """
    df = df.copy()
    # 计算下一日收益率，并转换为百分比
    df['next_return'] = df[price_col].shift(-1) / df[price_col] - 1
    df['next_return'] *= 100

    # 定义新的区间边界
    bins = [-np.inf, -0.1, -0.02, 0.02, 0.1, np.inf]
    labels = [-2, -1, 0, 1, 2]

    df['label'] = pd.cut(df['next_return'], bins=bins, labels=labels, right=False)
    return df['label']
