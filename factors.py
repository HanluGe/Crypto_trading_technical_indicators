import pandas as pd
import talib
import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter("ignore", PerformanceWarning)
import numpy as np

factor_categories = {
    "Overlap Studies": [
        "BBAND WIDTH", "BBAND UPPER SIGNAL", "BBAND LOWER SIGNAL", "RSI", "DEMA", "EMA", "H TRENDLINE", "KMAM",
        "MIDPOINT", "MIDPRICE", "SAR", "SAREXT", "SMA3", "SMA5", "SMA10", "SMA20", "T", "TEMA", "TRIMA", "WMA"
    ],
    "Momentum Indicators": [
        "ADX14", "ADX20", "ADXR", "APO", "AROONOSC", "BOP", "CCI3", "CCI5", "CCI10", "CCI14", "CMO", "DX",
        "MACD", "MACDSIGNAL", "MACDHIST", "MINUS_DI", "MINUS_DM", "MOM1", "MOM3", "MOM5", "MOM10", 
        "PLUSDI", "PLUSDM", "PPO", "ROC", "ROCP", "ROCR", "ROCR100", "RSI5", "RSI10", "RSI14", 
        "SLOWK", "SLOWD", "FASTK", "FASTD", "TRIX", "ULTOSC", "WILLR"
    ],
    "Volatility Indicators": ["ATR", "NATR", "TRANGE"],
    "Pattern Recognition": [
        "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDL3LINESTRIKE", "CDL3OUTSIDE", "CDL3STARSINSOUTH",
        "CDL3WHITESOLDIERS", "DLABANDONEDBABY", "CDLADVANCEBLOCK", "CDLBELTHOLD", "CDLBREAKAWAY", 
        "CDLCLOSINGMARUBOZU", "DLCONCEALBABYSWALL", "CDLCOUNTERATTACK", "CDLDARKCLOUDCOVER", "DLDOJI",
        "CDLDOJISTAR", "DLDRAGONFLYDOJI", "DLENGULFING", "CDLEVENINGDOJISTAR", "CDLEVENINGSTAR", 
        "CDLGAPSIDESIDEWHITE", "CDLGRAVESTONEDOJI", "CDLHAMMER", "CDLHANGINGMAN", "CDLHARAMI", "DLHARAMICROSS",
        "CDLHIGHWAVE", "CDLHIKKAKE", "CDLHIKKAKEMOD", "CDLHOMINGPIGEON", "CDLIDENTICAL3CROWS", "CDLINNECK", 
        "DLINVERTEDHAMMER", "DLKICKING", "CDLKICKINGBYLENGTH", "CDLLADDERBOTTOM", "CDLLONGLEGGEDDOJI", 
        "CDLLONGLINE", "CDLMARUBOZU", "CDLMATCHINGLOW", "CDLMATHOLD", "CDLMORNINGDOJISTAR", "CDLMORNINGSTAR", 
        "CDLONNECK", "CDLPIERCING", "DLRICKSHAWMAN", "CDLRISEFALL3METHODS", "CDLSEPARATINGLINES", 
        "CDLSHOOTINGSTAR", "CDLSHORTLINE", "CDLSPINNINGTOP", "CDLSTALLEDPATTERN", "CDLSTICKSANDWICH", 
        "CDLTAKURI", "CDLTASUKIGAP", "CDLTHRUSTING", "DLTRISTAR", "CDLUNIQUE3RIVER", "CDLUPSIDEGAP2CROWS", 
        "CDLXSIDEGAP3METHODS"
    ],
    "Cycle Indicators": ["HTDCPERIOD", "HT DCPHASE", "TRENDMODE"]
}

def restructure_dataframe(df):
    # Create a mapping of factor names to categories
    factor_to_category = {
        factor: category for category, factors in factor_categories.items() for factor in factors
    }
    
    # Identify factor columns in the DataFrame
    factor_columns = [col for col in df.columns if col in factor_to_category]

    # Create a MultiIndex for the columns
    multi_index_columns = pd.MultiIndex.from_tuples(
        [(factor_to_category[col], col) for col in factor_columns], 
        names=["Category", "Factor"]
    )

    # Restructure the DataFrame using the new MultiIndex
    df_restructured = pd.DataFrame(df[factor_columns].values, index=df.index, columns=multi_index_columns)
    
    return df_restructured

def compute_RSI(close, timeperiod=14):
    """Compute the Relative Strength Index (RSI) using EMA."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=timeperiod, adjust=False).mean()
    avg_loss = loss.ewm(span=timeperiod, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_sarext(high, low, acceleration_init=0.02, acceleration_max=0.2, offset_on_reverse=0):
    """
    Compute the Extended Parabolic SAR (an extension of the basic Parabolic SAR algorithm).
    
    Parameters:
        high, low: arrays representing the high and low prices for each period.
        acceleration_init: initial acceleration factor (default 0.02).
        acceleration_max: maximum acceleration factor (default 0.2).
        offset_on_reverse: offset when a reversal occurs (default 0).
    
    Returns:
        numpy array containing the SAREXT values for each period.
    """
    n = len(high)
    sar = np.zeros(n)
    
    # Determine the initial trend based on the first two periods
    if high[1] > high[0]:
        trend = 1  # Uptrend
        sar[0] = low[0]  # Initial SAR is set to the first period's low price
        ep = high[0]     # Extreme point is set to the first period's high price
    else:
        trend = -1  # Downtrend
        sar[0] = high[0]
        ep = low[0]
        
    acceleration = acceleration_init

    for i in range(1, n):
        # Calculate SAR
        sar[i] = sar[i-1] + acceleration * (ep - sar[i-1])
        # For an uptrend, SAR cannot exceed the low prices of the previous two periods
        if trend == 1 and i >= 2:
            sar[i] = min(sar[i], low[i-1], low[i-2])
        # For a downtrend, SAR cannot be lower than the high prices of the previous two periods
        if trend == -1 and i >= 2:
            sar[i] = max(sar[i], high[i-1], high[i-2])
        
        # Check for reversal
        if trend == 1:
            if low[i] < sar[i]:
                # Reverse to a downtrend
                trend = -1
                sar[i] = ep  # Set SAR to the extreme point of the previous trend upon reversal
                sar[i] *= (1 + offset_on_reverse)  # Apply offset
                ep = low[i]  # Reset the extreme point to the current low price
                acceleration = acceleration_init
            else:
                # In an uptrend, update the extreme point and acceleration factor if a new high is observed
                if high[i] > ep:
                    ep = high[i]
                    acceleration = min(acceleration + acceleration_init, acceleration_max)
        else:  # trend == -1
            if high[i] > sar[i]:
                # Reverse to an uptrend
                trend = 1
                sar[i] = ep
                sar[i] *= (1 - offset_on_reverse)
                ep = high[i]
                acceleration = acceleration_init
            else:
                if low[i] < ep:
                    ep = low[i]
                    acceleration = min(acceleration + acceleration_init, acceleration_max)
    return sar

def compute_technical_indicators(df, start_time, end_time):
    """
    Calculate the 124 technical indicators listed in the paper using TA-Lib.
    The input data must include the following columns: open, high, low, close, preclose, volume.
    Date information is stored in the "date" column or as the index.

    Parameters:
        df (pd.DataFrame): The raw data.
        start_time (str or pd.Timestamp): The start time (inclusive).
        end_time (str or pd.Timestamp): The end time (inclusive).

    Returns:
        pd.DataFrame: The DataFrame obtained by concatenating the raw data and the calculated indicators.
    """
    # Convert start and end times to Timestamp objects
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    # Determine whether date information is stored in a column or as the index
    if 'date' in df.columns:
        # If the "date" column exists, convert it to datetime type if necessary
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        df_filtered = df[(df['date'] >= start_time) & (df['date'] <= end_time)].copy()
    else:
        # If there is no "date" column, assume that the index contains date information
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df_filtered = df[(df.index >= start_time) & (df.index <= end_time)].copy()
    
    # Use a dictionary to store all indicator results to avoid memory fragmentation from multiple DataFrame insertions
    indicator_dict = {}

    # ====================
    # 1. Overlap Studies Indicators
    # ====================
    try:
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df_filtered['close'], timeperiod=20)
        indicator_dict['BBAND WIDTH'] = bb_upper - bb_lower
        indicator_dict['BBAND UPPER SIGNAL'] = (df_filtered['close'] > bb_upper).astype(int)
        indicator_dict['BBAND LOWER SIGNAL'] = (df_filtered['close'] < bb_lower).astype(int)
    except Exception:
        for col in ['BBAND WIDTH', 'BBAND UPPER SIGNAL', 'BBAND LOWER SIGNAL']:
            indicator_dict[col] = None

    indicator_dict['RSI'] = compute_RSI(df_filtered['close'], timeperiod=14)

    try:
        indicator_dict['DEMA'] = talib.DEMA(df_filtered['close'], timeperiod=20)
    except Exception:
        indicator_dict['DEMA'] = None

    try:
        indicator_dict['EMA'] = talib.EMA(df_filtered['close'], timeperiod=20)
    except Exception:
        indicator_dict['EMA'] = None

    try:
        indicator_dict['H TRENDLINE'] = talib.HT_TRENDLINE(df_filtered['close'])
    except Exception:
        indicator_dict['H TRENDLINE'] = None

    try:
        indicator_dict['KMAM'] = talib.KAMA(df_filtered['close'], timeperiod=10)
    except Exception:
        indicator_dict['KMAM'] = None

    try:
        indicator_dict['MIDPOINT'] = talib.MIDPOINT(df_filtered['close'], timeperiod=14)
    except Exception:
        indicator_dict['MIDPOINT'] = None

    try:
        indicator_dict['MIDPRICE'] = talib.MIDPRICE(df_filtered['high'], df_filtered['low'], timeperiod=14)
    except Exception:
        indicator_dict['MIDPRICE'] = None

    try:
        indicator_dict['SAR'] = talib.SAR(df_filtered['high'], df_filtered['low'], acceleration=0.02, maximum=0.2)
    except Exception:
        indicator_dict['SAR'] = None

    try:
        indicator_dict['SAREXT'] = compute_sarext(
            df_filtered['high'].values, df_filtered['low'].values,
            acceleration_init=0.02, acceleration_max=0.2, offset_on_reverse=0
        )
    except Exception:
        indicator_dict['SAREXT'] = None

    for period in [3, 5, 10, 20]:
        try:
            indicator_dict[f'SMA{period}'] = talib.SMA(df_filtered['close'], timeperiod=period)
        except Exception:
            indicator_dict[f'SMA{period}'] = None

    try:
        indicator_dict['T'] = talib.T3(df_filtered['close'], timeperiod=20)
    except Exception:
        indicator_dict['T'] = None

    try:
        indicator_dict['TEMA'] = talib.TEMA(df_filtered['close'], timeperiod=20)
    except Exception:
        indicator_dict['TEMA'] = None

    try:
        indicator_dict['TRIMA'] = talib.TRIMA(df_filtered['close'], timeperiod=20)
    except Exception:
        indicator_dict['TRIMA'] = None

    try:
        indicator_dict['WMA'] = talib.WMA(df_filtered['close'], timeperiod=20)
    except Exception:
        indicator_dict['WMA'] = None

    # ====================
    # 2. Momentum Indicators
    # ====================
    try:
        indicator_dict['ADX14'] = talib.ADX(df_filtered['high'], df_filtered['low'], df_filtered['close'], timeperiod=14)
    except Exception:
        indicator_dict['ADX14'] = None
    try:
        indicator_dict['ADX20'] = talib.ADX(df_filtered['high'], df_filtered['low'], df_filtered['close'], timeperiod=20)
    except Exception:
        indicator_dict['ADX20'] = None

    try:
        indicator_dict['ADXR'] = talib.ADXR(df_filtered['high'], df_filtered['low'], df_filtered['close'], timeperiod=14)
    except Exception:
        indicator_dict['ADXR'] = None

    try:
        indicator_dict['APO'] = talib.APO(df_filtered['close'], fastperiod=12, slowperiod=26, matype=0)
    except Exception:
        indicator_dict['APO'] = None

    try:
        indicator_dict['AROONOSC'] = talib.AROONOSC(df_filtered['high'], df_filtered['low'], timeperiod=14)
    except Exception:
        indicator_dict['AROONOSC'] = None

    try:
        indicator_dict['BOP'] = talib.BOP(df_filtered['open'], df_filtered['high'], df_filtered['low'], df_filtered['close'])
    except Exception:
        indicator_dict['BOP'] = None

    for period in [3, 5, 10, 14]:
        try:
            indicator_dict[f'CCI{period}'] = talib.CCI(df_filtered['high'], df_filtered['low'], df_filtered['close'], timeperiod=period)
        except Exception:
            indicator_dict[f'CCI{period}'] = None

    try:
        indicator_dict['CMO'] = talib.CMO(df_filtered['close'], timeperiod=14)
    except Exception:
        indicator_dict['CMO'] = None

    try:
        indicator_dict['DX'] = talib.DX(df_filtered['high'], df_filtered['low'], df_filtered['close'], timeperiod=14)
    except Exception:
        indicator_dict['DX'] = None

    try:
        macd, macdsignal, macdhist = talib.MACD(df_filtered['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        indicator_dict['MACD'] = macd
        indicator_dict['MACDSIGNAL'] = macdsignal
        indicator_dict['MACDHIST'] = macdhist
    except Exception:
        indicator_dict['MACD'] = indicator_dict['MACDSIGNAL'] = indicator_dict['MACDHIST'] = None

    try:
        indicator_dict['MINUS_DI'] = talib.MINUS_DI(df_filtered['high'], df_filtered['low'], df_filtered['close'], timeperiod=14)
    except Exception:
        indicator_dict['MINUS_DI'] = None

    try:
        indicator_dict['MINUS_DM'] = talib.MINUS_DM(df_filtered['high'], df_filtered['low'], timeperiod=14)
    except Exception:
        indicator_dict['MINUS_DM'] = None

    for period in [1, 3, 5, 10]:
        try:
            indicator_dict[f'MOM{period}'] = talib.MOM(df_filtered['close'], timeperiod=period)
        except Exception:
            indicator_dict[f'MOM{period}'] = None

    try:
        indicator_dict['PLUSDI'] = talib.PLUS_DI(df_filtered['high'], df_filtered['low'], df_filtered['close'], timeperiod=14)
    except Exception:
        indicator_dict['PLUSDI'] = None

    try:
        indicator_dict['PLUSDM'] = talib.PLUS_DM(df_filtered['high'], df_filtered['low'], timeperiod=14)
    except Exception:
        indicator_dict['PLUSDM'] = None

    try:
        indicator_dict['PPO'] = talib.PPO(df_filtered['close'], fastperiod=12, slowperiod=26, matype=0)
    except Exception:
        indicator_dict['PPO'] = None

    try:
        indicator_dict['ROC'] = talib.ROC(df_filtered['close'], timeperiod=10)
    except Exception:
        indicator_dict['ROC'] = None

    try:
        indicator_dict['ROCP'] = talib.ROCP(df_filtered['close'], timeperiod=10)
    except Exception:
        indicator_dict['ROCP'] = None

    try:
        indicator_dict['ROCR'] = talib.ROCR(df_filtered['close'], timeperiod=10)
    except Exception:
        indicator_dict['ROCR'] = None

    try:
        indicator_dict['ROCR100'] = talib.ROCR100(df_filtered['close'], timeperiod=10)
    except Exception:
        indicator_dict['ROCR100'] = None

    for period in [5, 10, 14]:
        try:
            indicator_dict[f'RSI{period}'] = talib.RSI(df_filtered['close'], timeperiod=period)
        except Exception:
            indicator_dict[f'RSI{period}'] = None

    try:
        slowk, slowd = talib.STOCH(df_filtered['high'], df_filtered['low'], df_filtered['close'],
                                   fastk_period=14, slowk_period=3, slowk_matype=0,
                                   slowd_period=3, slowd_matype=0)
        indicator_dict['SLOWK'] = slowk
        indicator_dict['SLOWD'] = slowd
    except Exception:
        indicator_dict['SLOWK'] = indicator_dict['SLOWD'] = None

    try:
        fastk, fastd = talib.STOCHF(df_filtered['high'], df_filtered['low'], df_filtered['close'],
                                   fastk_period=14, fastd_period=3, fastd_matype=0)
        indicator_dict['FASTK'] = fastk
        indicator_dict['FASTD'] = fastd
    except Exception:
        indicator_dict['FASTK'] = indicator_dict['FASTD'] = None

    try:
        indicator_dict['TRIX'] = talib.TRIX(df_filtered['close'], timeperiod=30)
    except Exception:
        indicator_dict['TRIX'] = None

    try:
        indicator_dict['ULTOSC'] = talib.ULTOSC(df_filtered['high'], df_filtered['low'], df_filtered['close'],
                                                  timeperiod1=7, timeperiod2=14, timeperiod3=28)
    except Exception:
        indicator_dict['ULTOSC'] = None

    try:
        indicator_dict['WILLR'] = talib.WILLR(df_filtered['high'], df_filtered['low'], df_filtered['close'], timeperiod=14)
    except Exception:
        indicator_dict['WILLR'] = None

    # ====================
    # 3. Volatility Indicators
    # ====================
    try:
        indicator_dict['ATR'] = talib.ATR(df_filtered['high'], df_filtered['low'], df_filtered['close'], timeperiod=14)
    except Exception:
        indicator_dict['ATR'] = None

    try:
        indicator_dict['NATR'] = talib.NATR(df_filtered['high'], df_filtered['low'], df_filtered['close'], timeperiod=14)
    except Exception:
        indicator_dict['NATR'] = None

    try:
        indicator_dict['TRANGE'] = talib.TRANGE(df_filtered['high'], df_filtered['low'], df_filtered['close'])
    except Exception:
        indicator_dict['TRANGE'] = None

    # ====================
    # 4. Pattern Recognition Indicators
    # ====================
    pattern_mapping = {
        "CDL2CROWS": "CDL2CROWS",
        "CDL3BLACKCROWS": "CDL3BLACKCROWS",
        "CDL3INSIDE": "CDL3INSIDE",
        "CDL3LINESTRIKE": "CDL3LINESTRIKE",
        "CDL3OUTSIDE": "CDL3OUTSIDE",
        "CDL3STARSINSOUTH": "CDL3STARSINSOUTH",
        "CDL3WHITESOLDIERS": "CDL3WHITESOLDIERS",
        "DLABANDONEDBABY": "CDLABANDONEDBABY",
        "CDLADVANCEBLOCK": "CDLADVANCEBLOCK",
        "CDLBELTHOLD": "CDLBELTHOLD",
        "CDLBREAKAWAY": "CDLBREAKAWAY",
        "CDLCLOSINGMARUBOZU": "CDLCLOSINGMARUBOZU",
        "DLCONCEALBABYSWALL": "CDLCONCEALBABYSWALL",
        "CDLCOUNTERATTACK": "CDLCOUNTERATTACK",
        "CDLDARKCLOUDCOVER": "CDLDARKCLOUDCOVER",
        "DLDOJI": "CDLDOJI",
        "CDLDOJISTAR": "CDLDOJISTAR",
        "DLDRAGONFLYDOJI": "CDLDRAGONFLYDOJI",
        "DLENGULFING": "CDLENGULFING",
        "CDLEVENINGDOJISTAR": "CDLEVENINGDOJISTAR",
        "CDLEVENINGSTAR": "CDLEVENINGSTAR",
        "CDLGAPSIDESIDEWHITE": "CDLGAPSIDESIDEWHITE",
        "CDLGRAVESTONEDOJI": "CDLGRAVESTONEDOJI",
        "CDLHAMMER": "CDLHAMMER",
        "CDLHANGINGMAN": "CDLHANGINGMAN",
        "CDLHARAMI": "CDLHARAMI",
        "DLHARAMICROSS": "CDLHARAMICROSS",
        "CDLHIGHWAVE": "CDLHIGHWAVE",
        "CDLHIKKAKE": "CDLHIKKAKE",
        "CDLHIKKAKEMOD": "CDLHIKKAKEMOD",
        "CDLHOMINGPIGEON": "CDLHOMINGPIGEON",
        "CDLIDENTICAL3CROWS": "CDLIDENTICAL3CROWS",
        "CDLINNECK": "CDLINNECK",
        "DLINVERTEDHAMMER": "CDLINVERTEDHAMMER",
        "DLKICKING": "CDLKICKING",
        "CDLKICKINGBYLENGTH": "CDLKICKINGBYLENGTH",
        "CDLLADDERBOTTOM": "CDLLADDERBOTTOM",
        "CDLLONGLEGGEDDOJI": "CDLLONGLEGGEDDOJI",
        "CDLLONGLINE": "CDLLONGLINE",
        "CDLMARUBOZU": "CDLMARUBOZU",
        "CDLMATCHINGLOW": "CDLMATCHINGLOW",
        "CDLMATHOLD": "CDLMATHOLD",
        "CDLMORNINGDOJISTAR": "CDLMORNINGDOJISTAR",
        "CDLMORNINGSTAR": "CDLMORNINGSTAR",
        "CDLONNECK": "CDLONNECK",
        "CDLPIERCING": "CDLPIERCING",
        "DLRICKSHAWMAN": "CDLRICKSHAWMAN",
        "CDLRISEFALL3METHODS": "CDLRISEFALL3METHODS",
        "CDLSEPARATINGLINES": "CDLSEPARATINGLINES",
        "CDLSHOOTINGSTAR": "CDLSHOOTINGSTAR",
        "CDLSHORTLINE": "CDLSHORTLINE",
        "CDLSPINNINGTOP": "CDLSPINNINGTOP",
        "CDLSTALLEDPATTERN": "CDLSTALLEDPATTERN",
        "CDLSTICKSANDWICH": "CDLSTICKSANDWICH",
        "CDLTAKURI": "CDLTAKURI",
        "CDLTASUKIGAP": "CDLTASUKIGAP",
        "CDLTHRUSTING": "CDLTHRUSTING",
        "DLTRISTAR": "DLTRISTAR",
        "CDLUNIQUE3RIVER": "CDLUNIQUE3RIVER",
        "CDLUPSIDEGAP2CROWS": "CDLUPSIDEGAP2CROWS",
        "CDLXSIDEGAP3METHODS": "CDLXSIDEGAP3METHODS"
    }
    for key, func_name in pattern_mapping.items():
        try:
            func = getattr(talib, func_name)
            indicator_dict[key] = func(df_filtered['open'], df_filtered['high'], df_filtered['low'], df_filtered['close'])
        except Exception:
            indicator_dict[key] = None

    # ====================
    # 5. Cycle Indicators
    # ====================
    try:
        indicator_dict['HTDCPERIOD'] = talib.HT_DCPERIOD(df_filtered['close'])
    except Exception:
        indicator_dict['HTDCPERIOD'] = None

    try:
        indicator_dict['HT DCPHASE'] = talib.HT_DCPHASE(df_filtered['close'])
    except Exception:
        indicator_dict['HT DCPHASE'] = None

    try:
        indicator_dict['TRENDMODE'] = talib.HT_TRENDMODE(df_filtered['close'])
    except Exception:
        indicator_dict['TRENDMODE'] = None

    # Construct the indicators DataFrame and call .copy() to optimize memory layout
    indicators = pd.DataFrame(indicator_dict, index=df_filtered.index).copy()
    
    # Merge the original data with the indicator data
    result_df = pd.concat([df_filtered, indicators], axis=1)
    return result_df
