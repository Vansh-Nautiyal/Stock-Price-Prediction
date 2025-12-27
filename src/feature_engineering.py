import numpy as np

# ----------------------------------------------------
# Add Returns
def add_returns(df):
    df['return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close']).diff()
    return df

# SMA
def add_sma(df, window):
    df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
    return df

# ----------------------------------------------------
# EMA
# ----------------------------------------------------
def add_ema(df, span):
    df[f'ema_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
    return df

# ----------------------------------------------------
# MACD
# ----------------------------------------------------
def add_macd(df, span_short=12, span_long=26, span_signal=9):
    ema_short = df['Close'].ewm(span=span_short, adjust=False).mean()
    ema_long  = df['Close'].ewm(span=span_long, adjust=False).mean()

    df['macd'] = ema_short - ema_long
    df['macd_signal'] = df['macd'].ewm(span=span_signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df

# ----------------------------------------------------
# RSI
# ----------------------------------------------------
def add_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

# ----------------------------------------------------
# Main Feature Engineering Function
# ----------------------------------------------------
def create_features(df):
    df = df.copy()

    df = add_returns(df)
    df = add_sma(df, 10)
    df = add_sma(df, 50)
    df = add_ema(df, 12)
    df = add_ema(df, 26)
    df = add_macd(df)
    df = add_rsi(df)

    df['close_lag1'] = df['Close'].shift(1)
    df['close_lag2'] = df['Close'].shift(2)

    return df.dropna()
