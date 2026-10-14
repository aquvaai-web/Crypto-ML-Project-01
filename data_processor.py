import pandas as pd
import numpy as np

def compute_technical_indicators(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Computes technical indicators for OHLCV financial time-series data
    without lookahead bias (anti-leakage guarantee).
    
    Target Variable:
    1 = Bullish (Future 'horizon'-day price return > +0.5%)
    0 = Bearish/Sideways (Future 'horizon'-day price return <= +0.5%)
    """
    data = df.copy()
    
    # 1. Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # 2. Exponential Moving Averages
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    
    # 3. Relative Strength Index (RSI 14)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 4. MACD & Signal Line
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # 5. Bollinger Bands
    rolling_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['SMA_20'] + (rolling_std * 2)
    data['BB_Lower'] = data['SMA_20'] - (rolling_std * 2)
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / (data['SMA_20'] + 1e-10)
    
    # 6. Volatility & Returns
    data['Price_Return_1d'] = data['Close'].pct_change(1)
    data['Price_Return_5d'] = data['Close'].pct_change(5)
    data['Volatility_14d'] = data['Price_Return_1d'].rolling(window=14).std()
    data['Volume_Change_1d'] = data['Volume'].pct_change(1)
    
    # 7. MA Crossover Signals
    data['EMA_Diff'] = (data['EMA_20'] - data['EMA_50']) / data['Close']
    
    # Target label: Future return looking ahead 'horizon' periods (Target generation ONLY for training)
    future_return = data['Close'].shift(-horizon) / data['Close'] - 1.0
    data['Target'] = np.where(future_return > 0.005, 1, 0)
    
    return data

FEATURE_COLUMNS = [
    'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'EMA_200',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Width', 'Price_Return_1d', 'Price_Return_5d',
    'Volatility_14d', 'Volume_Change_1d', 'EMA_Diff'
]
