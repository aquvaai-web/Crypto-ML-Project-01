import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_ohlcv(filename="btc_usdt_ohlcv.csv", num_days=730, initial_price=40000.0):
    """
    Generates realistic 2-year daily OHLCV dataset for BTC/USDT with trends, cycles, and noise.
    Used for machine learning model training & backtesting.
    """
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=num_days)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    # Generate realistic price path using geometric brownian motion + trend cycles
    t = np.linspace(0, 4 * np.pi, num_days)
    sine_wave = np.sin(t) * 0.15 # macro cycles
    returns = np.random.normal(0.0005, 0.025, num_days) + (sine_wave / num_days)
    
    price = initial_price
    prices = [price]
    for r in returns[:-1]:
        price = max(price * (1 + r), 5000.0)
        prices.append(price)
    
    close_prices = np.array(prices)
    # High, Low, Open relative to Close
    daily_volatility = np.abs(np.random.normal(0.015, 0.01, num_days))
    high_prices = close_prices * (1 + daily_volatility)
    low_prices = close_prices * (1 - daily_volatility)
    open_prices = low_prices + (high_prices - low_prices) * np.random.uniform(0.2, 0.8, num_days)
    volume = np.random.uniform(10000, 80000, num_days) * (1 + daily_volatility * 10)

    df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in dates],
        'Open': np.round(open_prices, 2),
        'High': np.round(high_prices, 2),
        'Low': np.round(low_prices, 2),
        'Close': np.round(close_prices, 2),
        'Volume': np.round(volume, 2)
    })
    
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"[Dataset] Saved synthetic OHLCV data with {len(df)} rows to {filename}")
    return df

if __name__ == "__main__":
    generate_synthetic_ohlcv("backend/dataset/btc_usdt_ohlcv.csv")
