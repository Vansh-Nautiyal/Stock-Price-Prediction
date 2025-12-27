import yfinance as yf #type: ignore
import pandas as pd

# Download stock data
def download_stock(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df = df.dropna()  # remove missing values
    return df

#Save Data
def save_data(df, filename):
    path = f"data/raw/{filename}"
    df.to_csv(path, index=True)
    print(f"Data saved to {path}")
