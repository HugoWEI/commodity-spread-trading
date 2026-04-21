import yfinance as yf
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw")


def download():
    tickers = {
        "henry_hub": "NG=F",
        "ttf": "TTF=F",
        "eurusd": "EURUSD=X",
        "brent": "BZ=F",
        "wti": "CL=F",
    }

    data = {}

    for name, ticker in tickers.items():
        df = yf.download(ticker, start="2015-01-01")

        # flatten columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # choose correct price column
        if 'Adj Close' in df.columns:
            price_col = 'Adj Close'
        elif 'Close' in df.columns:
            price_col = 'Close'
        else:
            raise ValueError(f"No price column found for {ticker}")

        df = df[[price_col]].rename(columns={price_col: name})
        data[name] = df

    merged = pd.concat(data.values(), axis=1, sort=True)
    merged.to_csv(RAW_PATH / "market_data.csv")


if __name__ == "__main__":
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    download()
