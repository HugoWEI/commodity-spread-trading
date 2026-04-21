import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw")


def check():
    df = pd.read_csv(RAW_PATH / "market_data.csv",
                     index_col=0, parse_dates=True)

    print("\nHead:\n")
    print(df.head())

    print("\nTail:\n")
    print(df.tail())

    print("\nInfo:\n")
    print(df.info())

    print("\nMissing values:\n")
    print(df.isna().sum())

    print("\nDate range:\n")
    print(df.index.min(), "->", df.index.max())


if __name__ == "__main__":
    check()
