import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")

MMBTU_PER_MWH = 3.412


def prepare():
    df = pd.read_csv(RAW_PATH / "market_data.csv",
                     index_col=0, parse_dates=True)

    print("\nMissing before cleaning:\n")
    print(df.isna().sum())

    # sort index
    df = df.sort_index()

    # forward-fill small gaps from market holiday mismatches
    df = df.ffill()

    # keep only overlapping sample where all series exist
    df = df.dropna(subset=["henry_hub", "ttf", "eurusd"]).copy()

    print("\nMissing after cleaning:\n")
    print(df.isna().sum())

    print("\nClean date range:\n")
    print(df.index.min(), "->", df.index.max())

    # convert TTF from EUR/MWh to USD/MMBtu
    df["ttf_usd_mmbtu"] = df["ttf"] * df["eurusd"] / MMBTU_PER_MWH

    # Henry Hub futures are already in USD/MMBtu
    df["hh_usd_mmbtu"] = df["henry_hub"]

    # naive spread
    df["spread"] = df["ttf_usd_mmbtu"] - df["hh_usd_mmbtu"]

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH / "clean_data.csv")

    print("\nPreview:\n")
    print(df[["hh_usd_mmbtu", "ttf_usd_mmbtu", "spread"]].head())
    print(df[["hh_usd_mmbtu", "ttf_usd_mmbtu", "spread"]].tail())


if __name__ == "__main__":
    prepare()
