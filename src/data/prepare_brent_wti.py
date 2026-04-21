from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")


def prepare():
    df = pd.read_csv(
        RAW_PATH / "market_data.csv",
        index_col=0,
        parse_dates=True,
    )

    df = df.sort_index().ffill()

    # keep only needed columns
    df = df[["brent", "wti"]].dropna().copy()

    # same unit ($/bbl already)
    df["spread"] = df["brent"] - df["wti"]

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH / "brent_wti_clean.csv")

    print("\nPreview:")
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    prepare()
