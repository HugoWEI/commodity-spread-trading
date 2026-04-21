import pandas as pd
from pathlib import Path

PROCESSED_PATH = Path("data/processed")


def compute_zscore():
    df = pd.read_csv(PROCESSED_PATH / "cointegration_data.csv",
                     index_col=0, parse_dates=True)

    # rolling stats (important: NOT full sample)
    window = 60

    df["mean"] = df["residual"].rolling(window).mean()
    df["std"] = df["residual"].rolling(window).std()

    df["zscore"] = (df["residual"] - df["mean"]) / df["std"]

    df.to_csv(PROCESSED_PATH / "signal_data.csv")

    print(df[["residual", "zscore"]].tail())


if __name__ == "__main__":
    compute_zscore()
