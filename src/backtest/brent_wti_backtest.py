from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_PATH = Path("data/processed")


def sharpe_ratio(pnl):
    pnl = pnl.dropna()
    if pnl.std() == 0:
        return np.nan
    return np.sqrt(252) * pnl.mean() / pnl.std()


def main():
    df = pd.read_csv(
        PROCESSED_PATH / "brent_wti_clean.csv",
        index_col=0,
        parse_dates=True,
    )

    # residual = spread directly
    df["residual"] = df["spread"]

    # z-score
    window = 60
    df["mean"] = df["residual"].rolling(window).mean()
    df["std"] = df["residual"].rolling(window).std()
    df["zscore"] = (df["residual"] - df["mean"]) / df["std"]

    # strategy
    entry = 2
    exit = 0.5

    position = 0
    positions = []

    for z in df["zscore"]:
        if np.isnan(z):
            position = 0
        elif position == 0:
            if z > entry:
                position = -1
            elif z < -entry:
                position = 1
        elif position == 1 and z > -exit:
            position = 0
        elif position == -1 and z < exit:
            position = 0

        positions.append(position)

    df["position"] = positions

    df["residual_change"] = df["residual"].diff()
    df["pnl"] = df["position"].shift(1) * df["residual_change"]
    df["cum_pnl"] = df["pnl"].cumsum()

    sharpe = sharpe_ratio(df["pnl"])

    print("\n=== BRENT–WTI BASELINE ===")
    print(f"Sharpe: {sharpe}")
    print(f"Final PnL: {df['cum_pnl'].iloc[-1]}")


if __name__ == "__main__":
    main()
