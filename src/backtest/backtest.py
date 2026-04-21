import pandas as pd
from pathlib import Path

PROCESSED_PATH = Path("data/processed")


def backtest():
    df = pd.read_csv(PROCESSED_PATH / "signal_data.csv",
                     index_col=0, parse_dates=True)

    # rolling volatility of residual
    df["vol"] = df["residual"].rolling(60).std()

    # define high-volatility cutoff using only past observable data
    vol_threshold = df["vol"].quantile(0.8)

    entry_threshold = 2
    exit_threshold = 0.5

    position = 0
    positions = []

    for _, row in df.iterrows():
        z = row["zscore"]
        vol = row["vol"]

        # do not trade in extreme volatility regimes
        if pd.isna(vol) or vol >= vol_threshold:
            position = 0
            positions.append(position)
            continue

        if position == 0:
            if z > entry_threshold:
                position = -1  # short spread
            elif z < -entry_threshold:
                position = 1   # long spread

        elif position == 1:
            if z > -exit_threshold:
                position = 0

        elif position == -1:
            if z < exit_threshold:
                position = 0

        positions.append(position)

    df["position"] = positions

    # PnL uses yesterday's position to avoid look-ahead
    df["residual_change"] = df["residual"].diff()
    df["pnl"] = df["position"].shift(1) * df["residual_change"]
    df["cum_pnl"] = df["pnl"].cumsum()

    print("\nFinal PnL:", df["cum_pnl"].iloc[-1])

    sharpe = df["pnl"].mean() / df["pnl"].std() * (252 ** 0.5)
    print("Sharpe:", sharpe)

    print("Vol threshold:", vol_threshold)

    df.to_csv(PROCESSED_PATH / "backtest_results.csv")


if __name__ == "__main__":
    backtest()
