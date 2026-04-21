from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_PATH = Path("data/processed")


def sharpe_ratio(pnl):
    pnl = pnl.dropna()
    if pnl.std() == 0:
        return np.nan
    return np.sqrt(252) * pnl.mean() / pnl.std()


def max_drawdown(cum_pnl):
    running_max = cum_pnl.cummax()
    return (cum_pnl - running_max).min()


# -----------------------------
# KALMAN FILTER
# -----------------------------
def kalman_filter(df, delta=1e-4, R=0.01):
    """
    delta: state noise (how fast beta can move)
    R: observation noise
    """

    n = len(df)

    beta = np.zeros(n)
    alpha = np.zeros(n)

    P = np.eye(2)  # covariance matrix
    Q = delta * np.eye(2)  # state noise

    for t in range(1, n):
        # observation matrix
        x = np.array([1.0, df["wti"].iloc[t]])

        # prediction
        theta = np.array([alpha[t-1], beta[t-1]])
        P = P + Q

        # observation
        y = df["brent"].iloc[t]

        # prediction error
        yhat = x @ theta
        e = y - yhat

        # Kalman gain
        S = x @ P @ x.T + R
        K = P @ x.T / S

        # update
        theta = theta + K * e
        P = P - np.outer(K, x) @ P

        alpha[t] = theta[0]
        beta[t] = theta[1]

    df["alpha"] = alpha
    df["beta"] = beta

    df["residual"] = df["brent"] - (df["alpha"] + df["beta"] * df["wti"])

    return df


# -----------------------------
# MAIN STRATEGY
# -----------------------------
def main():
    df = pd.read_csv(
        PROCESSED_PATH / "brent_wti_clean.csv",
        index_col=0,
        parse_dates=True,
    ).sort_index()

    # Kalman estimation
    df = kalman_filter(df)

    # features
    window = 60

    df["mean"] = df["residual"].rolling(window).mean()
    df["std"] = df["residual"].rolling(window).std()
    df["zscore"] = ((df["residual"] - df["mean"]) / df["std"]).shift(1)

    df["vol"] = df["residual"].rolling(window).std().shift(1)

    # strategy
    entry = 2
    exit = 0.5
    target_risk = 0.25

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

    df["position_raw"] = positions

    # risk scaling
    df["position"] = df["position_raw"] * (target_risk / df["vol"])
    df["position"] = df["position"].clip(-3, 3)

    # pnl
    df["residual_change"] = df["residual"].diff()
    df["pnl"] = df["position"].shift(1) * df["residual_change"]
    df["cum_pnl"] = df["pnl"].cumsum()

    # metrics
    sharpe = sharpe_ratio(df["pnl"])
    mdd = max_drawdown(df["cum_pnl"])

    print("\n=== BRENT–WTI KALMAN STRATEGY ===")
    print(f"Sharpe: {sharpe}")
    print(f"Final PnL: {df['cum_pnl'].iloc[-1]}")
    print(f"Max Drawdown: {mdd}")
    print(f"Beta mean: {df['beta'].mean()}")
    print(f"Beta std: {df['beta'].std()}")

    df.to_csv(PROCESSED_PATH / "brent_wti_kalman_results.csv")


if __name__ == "__main__":
    main()
