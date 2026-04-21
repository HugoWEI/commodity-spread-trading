from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROCESSED_PATH = Path("data/processed")


def sharpe_ratio(pnl: pd.Series) -> float:
    pnl = pnl.dropna()
    if len(pnl) < 2 or pnl.std() == 0:
        return np.nan
    return np.sqrt(252) * pnl.mean() / pnl.std()


def max_drawdown(cum_pnl: pd.Series) -> float:
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    return drawdown.min()


def main():
    df = pd.read_csv(
        PROCESSED_PATH / "ml_dataset.csv",
        index_col=0,
        parse_dates=True,
    )

    feature_cols = [
        "zscore_lag1",
        "zscore_lag5",
        "resid_mom_5",
        "resid_mom_20",
        "vol_short",
        "vol_long",
        "vol_ratio",
        "beta_lag1",
        "beta_change",
        "beta_vol",
        "abs_zscore",
    ]
    target_col = "target"

    # Safety check in case future edits reintroduce NaNs
    df = df.dropna(subset=feature_cols + [target_col]).copy()

    # -------------------------
    # TIME SPLIT (NO SHUFFLE)
    # -------------------------
    split = int(len(df) * 0.7)

    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    X_train = train[feature_cols].values
    y_train = train[target_col].values

    X_test = test[feature_cols].values
    y_test = test[target_col].values

    # -------------------------
    # SCALE FEATURES ON TRAIN ONLY
    # -------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------
    # MODEL
    # -------------------------
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Probability that the trade succeeds
    prob_success = model.predict_proba(X_test)[:, 1]

    test["prob_success"] = prob_success

    # -------------------------
    # TRADE FILTER
    # -------------------------
    threshold = 0.55
    test["trade"] = test["prob_success"] >= threshold

    # Base mean-reversion direction
    test["direction"] = -np.sign(test["zscore"])

    # Trade only when model likes the setup
    test["position"] = np.where(test["trade"], test["direction"], 0)

    # -------------------------
    # PNL
    # -------------------------
    test["residual_change"] = test["residual"].diff()
    test["pnl"] = test["position"].shift(1).fillna(0) * test["residual_change"]
    test["cum_pnl"] = test["pnl"].cumsum()

    sharpe = sharpe_ratio(test["pnl"])
    pnl = test["cum_pnl"].iloc[-1] if len(test) else np.nan
    mdd = max_drawdown(test["cum_pnl"]) if len(test) else np.nan

    print("\n=== LOGISTIC FILTER MODEL ===")
    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")
    print(f"Trade frequency: {test['trade'].mean():.3f}")
    print(f"Sharpe: {sharpe}")
    print(f"Final PnL: {pnl}")
    print(f"Max Drawdown: {mdd}")

    print("\nPredicted success probability summary:")
    print(test["prob_success"].describe())

    test.to_csv(PROCESSED_PATH / "logistic_results.csv")


if __name__ == "__main__":
    main()
