from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_PATH = Path("data/processed")


def build_dataset():
    df = pd.read_csv(
        PROCESSED_PATH / "rolling_beta_backtest_results.csv",
        index_col=0,
        parse_dates=True,
    )

    out = df.copy()

    # -------------------------
    # TARGET: trade success
    # -------------------------
    horizon = 5

    future_ret = out["residual"].shift(-horizon) - out["residual"]

    signal = np.abs(out["zscore"]) > 1.5
    success = np.sign(out["zscore"]) != np.sign(future_ret)

    out["target"] = np.where(signal, success.astype(float), np.nan)

    # -------------------------
    # FEATURES
    # -------------------------
    out["zscore_lag1"] = out["zscore"].shift(1)
    out["zscore_lag5"] = out["zscore"].shift(5)

    out["resid_mom_5"] = out["residual"].diff(5)
    out["resid_mom_20"] = out["residual"].diff(20)

    out["vol_short"] = out["residual"].rolling(20).std()
    out["vol_long"] = out["residual"].rolling(60).std()
    out["vol_ratio"] = out["vol_short"] / out["vol_long"]

    out["beta_lag1"] = out["beta"].shift(1)
    out["beta_change"] = out["beta"].diff(5)
    out["beta_vol"] = out["beta"].rolling(60).std()

    out["abs_zscore"] = out["zscore"].abs()

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

    keep_cols = feature_cols + ["target", "zscore", "residual"]

    # Keep only rows where we actually had a signal,
    # then drop any rows with missing engineered features.
    dataset = out.loc[signal, keep_cols].copy()
    dataset = dataset.dropna()

    # Make target integer 0/1
    dataset["target"] = dataset["target"].astype(int)

    dataset.to_csv(PROCESSED_PATH / "ml_dataset.csv")

    print("\nDataset shape:", dataset.shape)
    print("\nTarget distribution:")
    print(dataset["target"].value_counts(normalize=True))
    print("\nPreview:\n")
    print(dataset.head())


if __name__ == "__main__":
    build_dataset()
