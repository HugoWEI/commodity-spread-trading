import pandas as pd
import numpy as np


def compute_signal_features(df: pd.DataFrame, zscore_window: int, vol_window: int) -> pd.DataFrame:
    out = df.copy()

    out["mean"] = out["residual"].rolling(zscore_window).mean()
    out["std"] = out["residual"].rolling(zscore_window).std()
    out["zscore"] = (out["residual"] - out["mean"]) / out["std"]

    out["vol"] = out["residual"].rolling(vol_window).std()

    return out


def run_strategy_on_slice(
    df: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    vol_threshold: float | None = None,
) -> pd.DataFrame:
    out = df.copy()

    position = 0
    positions = []

    for _, row in out.iterrows():
        z = row["zscore"]
        vol = row["vol"]

        if pd.isna(z) or pd.isna(vol):
            position = 0
            positions.append(position)
            continue

        if vol_threshold is not None and vol >= vol_threshold:
            position = 0
            positions.append(position)
            continue

        if position == 0:
            if z > entry_threshold:
                position = -1
            elif z < -entry_threshold:
                position = 1

        elif position == 1:
            if z > -exit_threshold:
                position = 0

        elif position == -1:
            if z < exit_threshold:
                position = 0

        positions.append(position)

    out["position"] = positions
    out["residual_change"] = out["residual"].diff()
    out["pnl"] = out["position"].shift(1) * out["residual_change"]
    out["cum_pnl"] = out["pnl"].cumsum()

    return out


def sharpe_ratio(pnl: pd.Series) -> float:
    pnl = pnl.dropna()
    if len(pnl) < 2 or pnl.std() == 0:
        return np.nan
    return np.sqrt(252) * pnl.mean() / pnl.std()
