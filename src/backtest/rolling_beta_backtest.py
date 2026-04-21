from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

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


def fit_ols(train_df: pd.DataFrame) -> tuple[float, float]:
    y = train_df["ttf_usd_mmbtu"]
    x = sm.add_constant(train_df["hh_usd_mmbtu"])
    model = sm.OLS(y, x).fit()

    alpha = float(model.params["const"])
    beta = float(model.params["hh_usd_mmbtu"])
    return alpha, beta


def build_rolling_residuals(df: pd.DataFrame, regression_window: int) -> pd.DataFrame:
    out = df.copy()
    out["alpha"] = np.nan
    out["beta"] = np.nan
    out["residual"] = np.nan

    for i in range(regression_window, len(out)):
        train = out.iloc[i - regression_window:i]
        alpha, beta = fit_ols(train)

        hh_t = out.iloc[i]["hh_usd_mmbtu"]
        ttf_t = out.iloc[i]["ttf_usd_mmbtu"]
        residual_t = ttf_t - (alpha + beta * hh_t)

        out.iloc[i, out.columns.get_loc("alpha")] = alpha
        out.iloc[i, out.columns.get_loc("beta")] = beta
        out.iloc[i, out.columns.get_loc("residual")] = residual_t

    return out


def add_features(
    df: pd.DataFrame,
    zscore_window: int,
    vol_window: int,
) -> pd.DataFrame:
    out = df.copy()

    out["resid_mean"] = out["residual"].rolling(zscore_window).mean()
    out["resid_std"] = out["residual"].rolling(zscore_window).std()
    out["zscore"] = (out["residual"] - out["resid_mean"]) / out["resid_std"]

    out["resid_vol"] = out["residual"].rolling(vol_window).std()
    out["residual_change"] = out["residual"].diff()

    return out


def run_strategy(
    df: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    transaction_cost: float,
) -> pd.DataFrame:
    out = df.copy()

    position = 0
    positions = []

    for _, row in out.iterrows():
        z = row["zscore"]

        if pd.isna(z):
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
    out["position_lag"] = out["position"].shift(1).fillna(0)

    # gross pnl from yesterday's held position
    out["gross_pnl"] = out["position_lag"] * out["residual_change"]

    # transaction cost when changing position
    out["turnover"] = (out["position"] - out["position_lag"]).abs()
    out["cost"] = transaction_cost * out["turnover"]

    out["net_pnl"] = out["gross_pnl"] - out["cost"]
    out["cum_net_pnl"] = out["net_pnl"].cumsum()

    return out


def main():
    df = pd.read_csv(PROCESSED_PATH / "clean_data.csv",
                     index_col=0, parse_dates=True)

    regression_window = 252
    zscore_window = 60
    vol_window = 60

    entry_threshold = 2.0
    exit_threshold = 0.5

    # cost in "residual units" per one unit of position change
    transaction_cost = 0.05

    df = build_rolling_residuals(df, regression_window=regression_window)
    df = add_features(df, zscore_window=zscore_window, vol_window=vol_window)
    df = run_strategy(
        df,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        transaction_cost=transaction_cost,
    )

    result = df.dropna(subset=["residual", "zscore", "net_pnl"]).copy()

    sharpe = sharpe_ratio(result["net_pnl"])
    pnl = result["cum_net_pnl"].iloc[-1]
    mdd = max_drawdown(result["cum_net_pnl"])

    print("\n=== ROLLING BETA BACKTEST ===")
    print(f"Observations: {len(result)}")
    print(f"Final PnL: {pnl}")
    print(f"Sharpe: {sharpe}")
    print(f"Max Drawdown: {mdd}")
    print(f"Average beta: {result['beta'].mean()}")
    print(f"Beta std: {result['beta'].std()}")
    print(f"Average turnover: {result['turnover'].mean()}")

    result.to_csv(PROCESSED_PATH / "rolling_beta_backtest_results.csv")


if __name__ == "__main__":
    main()
