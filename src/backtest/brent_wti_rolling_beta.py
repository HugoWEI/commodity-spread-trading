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
    y = train_df["brent"]
    x = sm.add_constant(train_df["wti"])
    model = sm.OLS(y, x).fit()

    alpha = float(model.params["const"])
    beta = float(model.params["wti"])
    return alpha, beta


def build_rolling_residuals(df: pd.DataFrame, regression_window: int) -> pd.DataFrame:
    out = df.copy()
    out["alpha"] = np.nan
    out["beta"] = np.nan
    out["residual"] = np.nan

    alpha_col = out.columns.get_loc("alpha")
    beta_col = out.columns.get_loc("beta")
    residual_col = out.columns.get_loc("residual")

    for i in range(regression_window, len(out)):
        train = out.iloc[i - regression_window:i]
        alpha, beta = fit_ols(train)

        wti_t = out.iloc[i]["wti"]
        brent_t = out.iloc[i]["brent"]
        residual_t = brent_t - (alpha + beta * wti_t)

        out.iloc[i, alpha_col] = alpha
        out.iloc[i, beta_col] = beta
        out.iloc[i, residual_col] = residual_t

    return out


def add_features(
    df: pd.DataFrame,
    zscore_window: int,
    vol_window: int,
    beta_vol_window: int,
) -> pd.DataFrame:
    out = df.copy()

    out["resid_mean"] = out["residual"].rolling(zscore_window).mean()
    out["resid_std"] = out["residual"].rolling(zscore_window).std()
    out["raw_zscore"] = (
        out["residual"] - out["resid_mean"]) / out["resid_std"]

    # Risk / regime features
    out["resid_vol"] = out["residual"].rolling(vol_window).std()
    out["beta_vol"] = out["beta"].rolling(beta_vol_window).std()

    out["residual_change"] = out["residual"].diff()

    # Perfection fix:
    # only trade on information known after previous close
    out["zscore"] = out["raw_zscore"].shift(1)
    out["resid_vol_signal"] = out["resid_vol"].shift(1)
    out["beta_vol_signal"] = out["beta_vol"].shift(1)

    return out


def run_strategy(
    df: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    transaction_cost: float,
    vol_quantile: float,
    beta_vol_quantile: float,
    target_risk: float,
) -> pd.DataFrame:
    out = df.copy()

    # Regime thresholds from available history in this sample.
    # For a fully walk-forward production setup, estimate these on train only.
    vol_threshold = out["resid_vol_signal"].dropna().quantile(vol_quantile)
    beta_vol_threshold = out["beta_vol_signal"].dropna().quantile(
        beta_vol_quantile)

    out["regime_active"] = (
        (out["resid_vol_signal"] <= vol_threshold) &
        (out["beta_vol_signal"] <= beta_vol_threshold)
    )

    position = 0
    positions = []

    for _, row in out.iterrows():
        z = row["zscore"]
        active = row["regime_active"]

        if pd.isna(z) or pd.isna(active) or not active:
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

    out["position_raw"] = positions

    # Volatility scaling using lagged signal-time volatility
    out["position_scaled"] = np.where(
        out["resid_vol_signal"] > 0,
        out["position_raw"] * (target_risk / out["resid_vol_signal"]),
        0.0,
    )

    # Prevent extreme leverage when vol is tiny
    out["position_scaled"] = out["position_scaled"].clip(-3.0, 3.0)

    out["position_lag"] = out["position_scaled"].shift(1).fillna(0.0)

    out["gross_pnl"] = out["position_lag"] * out["residual_change"]

    out["turnover"] = (out["position_scaled"] - out["position_lag"]).abs()
    out["cost"] = transaction_cost * out["turnover"]

    out["net_pnl"] = out["gross_pnl"] - out["cost"]
    out["cum_net_pnl"] = out["net_pnl"].cumsum()

    out["vol_threshold"] = vol_threshold
    out["beta_vol_threshold"] = beta_vol_threshold

    return out


def main():
    df = pd.read_csv(
        PROCESSED_PATH / "brent_wti_clean.csv",
        index_col=0,
        parse_dates=True,
    )

    # Rolling hedge model
    regression_window = 252

    # Signal construction
    zscore_window = 60
    vol_window = 60
    beta_vol_window = 60

    # Trading rules
    entry_threshold = 2.0
    exit_threshold = 0.5

    # Risk controls
    vol_quantile = 0.80
    beta_vol_quantile = 0.80
    target_risk = 0.25

    # Cost in residual units per 1 unit of turnover
    transaction_cost = 0.02

    df = build_rolling_residuals(df, regression_window=regression_window)
    df = add_features(
        df,
        zscore_window=zscore_window,
        vol_window=vol_window,
        beta_vol_window=beta_vol_window,
    )
    df = run_strategy(
        df,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        transaction_cost=transaction_cost,
        vol_quantile=vol_quantile,
        beta_vol_quantile=beta_vol_quantile,
        target_risk=target_risk,
    )

    result = df.dropna(
        subset=[
            "residual",
            "zscore",
            "resid_vol_signal",
            "beta_vol_signal",
            "net_pnl",
        ]
    ).copy()

    sharpe = sharpe_ratio(result["net_pnl"])
    pnl = result["cum_net_pnl"].iloc[-1]
    mdd = max_drawdown(result["cum_net_pnl"])

    print("\n=== BRENT–WTI ROLLING BETA BACKTEST (UPGRADED) ===")
    print(f"Observations: {len(result)}")
    print(f"Final PnL: {pnl}")
    print(f"Sharpe: {sharpe}")
    print(f"Max Drawdown: {mdd}")
    print(f"Average beta: {result['beta'].mean()}")
    print(f"Beta std: {result['beta'].std()}")
    print(f"Average raw position: {result['position_raw'].abs().mean()}")
    print(f"Average scaled position: {result['position_scaled'].abs().mean()}")
    print(f"Average turnover: {result['turnover'].mean()}")
    print(f"Regime active rate: {result['regime_active'].mean()}")
    print(f"Residual vol threshold: {result['vol_threshold'].iloc[-1]}")
    print(f"Beta vol threshold: {result['beta_vol_threshold'].iloc[-1]}")

    result.to_csv(PROCESSED_PATH /
                  "brent_wti_rolling_beta_upgraded_results.csv")


if __name__ == "__main__":
    main()
