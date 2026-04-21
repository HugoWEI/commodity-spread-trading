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


def make_splits(
    df: pd.DataFrame,
    train_size: int = 252 * 2,
    test_size: int = 63,
) -> list[tuple[slice, slice]]:
    splits = []
    start = 0
    while start + train_size + test_size <= len(df):
        train_idx = slice(start, start + train_size)
        test_idx = slice(start + train_size, start + train_size + test_size)
        splits.append((train_idx, test_idx))
        start += test_size
    return splits


def compute_train_test_residuals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    alpha, beta = fit_ols(train_df)

    train_out = train_df.copy()
    test_out = test_df.copy()

    train_out["alpha"] = alpha
    train_out["beta"] = beta
    test_out["alpha"] = alpha
    test_out["beta"] = beta

    train_out["residual"] = train_out["brent"] - \
        (alpha + beta * train_out["wti"])
    test_out["residual"] = test_out["brent"] - (alpha + beta * test_out["wti"])

    return pd.concat([train_out, test_out]), alpha, beta


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

    out["resid_vol"] = out["residual"].rolling(vol_window).std()
    # beta is constant within a split, but this keeps the interface consistent
    out["beta_vol"] = out["beta"].rolling(beta_vol_window).std()

    out["residual_change"] = out["residual"].diff()

    # Production-grade timing: trade only on lagged information
    out["zscore"] = out["raw_zscore"].shift(1)
    out["resid_vol_signal"] = out["resid_vol"].shift(1)
    out["beta_vol_signal"] = out["beta_vol"].shift(1)

    return out


def run_strategy(
    df: pd.DataFrame,
    test_index: pd.Index,
    entry_threshold: float,
    exit_threshold: float,
    transaction_cost: float,
    target_risk: float,
    vol_threshold: float,
    beta_vol_threshold: float,
) -> pd.DataFrame:
    out = df.copy()

    out["regime_active"] = (
        (out["resid_vol_signal"] <= vol_threshold) &
        (out["beta_vol_signal"] <= beta_vol_threshold)
    )

    position = 0.0
    positions = []

    for _, row in out.iterrows():
        z = row["zscore"]
        active = row["regime_active"]

        if pd.isna(z) or pd.isna(active) or not active:
            position = 0.0
            positions.append(position)
            continue

        if position == 0.0:
            if z > entry_threshold:
                position = -1.0
            elif z < -entry_threshold:
                position = 1.0

        elif position > 0:
            if z > -exit_threshold:
                position = 0.0

        elif position < 0:
            if z < exit_threshold:
                position = 0.0

        positions.append(position)

    out["position_raw"] = positions

    out["position_scaled"] = np.where(
        out["resid_vol_signal"] > 0,
        out["position_raw"] * (target_risk / out["resid_vol_signal"]),
        0.0,
    )
    out["position_scaled"] = out["position_scaled"].clip(-3.0, 3.0)

    out["position_lag"] = out["position_scaled"].shift(1).fillna(0.0)
    out["gross_pnl"] = out["position_lag"] * out["residual_change"]

    out["turnover"] = (out["position_scaled"] - out["position_lag"]).abs()
    out["cost"] = transaction_cost * out["turnover"]

    out["net_pnl"] = out["gross_pnl"] - out["cost"]

    out["vol_threshold"] = vol_threshold
    out["beta_vol_threshold"] = beta_vol_threshold

    # Return only live/test period rows
    return out.loc[test_index].copy()


def main():
    df = pd.read_csv(
        PROCESSED_PATH / "brent_wti_clean.csv",
        index_col=0,
        parse_dates=True,
    ).sort_index()

    # Production-style configuration
    train_size = 252 * 2
    test_size = 63

    zscore_window = 60
    vol_window = 60
    beta_vol_window = 60

    entry_threshold = 2.0
    exit_threshold = 0.5

    vol_quantile = 0.80
    beta_vol_quantile = 0.80

    target_risk = 0.25
    transaction_cost = 0.02

    splits = make_splits(df, train_size=train_size, test_size=test_size)

    all_test_results = []
    split_summary = []

    for split_num, (train_idx, test_idx) in enumerate(splits, start=1):
        train_raw = df.iloc[train_idx].copy()
        test_raw = df.iloc[test_idx].copy()

        combined, alpha, beta = compute_train_test_residuals(
            train_raw, test_raw)
        combined = add_features(
            combined,
            zscore_window=zscore_window,
            vol_window=vol_window,
            beta_vol_window=beta_vol_window,
        )

        train_feat = combined.loc[train_raw.index].copy()

        # Thresholds estimated on train only
        train_vol = train_feat["resid_vol_signal"].dropna()
        train_beta_vol = train_feat["beta_vol_signal"].dropna()

        if len(train_vol) == 0 or len(train_beta_vol) == 0:
            continue

        vol_threshold = float(train_vol.quantile(vol_quantile))
        beta_vol_threshold = float(train_beta_vol.quantile(beta_vol_quantile))

        test_result = run_strategy(
            combined,
            test_index=test_raw.index,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            transaction_cost=transaction_cost,
            target_risk=target_risk,
            vol_threshold=vol_threshold,
            beta_vol_threshold=beta_vol_threshold,
        )

        test_result["split_num"] = split_num
        test_result["alpha"] = alpha
        test_result["beta"] = beta

        all_test_results.append(test_result)

        split_sharpe = sharpe_ratio(test_result["net_pnl"])
        split_summary.append(
            {
                "split_num": split_num,
                "alpha": alpha,
                "beta": beta,
                "vol_threshold": vol_threshold,
                "beta_vol_threshold": beta_vol_threshold,
                "test_rows": len(test_result),
                "trade_rate": float(test_result["position_raw"].ne(0).mean()),
                "split_sharpe": split_sharpe,
            }
        )

        print(
            f"Split {split_num}: "
            f"beta={beta:.3f}, "
            f"vol_th={vol_threshold:.3f}, "
            f"beta_vol_th={beta_vol_threshold:.5f}, "
            f"trade_rate={test_result['position_raw'].ne(0).mean():.3f}, "
            f"split_sharpe={split_sharpe:.3f}"
        )

    if not all_test_results:
        raise ValueError("No test results produced.")

    final_df = pd.concat(all_test_results).sort_index()
    final_df["cum_net_pnl"] = final_df["net_pnl"].cumsum()

    summary_df = pd.DataFrame(split_summary)

    final_sharpe = sharpe_ratio(final_df["net_pnl"])
    final_pnl = final_df["cum_net_pnl"].iloc[-1]
    final_mdd = max_drawdown(final_df["cum_net_pnl"])

    print("\n=== BRENT–WTI PRODUCTION WALK-FORWARD ===")
    print(f"Total live rows: {len(final_df)}")
    print(f"Final PnL: {final_pnl}")
    print(f"Sharpe: {final_sharpe}")
    print(f"Max Drawdown: {final_mdd}")
    print(f"Average beta: {final_df['beta'].mean()}")
    print(f"Beta std across splits: {summary_df['beta'].std()}")
    print(f"Average trade rate: {summary_df['trade_rate'].mean()}")

    final_df.to_csv(PROCESSED_PATH /
                    "brent_wti_production_walkforward_results.csv")
    summary_df.to_csv(
        PROCESSED_PATH / "brent_wti_production_walkforward_summary.csv", index=False)


if __name__ == "__main__":
    main()
