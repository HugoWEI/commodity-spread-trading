import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.backtest.strategy import compute_signal_features, run_strategy_on_slice, sharpe_ratio

PROCESSED_PATH = Path("data/processed")


def make_splits(df: pd.DataFrame, train_size: int = 252 * 2, test_size: int = 63):
    splits = []
    start = 0

    while start + train_size + test_size <= len(df):
        train_idx = slice(start, start + train_size)
        test_idx = slice(start + train_size, start + train_size + test_size)
        splits.append((train_idx, test_idx))
        start += test_size

    return splits


def fit_cointegration(train_df: pd.DataFrame):
    y = train_df["ttf_usd_mmbtu"]
    x = sm.add_constant(train_df["hh_usd_mmbtu"])
    model = sm.OLS(y, x).fit()

    alpha = model.params["const"]
    beta = model.params["hh_usd_mmbtu"]

    return alpha, beta, model


def add_residuals(df: pd.DataFrame, alpha: float, beta: float):
    out = df.copy()
    out["residual"] = out["ttf_usd_mmbtu"] - \
        (alpha + beta * out["hh_usd_mmbtu"])
    return out


def evaluate_params_on_train(
    train_df: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    zscore_window: int,
    vol_window: int,
    vol_quantile: float,
):
    feat = compute_signal_features(
        train_df, zscore_window=zscore_window, vol_window=vol_window)

    vol_series = feat["vol"].dropna()
    if len(vol_series) == 0:
        return np.nan, np.nan

    vol_threshold = vol_series.quantile(vol_quantile)

    bt = run_strategy_on_slice(
        feat,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        vol_threshold=vol_threshold,
    )
    score = sharpe_ratio(bt["pnl"])
    return score, vol_threshold


def run_walk_forward_cointegration():
    df = pd.read_csv(PROCESSED_PATH / "clean_data.csv",
                     index_col=0, parse_dates=True)

    param_grid = {
        "entry_threshold": [1.5, 2.0, 2.5],
        "exit_threshold": [0.0, 0.5, 1.0],
        "zscore_window": [40, 60, 90],
        "vol_window": [40, 60, 90],
        "vol_quantile": [0.7, 0.8, 0.9],
    }

    all_params = list(itertools.product(
        param_grid["entry_threshold"],
        param_grid["exit_threshold"],
        param_grid["zscore_window"],
        param_grid["vol_window"],
        param_grid["vol_quantile"],
    ))

    splits = make_splits(df, train_size=252 * 2, test_size=63)

    all_test_results = []
    chosen_params_by_split = []
    hedge_params_by_split = []

    for split_num, (train_idx, test_idx) in enumerate(splits, start=1):
        train_raw = df.iloc[train_idx].copy()
        test_raw = df.iloc[test_idx].copy()

        alpha, beta, model = fit_cointegration(train_raw)

        train_df = add_residuals(train_raw, alpha, beta)
        test_df = add_residuals(test_raw, alpha, beta)

        best_score = -np.inf
        best = None

        for params in all_params:
            entry_threshold, exit_threshold, zscore_window, vol_window, vol_quantile = params

            if exit_threshold >= entry_threshold:
                continue

            score, vol_threshold = evaluate_params_on_train(
                train_df=train_df,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                zscore_window=zscore_window,
                vol_window=vol_window,
                vol_quantile=vol_quantile,
            )

            if pd.notna(score) and score > best_score:
                best_score = score
                best = {
                    "entry_threshold": entry_threshold,
                    "exit_threshold": exit_threshold,
                    "zscore_window": zscore_window,
                    "vol_window": vol_window,
                    "vol_quantile": vol_quantile,
                    "vol_threshold": vol_threshold,
                    "train_sharpe": score,
                }

        if best is None:
            continue

        # Use train history + test slice so rolling windows in test have proper past context
        combined = pd.concat([train_df, test_df]).copy()
        feat_all = compute_signal_features(
            combined,
            zscore_window=best["zscore_window"],
            vol_window=best["vol_window"],
        )

        feat_test = feat_all.loc[test_df.index].copy()

        bt_test = run_strategy_on_slice(
            feat_test,
            entry_threshold=best["entry_threshold"],
            exit_threshold=best["exit_threshold"],
            vol_threshold=best["vol_threshold"],
        )

        bt_test["split_num"] = split_num
        bt_test["alpha"] = alpha
        bt_test["beta"] = beta

        all_test_results.append(bt_test)

        chosen_params_by_split.append({
            "split_num": split_num,
            **best,
        })

        hedge_params_by_split.append({
            "split_num": split_num,
            "alpha": alpha,
            "beta": beta,
            "r_squared": model.rsquared,
        })

        print(
            f"Split {split_num}: "
            f"alpha={alpha:.3f}, beta={beta:.3f}, "
            f"train_sharpe={best['train_sharpe']:.3f}, "
            f"entry={best['entry_threshold']}, exit={best['exit_threshold']}, "
            f"zwin={best['zscore_window']}, vwin={best['vol_window']}, "
            f"vq={best['vol_quantile']}, vth={best['vol_threshold']:.3f}"
        )

    if not all_test_results:
        raise ValueError("No valid walk-forward test results produced.")

    final_df = pd.concat(all_test_results).sort_index()
    final_df["cum_pnl"] = final_df["pnl"].cumsum()

    final_sharpe = sharpe_ratio(final_df["pnl"])
    final_pnl = final_df["cum_pnl"].iloc[-1]

    params_df = pd.DataFrame(chosen_params_by_split)
    hedge_df = pd.DataFrame(hedge_params_by_split)

    final_df.to_csv(PROCESSED_PATH / "walk_forward_cointegration_results.csv")
    params_df.to_csv(PROCESSED_PATH /
                     "walk_forward_cointegration_params.csv", index=False)
    hedge_df.to_csv(PROCESSED_PATH /
                    "walk_forward_cointegration_hedge_params.csv", index=False)

    print("\n=== WALK-FORWARD COINTEGRATION RESULTS ===")
    print(f"Final PnL: {final_pnl}")
    print(f"Sharpe: {final_sharpe}")


if __name__ == "__main__":
    run_walk_forward_cointegration()
