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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

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

    return out


def make_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy()
    future_ret = out["residual"].shift(-horizon) - out["residual"]

    signal_direction = np.sign(out["zscore"])
    future_direction = np.sign(future_ret)

    # Success means future move is opposite current z-score,
    # i.e. mean reversion happened over the horizon.
    out["target"] = (signal_direction != future_direction).astype(float)

    return out


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


def prepare_candidate_dataset(
    df: pd.DataFrame,
    z_entry_threshold: float,
    horizon: int,
) -> pd.DataFrame:
    out = build_features(df)
    out = make_target(out, horizon=horizon)

    candidate_mask = out["abs_zscore"] >= z_entry_threshold

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
    out = out.loc[candidate_mask, keep_cols].copy()
    out = out.dropna()
    out["target"] = out["target"].astype(int)

    return out


def evaluate_threshold_on_train(
    train_df: pd.DataFrame,
    prob_threshold: float,
) -> tuple[float, pd.DataFrame]:
    eval_df = train_df.copy()

    eval_df["trade"] = eval_df["prob_success"] >= prob_threshold
    eval_df["direction"] = -np.sign(eval_df["zscore"])
    eval_df["position"] = np.where(eval_df["trade"], eval_df["direction"], 0)

    eval_df["residual_change"] = eval_df["residual"].diff()
    eval_df["pnl"] = eval_df["position"].shift(
        1).fillna(0) * eval_df["residual_change"]

    score = sharpe_ratio(eval_df["pnl"])
    return score, eval_df


def main():
    df = pd.read_csv(
        PROCESSED_PATH / "rolling_beta_backtest_results.csv",
        index_col=0,
        parse_dates=True,
    )

    # Keep only rows where the rolling-beta baseline is actually defined
    df = df.dropna(subset=["residual", "zscore", "beta"]).copy()

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

    horizon = 5
    z_entry_grid = [1.5, 2.0, 2.5]
    prob_threshold_grid = [0.50, 0.55, 0.60, 0.65, 0.70]

    splits = make_splits(df, train_size=252 * 2, test_size=63)

    all_test_results = []
    chosen_params = []

    for split_num, (train_idx, test_idx) in enumerate(splits, start=1):
        train_raw = df.iloc[train_idx].copy()
        test_raw = df.iloc[test_idx].copy()

        best_score = -np.inf
        best_config = None
        best_model = None
        best_scaler = None

        for z_entry_threshold in z_entry_grid:
            train_candidates = prepare_candidate_dataset(
                train_raw,
                z_entry_threshold=z_entry_threshold,
                horizon=horizon,
            )

            # Need enough rows and both classes present
            if len(train_candidates) < 50:
                continue
            if train_candidates["target"].nunique() < 2:
                continue

            X_train = train_candidates[feature_cols].values
            y_train = train_candidates["target"].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )
            model.fit(X_train_scaled, y_train)

            train_candidates = train_candidates.copy()
            train_candidates["prob_success"] = model.predict_proba(X_train_scaled)[
                :, 1]

            for prob_threshold in prob_threshold_grid:
                score, _ = evaluate_threshold_on_train(
                    train_df=train_candidates,
                    prob_threshold=prob_threshold,
                )

                if pd.notna(score) and score > best_score:
                    best_score = score
                    best_config = {
                        "z_entry_threshold": z_entry_threshold,
                        "prob_threshold": prob_threshold,
                        "train_sharpe": score,
                        "train_rows": len(train_candidates),
                    }
                    best_model = model
                    best_scaler = scaler

        if best_config is None:
            print(f"Split {split_num}: no valid train configuration found.")
            continue

        # Build features on train+test chronology so test rows have proper lagged context
        combined_raw = pd.concat([train_raw, test_raw]).copy()
        combined_candidates = prepare_candidate_dataset(
            combined_raw,
            z_entry_threshold=best_config["z_entry_threshold"],
            horizon=horizon,
        )

        test_candidates = combined_candidates.loc[
            combined_candidates.index.intersection(test_raw.index)
        ].copy()

        # Some splits may have very few candidate trades in the test block
        if len(test_candidates) == 0:
            print(
                f"Split {split_num}: no test candidates at z >= {best_config['z_entry_threshold']}."
            )
            continue

        X_test = test_candidates[feature_cols].values
        X_test_scaled = best_scaler.transform(X_test)

        test_candidates["prob_success"] = best_model.predict_proba(X_test_scaled)[
            :, 1]
        test_candidates["trade"] = (
            test_candidates["prob_success"] >= best_config["prob_threshold"]
        )
        test_candidates["direction"] = -np.sign(test_candidates["zscore"])
        test_candidates["position"] = np.where(
            test_candidates["trade"],
            test_candidates["direction"],
            0,
        )

        test_candidates["residual_change"] = test_candidates["residual"].diff()
        test_candidates["pnl"] = (
            test_candidates["position"].shift(1).fillna(0)
            * test_candidates["residual_change"]
        )
        test_candidates["split_num"] = split_num

        all_test_results.append(test_candidates)

        test_sharpe = sharpe_ratio(test_candidates["pnl"])
        chosen_params.append(
            {
                "split_num": split_num,
                "z_entry_threshold": best_config["z_entry_threshold"],
                "prob_threshold": best_config["prob_threshold"],
                "train_sharpe": best_config["train_sharpe"],
                "test_sharpe": test_sharpe,
                "train_rows": best_config["train_rows"],
                "test_rows": len(test_candidates),
                "trade_rate_test": float(test_candidates["trade"].mean()),
            }
        )

        print(
            f"Split {split_num}: "
            f"z={best_config['z_entry_threshold']}, "
            f"p={best_config['prob_threshold']}, "
            f"train_sharpe={best_config['train_sharpe']:.3f}, "
            f"test_sharpe={test_sharpe:.3f}, "
            f"test_rows={len(test_candidates)}"
        )

    if not all_test_results:
        raise ValueError("No walk-forward test results produced.")

    final_df = pd.concat(all_test_results).sort_index()
    final_df["cum_pnl"] = final_df["pnl"].cumsum()

    params_df = pd.DataFrame(chosen_params)

    final_sharpe = sharpe_ratio(final_df["pnl"])
    final_pnl = final_df["cum_pnl"].iloc[-1]
    final_mdd = max_drawdown(final_df["cum_pnl"])

    print("\n=== WALK-FORWARD LOGISTIC RESULTS ===")
    print(f"Total test observations: {len(final_df)}")
    print(f"Final PnL: {final_pnl}")
    print(f"Sharpe: {final_sharpe}")
    print(f"Max Drawdown: {final_mdd}")
    print(f"Average trade rate: {final_df['trade'].mean():.3f}")

    final_df.to_csv(PROCESSED_PATH / "walk_forward_logistic_results.csv")
    params_df.to_csv(PROCESSED_PATH /
                     "walk_forward_logistic_params.csv", index=False)


if __name__ == "__main__":
    main()
