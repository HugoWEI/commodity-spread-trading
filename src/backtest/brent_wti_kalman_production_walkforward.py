from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_PATH = Path("data/processed")


# =========================
# METRICS
# =========================
def sharpe_ratio(pnl: pd.Series) -> float:
    pnl = pnl.dropna()
    if len(pnl) < 2 or pnl.std() == 0:
        return np.nan
    return np.sqrt(252) * pnl.mean() / pnl.std()


def max_drawdown(cum_pnl: pd.Series) -> float:
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    return drawdown.min()


# =========================
# SPLIT GENERATOR
# =========================
def make_splits(df, train_size=252 * 2, test_size=63):
    splits = []
    start = 0
    while start + train_size + test_size <= len(df):
        splits.append(
            (slice(start, start + train_size),
             slice(start + train_size, start + train_size + test_size))
        )
        start += test_size
    return splits


# =========================
# KALMAN FILTER
# =========================
def kalman_filter(df, delta, R):
    out = df.copy()
    n = len(out)

    alpha = np.zeros(n)
    beta = np.zeros(n)

    beta[0] = 1.0
    P = np.eye(2)
    Q = delta * np.eye(2)

    for t in range(1, n):
        x = np.array([1.0, out["wti"].iloc[t]])
        theta_prev = np.array([alpha[t - 1], beta[t - 1]])

        # Predict
        theta_pred = theta_prev
        P = P + Q

        # Observe
        y = out["brent"].iloc[t]
        e = y - (x @ theta_pred)

        # Update
        S = x @ P @ x.T + R
        K = (P @ x.T) / S
        theta = theta_pred + K * e
        P = P - np.outer(K, x) @ P

        alpha[t], beta[t] = theta

    out["alpha"] = alpha
    out["beta"] = beta
    out["residual"] = out["brent"] - (out["alpha"] + out["beta"] * out["wti"])

    return out


# =========================
# FEATURE ENGINEERING
# =========================
def add_features(df, z_win, vol_win, beta_win):
    out = df.copy()

    out["mean"] = out["residual"].rolling(z_win).mean()
    out["std"] = out["residual"].rolling(z_win).std()
    out["raw_z"] = (out["residual"] - out["mean"]) / out["std"]

    out["vol"] = out["residual"].rolling(vol_win).std()
    out["beta_vol"] = out["beta"].diff().rolling(beta_win).std()

    out["residual_change"] = out["residual"].diff()

    # production timing
    out["zscore"] = out["raw_z"].shift(1)
    out["vol_signal"] = out["vol"].shift(1)
    out["beta_vol_signal"] = out["beta_vol"].shift(1)

    return out


# =========================
# STRATEGY
# =========================
def run_strategy(
    df,
    test_index,
    entry,
    exit,
    cost,
    target_risk,
    vol_th,
    beta_th,
):
    out = df.copy()

    out["active"] = (
        (out["vol_signal"] <= vol_th) &
        (out["beta_vol_signal"] <= beta_th)
    )

    pos = 0.0
    positions = []

    for _, r in out.iterrows():
        z = r["zscore"]
        active = r["active"]

        if pd.isna(z) or pd.isna(active) or not active:
            pos = 0.0
        elif pos == 0:
            if z > entry:
                pos = -1
            elif z < -entry:
                pos = 1
        elif pos > 0 and z > -exit:
            pos = 0
        elif pos < 0 and z < exit:
            pos = 0

        positions.append(pos)

    out["pos_raw"] = positions

    # scaling
    out["pos"] = np.where(
        out["vol_signal"] > 0,
        out["pos_raw"] * (target_risk / out["vol_signal"]),
        0,
    ).clip(-3, 3)

    out["pos_lag"] = out["pos"].shift(1).fillna(0)

    out["pnl"] = out["pos_lag"] * out["residual_change"]

    out["turnover"] = (out["pos"] - out["pos_lag"]).abs()
    out["cost"] = cost * out["turnover"]
    out["net_pnl"] = out["pnl"] - out["cost"]

    return out.loc[test_index].copy()


# =========================
# TRAIN EVALUATION
# =========================
def evaluate_train(train, delta, R, cfg):
    feat = kalman_filter(train, delta, R)
    feat = add_features(feat, cfg["z_win"], cfg["vol_win"], cfg["beta_win"])

    vol = feat["vol_signal"].dropna()
    beta_vol = feat["beta_vol_signal"].dropna()

    if len(vol) == 0 or len(beta_vol) == 0:
        return np.nan, None

    vol_th = float(vol.quantile(cfg["vol_q"]))
    beta_th = float(beta_vol.quantile(cfg["beta_q"]))

    bt = run_strategy(
        feat,
        feat.index,
        cfg["entry"],
        cfg["exit"],
        cfg["cost"],
        cfg["risk"],
        vol_th,
        beta_th,
    )

    return sharpe_ratio(bt["net_pnl"]), {
        "delta": delta,
        "R": R,
        "vol_th": vol_th,
        "beta_th": beta_th,
    }


# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv(
        PROCESSED_PATH / "brent_wti_clean.csv",
        index_col=0,
        parse_dates=True,
    ).sort_index()

    cfg = dict(
        z_win=60,
        vol_win=60,
        beta_win=60,
        entry=2.0,
        exit=0.5,
        vol_q=0.8,
        beta_q=0.8,
        risk=0.25,
        cost=0.02,
    )

    delta_grid = [1e-6, 1e-5, 1e-4]
    R_grid = [1e-2, 1e-1, 1.0]

    splits = make_splits(df)

    results = []
    summary = []

    for i, (tr, te) in enumerate(splits, 1):
        train = df.iloc[tr]
        test = df.iloc[te]

        best_score = -np.inf
        best = None

        for d in delta_grid:
            for r in R_grid:
                score, details = evaluate_train(train, d, r, cfg)
                if pd.notna(score) and score > best_score:
                    best_score = score
                    best = details | {"train_sharpe": score}

        if best is None:
            continue

        combined = pd.concat([train, test])
        combined = kalman_filter(combined, best["delta"], best["R"])
        combined = add_features(
            combined, cfg["z_win"], cfg["vol_win"], cfg["beta_win"])

        test_res = run_strategy(
            combined,
            test.index,
            cfg["entry"],
            cfg["exit"],
            cfg["cost"],
            cfg["risk"],
            best["vol_th"],
            best["beta_th"],
        )

        test_res["split"] = i
        results.append(test_res)

        trade_days = int(test_res["pos_raw"].ne(0).sum())
        nonzero_pnl = int((test_res["net_pnl"] != 0).sum())

        test_sharpe = sharpe_ratio(test_res["net_pnl"])
        display_sharpe = "NA" if pd.isna(test_sharpe) else f"{test_sharpe:.3f}"

        print(
            f"Split {i}: "
            f"delta={best['delta']:.0e}, R={best['R']:.0e}, "
            f"train_sharpe={best['train_sharpe']:.3f}, "
            f"test_sharpe={display_sharpe}, "
            f"trades={trade_days}, pnl_days={nonzero_pnl}"
        )

        summary.append({
            "split": i,
            "delta": best["delta"],
            "R": best["R"],
            "train_sharpe": best["train_sharpe"],
            "test_sharpe": test_sharpe,
            "trade_days": trade_days,
            "pnl_days": nonzero_pnl,
        })

    final = pd.concat(results).sort_index()
    final["cum"] = final["net_pnl"].cumsum()

    summary_df = pd.DataFrame(summary)

    valid_sharpes = summary_df.loc[summary_df["pnl_days"]
                                   >= 10, "test_sharpe"].dropna()
    empty_split_rate = (summary_df["pnl_days"] == 0).mean()

    print("\n=== FINAL ===")
    print("Sharpe:", sharpe_ratio(final["net_pnl"]))
    print("PnL:", final["cum"].iloc[-1])
    print("MaxDD:", max_drawdown(final["cum"]))
    print("Mean trade days per split:", summary_df["trade_days"].mean())
    print("Median trade days per split:", summary_df["trade_days"].median())
    print("Mean pnl days per split:", summary_df["pnl_days"].mean())
    print("Median pnl days per split:", summary_df["pnl_days"].median())
    print("Empty split rate:", empty_split_rate)

    if len(valid_sharpes) > 0:
        print("Median split Sharpe (pnl_days >= 10):", valid_sharpes.median())
        print("Mean split Sharpe (pnl_days >= 10):", valid_sharpes.mean())
    else:
        print("Median split Sharpe (pnl_days >= 10): NA")
        print("Mean split Sharpe (pnl_days >= 10): NA")

    final.to_csv(PROCESSED_PATH / "kalman_final.csv")
    pd.DataFrame(summary).to_csv(PROCESSED_PATH /
                                 "kalman_summary.csv", index=False)


if __name__ == "__main__":
    main()
