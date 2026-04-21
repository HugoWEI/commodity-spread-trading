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
# SPLITS
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

        theta_pred = theta_prev
        P = P + Q

        y = out["brent"].iloc[t]
        e = y - (x @ theta_pred)

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
# FEATURES
# =========================
def add_features(df, z_win, vol_win, beta_win):
    out = df.copy()

    out["mean"] = out["residual"].rolling(z_win).mean()
    out["std"] = out["residual"].rolling(z_win).std()
    out["raw_z"] = (out["residual"] - out["mean"]) / out["std"]

    out["vol"] = out["residual"].rolling(vol_win).std()
    out["beta_vol"] = out["beta"].diff().rolling(beta_win).std()

    out["spread_change"] = out["residual"].diff()

    # production timing
    out["zscore"] = out["raw_z"].shift(1)
    out["vol_signal"] = out["vol"].shift(1)
    out["beta_vol_signal"] = out["beta_vol"].shift(1)

    return out


# =========================
# TRAIN EVAL
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

    bt = run_strategy_usd(
        feat,
        feat.index,
        cfg,
        vol_th,
        beta_th,
    )

    return sharpe_ratio(bt["net_pnl_usd"]), {
        "delta": delta,
        "R": R,
        "vol_th": vol_th,
        "beta_th": beta_th,
    }


# =========================
# STRATEGY IN USD
# =========================
def run_strategy_usd(df, test_index, cfg, vol_th, beta_th):
    out = df.copy()

    out["active"] = (
        (out["vol_signal"] <= vol_th) &
        (out["beta_vol_signal"] <= beta_th)
    )

    pos = 0
    positions = []

    for _, r in out.iterrows():
        z = r["zscore"]
        active = r["active"]

        if pd.isna(z) or pd.isna(active) or not active:
            pos = 0
        elif pos == 0:
            if z > cfg["entry"]:
                pos = -1
            elif z < -cfg["entry"]:
                pos = 1
        elif pos > 0 and z > -cfg["exit"]:
            pos = 0
        elif pos < 0 and z < cfg["exit"]:
            pos = 0

        positions.append(pos)

    out["signal_side"] = positions

    # Contract sizing:
    # target daily USD risk / (spread vol in $/bbl * contract size)
    # round down to whole contracts
    out["contracts_target"] = np.where(
        out["vol_signal"] > 0,
        cfg["target_daily_risk_usd"] /
        (out["vol_signal"] * cfg["contract_size_bbl"]),
        0.0,
    )

    out["contracts"] = np.floor(
        out["contracts_target"]).clip(0, cfg["max_contracts"])
    out["contracts_signed"] = out["signal_side"] * out["contracts"]

    # lagged execution
    out["contracts_lag"] = out["contracts_signed"].shift(1).fillna(0)

    # USD PnL:
    # spread change ($/bbl) * barrels/contract * number of spread units
    out["gross_pnl_usd"] = (
        out["contracts_lag"] * out["spread_change"] * cfg["contract_size_bbl"]
    )

    out["turnover_contracts"] = (
        out["contracts_signed"] - out["contracts_lag"]).abs()

    # flat per-contract round-turn approximation on each rebalance leg
    out["cost_usd"] = out["turnover_contracts"] * cfg["cost_per_contract_usd"]

    out["net_pnl_usd"] = out["gross_pnl_usd"] - out["cost_usd"]

    return out.loc[test_index].copy()


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
        # feature windows
        z_win=60,
        vol_win=60,
        beta_win=60,

        # signal
        entry=2.0,
        exit=0.5,

        # regime filter quantiles
        vol_q=0.80,
        beta_q=0.80,

        # contract plumbing
        contract_size_bbl=1000,        # ICE Brent / NYMEX WTI typical multiplier
        cost_per_contract_usd=20.0,    # simple placeholder cost
        target_daily_risk_usd=1000.0,  # target daily spread risk budget
        max_contracts=10,
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

        test_res = run_strategy_usd(
            combined,
            test.index,
            cfg,
            best["vol_th"],
            best["beta_th"],
        )

        test_res["split"] = i
        results.append(test_res)

        trade_days = int(test_res["signal_side"].ne(0).sum())
        pnl_days = int((test_res["net_pnl_usd"] != 0).sum())

        test_sharpe = sharpe_ratio(test_res["net_pnl_usd"])
        display_sharpe = "NA" if pd.isna(test_sharpe) else f"{test_sharpe:.3f}"

        print(
            f"Split {i}: "
            f"delta={best['delta']:.0e}, R={best['R']:.0e}, "
            f"train_sharpe={best['train_sharpe']:.3f}, "
            f"test_sharpe={display_sharpe}, "
            f"trades={trade_days}, pnl_days={pnl_days}"
        )

        summary.append({
            "split": i,
            "delta": best["delta"],
            "R": best["R"],
            "train_sharpe": best["train_sharpe"],
            "test_sharpe": test_sharpe,
            "trade_days": trade_days,
            "pnl_days": pnl_days,
            "avg_contracts": float(test_res["contracts"].mean()),
            "avg_abs_contracts": float(test_res["contracts_signed"].abs().mean()),
            "gross_pnl_usd": float(test_res["gross_pnl_usd"].sum()),
            "cost_usd": float(test_res["cost_usd"].sum()),
            "net_pnl_usd": float(test_res["net_pnl_usd"].sum()),
        })

    final = pd.concat(results).sort_index()
    final["cum_usd"] = final["net_pnl_usd"].cumsum()

    summary_df = pd.DataFrame(summary)
    valid_sharpes = summary_df.loc[summary_df["pnl_days"]
                                   >= 10, "test_sharpe"].dropna()
    empty_split_rate = (summary_df["pnl_days"] == 0).mean()

    print("\n=== FINAL USD RESULTS ===")
    print("Sharpe:", sharpe_ratio(final["net_pnl_usd"]))
    print("Net PnL USD:", final["cum_usd"].iloc[-1])
    print("MaxDD USD:", max_drawdown(final["cum_usd"]))
    print("Gross PnL USD:", final["gross_pnl_usd"].sum())
    print("Total Cost USD:", final["cost_usd"].sum())
    print("Mean trade days per split:", summary_df["trade_days"].mean())
    print("Median trade days per split:", summary_df["trade_days"].median())
    print("Mean pnl_days per split:", summary_df["pnl_days"].mean())
    print("Median pnl_days per split:", summary_df["pnl_days"].median())
    print("Mean abs contracts:", summary_df["avg_abs_contracts"].mean())
    print("Empty split rate:", empty_split_rate)

    if len(valid_sharpes) > 0:
        print("Median split Sharpe (pnl_days >= 10):", valid_sharpes.median())
        print("Mean split Sharpe (pnl_days >= 10):", valid_sharpes.mean())
    else:
        print("Median split Sharpe (pnl_days >= 10): NA")
        print("Mean split Sharpe (pnl_days >= 10): NA")

    final.to_csv(PROCESSED_PATH / "kalman_final_usd.csv")
    summary_df.to_csv(PROCESSED_PATH / "kalman_summary_usd.csv", index=False)


if __name__ == "__main__":
    main()
