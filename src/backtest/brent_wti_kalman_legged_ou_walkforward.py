from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROCESSED_PATH = Path("data/processed")


# =========================
# METRICS
# =========================
def sharpe_ratio(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    return np.sqrt(252) * returns.mean() / returns.std()


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    return drawdown.min()


# =========================
# SPLITS
# =========================
def make_splits(df, train_size=252 * 2, test_size=63):
    splits = []
    start = 0
    while start + train_size + test_size <= len(df):
        splits.append(
            (
                slice(start, start + train_size),
                slice(start + train_size, start + train_size + test_size),
            )
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
# OU HALF-LIFE
# =========================
def estimate_half_life(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 30:
        return np.nan

    x_lag = s.shift(1).dropna()
    dx = s.diff().dropna()

    aligned = pd.concat([x_lag, dx], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan

    x = aligned.iloc[:, 0].values
    y = aligned.iloc[:, 1].values
    X = np.column_stack([np.ones(len(x)), x])

    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        b = beta[1]
    except np.linalg.LinAlgError:
        return np.nan

    if b >= 0:
        return np.nan

    hl = -np.log(2.0) / b
    if not np.isfinite(hl) or hl <= 0:
        return np.nan

    return float(hl)


def add_ou_half_life(df: pd.DataFrame, ou_window: int, step: int = 5) -> pd.DataFrame:
    out = df.copy()
    half_lives = pd.Series(np.nan, index=out.index)

    resid = out["residual"]

    for i in range(ou_window, len(out), step):
        window_series = resid.iloc[i - ou_window:i]
        hl = estimate_half_life(window_series)
        half_lives.iloc[i] = hl

    out["ou_half_life"] = half_lives.ffill()
    out["ou_half_life_signal"] = out["ou_half_life"].shift(1)
    return out


# =========================
# FEATURES
# =========================
def add_features(df, z_win, vol_win, beta_win, ou_window):
    out = df.copy()

    out["mean"] = out["residual"].rolling(z_win).mean()
    out["std"] = out["residual"].rolling(z_win).std()
    out["raw_z"] = (out["residual"] - out["mean"]) / out["std"]

    out["vol"] = out["residual"].rolling(vol_win).std()
    out["beta_vol"] = out["beta"].diff().rolling(beta_win).std()

    out["brent_change"] = out["brent"].diff()
    out["wti_change"] = out["wti"].diff()

    out = add_ou_half_life(out, ou_window=ou_window, step=5)

    out["zscore"] = out["raw_z"].shift(1)
    out["vol_signal"] = out["vol"].shift(1)
    out["beta_signal"] = out["beta"].shift(1)
    out["beta_vol_signal"] = out["beta_vol"].shift(1)

    return out


# =========================
# STRATEGY
# =========================
def run_strategy_legged(df, test_index, cfg, vol_th, beta_th):
    out = df.copy()

    out["active"] = (
        (out["vol_signal"] <= vol_th) &
        (out["beta_vol_signal"] <= beta_th) &
        (out["ou_half_life_signal"] >= cfg["min_half_life"]) &
        (out["ou_half_life_signal"] <= cfg["max_half_life"])
    )

    side = 0
    holding_days = 0
    sides = []
    hold_list = []

    for _, r in out.iterrows():
        z = r["zscore"]
        active = r["active"]

        if side != 0:
            holding_days += 1
        else:
            holding_days = 0

        if pd.isna(z) or pd.isna(active) or not active:
            side = 0
            holding_days = 0

        elif side == 0:
            if z > cfg["entry"]:
                side = -1
                holding_days = 0
            elif z < -cfg["entry"]:
                side = 1
                holding_days = 0

        elif side > 0:
            if z > -cfg["exit"] or holding_days >= cfg["max_holding_days"]:
                side = 0
                holding_days = 0

        elif side < 0:
            if z < cfg["exit"] or holding_days >= cfg["max_holding_days"]:
                side = 0
                holding_days = 0

        sides.append(side)
        hold_list.append(holding_days)

    out["signal_side"] = sides
    out["holding_days"] = hold_list

    out["spread_unit_target"] = np.where(
        out["vol_signal"] > 0,
        cfg["target_daily_risk_usd"] /
        (out["vol_signal"] * cfg["contract_size_bbl"]),
        0.0,
    )
    out["spread_units"] = np.floor(
        out["spread_unit_target"]).clip(0, cfg["max_spread_units"])

    out["spread_units"] = np.where(
        out["spread_units"] >= cfg["min_spread_units"],
        out["spread_units"],
        0.0,
    )

    beta_abs = out["beta_signal"].abs().fillna(1.0)
    beta_abs = beta_abs.clip(
        lower=cfg["min_beta_abs"], upper=cfg["max_beta_abs"])

    out["brent_contracts"] = out["signal_side"] * out["spread_units"]
    out["wti_contracts"] = -out["signal_side"] * \
        np.round(out["spread_units"] * beta_abs)

    out["brent_contracts_lag"] = out["brent_contracts"].shift(1).fillna(0)
    out["wti_contracts_lag"] = out["wti_contracts"].shift(1).fillna(0)

    out["brent_pnl_usd"] = (
        out["brent_contracts_lag"] *
        out["brent_change"] * cfg["contract_size_bbl"]
    )
    out["wti_pnl_usd"] = (
        out["wti_contracts_lag"] * out["wti_change"] * cfg["contract_size_bbl"]
    )
    out["gross_pnl_usd"] = out["brent_pnl_usd"] + out["wti_pnl_usd"]

    out["brent_turnover"] = (out["brent_contracts"] -
                             out["brent_contracts_lag"]).abs()
    out["wti_turnover"] = (out["wti_contracts"] -
                           out["wti_contracts_lag"]).abs()

    out["cost_usd"] = (
        out["brent_turnover"] * cfg["brent_cost_per_contract_usd"] +
        out["wti_turnover"] * cfg["wti_cost_per_contract_usd"]
    )

    out["net_pnl_usd"] = out["gross_pnl_usd"] - out["cost_usd"]
    out["returns"] = out["net_pnl_usd"] / cfg["capital_usd"]

    out["gross_notional_usd"] = (
        out["brent_contracts_lag"].abs() * out["brent"] * cfg["contract_size_bbl"] +
        out["wti_contracts_lag"].abs() * out["wti"] * cfg["contract_size_bbl"]
    )

    return out.loc[test_index].copy()


# =========================
# TRAIN EVALUATION
# =========================
def evaluate_train(train, delta, R, cfg):
    feat = kalman_filter(train, delta, R)
    feat = add_features(
        feat,
        cfg["z_win"],
        cfg["vol_win"],
        cfg["beta_win"],
        cfg["ou_window"],
    )

    vol = feat["vol_signal"].dropna()
    beta_vol = feat["beta_vol_signal"].dropna()

    if len(vol) == 0 or len(beta_vol) == 0:
        return {
            "score": np.nan,
            "details": None,
        }

    vol_th = float(vol.quantile(cfg["vol_q"]))
    beta_th = float(beta_vol.quantile(cfg["beta_q"]))

    bt = run_strategy_legged(feat, feat.index, cfg, vol_th, beta_th)
    score = sharpe_ratio(bt["returns"])

    return {
        "score": score,
        "details": {
            "delta": delta,
            "R": R,
            "vol_th": vol_th,
            "beta_th": beta_th,
        },
    }


def select_best_train_config(train, delta_grid, R_grid, cfg, n_jobs=-1):
    grid = [(d, r) for d in delta_grid for r in R_grid]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(evaluate_train)(train, d, r, cfg)
        for d, r in grid
    )

    best_score = -np.inf
    best = None

    for res in results:
        score = res["score"]
        details = res["details"]
        if pd.notna(score) and score > best_score:
            best_score = score
            best = details | {"train_sharpe": score}

    return best


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
        ou_window=40,
        entry=2.0,
        exit=0.5,
        vol_q=0.80,
        beta_q=0.80,
        min_half_life=1.0,
        max_half_life=20.0,
        max_holding_days=15,
        contract_size_bbl=1000,
        target_daily_risk_usd=1000.0,
        max_spread_units=10,
        min_spread_units=1,
        min_beta_abs=0.25,
        max_beta_abs=3.0,
        brent_cost_per_contract_usd=20.0,
        wti_cost_per_contract_usd=20.0,
        capital_usd=100_000.0,
    )

    # Smaller grid for faster systematic search
    delta_grid = [1e-6, 1e-5]
    R_grid = [1e-1, 1.0]

    splits = make_splits(df)

    results = []
    summary = []

    for i, (tr, te) in enumerate(splits, 1):
        train = df.iloc[tr]
        test = df.iloc[te]

        best = select_best_train_config(
            train=train,
            delta_grid=delta_grid,
            R_grid=R_grid,
            cfg=cfg,
            n_jobs=-1,
        )

        if best is None:
            print(f"Split {i}: no valid train configuration.")
            continue

        combined = pd.concat([train, test])
        combined = kalman_filter(combined, best["delta"], best["R"])
        combined = add_features(
            combined,
            cfg["z_win"],
            cfg["vol_win"],
            cfg["beta_win"],
            cfg["ou_window"],
        )

        test_res = run_strategy_legged(
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
        test_sharpe = sharpe_ratio(test_res["returns"])
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
            "avg_brent_contracts": float(test_res["brent_contracts"].abs().mean()),
            "avg_wti_contracts": float(test_res["wti_contracts"].abs().mean()),
            "avg_gross_notional_usd": float(test_res["gross_notional_usd"].mean()),
            "gross_pnl_usd": float(test_res["gross_pnl_usd"].sum()),
            "cost_usd": float(test_res["cost_usd"].sum()),
            "net_pnl_usd": float(test_res["net_pnl_usd"].sum()),
            "mean_half_life": float(test_res["ou_half_life_signal"].dropna().mean())
            if test_res["ou_half_life_signal"].dropna().size > 0 else np.nan,
        })

    if not results:
        raise ValueError("No results produced.")

    final = pd.concat(results).sort_index()
    final["cum_usd"] = final["net_pnl_usd"].cumsum()
    final["equity_curve"] = cfg["capital_usd"] + final["cum_usd"]

    summary_df = pd.DataFrame(summary)
    valid_sharpes = summary_df.loc[summary_df["pnl_days"]
                                   >= 10, "test_sharpe"].dropna()
    empty_split_rate = (summary_df["pnl_days"] == 0).mean()

    returns = final["returns"]
    daily_mean = returns.mean()
    daily_std = returns.std()
    ann_return = daily_mean * 252
    ann_vol = daily_std * np.sqrt(252)

    print("\n=== FINAL LEGGED OU USD RESULTS ===")
    print("Sharpe:", sharpe_ratio(returns))
    print("Net PnL USD:", final["cum_usd"].iloc[-1])
    print("MaxDD USD:", max_drawdown(final["cum_usd"]))
    print("Mean daily return:", daily_mean)
    print("Std daily return:", daily_std)
    print("Annualized return:", ann_return)
    print("Annualized volatility:", ann_vol)
    print("Gross PnL USD:", final["gross_pnl_usd"].sum())
    print("Total Cost USD:", final["cost_usd"].sum())
    print("Mean trade days per split:", summary_df["trade_days"].mean())
    print("Median trade days per split:", summary_df["trade_days"].median())
    print("Mean pnl_days per split:", summary_df["pnl_days"].mean())
    print("Median pnl_days per split:", summary_df["pnl_days"].median())
    print("Mean avg Brent contracts:",
          summary_df["avg_brent_contracts"].mean())
    print("Mean avg WTI contracts:", summary_df["avg_wti_contracts"].mean())
    print("Mean gross notional USD:",
          summary_df["avg_gross_notional_usd"].mean())
    print("Mean half-life:", summary_df["mean_half_life"].mean())
    print("Empty split rate:", empty_split_rate)

    if len(valid_sharpes) > 0:
        print("Median split Sharpe (pnl_days >= 10):", valid_sharpes.median())
        print("Mean split Sharpe (pnl_days >= 10):", valid_sharpes.mean())
    else:
        print("Median split Sharpe (pnl_days >= 10): NA")
        print("Mean split Sharpe (pnl_days >= 10): NA")

    final.to_csv(PROCESSED_PATH / "kalman_legged_ou_final_usd.csv")
    summary_df.to_csv(
        PROCESSED_PATH / "kalman_legged_ou_summary_usd.csv", index=False)


if __name__ == "__main__":
    main()
