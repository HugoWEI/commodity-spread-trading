import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from pathlib import Path

PROCESSED_PATH = Path("data/processed")


def run_cointegration():
    df = pd.read_csv(PROCESSED_PATH / "clean_data.csv",
                     index_col=0, parse_dates=True)

    y = df["ttf_usd_mmbtu"]
    x = df["hh_usd_mmbtu"]

    # add constant for regression
    x_const = sm.add_constant(x)

    # OLS regression: TTF = alpha + beta * HH + residual
    model = sm.OLS(y, x_const).fit()

    print("\n=== OLS Regression ===")
    print(model.summary())

    # extract residual (this is your "spread")
    df["residual"] = model.resid

    # cointegration test
    score, pvalue, _ = coint(y, x)

    print("\n=== Cointegration Test ===")
    print(f"Test statistic: {score:.4f}")
    print(f"P-value: {pvalue:.4f}")

    # save residual
    df.to_csv(PROCESSED_PATH / "cointegration_data.csv")

    print("\nResidual preview:\n")
    print(df["residual"].head())


if __name__ == "__main__":
    run_cointegration()
