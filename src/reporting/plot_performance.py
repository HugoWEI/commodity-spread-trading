from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_PATH = Path("data/processed")
REPORT_PATH = Path("reports")
REPORT_PATH.mkdir(exist_ok=True)


df = pd.read_csv(
    PROCESSED_PATH / "kalman_final_usd.csv",
    index_col=0,
    parse_dates=True,
)

# cumulative pnl
df["cum_pnl"] = df["net_pnl_usd"].cumsum()

# cumulative cost
df["cum_cost"] = df["cost_usd"].cumsum()

# cumulative gross pnl
df["cum_gross"] = df["gross_pnl_usd"].cumsum()

# drawdown
running_max = df["cum_pnl"].cummax()
df["drawdown"] = df["cum_pnl"] - running_max

# =========================
# PLOT
# =========================
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- Top: PnL lines ---
axes[0].plot(df.index, df["cum_pnl"], label="Net PnL", linewidth=2)
axes[0].plot(df.index, df["cum_gross"], label="Gross PnL", linestyle="--")
axes[0].plot(df.index, df["cum_cost"], label="Costs", linestyle=":")

axes[0].set_title("Brent–WTI Kalman Strategy Performance (USD)")
axes[0].set_ylabel("USD")
axes[0].legend()
axes[0].grid(True)

# --- Bottom: Drawdown ---
axes[1].plot(df.index, df["drawdown"], color="red", label="Drawdown")
axes[1].set_title("Drawdown")
axes[1].set_ylabel("USD")
axes[1].set_xlabel("Date")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

# save
output_file = REPORT_PATH / "performance.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")

plt.show()
