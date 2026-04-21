from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_PATH = Path("data/processed")
REPORT_PATH = Path("reports")
REPORT_PATH.mkdir(exist_ok=True)

CAPITAL_USD = 100_000.0

df = pd.read_csv(
    PROCESSED_PATH / "kalman_final_usd.csv",
    index_col=0,
    parse_dates=True,
)

df["returns"] = df["net_pnl_usd"] / CAPITAL_USD
df["mean_20"] = df["returns"].rolling(20).mean()
df["mean_60"] = df["returns"].rolling(60).mean()
df["mean_120"] = df["returns"].rolling(120).mean()

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].plot(df.index, df["returns"], label="Daily return", alpha=0.6)
axes[0].axhline(0, linestyle="--")
axes[0].set_title("Daily Returns")
axes[0].set_ylabel("Return")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(df.index, df["mean_20"], label="20-day mean")
axes[1].plot(df.index, df["mean_60"], label="60-day mean")
axes[1].plot(df.index, df["mean_120"], label="120-day mean")
axes[1].axhline(0, linestyle="--")
axes[1].set_title("Rolling Mean Returns")
axes[1].set_ylabel("Return")
axes[1].set_xlabel("Date")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(REPORT_PATH / "returns.png", dpi=150, bbox_inches="tight")
plt.show()
