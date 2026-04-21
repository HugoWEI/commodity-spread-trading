from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_PATH = Path("data/processed")

df = pd.read_csv(
    PROCESSED_PATH / "kalman_final.csv",
    index_col=0,
    parse_dates=True,
)

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["cum"], label="Kalman Strategy")
plt.title("Brent–WTI Kalman Strategy Cumulative PnL")
plt.xlabel("Date")
plt.ylabel("PnL")
plt.legend()
plt.grid()

# save
output_path = Path("reports")
output_path.mkdir(exist_ok=True)
plt.savefig(output_path / "pnl.png", dpi=150, bbox_inches="tight")

plt.show()
