import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_PATH = Path("data/processed")

df = pd.read_csv(PROCESSED_PATH / "signal_data.csv",
                 index_col=0, parse_dates=True)

df["zscore"].plot(figsize=(12, 6), title="Z-score of Residual")
plt.axhline(2, linestyle="--", color="red")
plt.axhline(-2, linestyle="--", color="red")
plt.axhline(0, linestyle="--")
plt.show()
