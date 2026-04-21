import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_PATH = Path("data/processed")

df = pd.read_csv(PROCESSED_PATH / "cointegration_data.csv",
                 index_col=0, parse_dates=True)

df["residual"].plot(figsize=(12, 6), title="Cointegration Residual")
plt.axhline(0, linestyle="--")
plt.show()
