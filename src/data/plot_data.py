import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_PATH = Path("data/processed")


def plot():
    df = pd.read_csv(PROCESSED_PATH / "clean_data.csv",
                     index_col=0, parse_dates=True)

    df[["hh_usd_mmbtu", "ttf_usd_mmbtu"]].plot(
        figsize=(12, 6), title="Henry Hub vs TTF in USD/MMBtu")
    plt.tight_layout()
    plt.show()

    df["spread"].plot(figsize=(12, 6), title="Naive TTF - Henry Hub Spread")
    plt.axhline(0, linestyle="--")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot()
