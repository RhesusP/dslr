import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from describe import load
import seaborn as sns

attributes = [
    "Arithmancy",
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying"
]

def splited_plot(df: pd.DataFrame, attributes: tuple[str, str]):
    attribute_x, attribute_y = attributes
    plt.title(f"{attribute_x} vs {attribute_y}")
    plt.scatter(
        df[attribute_x], df[attribute_y], color='green',
        alpha=0.35, edgecolors='none', label='House'
        )
    plt.legend()
    plt.xlabel(attribute_x)
    plt.ylabel(attribute_y)
    plt.show()

def heatmap(df: pd.DataFrame):
    house_mapping = {"Gryffindor": 0,
                     "Hufflepuff": 1,
                     "Slytherin": 2,
                     "Ravenclaw": 3}
    df["house_numeric"] = df["Hogwarts House"].map(house_mapping)
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 12))
    sns.heatmap(corr_matrix,
                annot=False,
                cmap="afmhot",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=.5
                )
    plt.tight_layout(pad=4.0, w_pad=2.0, h_pad=2.0)
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: describe.py <path>")
        sys.exit(1)
    df = load(sys.argv[1])
    if df is None:
        print("No data found")
        sys.exit(1)

    splited_plot(df, ["Astronomy", "Defense Against the Dark Arts"])
    heatmap(df)

    return



if __name__ == "__main__":
    main()
