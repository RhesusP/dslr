import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from describe import describe, load

def second_plot(df: pd.DataFrame):
    ravenclaw = df[df["Hogwarts House"].str.lower() == "ravenclaw"]
    gryffindor = df[df["Hogwarts House"].str.lower() == "gryffindor"]
    slytherin = df[df["Hogwarts House"].str.lower() == "slytherin"]
    hufflepuff = df[df["Hogwarts House"].str.lower() == "hufflepuff"]

    fig, ax = plt.subplots()
    ax.hist(hufflepuff["Care of Magical Creatures"], bins=50, color="yellow", alpha=0.5, label="hufflepuff")
    ax.hist(ravenclaw["Care of Magical Creatures"], bins=50, color="mediumblue", alpha=0.5, label="ravenclaw")
    ax.hist(gryffindor["Care of Magical Creatures"], bins=50, color="red", alpha=0.5, label="gryffindor")
    ax.hist(slytherin["Care of Magical Creatures"], bins=50, color="green", alpha=0.5, label="slytherin")

    ax.set_title("Care of Magical Creatures")
    ax.grid(False)
    ax.legend(loc="upper right")
    plt.show()


def first_plot(df: pd.DataFrame):
    hufflepuff = df[df["Hogwarts House"].str.lower() == "hufflepuff"]
    ravenclaw = df[df["Hogwarts House"].str.lower() == "ravenclaw"]
    gryffindor = df[df["Hogwarts House"].str.lower() == "gryffindor"]
    slytherin = df[df["Hogwarts House"].str.lower() == "slytherin"]

    numeric_df = df.select_dtypes(include=['number'])
    num_cols = numeric_df.columns.tolist()

    cols = 7
    rows = math.ceil(len(num_cols) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(50, 15))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]

        ax.hist(hufflepuff[col].dropna(), bins=20, color="yellow", alpha=0.5, label="hufflepuff")
        ax.hist(ravenclaw[col].dropna(), bins=20, color="mediumblue", alpha=0.5, label="ravenclaw")
        ax.hist(gryffindor[col].dropna(), bins=20, color="red", alpha=0.5, label="gryffindor")
        ax.hist(slytherin[col].dropna(), bins=20, color="green", alpha=0.5, label="slytherin")

        ax.set_title(col)
        ax.grid(False)
        ax.legend(loc='upper right', fontsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=10.0, w_pad=6.0, h_pad=4.0)
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: describe.py <path>")
        sys.exit(1)
    df = load(sys.argv[1])
    if df is None:
        print("No data found")
        sys.exit(1)

    first_plot(df)
    second_plot(df)

    return



if __name__ == "__main__":
    main()
