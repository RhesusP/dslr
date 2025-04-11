import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load(path: str) -> pd.DataFrame:
    """
    Open a csv file and return a pandas DataFrame

    :param path: path to the csv file
    :return: pandas DataFrame
    """
    try:
        df = pd.read_csv(path, header=[0], index_col=0)
    except Exception as e:
        print(type(e).__name__ + ": " + str(e))
        return None
    return df


def get_args() -> tuple:
    """
    Check and return the command line arguments

    :return: (testing_path, weights_path)
    """
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py <path_to_testing_csv>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Error: File {path} does not exist.")
        sys.exit(1)
    if not os.path.isfile(path):
        print(f"Error: {path} is not a file.")
        sys.exit(1)
    print(f"Loading testing data from '{path}'...")
    if not path.endswith(".csv"):
        print(f"Error: {path} is not a csv file.")
        sys.exit(1)
    return path


def main():
    path = get_args()
    df = load(path)
    if df is None:
        print("Error: Could not load data.")
        sys.exit(1)

    g = sns.pairplot(df,
                     hue="Hogwarts House",
                     markers='o',
                     palette="husl",
                     diag_kind="kde",
                     plot_kws={'s': 5})
    g.figure.suptitle("Pair Plot of Hogwarts Houses", fontsize=16, y=1.02)
    g.figure.subplots_adjust(top=0.95, right=0.95, left=0.05, bottom=0.05)
    plt.show()


if __name__ == "__main__":
    main()

# The features that are going to be used for the logistic regression are:
# - Astronomy
# - Herbology
# - Divination
# - Ancient Runes
# - Muggle Studies
# - History of Magic
# - Transfiguration
# - Charms
