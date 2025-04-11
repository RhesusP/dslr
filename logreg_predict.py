import pandas as pd
import numpy as np
import sys
import os


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
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py <path_to_testing_csv> <weights.csv>")
        sys.exit(1)
    testing_path = sys.argv[1]
    weights_path = sys.argv[2]
    if not os.path.exists(testing_path):
        print(f"Error: File {testing_path} does not exist.")
        sys.exit(1)
    if not os.path.isfile(testing_path):
        print(f"Error: {testing_path} is not a file.")
        sys.exit(1)
    print(f"Loading testing data from '{testing_path}'...")
    if not testing_path.endswith(".csv"):
        print(f"Error: {testing_path} is not a csv file.")
        sys.exit(1)

    if not os.path.exists(weights_path):
        print(f"Error: File {weights_path} does not exist.")
        sys.exit(1)
    if not os.path.isfile(weights_path):
        print(f"Error: {weights_path} is not a file.")
        sys.exit(1)
    if not weights_path.endswith(".csv"):
        print(f"Error: {weights_path} is not a csv file.")
        sys.exit(1)
    return testing_path, weights_path


def keep_pertinent_columns(df: pd.DataFrame) -> tuple:
    """
    Keep only the pertinent columns of the DataFrame

    :param df: pandas DataFrame
    :return: tuple of (X features and y target)
    """
    target = "Hogwarts House"
    y = df[target]
    df = df.drop(columns=[target])
    useful_cols = ["Astronomy", "Herbology", "Divination", "Ancient Runes", "Muggle Studies", "History of Magic",
                   "Transfiguration", "Charms"]
    return df[useful_cols], y


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    for var in df:
        mean = df[var].mean()
        std = df[var].std()
        df[var] = (df[var] - mean) / std
    return df


def replace_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the NaN values in the DataFrame with the mean of the column (feature)

    :param df: pandas DataFrame
    :return: pandas DataFrame with NaN values replaced
    """
    for var in df:
        if df[var].isnull().sum() > 0:
            mean = df[var].mean()
            df[var] = df[var].fillna(mean)
    return df


def preprocess(df: pd.DataFrame) -> tuple:
    test_df, _ = keep_pertinent_columns(df)
    test_df = replace_na(test_df)
    test_df = standardize(test_df)
    X_test = np.hstack((np.ones((test_df.shape[0], 1)), test_df.to_numpy()))
    return X_test


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def predict_one_vs_all(all_theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Predict the class labels for a multi-class classification problem using one-vs-all logistic regression

    Predictions are made by computing the probabilities for each class and selecting the class with the highest probability.
    :param all_theta: weights for each class
    :param X: input features
    :return: predictions (class identifiers)
    """
    probabilities = sigmoid(np.dot(X, all_theta.T))
    return np.argmax(probabilities, axis=1)


def main():
    testing_path, weights_path = get_args()
    testing_df = load(testing_path)
    if testing_df is None:
        print("Error: Could not load training data.")
        sys.exit(1)
    all_theta = None
    try:
        all_theta = np.loadtxt(weights_path, delimiter=",")
    except Exception as e:
        print("Error: Could not load weights file.")
        sys.exit(1)
    labels = {0: "Gryffindor", 1: "Hufflepuff", 2: "Ravenclaw", 3: "Slytherin"}
    X = preprocess(testing_df)
    predictions = predict_one_vs_all(all_theta, X)

    try:
        file = open("houses.csv", "w")
        file.write("Index,Hogwarts House\n")
        for i in range(len(predictions)):
            file.write(str(i) + "," + labels[predictions[i]] + "\n")
        file.close()
        print("✅ Predictions saved to houses.csv")
    except Exception as e:
        print("Error: Could not write to file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
