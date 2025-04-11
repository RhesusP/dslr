import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import accuracy_score


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


def get_args() -> str:
    """
    Check and return the command line arguments

    :return: path to the training data
    """
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <path_to_training_csv>")
        sys.exit(1)
    training_path = sys.argv[1]
    if not os.path.exists(training_path):
        print(f"Error: File {training_path} does not exist.")
        sys.exit(1)
    if not os.path.isfile(training_path):
        print(f"Error: {training_path} is not a file.")
        sys.exit(1)
    print(f"Loading training data from '{training_path}'...")
    if not training_path.endswith(".csv"):
        print(f"Error: {training_path} is not a csv file.")
        sys.exit(1)
    return training_path


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


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def predict(X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """
    Compute the hypothesis function for logistic regression

    :param X: input features
    :param thetas: weights
    :return: predictions
    """
    return sigmoid(np.dot(X, thetas))


def compute_cost(X: np.ndarray, y: np.ndarray, thetas: np.ndarray) -> float:
    """
    Compute the cost function for logistic regression using the cross-entropy loss
    :param X: input features
    :param y: target
    :param thetas: weights
    :return: cost
    """
    m = len(y)
    h = predict(X, thetas)
    return -1 / m * (np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h)))


def fit(X: np.ndarray, y: np.ndarray, thetas: np.ndarray, learning_rate: float, nb_iters: int) -> np.ndarray:
    """
    Fit the logistic regression model to the training data using gradient descent

    :param X: input features
    :param y: target
    :param thetas: weights
    :param learning_rate: learning rate
    :param nb_iters: number of iterations
    :return:
    """
    m = len(y)
    cost_history = np.zeros(nb_iters)
    for i in range(nb_iters):
        h = predict(X, thetas)
        thetas -= learning_rate / m * np.dot(X.T, (h - y))
        cost_history[i] = compute_cost(X, y, thetas)
    return cost_history


def split_training_data(df: pd.DataFrame) -> tuple:
    """
    Split the DataFrame into training and validation sets
    :param df: pandas DataFrame
    :return: tuple of (training set, validation set)
    """
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    return train_df, val_df


def preprocess(df: pd.DataFrame, labels: dict = None) -> tuple:
    """
    Preprocess the DataFrame by splitting it into training and validation sets,
    keeping only the pertinent columns, replacing NaN values, and standardizing the features

    :param df: raw pandas DataFrame
    :param labels: dictionary of labels
    :return: tuple of (X training, y training, X validation, y validation)
    """
    train_df, val_df = split_training_data(df)

    train_df, y_train = keep_pertinent_columns(train_df)
    train_df = replace_na(train_df)
    train_df = standardize(train_df)
    X_train = np.hstack((np.ones((train_df.shape[0], 1)), train_df.to_numpy()))
    y_train = y_train.to_numpy()
    y_train = np.array([labels[label] for label in y_train])

    val_df, y_val = keep_pertinent_columns(val_df)
    val_df = replace_na(val_df)
    val_df = standardize(val_df)
    X_val = np.hstack((np.ones((val_df.shape[0], 1)), val_df.to_numpy()))
    y_val = y_val.to_numpy()
    y_val = np.array([labels[label] for label in y_val])

    return X_train, y_train, X_val, y_val


def train_one_vs_all(X: np.ndarray, y: np.ndarray, nb_labels: int,
                     learning_rate: float, nb_iters: int) -> tuple:
    """
    Train a one-vs-all logistic regression model for multi-class classification

    Here, we train one logistic regression model for each class (Hogwarts house).
    :param X: input features
    :param y: target
    :param nb_labels: number of Hogwarts houses
    :param learning_rate: learning rate
    :param nb_iters: number of iterations
    :return: tuple of (all_theta, cost_histories)
    """
    all_theta = np.zeros((nb_labels, X.shape[1]))
    cost_histories = dict()
    for c in range(nb_labels):
        y_binary = np.where(y == c, 1, 0)
        theta = np.zeros(X.shape[1])
        cost_history = fit(X, y_binary, theta, learning_rate, nb_iters)
        all_theta[c] = theta
        cost_histories[c] = cost_history
        print(f"Entraînement terminé pour la classe {c} (coût final = {cost_history[-1]:.4f})")
    return all_theta, cost_histories


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
    training_path = get_args()
    df = load(training_path)
    if df is None:
        print("Failed to load the training data.")
        sys.exit(1)
    print("✅ Data loaded successfully.")

    labels = {"Gryffindor": 0, "Hufflepuff": 1, "Ravenclaw": 2, "Slytherin": 3}
    X_train, y_train, X_val, y_val = preprocess(df, labels)

    all_theta, cost_histories = train_one_vs_all(X_train, y_train, len(labels), 0.01, 2000)
    pred_train = predict_one_vs_all(all_theta, X_train)
    pred_val = predict_one_vs_all(all_theta, X_val)

    acc_train = accuracy_score(y_train, pred_train)
    acc_val = accuracy_score(y_val, pred_val)
    print(f"Training Accuracy: {acc_train * 100:.2f}%")
    print(f"Validation Accuracy: {acc_val * 100:.2f}%")

    # Save weights in a csv file
    try:
        np.savetxt(".weights.csv", all_theta, delimiter=",")
    except Exception as e:
        print(f"Error: Failed to save the weights {e}")
        sys.exit(1)

    plt.figure(figsize=(10, 6))
    for c in range(len(labels)):
        plt.plot(cost_histories[c], label=f"Class {c}")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function Evolution for each Classifier")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
