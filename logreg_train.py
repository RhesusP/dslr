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


def keep_pertinent_columns(df: pd.DataFrame) -> pd.DataFrame:
    target = "Hogwarts House"
    y = df[target]
    df = df.drop(columns=[target])
    useless_cols = ["First Name", "Last Name", "Birthday", "Best Hand"]
    return df.drop(columns=useless_cols), y


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the numerical columns of the DataFrame
    :param df: pandas DataFrame
    :return: standardized pandas DataFrame
    """
    for var in df:
        mean = df[var].mean()
        std = df[var].std()
        df[var] = (df[var] - mean) / std
    return df


def replace_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the NaN values in the DataFrame with the mean of the column
    :param df: pandas DataFrame
    :return: pandas DataFrame with NaN values replaced
    """
    for var in df:
        if df[var].isnull().sum() > 0:
            mean = df[var].mean()
            df[var] = df[var].fillna(mean)
    return df


def f_model(x_matrix: np.ndarray, theta_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the model F = X.theta
    :param x_matrix: matrix of features (input data)
    :param theta_matrix: matrix of parameters (weights of the model)
    :return:
    """
    return x_matrix.dot(theta_matrix)


def cost(x_matrix: np.ndarray, y_matrix: np.ndarray, theta_matrix: np.ndarray) -> float:
    """
    Calculate the cost function J(theta) = 1/2m * sum((X * theta) - Y)^2
    This function is used to measure the error of the model thanks to the MSE (Mean Squared Error)
    :param x_matrix: matrix of features
    :param y_matrix: matrix of target values
    :param theta_matrix: matrix of parameters
    :return:
    """
    m = len(y_matrix)
    # TODO: change the cost function
    return 1 / (2 * m) * np.sum((f_model(x_matrix, theta_matrix) - y_matrix) ** 2)


def gradient(x_matrix: np.ndarray, y_matrix: np.ndarray, theta_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate a gradient
    :param x_matrix: matrix of features
    :param y_matrix: matrix of target values
    :param theta_matrix: matrix of parameters
    :return:
    """
    m = len(y_matrix)
    # TODO: add sigmoid function
    return 1 / m * x_matrix.T.dot(f_model(x_matrix, theta_matrix) - y_matrix)


def gradient_descent(x_matrix: np.ndarray, y_matrix: np.ndarray, theta_matrix: np.ndarray,
                     learning_rate: float) -> np.ndarray:
    """
    Gradient descent algorithm
    This is a minimization algorithm that aims to find the best parameters theta that minimize the cost function.
    The algorithm loops until the cost difference between two iterations is negligible.
    :param x_matrix: matrix of features
    :param y_matrix: matrix of target values
    :param theta_matrix: matrix of parameters
    :param learning_rate: the step between each iteration
    :return: the best theta that minimize the cost function
    """
    cost_history = np.empty((0, 1))
    while True:
        theta_matrix = theta_matrix - learning_rate * gradient(x_matrix, y_matrix, theta_matrix)
        # Stop the loop if the cost function converges
        if len(cost_history) > 1 and cost_history[-2] - cost_history[-1] < 0.01:
            break
    print("🔄 Number of iterations: ", iter)
    return theta_matrix


def main():
    training_path = get_args()
    df = load(training_path)
    if df is None:
        print("Failed to load the training data.")
        sys.exit(1)
    print("✅ Data loaded successfully.")
    df, y = keep_pertinent_columns(df)
    df = replace_na(df)
    df = standardize(df)
    print(df.shape)
    X = np.hstack((np.ones((df.shape[0], 1)), df.to_numpy()))
    y = y.to_numpy()

    # Convert y to 0 and 1
    y = np.where(y == "Gryffindor", 1, 0)

    print(X.shape)
    weights = np.zeros((X.shape[1]))

    theta_final = gradient_descent(X, y, weights, 0.01)
    print("Final theta: ", theta_final)


if __name__ == "__main__":
    main()
