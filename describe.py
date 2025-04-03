import pandas as pd
import numpy as np
import sys

def load(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame
    """
    try:
        if not isinstance(path, str):
            raise ValueError("Path must be a string")
        if not path.endswith(".csv"):
            raise ValueError("Path must be a CSV file")
        dataset = pd.read_csv(path, index_col=0)
    except Exception as e:
        print(e)
        return None
    return dataset

def getstd(values):
    a = sorted(values)
    mean = sum(a) / len(a)
    squared_differences = [(x - mean) ** 2 for x in a]
    std = (sum(squared_differences) / len(a)) ** 0.5
    return std

def getmin(values):
    a = sorted(values)
    min = a[0]
    for value in a:
        if value < min:
            min = value
    return min

def getmax(values):
    a = sorted(values)
    max = a[0]
    for value in a:
        if value > max:
            max = value
    return max

def get50(values):
    result = sorted(values)[len(values) // 2]
    return result

def get25(values):
    result = sorted(values)[len(values) // 4]
    return result

def get75(values):
    result = sorted(values)[3 * len(values) // 4]
    return result

def getvar(values):
    a = sorted(values)
    mean = sum(a) / len(a)
    squared_differences = [(x - mean) ** 2 for x in a]
    variance = sum(squared_differences) / len(a)
    return variance

def describe(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = {}

    for col in numeric_cols:
        values = df[col].dropna()
        std = getstd(values)
        stats[col] = {
            "Count" : len(values),
            "Mean" : sum(values) / len(values),
            "Std" : std,
            "Variance" : getvar(values),
            "Min" : getmin(values),
            "25%" : get25(values),
            "50%" : get50(values),
            "75%" : get75(values),
            "Max" : getmax(values),
        }
    return pd.DataFrame(stats)


def main():
    if len(sys.argv) < 2:
        print("Usage: describe.py <path>")
        sys.exit(1)
    df = load(sys.argv[1])
    result = describe(df)
    print(result)
    # for col, values in df.items():
    return


if __name__ == "__main__":
    main()
