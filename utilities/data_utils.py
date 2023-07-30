import pandas as pd

def load_data(file_path):
    """
    Load data from a file into a pandas DataFrame.

    Args:
        file_path (str): The path to the data file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

def check_missing_values(data):
    """
    Check for missing values in a DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame to check for missing values.

    Returns:
        pd.Series: A Series containing the count of missing values for each column.
    """
    return data.isnull().sum()

def normalize_data(data):
    """
    Perform data normalization on a DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame to be normalized.

    Returns:
        pd.DataFrame: The normalized DataFrame.
    """
    return (data - data.min()) / (data.max() - data.min())

def encode_categorical_data(data):
    """
    Encode categorical variables in the DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing categorical variables.

    Returns:
        pd.DataFrame: The DataFrame with categorical variables encoded.
    """
    return pd.get_dummies(data, drop_first=True)

def split_train_test_data(data, test_ratio=0.2):
    """
    Split data into training and testing sets.

    Args:
        data (pd.DataFrame): The DataFrame to be split.
        test_ratio (float, optional): The proportion of data to be used as the test set.

    Returns:
        tuple: A tuple containing the training and testing DataFrames.
    """
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=42)
    return train_data, test_data