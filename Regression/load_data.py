import pandas as pd

def load_data(path):
    """
       Loads data from a specified CSV file while handling custom defined missing values.

       Args:
           path (str): Path to the CSV file.
           na_values (list, optional): List of strings to recognize as NA/NaN. By default, it is ["", " "].
           **kwargs: Additional keyword arguments to be passed to pd.read_csv().

       Returns:
           DataFrame: A pandas DataFrame loaded from the specified file.
    """

    #pandas automatically referred to 'NA' as missing values, while we have 'NA' values that are important,
    # so we don't want want them to be treated as missing values.
    missing_values = ["", " "]
    df = pd.read_csv(path, na_values=missing_values, keep_default_na=False)
    return df