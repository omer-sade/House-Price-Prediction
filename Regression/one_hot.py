from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def one_hot(X_train, X_test):
    """
     Applies one-hot encoding to the categorical variables of both training and testing datasets.

     Args:
         X_train (DataFrame): Training dataset.
         X_test (DataFrame): Testing dataset.

     Returns:
         tuple: A tuple containing the transformed training and testing datasets.
     """

    # Creating instance of OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')

    # Identify categorical columns
    categorical_cols_train = X_train.select_dtypes(include=['object'])

    # Fit encoder to the categorical columns of training data
    enc.fit(categorical_cols_train)

    # Transforming the categorical columns in X_train, saving result as a df
    one_hot_train = enc.transform(categorical_cols_train)

    # Saving all the numeric columns of X_train
    X_train_numeric = X_train.select_dtypes(include=['number'])

    # Adding the one-hot encoded dataframe to the numeric dataframe
    X_train = pd.concat([X_train_numeric, one_hot_train], axis=1)

    # Saving subset of X_test - only categorical
    categorical_test = X_test.select_dtypes(include=['object'])

    # Saving subset of X_test - only numerical
    numerical_test = X_test.select_dtypes(include=['number'])

    # Transforming all the categorical columns in X_test
    one_hot_test = enc.transform(categorical_test)

    # Adding the one-hot encoded dataframe to the numeric dataframe
    X_test = pd.concat([numerical_test, one_hot_test], axis=1)

    return X_train, X_test

