from sklearn.preprocessing import MinMaxScaler

def rescale_data(X_train, X_test):
    """
    Rescales numerical features in the training and test datasets to a [0, 1] range using MinMax scaling.

    Args:
        X_train (DataFrame): Training data features.
        X_test (DataFrame): Test data features.

    Returns:
        tuple: A tuple containing the rescaled training and testing DataFrames.
    """

    #creating instance of MinMaxScaler
    scaler = MinMaxScaler().set_output(transform='pandas')

    #learning X_train dataframe and rescaling it
    X_train_scaled = scaler.fit_transform(X_train)

    # Ensuring column order in X_test matches that of X_train
    X_test = X_test[X_train.columns]

    #rescaling X_test dataframe
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled
