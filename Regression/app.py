from feature_engineer import feature_engineer
from load_data import load_data
from missing_data import deal_with_NA_values, fill_missing_values
from one_hot import one_hot
from rescaling import rescale_data
from regression_file import predict
import pandas as pd

if __name__ == '__main__':
    # Loading data
    try:
        train = load_data('train.csv')
        test = load_data('test.csv')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")

    # Spliting the data
    X_train = train.drop(['SalePrice', 'Id'], axis=1)
    y_train = train['SalePrice']
    X_test = test.drop('Id', axis=1)

    # Changing features and columns in both dataframes
    X_train = feature_engineer(X_train)
    X_test = feature_engineer(X_test)

    # Converting Strings 'NA' to NaN (null)
    X_train = deal_with_NA_values(X_train)
    X_test =  deal_with_NA_values(X_test)

    # Making sure no missing values are in our dataframes
    X_train = fill_missing_values(X_train)
    X_test = fill_missing_values(X_test)

    # Performing one-hot on categorical features in both dataframes
    X_train, X_test = one_hot(X_train,X_test)

    # Rescale both dataframes to have the same scale
    X_train, X_test = rescale_data(X_train, X_test)

    # Predicting the results
    y_pred = predict(X_train,X_test,y_train)

    # Saving the result to a file
    Id_lst = test['Id'].tolist()
    dict_res = {'Id': Id_lst, 'SalePrice': y_pred}

    df_res = pd.DataFrame(dict_res)

    # Saving result as a file
    output_file_path = r'prediction file.csv'
    df_res.to_csv(output_file_path, index=False)
    print(f"Results saved to '{output_file_path}'")
