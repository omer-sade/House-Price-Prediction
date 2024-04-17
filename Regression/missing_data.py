import pandas as pd


# In some columns 'NA' is a valid value, while in other columns it means 'Nan' (null).
# Hence, we will convert some of the 'NA' values to 'Nan' (null) values
def deal_with_NA_values(df):
    # replacing all remaining 'NA' with NaN
    # Firstly, we will convert every 'NA' (String) value to be missing value

    columns_with_NA = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu'
        , 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

    all_columns = df.columns.tolist()

    # columns where 'NA' means missing value
    columns_without_NA = [col_name for col_name in all_columns if col_name not in columns_with_NA]

    # converting 'NA' in both train and test to NaN
    for col in columns_without_NA:
        df[col] = df[col].replace('NA', pd.NA)

    return df

#Adding values to 'df' so that it wont have missing values.
def fill_missing_values(df):
    # dealing with missing values - numerical data. adding the average value
    numerical_df = df.select_dtypes(include=['number'])
    numerical_df.fillna(numerical_df.mean(), inplace=True)


    # dealing with missing values - categorical data. Adding 'Others'
    categorical_df = df.select_dtypes(include=['object'])
    categorical_df.fillna('Others', inplace=True)
    df = pd.concat([numerical_df, categorical_df], axis=1)

    return df

