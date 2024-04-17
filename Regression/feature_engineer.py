import pandas as pd
""""
This file changes the dataframes in order for them to better fit the regression model. 
It has 3 functions in it.

1:  feature_engineer - The main function in this script. It transforms 2 columns manually, and calles the 
    other 2 functions. 

2: convert_data - some of the data that should be numeric was read by pandas as categorical, vice versa.
   Hence, we will change several columns from categorical type to numeric type, and 1 column from numeric type 
   to categorical type.

3: map_values: Several columns are categorical, but contains values such as "excellent", "good", "average" 
   that have explicit hierarchy between them. Therefore, we can convert them to numbeic values to increase
   the model's accuracy. For example:  "excellent" -> 5, "good" -> 4, "average" -> 3 
"""
#help function for 'feature_engineer' method.
#converts relevant 'object' type columns to 'number' type columns, vice versa
def convert_data(df):
    """
      Converts relevant columns between object and numeric types.

      Args:
          df (DataFrame): The input DataFrame.

      Returns:
          DataFrame: The DataFrame with converted column types.
      """
    # Columns that should be numeric but are read as object
    convert_to_numeric = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath'
        , 'BsmtHalfBath', 'BsmtUnfSF', 'GarageArea', 'GarageCars', 'TotalBsmtSF']

    for col in convert_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    """"
    we will want to perform one-hot encoding on 'MSSubClass' column.
    It has numerical data but operates as categorical.
    see 'data_decription' file for further explaination
    """

    df['MSSubClass'] = df['MSSubClass'].astype(str)

    return df
"""
Help function for 'feature_engineer'
In some 'object' type columns there are values such as "Ex" (Excellent), "Po" (Poor).
Lets convert the values to numers, for instance "Ex" -> 5 (best), "Po" -> 1
"""
def map_values(df):
    """
     Maps ordinal and categorical values to numerical scales.

     Args:
         df (DataFrame): The input DataFrame.

     Returns:
         DataFrame: The DataFrame with mapped values.
     """

    columns_to_map = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond'
    ]
    mapping_values1 = {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': 0
    }
    for column in columns_to_map:
        # Apply the mapping
        df[column] = df[column].map(mapping_values1)

    # replacing values in columns: BsmtExposure
    columns_to_map = [
        'BsmtExposure'
    ]
    mapping_values2 = {
        'Gd': 4,
        'Av': 3,
        'Mn': 2,
        'No': 1,
        'NA': 0
    }
    for column in columns_to_map:
        # Apply the mapping
        df[column] = df[column].map(mapping_values2)

    # replacing values in columns: BsmtFinType1, BsmtFinType2

    columns_to_map = [
        'BsmtFinType1', 'BsmtFinType2'
    ]
    mapping_values3 = {
        'GLQ': 6,
        'ALQ': 5,
        'BLQ': 4,
        'Rec': 3,
        'LwQ': 2,
        'Unf': 1,
        'NA': 0
    }
    for column in columns_to_map:
        # Apply the mapping
        df[column] = df[column].map(mapping_values3)

    # replacing values in columns: CentralAir
    columns_to_map = [
        'CentralAir'
    ]
    mapping_values4 = {
        'N': 0,
        'Y': 1
    }
    for column in columns_to_map:
        # Apply the mapping
        df[column] = df[column].map(mapping_values4)

    # replacing values in columns: Functional
    columns_to_map = [
        'Functional'
    ]
    mapping_values5 = {
        'NA': 8,
        'Typ': 8,
        'Min1': 7,
        'Min2': 6,
        'Mod': 5,
        'Maj1': 4,
        'Maj2': 3,
        'Sev': 2,
        'Sal': 1
    }
    for column in columns_to_map:
        # Apply the mapping
        df[column] = df[column].map(mapping_values5)

    # replacing values in columns: GarageFinish
    columns_to_map = [
        'GarageFinish'
    ]
    mapping_values6 = {
        'Fin': 3,
        'RFn': 2,
        'Unf': 1,
        'NA': 0
    }
    for column in columns_to_map:
        # Apply the mapping
        df[column] = df[column].map(mapping_values6)

    # replacing values in columns: PavedDrive
    columns_to_map = [
        'PavedDrive'
    ]
    mapping_values7 = {
        'Y': 3,
        'P': 2,
        'N': 1,
    }
    for column in columns_to_map:
        # Apply the mapping
        df[column] = df[column].map(mapping_values7)

    # replacing values in columns: PoolQC
    columns_to_map = [
        'PoolQC'
    ]
    mapping_values8 = {
        'Ex': 4,
        'Gd': 3,
        'TA': 2,
        'Fa': 1,
        'NA': 0
    }
    for column in columns_to_map:
        # Apply the mapping
        df[column] = df[column].map(mapping_values8)

    # replacing values in columns: Fence
    # The idea here is assuming that "good privacy" is similar to "good wood",
    # and that "minimum privacy" is similar to "minimum wood / wire"
    columns_to_map = [
        'Fence'
    ]
    mapping_values9 = {
        'GdPrv': 2,
        'GdWo': 2,
        'MnPrv': 1,
        'MnWw': 1,
        'NA': 0
    }
    for column in columns_to_map:
        # Apply the mapping
        df[column] = df[column].map(mapping_values9)

    return df


def feature_engineer(df):
    """
      Applies feature engineering to the dataset including age calculations and
      data type conversions.

      Args:
          df (DataFrame): The original DataFrame.
          current_year (int): The year to use for calculating ages.

      Returns:
          DataFrame: The modified DataFrame.
      """
    #changing values for 2 columns to make them more readable
    df['YearsOld'] = 2024 - df['YearBuilt']
    df = df.drop('YearBuilt', axis=1)

    df['YearSinceRemodel'] = 2024 - df['YearRemodAdd']
    df = df.drop('YearRemodAdd', axis=1)


    df = convert_data(df)
    df = map_values(df)
    return df