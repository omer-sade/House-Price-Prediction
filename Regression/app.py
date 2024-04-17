import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV




#pandas automatically referd to 'NA' as missing values, while we have 'NA' values that are not concidered
# as missing values.
missing_values = ["", " "]

train_path = 'train.csv'
train = pd.read_csv(train_path, na_values=missing_values, keep_default_na=False)

test_path = 'test.csv'
test = pd.read_csv(test_path, na_values=missing_values, keep_default_na=False)


X_train = train.drop(['SalePrice', 'Id'], axis =1)
y_train = train['SalePrice']

X_test = test.drop('Id', axis =1)

#feature engineering
X_train['YearsOld'] = 2024 - X_train['YearBuilt']
X_test['YearsOld'] = 2024 - X_test['YearBuilt']

X_train = X_train.drop('YearBuilt', axis = 1)
X_test = X_test.drop('YearBuilt', axis = 1)

X_train['YearSinceRemodel'] = 2024 - X_train['YearRemodAdd']
X_test['YearSinceRemodel'] = 2024 - X_test['YearRemodAdd']

#dropping the column 'YearRemodAdd'
X_train = X_train.drop('YearRemodAdd', axis = 1)
X_test = X_test.drop('YearRemodAdd', axis = 1)

#moving numerical values that pandas identified as categorical to numeric

convert_to_numeric = ['LotFrontage','MasVnrArea','GarageYrBlt','BsmtFinSF1','BsmtFinSF2','BsmtFullBath'
                    , 'BsmtHalfBath','BsmtUnfSF','GarageArea','GarageCars','TotalBsmtSF']

for col in convert_to_numeric:
  X_train[col] =pd.to_numeric(X_train[col], errors='coerce')
  X_test[col] =pd.to_numeric(X_test[col], errors='coerce')

  # we also want to perform onehot on 'MSSubClass' column. It has numerical data but if operates as categorical.
  # see 'data_decription' file for further explaination
X_train['MSSubClass'] = X_train['MSSubClass'].astype(str)
X_test['MSSubClass'] = X_test['MSSubClass'].astype(str)

#replacing values in columns: ExterQual,,ExterCond,BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu
#                             GarageQual, GarageCond

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
    X_train[column] = X_train[column].map(mapping_values1)
    X_test[column] = X_test[column].map(mapping_values1)

#replacing values in columns: BsmtExposure,
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
    X_train[column] = X_train[column].map(mapping_values2)
    X_test[column] = X_test[column].map(mapping_values2)

#replacing values in columns: BsmtFinType1, BsmtFinType2

columns_to_map = [
    'BsmtFinType1','BsmtFinType2'
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
    X_train[column] = X_train[column].map(mapping_values3)
    X_test[column] = X_test[column].map(mapping_values3)

#replacing values in columns: CentralAir
columns_to_map = [
    'CentralAir'
]
mapping_values4 = {
    'N':0,
    'Y':1
}
for column in columns_to_map:
    # Apply the mapping
    X_train[column] = X_train[column].map(mapping_values4)
    X_test[column] = X_test[column].map(mapping_values4)

#replacing values in columns: Functional
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
    X_train[column] = X_train[column].map(mapping_values5)
    X_test[column] = X_test[column].map(mapping_values5)

#replacing values in columns: GarageFinish
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
    X_train[column] = X_train[column].map(mapping_values6)
    X_test[column] = X_test[column].map(mapping_values6)


#replacing values in columns: PavedDrive
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
    X_train[column] = X_train[column].map(mapping_values7)
    X_test[column] = X_test[column].map(mapping_values7)

#replacing values in columns: PoolQC
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
    X_train[column] = X_train[column].map(mapping_values8)
    X_test[column] = X_test[column].map(mapping_values8)


#replacing values in columns: Fence
#The idea here is assuming that "good privacy" is similar to "good wood",
#and that "minimum privacy" is similar to "minimum wood / wire"
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
    X_train[column] = X_train[column].map(mapping_values9)
    X_test[column] = X_test[column].map(mapping_values9)


#replacing all remaining 'NA' with NaN
#Firstly, we will convert every 'NA' (String) value to be missing value

columns_with_NA = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu'
                   ,'GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']

all_columns = X_train.columns.tolist()

#columns where 'NA' means missing value
columns_without_NA = [col_name for col_name in all_columns if col_name not in columns_with_NA ]

#converting 'NA' in both train and test to NaN
for col in columns_without_NA:
  X_train[col] =X_train[col].replace('NA', pd.NA)
  X_test[col] =X_test[col].replace('NA', pd.NA)


#dealing with missing values in 'train' dataset - numerical data
numerical_train = X_train.select_dtypes(include=['number'])

numerical_train.fillna(numerical_train.mean(), inplace=True)


#dealing with missing values in 'test' dataset - numerical data

# filling the mean value found in TRAIN data set! not test
numerical_test = X_test.select_dtypes(include=['number'])

numerical_test.fillna(numerical_train.mean(), inplace=True)


#filling "others" instead of missing values in categorical columns in X_train
categorical_train = X_train.select_dtypes(include=['object'])
categorical_train.fillna('Others', inplace=True)
X_train = pd.concat([numerical_train, categorical_train], axis=1)



#filling "others" instead of missing values in categorical columns in X_test
categorical_test = X_test.select_dtypes(include=['object'])
categorical_test.fillna('Others', inplace=True)
X_test = pd.concat([numerical_test, categorical_test], axis=1)

#creating instance of OneHotEncoder
#ignore is for when encountering unkown categories during the transforming of 'test'
enc = OneHotEncoder(handle_unknown='ignore', sparse_output = False).set_output(transform='pandas')

#saving categirocal columns
categorical_cols_train = X_train.select_dtypes(include=['object'])

#learning all the relevant columns names in 'categorical train'
enc.fit(categorical_cols_train)

#transforming all the dataset, saving result as a df
one_hot_train = enc.transform(categorical_cols_train)

#dropping all the categorical columns - dont need them anymore
X_train = X_train.select_dtypes(include=['number'])

#adding the onehot encoded dataframe
X_train = pd.concat([X_train, one_hot_train], axis = 1)

#saving subset of X_test - only categorical
categorical_test = X_test.select_dtypes(include=['object'])

#saving subset of X_test - only numerical
numerical_test = X_test.select_dtypes(include=['number'])

one_hot_test = enc.transform(categorical_test)

X_test = pd.concat([numerical_test, one_hot_test], axis = 1)


scaler = MinMaxScaler().set_output(transform='pandas')

X_train = scaler.fit_transform(X_train)

#before rescaling 'X_test', lets make sure the columns are in the same order
# else the will be an error
X_test = X_test[X_train.columns]

X_test = scaler.transform(X_test)


# Create a LassoCV object -  to find the best alpha value from given values
lasso_cv = LassoCV(alphas=[131.235,131.24, 131.245, ], cv=5, random_state=42)

# Fit it to your data - find the best alpha value
lasso_cv.fit(X_train, y_train)

#create an instance of Lasso regression with the alpha value we found
lasso = Lasso(alpha=lasso_cv.alpha_)

#training the model
lasso.fit(X_train, y_train)

# calculating the prediction
y_pred_lasso = lasso.predict(X_test)


Id_lst = [x for x in range(1461,2920)]
dict_res = {'Id': Id_lst,
      'SalePrice': y_pred_lasso}

df_res = pd.DataFrame(dict_res)
df_res
df_res.to_csv(r'prediction file.csv', index=False)