from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

def predict(X_train, X_test, y_train):
    """
     Trains a Lasso regression model using cross-validation to predict outcomes based on the provided training and testing data.

     Args:
         X_train (DataFrame): The training feature dataset.
         X_test (DataFrame): The testing feature dataset.
         y_train (Series): The target values for the training dataset.
         alphas (list, optional): List of alphas to consider for model tuning. Defaults to a predefined range.
         cv (int): The number of cross-validation folds.
         random_state (int): The seed used by the random number generator.

     Returns:
         array: Predicted values for the testing dataset.
     """
    # Create a LassoCV object -  to find the best alpha value from given values
    # The given values were found after experimenting with a wide range of values [0.001, 1000]
    lasso_cv = LassoCV(alphas=[131.235, 131.24, 131.245 ], cv=5, random_state=42)

    # Fit it to our data - find the best alpha value
    lasso_cv.fit(X_train, y_train)

    # calculating the prediction
    y_pred_lasso = lasso_cv.predict(X_test)

    return y_pred_lasso

