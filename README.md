# House Price Prediction

## Project Overview
This project is developed as part of a Kaggle competition and is designed to predict house prices based on various features like size, condition, number of bedrooms, etc. The model takes in data from two CSV files (`test.csv` and `train.csv`), learns how each feature influences the house prices from the `train.csv` file, and then predicts the prices for the houses in the `test.csv` file. The results are outputted to a file called `prediction file.csv`.

## File Descriptions
- **app.py**: The main script that orchestrates the data loading, preprocessing, modeling, and prediction processes. It utilizes functions from the other modules to execute the model pipeline.
- **load_data.py**: Contains the `load_data` function which loads the CSV files into pandas DataFrames, handling any specific nuances of the CSV format and missing data.
- **feature_engineer.py**: Implements the `feature_engineer` function to transform raw dataset features into formats better suited for modeling (e.g., converting categories to numerical values, etc.).
- **missing_data.py**: Provides functions to handle missing values in the dataset. It differentiates between missing data that are 'NA' by design and those that are genuinely missing.
- **one_hot.py**: Uses `OneHotEncoder` from scikit-learn to convert categorical variables into a format that can be provided to machine learning models.
- **rescaling.py**: Contains functionality to scale numerical features using MinMaxScaler, ensuring that all numeric inputs have comparable scales.
- **regression_file.py**: Defines the regression model using `LassoCV` from scikit-learn to find the optimal alpha parameter and predict house prices based on the trained model.

## Features
- Load and preprocess data from CSV files.
- Feature engineering to prepare data for modeling.
- Use of advanced regression techniques to predict house prices.
- Output predictions to a CSV file.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/omer-sade/house-price-prediction.git
