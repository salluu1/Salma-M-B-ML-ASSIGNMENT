# ============================================
# MULTIPLE LINEAR REGRESSION
# Profit Prediction
# ============================================


# =========================
# IMPORTING LIBRARIES
# =========================

import pandas as pd
import pickle


# =========================
# DATA COLLECTION
# =========================

# Reading dataset from CSV file

dataset = pd.read_csv(
    r"C:\Users\kanil\Downloads\assignment2\50_Startups (1).csv"
)

# Display first 5 rows

print(dataset.head())


# =========================
# DATA PREPROCESSING
# =========================

# Dataset contains categorical data
# Example: State column

# Converting categorical data into numerical data
# using One Hot Encoding

dataset = pd.get_dummies(dataset, drop_first=True)

# Convert boolean columns into integers

dataset = dataset.astype(int, errors='ignore')

# Display processed dataset

print(dataset.head())

# Display column names

print(dataset.columns)


# =========================
# INPUT AND OUTPUT SPLIT
# =========================

# Independent variables (Inputs)

independent = dataset[
    [
        'R&D Spend',
        'Administration',
        'Marketing Spend',
        'State_Florida',
        'State_New York'
    ]
]

# Display input data

print(independent)


# Dependent variable (Output)

dependent = dataset[['Profit']]

# Display output data

print(dependent)


# =========================
# TRAIN TEST SPLIT
# =========================

from sklearn.model_selection import train_test_split

# Splitting data into training and testing sets

X_train, X_test, Y_train, Y_test = train_test_split(
    independent,
    dependent,
    test_size=0.30,
    random_state=0
)


# =========================
# MODEL CREATION
# =========================

from sklearn.linear_model import LinearRegression

# Creating Linear Regression model

regressor = LinearRegression()

# Training the model

regressor.fit(X_train, Y_train)


# =========================
# MODEL PARAMETERS
# =========================

# Coefficients / Weights

weight = regressor.coef_

print("Weights :")
print(weight)


# Bias / Intercept

bias = regressor.intercept_

print("Bias :")
print(bias)


# =========================
# MODEL EVALUATION
# =========================

# Predicting test results

Y_pred = regressor.predict(X_test)

# Importing evaluation metric

from sklearn.metrics import r2_score

# Calculating R2 Score

r_score = r2_score(Y_test, Y_pred)

# Display R2 Score

print("R2 Score :", r_score)


# =========================
# SAVING THE MODEL
# =========================

# File name for saving model

filename = "finalized_model_multiple_linear.sav"

# Saving trained model

pickle.dump(regressor, open(filename, 'wb'))


# =========================
# LOADING SAVED MODEL
# =========================

# Loading saved model

loaded_model = pickle.load(
    open("finalized_model_multiple_linear.sav", 'rb')
)


# =========================
# TESTING WITH SAMPLE INPUT
# =========================

# Creating sample input with proper feature names

sample_input = pd.DataFrame(
    [[
        1234,   # R&D Spend
        345,    # Administration
        346,    # Marketing Spend
        1,      # State_Florida
        0       # State_New York
    ]],
    columns=[
        'R&D Spend',
        'Administration',
        'Marketing Spend',
        'State_Florida',
        'State_New York'
    ]
)

# Predicting profit

result = loaded_model.predict(sample_input)

# Display prediction result

print("Predicted Profit :")
print(result)