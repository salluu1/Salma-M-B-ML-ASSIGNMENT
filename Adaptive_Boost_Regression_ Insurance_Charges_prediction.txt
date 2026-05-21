# ============================================
# Adaptive Boost Regression
# Insurance Charges Prediction
# ============================================


# =========================
# IMPORTING LIBRARIES
# =========================

import pandas as pd
import numpy as np
import pickle


# =========================
# DATA COLLECTION
# =========================

# Reading dataset from CSV file

dataset = pd.read_csv(
    r"C:\Users\kanil\Downloads\assignment3\insurance_pre (1).csv"
)

# Display first 5 rows
print(dataset.head())


# =========================
# DATA PREPROCESSING
# =========================

# Checking dataset information
print(dataset.info())

# Checking null values
print(dataset.isnull().sum())


# ============================================
# ENCODING CATEGORICAL DATA
# ============================================

# Converting categorical columns into numerical format
# using One Hot Encoding

dataset = pd.get_dummies(dataset, drop_first=True)

# Convert boolean columns into integers (0 and 1)

dataset = dataset.astype(int, errors='ignore')

# Display processed dataset
print(dataset.head())


# =========================
# INPUT AND OUTPUT SPLIT
# =========================

# Independent variables (Input features)

independent = dataset.drop("charges", axis=1)

# Dependent variable (Target)

dependent = dataset["charges"]

# Display inputs
print(independent)

# Display outputs
print(dependent)


# =========================
# TRAIN TEST SPLIT
# =========================

from sklearn.model_selection import train_test_split

# Splitting dataset into training and testing data

X_train, X_test, Y_train, Y_test = train_test_split(
    independent,
    dependent,
    test_size=0.30,
    random_state=0
)


# =========================
# MODEL CREATION
# =========================

from sklearn.ensemble import AdaBoostRegressor

# Creating AdaBoost Regression model

regressor = AdaBoostRegressor(
    n_estimators=100,
    random_state=0
)

# Training the model

regressor.fit(X_train, Y_train)


# =========================
# MODEL PREDICTION
# =========================

# Predicting test values

Y_pred = regressor.predict(X_test)

# Display predictions

print(Y_pred)


# =========================
# MODEL EVALUATION
# =========================

from sklearn.metrics import r2_score

# Calculating R2 Score

r_score = r2_score(Y_test, Y_pred)

# Display model accuracy

print("R2 Score :", r_score)


# =========================
# SAVING THE MODEL
# =========================

# File name for saving model

filename = "finalized_model_adaboost.sav"

# Saving trained model

pickle.dump(regressor, open(filename, 'wb'))


# =========================
# LOADING SAVED MODEL
# =========================

# Loading saved model

loaded_model = pickle.load(
    open("finalized_model_adaboost.sav", 'rb')
)


# =========================
# TESTING WITH SAMPLE INPUT
# =========================

# IMPORTANT:
# Model expects only 5 features:
# age, bmi, children, sex_male, smoker_yes

sample_input = pd.DataFrame(
    [[25, 28.5, 2, 1, 0]],
    columns=[
        'age',
        'bmi',
        'children',
        'sex_male',
        'smoker_yes'
    ]
)

# Predicting insurance charges

result = loaded_model.predict(sample_input)

# Display prediction result

print("Predicted Insurance Charges :", result)