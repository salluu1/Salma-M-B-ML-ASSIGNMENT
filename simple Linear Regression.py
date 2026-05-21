# ============================================
# SIMPLE LINEAR REGRESSION
# Salary Prediction
# ============================================


# =========================
# IMPORT REQUIRED LIBRARIES
# =========================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle


# =========================
# LOAD DATASET
# =========================

# Reading dataset from CSV file

dataset = pd.read_csv(
    r"C:\Users\kanil\Downloads\assginment 1\Salary_Data.csv"
)

# Display first 5 rows

print(dataset.head())


# =========================
# INPUT AND OUTPUT SPLIT
# =========================

# Independent variable (Input)

X = dataset[['YearsExperience']]

# Dependent variable (Output)

Y = dataset[['Salary']]

# Display input data

print("\nInput (X):")
print(X.head())

# Display output data

print("\nOutput (Y):")
print(Y.head())


# =========================
# TRAIN TEST SPLIT
# =========================

# Splitting dataset into training and testing data
# 80% training and 20% testing

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=0
)

# Display dataset sizes

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)


# =========================
# MODEL CREATION
# =========================

# Creating Linear Regression model

model = LinearRegression()

# Training the model

model.fit(X_train, Y_train)


# =========================
# MODEL PARAMETERS
# =========================

# Weight / Slope

weight = model.coef_

print("\nModel Weight (Slope):")
print(weight)

# Bias / Intercept

bias = model.intercept_

print("\nModel Bias (Intercept):")
print(bias)


# =========================
# MODEL PREDICTION
# =========================

# Predicting test values

Y_pred = model.predict(X_test)

# Display predictions

print("\nPredicted Values:")
print(Y_pred)


# =========================
# MODEL EVALUATION
# =========================

# Calculating R2 Score

r_score = r2_score(Y_test, Y_pred)

# Display model accuracy

print("\nR2 Score (Model Accuracy):", r_score)


# =========================
# SAVING THE MODEL
# =========================

# File name for saving model

filename = "finalized_model_linear.sav"

# Saving trained model

pickle.dump(model, open(filename, 'wb'))

print("\nModel saved successfully!")


# =========================
# LOADING SAVED MODEL
# =========================

# Loading saved model

loaded_model = pickle.load(
    open(filename, 'rb')
)


# =========================
# TESTING WITH SAMPLE INPUT
# =========================

# Creating sample input using DataFrame
# to avoid feature name warning

sample_input = pd.DataFrame(
    [[13]],
    columns=['YearsExperience']
)

# Predicting salary

result = loaded_model.predict(sample_input)

# Display prediction result

print("\nPredicted Salary for 13 years experience:")
print(result)