# ============================================
# SUPPORT VECTOR MACHINE (SVM) CLASSIFICATION
# Social Network Ads Prediction
# ============================================


# =========================
# IMPORTING LIBRARIES
# =========================

import pandas as pd
import numpy as np


# =========================
# DATA COLLECTION
# =========================

# Reading dataset from CSV file

dataset = pd.read_csv("C:\\Users\\kanil\\Downloads\\assignment5\\Social_Network_Ads.csv")

# Display first 5 rows

print(dataset.head())


# =========================
# DATA PREPROCESSING
# =========================

# Checking dataset information

print(dataset.info())

# Checking null values

print(dataset.isnull().sum())


# =========================
# INPUT AND OUTPUT SPLIT
# =========================

# Independent variables (Input features)

independent = dataset[[
    'Age',
    'EstimatedSalary'
]]

# Dependent variable (Target)

dependent = dataset['Purchased']

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
    test_size=0.25,
    random_state=0
)


# =========================
# FEATURE SCALING
# =========================

from sklearn.preprocessing import StandardScaler

# Creating scaler object

sc = StandardScaler()

# Scaling training data

X_train = sc.fit_transform(X_train)

# Scaling testing data

X_test = sc.transform(X_test)


# =========================
# MODEL CREATION
# =========================

from sklearn.svm import SVC

# Creating SVM classifier model

classifier = SVC(
    kernel='linear',
    random_state=0
)

# Training the model

classifier.fit(X_train, Y_train)


# =========================
# MODEL PREDICTION
# =========================

# Predicting test values

Y_pred = classifier.predict(X_test)

# Display predictions

print(Y_pred)


# =========================
# MODEL EVALUATION
# =========================

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Calculating accuracy

accuracy = accuracy_score(Y_test, Y_pred)

# Display accuracy

print("Accuracy :", accuracy)

# Confusion Matrix

cm = confusion_matrix(Y_test, Y_pred)

print("Confusion Matrix :")
print(cm)

# Classification Report

report = classification_report(Y_test, Y_pred)

print("Classification Report :")
print(report)


# =========================
# SAVING THE MODEL
# =========================

import pickle

# File name for saving model

filename = "finalized_model_svm.sav"

# Saving trained model

pickle.dump(classifier, open(filename, 'wb'))


# =========================
# LOADING SAVED MODEL
# =========================

# Loading saved model

loaded_model = pickle.load(
    open("finalized_model_svm.sav", 'rb')
)


# =========================
# TESTING WITH SAMPLE INPUT
# =========================

# Sample input values:
# Age, EstimatedSalary

sample_input = pd.DataFrame(
    [[
        35,      # Age
        50000    # Estimated Salary
    ]],
    columns=[
        'Age',
        'EstimatedSalary'
    ]
)

# Scaling sample input

sample_input = sc.transform(sample_input)

# Predicting output

result = loaded_model.predict(sample_input)

# Display prediction result

if result[0] == 1:
    print("Customer will Purchase")
else:
    print("Customer will Not Purchase")