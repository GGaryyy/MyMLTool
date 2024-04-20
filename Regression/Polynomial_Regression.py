#Import Package
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the Dataset
dataset = pd.read_csv("/data_path/data_name.csv")

# Suppose the data output is at the end of columns
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splite the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Polynomial Regression model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)

# Predict the test dataset
y_pred = regressor.predict(poly_reg.transform(X_test))

# r2_score
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
