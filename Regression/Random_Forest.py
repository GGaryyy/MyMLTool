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

# Training Rrandom Forest model 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predict the test dataset
y_pred = regressor.predict(X_test)

# r2_score
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
