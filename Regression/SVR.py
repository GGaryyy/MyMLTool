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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Training SVR model 
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Predict the test dataset
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1,1))

# r2_score
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
