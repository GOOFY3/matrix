# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 12:31:33 2018

@author: Aquib Baig
"""

#importing packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("./Salaries.csv")

#data splitting
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,4].values

#encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding dummy variable trap
X = X[:,1:]


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


