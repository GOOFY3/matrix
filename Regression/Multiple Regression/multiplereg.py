# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("./50_Startups.csv")


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

#predict
y_pred = regressor.predict(X_test)

plt.plot(y_test, "ro", color="blue")
plt.plot(y_pred, "ro", color="red")

#backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values=X, axis=1)
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()   
regressor_OLS.summary()


# feature scaling(Optional)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScalar()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

