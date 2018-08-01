
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("./Position_Salaries.csv")


#data splitting
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

# feature scaling(Optional)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)



#encoding
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
#avoiding dummy variable trap
X = X[:,1:]'''


'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

result = regressor.predict(6.5)
if result < 155000 or result > 165000:
    print("The Interviewer is bluffing!")
    
plt.scatter(X,y, color="red")    
plt.plot(X, regressor.predict(X), color="blue")    