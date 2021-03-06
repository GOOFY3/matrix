import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)




y_pred = regressor.predict(6.5)


#plot for lower resolution
'''plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), 'blue')
plt.xlabel("Position levels")
plt.ylabel("Salaries")
plt.show()'''    

#plot for higher resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), 'blue')
plt.xlabel("Position levels")
plt.ylabel("Salaries")
plt.show()


