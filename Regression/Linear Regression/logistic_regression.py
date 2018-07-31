import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

# pi = 1/1+exp[-(b0 + b1*x)]

x1 = np.array([0, 0.6, 1.1, 1.5, 1.8, 2.5, 3, 3.1, 3.9, 4, 4.9, 5, 5.1])
y1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

x2 = np.array([3, 3.8, 4.4, 5.2, 5.5, 6.5, 6, 6.1, 6.9, 7, 7.9, 8, 8.1])
y2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

X = np.array([0, 0.6, 1.1, 1.5, 1.8, 2.5, 3, 3.1, 3.9, 4, 4.9, 5, 5.1, 3, 3.8, 4.4, 5.2, 5.5, 6.5, 6, 6.1, 6.9, 7, 7.9, 8, 8.1])
y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1])


plt.plot(x1, y1, 'ro', blue)
plt.plot(x2, y2, 'ro', red)


classifier = LogisticRegression()
classifier.fit(X, y)

pred = classifier.predict_proba(8)
print(pred)

plt.axis([-2, 10, -1, 2])

plt.show()
