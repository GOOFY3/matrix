from scipy import stats
from matplotlib import pyplot as plt
import numpy as np

x = np.array([112, 345, 198, 305, 372, 550, 302, 420, 578])
y = np.array([1120, 1523, 2102, 2230, 2600, 3200, 3409, 3689, 4460])

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

plt.plot(x, y, 'ro', color='black')

plt.ylabel('Price')
plt.xlabel('Size of Houses')

plt.axis([0, 600, 0, 5000])

plt.plot()
plt.show()
