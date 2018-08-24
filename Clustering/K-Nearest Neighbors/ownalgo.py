import numpy as np
from math import sqrt
import matplotlib.pyplot as plt 
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

#euclidean_distance = sqrt((plot1[0] - plot2[0])**2 + plot1[1] - plot2[1)**2)

dataset = {'k':[[1,2], [2,3], [3,1]], 'r':[[6,5],[7,7],[8,6]]}

new_features = [5,7]

#for i in dataset:
#   for ii in dataset[i]:
#       [plt.scatter(ii[0], ii[1], s = 100, color = i)]
# plt.show()

def k_nearest_neighbors(dataset, predict, k =3):
    if len(dataset) >= k:
        warnings.warn("Idiot!")
    distances = []
    for i in dataset:
        for features in dataset[i]:
          #euclidean_distance = np.sqrt(np.sum((np.array(features) - npm.array(predict))**2))
          euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
          distances.append([euclidean_distance, i])
    votes = [j[1] for j in sorted(distances)[:k]]
    
    vote_result = Counter(votes).most_common(1)[0][0]
     
    return vote_result

result = k_nearest_neighbors(dataset, new_features, k =3)
print(result)      
print(distances)  