import numpy as np
from numpy.random import *
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

mu = np.array([[0, 0], [20, 20], [40, 40]])
sigma = np.array([[[3, 2], 
                   [2, 5]],
                   [[4, 1],
                   [1, 7]],
                   [[5, 3],
                   [3, 10]]])
data = []
for i in range(len(mu)):
    values = multivariate_normal(mu[i], sigma[i], 1000)
    data.extend(values)
data = np.array(data)
print(data.shape)

plt.scatter(data[:, 0], data[:, 1])
plt.show()