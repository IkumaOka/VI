import numpy as np
from numpy.random import *
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

mu = np.array([[0, 0], [2, 2], [4, 4]])
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

# mu = np.array([0, 0])
# sigma = np.array([[30, 20], [20, 50]])
 
# # 2次元正規乱数を1万個生成
# values = multivariate_normal(mu, sigma, 100)
# print(values.shape)