# import sys
# import os
# print(sys.path)
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, gamma, gammaln
from numpy.random import *
from sklearn.mixture import BayesianGaussianMixture


def create_toy_data():
    mu = np.array([[5, 5], [20, 20], [50, 10]])
    sigma = np.array([[[3, 1], 
                       [1, 3]],
                       [[3, 1],
                       [1, 3]],
                       [[3, 1],
                       [1, 3]]])
    data = []
    for i in range(len(mu)):
        values = multivariate_normal(mu[i], sigma[i], 100)
        data.extend(values)
    data = np.array(data)
    return data

np.random.seed(6)
X = create_toy_data()
gmm = BayesianGaussianMixture(n_components=10, verbose=1, max_iter=1000)
lower_bounds = gmm.fit(X)[1][1]
plt.plot(lower_bounds, color='#ffff00', linestyle='solid')
plt.show()