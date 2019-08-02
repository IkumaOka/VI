import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt
from sklearn import cluster, preprocessing, mixture
from numpy.random import *
import matplotlib.cm as cm


def create_data(N):
    mu = np.array([[0, 0], [20, 20], [40, 10]])
    sigma = np.array([[[3, 1], 
                       [1, 3]],
                       [[3, 1],
                       [1, 3]],
                       [[3, 1],
                       [1, 3]]])
    data = []
    for i in range(len(mu)):
        values = multivariate_normal(mu[i], sigma[i], N)
        data.extend(values)
    data = np.array(data)
    return data


N = 100
X = create_data(N)
average_X = np.mean(X, axis=0)
m = np.tile(average_X, (5, 1)).T

vbgm = mixture.BayesianGaussianMixture(n_components=5, random_state=6)
vbgm=vbgm.fit(X)
labels=vbgm.predict(X)
print(labels)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=cm.get_cmap())
plt.show()