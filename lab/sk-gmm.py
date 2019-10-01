import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, gamma, gammaln
from numpy.random import *
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture_stochastic import BayesianGaussianMixtureStochastic
from sklearn.mixture_sgd import BayesianGaussianMixtureSgd


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
# normal
# gmm = BayesianGaussianMixture(n_components=10, verbose=1, max_iter=1000)
# lower_bounds = gmm.fit(X)[1][1]
# #ffff00は黄色
# plt.plot(lower_bounds, color='#ffff00', linestyle='solid')
# plt.show()

# stochastic
# stochastic_gmm = BayesianGaussianMixtureStochastic(n_components=10, verbose=1, max_iter=1000)
# stochastic_lower_bounds = stochastic_gmm.fit(X)[1][1]
# plt.plot(stochastic_lower_bounds, color='#2971e5', linestyle='solid')
# plt.show()

# sgd
sgd_gmm = BayesianGaussianMixtureSgd(n_components=10, verbose=1, max_iter=1000)
sgd_lower_bounds = sgd_gmm.fit(X)[1][1]
plt.plot(sgd_lower_bounds, color='#2971e5', linestyle='solid')
plt.show()
