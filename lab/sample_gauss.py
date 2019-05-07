import numpy as np
import sys

def create_data(N, K):
    X, mu_star, sigma_star = [], [], []
    for i in range(K):
        loc = (np.random.rand() - 0.5) * 10.0 # range: -5.0 - 5.0
        scale = np.random.rand() * 3.0 # range: 0.0 - 3.0
        X = np.append(X, np.random.normal(loc = loc, scale = scale, size = int(N / K)))
        # print(X.shape)
        mu_star = np.append(mu_star, loc)
        sigma_star = np.append(sigma_star, scale)
    return (X, mu_star, sigma_star)


def gaussian(mu, sigma):
    def f(x):
        return np.exp(-0.5 * (x - mu) ** 2 / sigma) / np.sqrt(2 * np.pi * sigma)
    return f



K = 2
# initialize gmm parameter
pi = np.random.rand(K)
mu = np.random.randn(K)
sigma = np.abs(np.random.randn(K))
N = 1000 * K
X, mu_star, sigma_star = create_data(N, K)


gf = gaussian(mu, sigma)
# l = np.zeros((X.size, pi.size))
# for (i, x) in enumerate(X):
#     l[i, :] = pi * gf(x)

print(X[0] - sigma)





