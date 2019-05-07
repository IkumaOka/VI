import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.special import digamma, gamma

def create_data(N, K):
    X = []
    for i in range(K):
        loc = (np.random.rand() - 0.5) * 10.0 # range: -5.0 - 5.0
        scale = np.random.rand() * 3.0 # range: 0.0 - 3.0
        X = np.append(X, np.random.normal(loc = loc, scale = scale, size = int(N / K)))
    return X

def gaussian(mu, sigma):
    def f(x):
        return np.exp(-0.5 * (x - mu) ** 2 / sigma) / np.sqrt(2 * np.pi * sigma)
    return f


# 対数尤度を計算
def calc_log_likelihood(X, pi, gf):
    l = np.zeros((X.size, pi.size))
    for (i, x) in enumerate(X):
       l[i, :] = pi * gf(x)
    # Xのi番目について, 式6.5を用いて各π_jのsumを求める
    p_i = l.sum(axis = 1).reshape(-1, 1)
    # print(p_i.shape)
    log_p_i = np.log(p_i)
    log_p = log_p_i.sum(axis=0)
    # xのsumを求める
    p = p_i.sum(axis=0)
    return log_p


K = 2
N = 1000 * K
D = 2
alpha0 = 1.0
beta0 = 1.0
nu0 = 2.0
m0 = np.zeros(2)
W0 = np.eye(2)
X = create_data(N, K)
W = 1.0/np.var(X, axis=0)
rnd = np.random.RandomState(seed=None)
alpha = (alpha0 + N / K) * np.ones(K)
beta = (beta0 + N / K) * np.ones(K)
nu = (nu0 + N / K) * np.ones(K)
m = X[rnd.randint(low=0, high=N, size=K)]

# 負担率を計算
# 変分ベイズではηを計算する式
def estimate_posterior_likelihood(X, D, W):
    n = len(X)
    tpi = np.exp( digamma(alpha) - digamma(alpha.sum()))
    arg_digamma = np.reshape(nu, (K, 1)) - np.reshape(np.arange(0, D, 1), (1, D))
    tlam = np.exp( digamma(arg_digamma/2).sum(axis=1) + D * np.log(2) + np.log(W))
    diff = []
    for i in range(len(X)):
        a = X[i] - m
        diff.append(a)
    diff = np.array(diff)
    diff_2 = diff ** 2
    diff_2_w = diff_2 * W
    exponent = D / beta + nu * diff_2_w
    exponent_subtracted = exponent - np.reshape(exponent.min(axis=1), (2000, 1))
    rho = tpi * np.sqrt(tlam) * np.exp(-0.5 * exponent_subtracted)
    r = rho/np.reshape(rho.sum(axis=1), (2000, 1))

    return r, tpi, tlam

def estimate_gmm_parameter(X, r):
    n = len(X)
    n_samples_in_component = r.sum(axis=0)
    # print(n_samples_in_component)
    barx = r.T @ X / np.reshape(n_samples_in_component, (K, 1))
    # print(barx)
    diff = []
    for i in range(len(X)):
        a = X[i] - barx
        diff.append(a)
    diff = np.array(diff)
    # print(diff.shape)
    S = np.einsum("nki,nkj->kij", np.einsum("nk,nki->nki", r, diff), diff) / np.reshape(n_samples_in_component, (K, 1, 1))
    # print(S.shape)

    alpha = alpha0 + n_samples_in_component
    beta = beta0 + n_samples_in_component
    nu = nu0 + n_samples_in_component
    m = (m0 * beta0 + barx * np.reshape(n_samples_in_component, (K, 1)))/np.reshape(beta, (K, 1))
    diff2 = barx - m0
    Winv = np.reshape(np.linalg.inv( W0 ), (1, D, D)) + \
        S * np.reshape(n_samples_in_component, (K, 1, 1)) + \
        np.reshape( beta0 * n_samples_in_component / (beta0 + n_samples_in_component), (K, 1, 1)) * np.einsum("ki,kj->kij",diff2,diff2)
    # print(Winv)
    W = np.linalg.inv(Winv)

    return alpha, beta, nu, m, W



r, tpi, tlam = estimate_posterior_likelihood(X, D, W)
print(W)
# print(tpi)
# print(tlam)
alpha, beta, nu, m, W = estimate_gmm_parameter(X, r)
print(W.shape)
print(W)
# for iter in range(1000):
#     r, tpi, tlam = estimate_posterior_likelihood(X, D, W)
#     alpha, beta, nu, m, W = estimate_gmm_parameter(X, r)











# y = []
# for iter in range(1000):
#     gf = gaussian(mu, sigma)
#     gamma = estimate_posterior_likelihood(X, pi, gf)
#     hard_gamma = hard_cluster(gamma)
#     # print("gamma:\n", gamma)
#     # print(gamma.shape)
#     mu, sigma, pi = estimate_gmm_parameter(X, gamma)
#     gf = gaussian(mu, sigma)
#     log_likelihood = calc_log_likelihood(X, pi, gf)
#     print("log_likelihood: ", log_likelihood)
#     y.append(log_likelihood[0])

# print("gamma:\n", gamma)
# np_y = np.array(y)
# x = np.linspace(0, 10, 1000)
# plt.plot(x, np_y)
# plt.show()