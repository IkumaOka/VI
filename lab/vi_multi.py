import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.special import digamma, gamma, loggamma
from scipy.misc import logsumexp
import math
import numpy.linalg as LA


def create_data(N, K):
    x1 = np.random.normal(size=(N, K))
    x1 += np.array([-5, -5])
    x2 = np.random.normal(size=(N, K))
    x2 += np.array([5, -5])
    x3 = np.random.normal(size=(N, K))
    x3 += np.array([0, 5])
    return np.vstack((x1, x2, x3))


def gaussian(X, W, m, nu, dim, beta):
    gauss = []
    for i in range(len(X)):
        a = np.dot((X[i] - m), W)
        b = np.exp(
                -0.5 * dim / beta
                - 0.5 * nu * np.dot(a, (X[i] - m))
            )
        gauss.append(b)
    gauss = np.array(gauss)
    return gauss


def calc_r(X, W, m, nu, dim, beta, u_dir_param):
    gauss = gaussian(X, W, m, nu, dim, beta)
    # PRML式10.65
    log_lambda = digamma((nu + 1 - np.arange(1, dim+1)[:, None]) / 2).sum(axis=0) + dim*np.log(2) + LA.det(W)
    # PRML式10.66
    log_pi = np.exp(digamma(u_dir_param)) - digamma(u_dir_param.sum(axis=0))
    Lambda = np.exp(log_lambda)
    pi = np.exp(log_pi)
    r = pi * np.sqrt(Lambda) * np.tile(gauss, (2, 1)).T
    r = r / np.tile(r.sum(axis=1), (2, 1)).T
    return r


# Mステップ
# def estimate_gmm_parameter(X, r, u_psi, u_beta, u_kappa, u_xi, u_dir_param):
#     # 負担率の総和
#     n_j = r.sum(axis=0)
#     # 負担率による観測値の重み付き平均
#     barx_j = np.dot(X.T, r) / n_j
#     s = np.tile(X, (2, 1)) - barx_j.reshape((2, 1))
#     s_ = s**2
#     # 観測値の重み付き分散
#     s_j = (s_.T * r).sum(axis=0) / n_j

#     # q(π)のパラメータを更新(式7.44)
#     u_dir_param = dir_param + n_j

#     #q(μ,λ)のパラメータを更新(式7.52)
#     u_psi = ( n_j * barx_j + beta * psi ) / ( n_j + beta )
#     u_beta = n_j + beta
#     u_kappa = n_j / 2 + kappa
#     u_xi = ( n_j * s_j + (n_j * beta * ((barx_j - psi) ** 2 / (n_j + beta)))) / 2 + xi
#     return u_psi, u_beta, u_kappa, u_xi, u_dir_param


# def calc_log_likelihood(X, u_psi, u_beta, u_kappa, u_xi, r):
#     log_likelihood = 0.0
#     diff = np.tile(X, (2, 1)) - psi.reshape((2, 1))
#     log_u = np.log(u_beta * diff.T**2 / (2 * (1 + u_beta)) + u_xi)
#     elm1 = np.log(r)
#     elm2 = np.log(u_beta / (2 * math.pi * (1 + u_beta))) / 2
#     elm3 = u_kappa * np.log(u_xi)
#     elm4 = (u_kappa + 0.5) * log_u
#     elm5 = loggamma(u_kappa + 0.5) - loggamma(u_kappa)
#     s = elm1 + elm2 + elm3 - elm4 + elm5
#     for i in range(len(s)):
#         log_likelihood += logsumexp(s[i])
#     print(log_likelihood)
#     return log_likelihood


np.random.seed(20)
K = 2 # クラス数 
N = 1000 # データ数
# π,μ,σの値を初期化
pi = np.random.rand(K)
mu = np.random.randn(K)
sigma = np.abs(np.random.randn(K))
lamda = 1.0 / sigma
# データは2次元 × 3000
X = create_data(N, K)
# ウィシャート分布のパラメータを定義
dim = 2
nu0 = dim
m0 = [0.0, 0.0]
beta0 = 1.0
nu = nu0
m = [1.0, 1.0] # mの初期値（てきとう）
beta = beta0
W = np.array([[10, 1],
              [1, 10]
            ])

# print(digamma((nu + 1 - np.arange(1, dim+1)[:, None]) / 2).sum(axis=0) + dim*np.log(2) + LA.det(W))

# ディリクレ分布のパラメータを定義
dir_param = np.array([10.0, 8.0]) # [1.0, 0.1]だとdigammaに代入した時にマイナスになる
u_dir_param = dir_param

r = calc_r(X, W, m, nu, dim, beta, u_dir_param)
print(r)





