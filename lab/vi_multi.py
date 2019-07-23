import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.special import digamma, gamma, loggamma
from scipy.misc import logsumexp
import math
import numpy.linalg as LA


def create_data(N):
    x1 = np.random.normal(size=(N, 2))
    x1 += np.array([-5, -5])
    x2 = np.random.normal(size=(N, 2))
    x2 += np.array([5, -5])
    x3 = np.random.normal(size=(N, 2))
    x3 += np.array([0, 5])
    return np.vstack((x1, x2, x3))


def gaussian(X, W, m, nu, dim, beta):
    d = X[:, :, None] - m
    gauss = np.exp(
            -0.5 * dim / beta
            - 0.5 * nu * np.sum(
                np.einsum('ijk,njk->nik', W, d) * d,
                axis=1)
        )
    return gauss


def calc_r(X, W, m, nu, dim, beta, dir_param):
    gauss = gaussian(X, W, m, nu, dim, beta)
    # PRML式10.65
    log_lambda = digamma((nu + 1 - np.arange(dim)[:, None]) / 2).sum(axis=0) + dim*np.log(2) + LA.slogdet(W.T)[1]
    # PRML式10.66
    log_pi = digamma(dir_param) - digamma(dir_param.sum(axis=0))
    Lambda = np.exp(log_lambda)
    pi = np.exp(log_pi)
    r = pi * np.sqrt(Lambda) * gauss
    # r[np.where(r == 0)] = 0.5
    r /= np.sum(r, axis=-1, keepdims=True)
    # r[np.isnan(r)] = 1.0 / 2.0
    return r


def update_param(X, W0, m0, nu0, beta0, dir_param0, r):
    # PRML式10.51
    N = r.sum(axis=0)
    # PRML式10.52
    bar_x = np.dot(X.T, r) / N
    d = X[:, :, None] - bar_x
    # PRML式10.53
    S = np.einsum('nik,njk->ijk', d, r[:, None, :] * d) / N
    # PRML式10.58
    dir_param = dir_param0 + N
    # PRML式10.60
    beta = beta0 + N
    # PRML式10.61
    m = (beta0 * m0[:, None] + N * bar_x) / beta
    # PRML式10.62
    d_ = bar_x - m0[:, None]
    W = LA.inv(
        LA.inv(W0)
        +  (N * S).T
        + (beta0 * N * np.einsum('ik,jk->ijk', d_, d_) / (beta0 + N)).T
        ).T
    # PRML式10.63
    nu = nu0 + N
    return dir_param, beta, m, W, nu


np.random.seed(20)
K = 5 # クラス数 
N = 1000 # データ数
# データは2次元 × 3000
X = create_data(N)
sample_size = 3000
# ウィシャート分布のパラメータを定義
dim = 2
nu0 = dim
m0 = np.array([0.0, 0.0])
beta0 = 1.0
nu = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
m = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]).T # mの初期値（てきとう）
beta = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
W0 = np.array([[1.0, 0.0],
              [0.0, 1.0]
            ])
W = np.tile(W0, (K, 1, 1)).T
# ディリクレ分布のパラメータを定義
dir_param0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) # [1.0, 0.1]だとdigammaに代入した時にマイナスになる
dir_param = dir_param0
# r = calc_r(X, W, m, nu, dim, beta, dir_param)
# dir_param, beta, m, W, nu = update_param(X, W0, m0, nu0, beta0, dir_param0, r)
for iter in range(15):
    r = calc_r(X, W, m, nu, dim, beta, dir_param)
    dir_param, beta, m, W, nu = update_param(X, W0, m0, nu0, beta0, dir_param0, r)
print(r)