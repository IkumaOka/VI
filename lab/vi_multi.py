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


def calc_r(X, W, m, nu, dim, beta, dir_param):
    gauss = gaussian(X, W, m, nu, dim, beta)
    # PRML式10.65
    log_lambda = digamma((nu + 1 - np.arange(1, dim+1)[:, None]) / 2).sum(axis=0) + dim*np.log(2) + LA.det(W)
    # PRML式10.66
    print(dir_param.sum(axis=0))
    log_pi = np.exp(digamma(dir_param)) - digamma(dir_param.sum(axis=0))
    Lambda = np.exp(log_lambda)
    pi = np.exp(log_pi)
    r = pi * np.sqrt(Lambda) * gauss
    r[np.where(r == 0)] = 0.5
    r = r / np.tile(r.sum(axis=1), (2, 1)).T
    r[np.isnan(r)] = 1.0 / 2.0
    return r


def update_param(X, W0, m0, nu0, beta0, dir_param0, r):
    # PRML式10.51
    N = r.sum(axis=0)
    # PRML式10.52
    bar_x = (r * X).sum(axis=0) / N
    elm1 = r * (X - np.tile(bar_x, (3000, 1)))
    # PRML式10.53
    S = np.dot(elm1.T, (X - np.tile(bar_x, (3000, 1)))) / N
    # PRML式10.58
    dir_param = dir_param0 + N
    # PRML式10.60
    beta = beta0 + N
    # PRML式10.61
    m = (beta0*m0 + N*bar_x) / beta
    # PRML式10.62
    W = LA.inv(
        LA.inv(W0)
        +  N * S
        + (beta0*N) * np.dot((bar_x - m0), (bar_x - m0).T) / (beta0 + N)
        )
    # PRML式10.63
    nu = nu0 + N
    return dir_param, beta, m, W, nu


np.random.seed(20)
K = 2 # クラス数 
N = 1000 # データ数
# データは2次元 × 3000
X = create_data(N, K)
sample_size = 3000
component_size = sample_size / K + np.zeros(K)
# ウィシャート分布のパラメータを定義
dim = 2
nu0 = dim
m0 = np.array([1.0, 0.0])
beta0 = 1.0
nu = np.array([2.0, 2.0])
m = [1.0, 3.0] # mの初期値（てきとう）
beta = beta0
W0 = np.array([[0.01, 0.05],
              [0.05, 0.04]
            ])
W = W0
# ディリクレ分布のパラメータを定義
dir_param0 = np.array([1.0, 2.0]) # [1.0, 0.1]だとdigammaに代入した時にマイナスになる
dir_param = dir_param0
r = calc_r(X, W, m, nu, dim, beta, dir_param)
dir_param, beta, m, W, nu = update_param(X, W0, m0, nu0, beta0, dir_param0, r)
for iter in range(1000):
    r = calc_r(X, W, m, nu, dim, beta, dir_param)
    dir_param, beta, m, W, nu = update_param(X, W0, m0, nu0, beta0, dir_param0, r)
