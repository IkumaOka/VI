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
    # print(pi)
    # print(Lambda)
    r = pi * np.sqrt(Lambda) * gauss
    r[np.where(r == 0)] = 0.5
    r = r / np.tile(r.sum(axis=1), (2, 1)).T
    r[np.isnan(r)] = 1.0 / 2.0
    return r

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
W = np.array([[0.01, 0.05],
              [0.05, 0.04]
            ])

# ディリクレ分布のパラメータを定義
dir_param = np.array([1.0, 2.0]) # [1.0, 0.1]だとdigammaに代入した時にマイナスになる
u_dir_param = dir_param
r = calc_r(X, W, m, nu, dim, beta, u_dir_param)
for x in r:
    print(x)
# print(X)
