# 負担率をハードに割り当てる（1, 0）
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import digamma, gamma, loggamma
from scipy.misc import logsumexp
import math
import numpy.linalg as LA
from numpy.random import *
import warnings
warnings.filterwarnings('ignore')


def create_data(N):
    mu = np.array([[0, 0], [10, 20], [45, 10]])
    sigma = np.array([[[3, 2], 
                       [2, 5]],
                       [[3, 2],
                       [2, 5]],
                       [[3, 2],
                       [2, 5]]])
    data = []
    for i in range(len(mu)):
        values = multivariate_normal(mu[i], sigma[i], N)
        data.extend(values)
    data = np.array(data)
    return data


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
    # print(gauss)
    # PRML式10.65
    # print(np.arange(1,dim+1)[:, None])
    log_lambda = (digamma((nu + 1 - np.arange(1, dim+1)[:, None]) / 2)).sum(axis=0) + dim*np.log(2) + LA.slogdet(W.T)[1]
    # PRML式10.66
    log_pi = digamma(dir_param) - digamma(dir_param.sum())
    Lambda = np.exp(log_lambda)
    pi = np.exp(log_pi)
    r = pi * np.sqrt(Lambda) * gauss
    # r[np.where(r == 0)] = 0.5
    # r[np.isnan(r)] = 1.0 / 2.0
    r /= np.sum(r, axis=-1, keepdims=True)
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


def classify(r):
        return np.argmax(r, 1)


def student_t(nu, dim, beta, W, m, X_test):
    nu = nu + 1 - dim
    L = nu * beta * W / (1 + beta)
    d = X_test[:, :, None] - m
    maha_sq = np.sum(np.einsum('nik,ijk->njk', d, L) * d, axis=1)
    return (
        gamma(0.5 * (nu + dim))
        * np.sqrt(np.linalg.det(L.T))
        * (1 + maha_sq / nu) ** (-0.5 * (nu + dim))
        / (gamma(0.5 * nu) * (nu * np.pi) ** (0.5 * dim)))

def predict_dist(dir_param, nu, dim, beta, W, m, X_test):
        return (dir_param * student_t(nu, dim, beta, W, m, X_test)).sum(axis=-1) / dir_param.sum()


# 負担率は確率分布で、その確率分布に従うコインを投げて0か1を代入
def stochastic_cluster(r):
    clusters = [0, 0, 0, 0, 1]
    stochastic_r = []
    for i in range(len(r)):
        p = np.random.choice(a=clusters, p=r[i])
        a = np.zeros(len(r[i]))
        a[p] = 1.0
        stochastic_r.append(a)
    stochastic_r = np.array(stochastic_r)

    return stochastic_r


np.random.seed(20)
K = 5 # クラス数 
N = 100 # データ数
# データは2次元 × 3000
X = create_data(N)
# ウィシャート分布のパラメータを定義
dim = 2
nu0 = dim
m0 = np.array([0.0, 0.0])
beta0 = 1.0
nu = np.array([35.0, 36.0, 37.0, 38.0, 39.0])
m = np.array([[1.0, 1.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [7.0, 7.0]]).T # mの初期値（てきとう）
beta = np.array([1.0, 5.0, 2.0, 4.0, 5.0])
W0 = np.array([[0.001, 0.0],
              [0.0, 0.001]
            ])
W = np.tile(W0, (K, 1, 1)).T
# ディリクレ分布のパラメータを定義
dir_param0 = np.array([0.01, 0.02, 0.03, 0.03, 0.03])
dir_param = dir_param0
for iter in range(50):
    r = calc_r(X, W, m, nu, dim, beta, dir_param)
    stochastic_r = stochastic_cluster(r)
    print(stochastic_r)
    dir_param, beta, m, W, nu = update_param(X, W0, m0, nu0, beta0, dir_param0, stochastic_r)
labels = classify(r)
# print(labels)
x_test, y_test = np.meshgrid(
        np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
X_test = np.array([x_test, y_test]).reshape(2, -1).transpose()
probs = predict_dist(dir_param, nu, dim, beta, W, m, X_test)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=cm.get_cmap())
plt.show()
