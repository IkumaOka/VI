import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.special import digamma, gamma

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

# 負担率を計算
# ηを計算する式（式7.29参照）PRMLだと式10.46
# 式7.29第1項において、多変量では正規-ウィシャート分布を使うが、一変量の場合は正規-ガンマ分布を使う
def estimate_posterior_likelihood(X):
    # 正規-ガンマ分布の期待値を求める(wiki参照)
    ex_T = kappa / xi
    ex_T_X = psi * kappa / xi
    ex_T_X_2 = 1 / beta + psi ** 2 * kappa / xi
    eta = []
    for i in range(len(X)):
        elm = ( X[i]*ex_T - 2*X[i]*ex_T_X + ex_T_X_2 ) / 2 + digamma(dir_param)- digamma(dir_param.sum(axis=0))
        eta.append(elm)

    eta = np.array(eta)
    # print(eta)
    r = []
    for i in range(len(eta)):
        a = eta[i] / eta[i].sum()
        r.append(a)
    r = np.array(r)
    return r

# Mステップ
def estimate_gmm_parameter(X, r, psi, beta, kappa, xi, dir_param):
    # 負担率の総和
    n_j = r.sum(axis=0)
    # 負担率による観測値の重み付き平均
    barx_j = np.dot(X.T, r) / n_j

    # q(π)のパラメータを更新(式7.44)
    dir_param = dir_param + n_j

    #q(μ,λ)のパラメータを更新(式7.54)
    psi = ( n_j * barx_j + beta * psi ) / ( n_j + beta )
    beta = n_j + beta
    kappa = n_j / 2 + kappa
    xi = ( n_j * barx_j + (n_j * beta + ((barx_j - psi) ** 2 / (n_j + beta))) / 2 ) + xi
    return psi, beta, kappa, xi, dir_param

def calc_log_likelihood(X, pi, gf):
    l = np.zeros((X.size, pi.size))
    print(l.shape)
    for (i, x) in enumerate(X):
       l[i, :] = pi * gf(x)
    # Xのi番目について, 式6.5を用いて各π_jのsumを求める
    p_i = l.sum(axis = 1).reshape(-1, 1)
    log_p_i = np.log(p_i)
    log_p = log_p_i.sum(axis=0)
    # xのsumを求める
    p = p_i.sum(axis=0)
    return log_p


K = 2
N = 1000 * K
# π,μ,σの値を初期化
pi = np.random.rand(K)
mu = np.random.randn(K)
sigma = np.abs(np.random.randn(K))
lamda = 1.0 / sigma
X, mu_star, sigma_star = create_data(N, K)
# 正規-ガンマ分布のパラメータを定義
psi = np.array([0.0, 0.0])
beta = np.array([1.0, 0.9])
kappa = np.array([1.0, 0.9])
xi = np.array([1.0, 0.9])
# ディリクレ分布のパラメータを定義
dir_param = np.array([1.0, 0.1])

for iter in range(100):
    r = estimate_posterior_likelihood(X)
    psi, beta, kappa, xi, dir_param = estimate_gmm_parameter(X, r, psi, beta, kappa, xi, dir_param)
    print("psi: ", psi)
    print("beta: ", beta)
    print("kappa: ", kappa)
    print("xi: ", xi)
    print("dir_param: ", dir_param)
    print()



