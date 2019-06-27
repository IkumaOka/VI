# プログラムの高速化目当て
# 本体はva_ga_mix_2.py
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.special import digamma, gamma, loggamma
from scipy.misc import logsumexp
import math


def create_data(N, K):
    loc = np.array([1.0, 10.0]) # 平均の正解
    scale = np.array([1.0, 2.0]) # 標準偏差の正解
    X, mu_star, sigma_star = [], [], []
    for i in range(K):
        X = np.append(X, np.random.normal(loc = loc[i], scale = scale[i], size = int(N / K)))
        mu_star = np.append(mu_star, loc[i])
        sigma_star = np.append(sigma_star, scale[i])
    return (X, mu_star, sigma_star)


def gaussian(mu, sigma):
    def f(x):
        return np.exp(-0.5 * (x - mu) ** 2 / sigma) / np.sqrt(2 * np.pi * sigma)
    return f


# 負担率を計算
# ηを計算する式（式7.29参照）PRMLだと式10.46
# 式7.29第1項において、多変量では正規-ウィシャート分布を使うが、一変量の場合は正規-ガンマ分布を使う
def calc_r(X, psi, beta, kappa, xi, dir_param):
    # 正規-ガンマ分布の期待値を求める(wiki参照)
    ex_T = kappa / xi
    ex_T_X = psi * kappa / xi
    ex_T_X_2 = 1 / beta + psi ** 2 * kappa / xi
    log_eta = - (np.tile(X**2, (2, 1)) * ex_T.reshape((2, 1)) - 2*np.tile(X, (2, 1))*ex_T_X.reshape((2, 1)) + ex_T_X_2.reshape((2, 1))) / 2 + digamma(dir_param).reshape((2, 1)) - digamma(dir_param.sum(axis=0))
    eta = np.exp(log_eta.T)
    r = eta / np.tile(eta.sum(axis=1), (2, 1)).T
    return r


# Mステップ
def estimate_gmm_parameter(X, r, u_psi, u_beta, u_kappa, u_xi, u_dir_param):
    # 負担率の総和
    n_j = r.sum(axis=0)
    # 負担率による観測値の重み付き平均
    barx_j = np.dot(X.T, r) / n_j
    s = np.tile(X, (2, 1)) - barx_j.reshape((2, 1))
    s_ = s**2
    # 観測値の重み付き分散
    s_j = (s_.T * r).sum(axis=0) / n_j

    # q(π)のパラメータを更新(式7.44)
    u_dir_param = dir_param + n_j

    #q(μ,λ)のパラメータを更新(式7.52)
    u_psi = ( n_j * barx_j + beta * psi ) / ( n_j + beta )
    u_beta = n_j + beta
    u_kappa = n_j / 2 + kappa
    u_xi = ( n_j * s_j + (n_j * beta * ((barx_j - psi) ** 2 / (n_j + beta)))) / 2 + xi
    return u_psi, u_beta, u_kappa, u_xi, u_dir_param


def calc_log_likelihood(X, u_psi, u_beta, u_kappa, u_xi, r):
    log_likelihood = 0.0
    diff = np.tile(X, (2, 1)) - psi.reshape((2, 1))
    log_u = np.log(u_beta * diff.T**2 / (2 * (1 + u_beta)) + u_xi)
    elm1 = np.log(r)
    elm2 = np.log(u_beta / (2 * math.pi * (1 + u_beta))) / 2
    elm3 = u_kappa * np.log(u_xi)
    elm4 = (u_kappa + 0.5) * log_u
    elm5 = loggamma(u_kappa + 0.5) - loggamma(u_kappa)
    s = elm1 + elm2 + elm3 - elm4 + elm5
    for i in range(len(s)):
        log_likelihood += logsumexp(s[i])
    print(log_likelihood)
    return log_likelihood


np.random.seed(19)
K = 2 # クラス数 
N = 1000 * K # データ数
# π,μ,σの値を初期化
pi = np.random.rand(K)
mu = np.random.randn(K)
sigma = np.abs(np.random.randn(K))
lamda = 1.0 / sigma
X, mu_star, sigma_star = create_data(N, K)
# 正規-ガンマ分布のパラメータを定義
psi = np.array([-1.0, 7.0])
beta = np.array([1.0, 0.9])
kappa = np.array([1.0, 0.9])
xi = np.array([1.0, 0.9])
# ディリクレ分布のパラメータを定義
dir_param = np.array([10.0, 8.0]) # [1.0, 0.1]だとdigammaに代入した時にマイナスになる
u_psi = psi
u_beta = beta
u_kappa = kappa
u_xi = xi
u_dir_param = dir_param
log_likelihoods = []
for iter in range(1000):
    r = calc_r(X, u_psi, u_beta, u_kappa, u_xi, u_dir_param)
    u_psi, u_beta, u_kappa, u_xi, u_dir_param = estimate_gmm_parameter(X, r, u_psi, u_beta, u_kappa, u_xi, u_dir_param)
    ave_dir_param = np.average(u_dir_param)
    ave_sigma = u_xi / u_kappa # 後でσを使うから期待値の逆数をとった
    gf = gaussian(u_psi, ave_sigma)
    log_likelihood = calc_log_likelihood(X, u_psi, u_beta, u_kappa, u_xi, r)
    log_likelihoods.append(log_likelihood)
    # print(u_psi)

log_likelihoods = np.array(log_likelihoods)

plt.plot(log_likelihoods, color='#4169e1', linestyle='solid')
plt.show()



