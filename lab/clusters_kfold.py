# ノーマルEMとSEMのみ（ハードクラスタリングはしない）
# データの8割をテストデータとしてパラメータを推定し、残りの2割を使って尤度を計算
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from statistics import mean

# loc:平均, scale:標準偏差, size:出力配列のサイズ
# for i in range(K)のところは, K=2の場合、正規分布が二つとなるデータXを生成する
def create_data(N, K):
    loc = np.array([1.0, 3.0]) # 平均の初期値
    scale = np.array([1.0, 2.0]) # 標準偏差の初期値
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
def estimate_posterior_likelihood(X, pi, gf):
    l = np.zeros((X.size, pi.size))
    for (i, x) in enumerate(X):
        l[i, :] = pi * gf(x)
    return l * np.vectorize(lambda y: 1 / y)(l.sum(axis = 1).reshape(-1, 1))


def normal_cluster(gamma):
    return gamma


# 負担率は確率分布で、その確率分布に従うコインを投げて0か1を代入
def stochastic_cluster(gamma):
    clusters = [0, 1]
    stochastic_gamma = []
    for i in range(len(gamma)):
        p = np.random.choice(a=clusters, p=gamma[i])
        a = np.zeros(len(gamma[i]))
        a[p] = 1.0
        stochastic_gamma.append(a)
    stochastic_gamma = np.array(stochastic_gamma)
    return stochastic_gamma


# Q関数を最大にするようなパラメータを求めてπ、μ、σを更新
def estimate_gmm_parameter(X, gamma):
    N = gamma.sum(axis = 0)
    mu = (gamma * X.reshape((-1, 1))).sum(axis = 0) / N
    sigma = (gamma * (X.reshape(-1, 1) - mu) ** 2).sum(axis = 0) / N
    pi = N / X.size
    return (mu, sigma, pi)


# 対数尤度を計算
def calc_log_likelihood(X, pi, gf):
    l = np.zeros((X.size, pi.size))

    for (i, x) in enumerate(X):
       l[i, :] = pi * gf(x)

    # Xのi番目について, 式6.5を用いて各π_jのsumを求める
    p_i = l.sum(axis = 1).reshape(-1, 1)
    log_p_i = np.log(p_i)
    log_p = log_p_i.sum(axis=0)
    # xのsumを求める
    p = p_i.sum(axis=0)
    return log_p
    

def clustering(X_train, X_test, pi, mu, sigma, allocation=normal_cluster):
    log_likelihoods = []
    for iter in range(1000):
        gf = gaussian(mu, sigma)
        gamma = estimate_posterior_likelihood(X, pi, gf)
        update_gamma = allocation(gamma)
        mu, sigma, pi = estimate_gmm_parameter(X, update_gamma)
        gf = gaussian(mu, sigma)
        log_likelihood = calc_log_likelihood(X_test, pi, gf)
        log_likelihoods.append(log_likelihood)
    return log_likelihoods

K = 2
N = 1000 * K
pi = np.random.rand(K)
mu = np.random.randn(K)
sigma = np.abs(np.random.randn(K))
print("pi0: ", pi)
print("mu0: ", mu)
print("sigma0: ", sigma)
X, mu_star, sigma_star = create_data(N, K)
kf = KFold(n_splits=5)
print("normal_EM:")
normal_em_log_likelihoods = []
for X_train_index, X_test_index in kf.split(X):
    log_likelihoods = clustering(X[X_train_index], X[X_test_index], pi, mu, sigma, normal_cluster)
    log_likelihoods = np.array(log_likelihoods)
    normal_em_log_likelihoods.append(log_likelihoods)

normal_em_log_likelihoods = np.array(normal_em_log_likelihoods)
print(normal_em_log_likelihoods.shape)
print("stochastic_EM:")
stochastic_em_log_likelihoods = []
for X_train_index, X_test_index in kf.split(X):
    log_likelihoods = clustering(X[X_train_index], X[X_test_index], pi, mu, sigma, stochastic_cluster)
    log_likelihoods = np.array(log_likelihoods)
    stochastic_em_log_likelihoods.append(log_likelihoods)

stochastic_em_log_likelihoods = np.array(stochastic_em_log_likelihoods)
print(stochastic_em_log_likelihoods.shape)

