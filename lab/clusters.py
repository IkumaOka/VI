import numpy as np
import sys
import matplotlib.pyplot as plt

def create_data(N, K):
    X, mu_star, sigma_star = [], [], []
    for i in range(K):
        loc = (np.random.rand() - 0.5) * 10.0 # range: -5.0 - 5.0
        scale = np.random.rand() * 3.0 # range: 0.0 - 3.0
        X = np.append(X, np.random.normal(loc = loc, scale = scale, size = int(N / K)))
        mu_star = np.append(mu_star, loc)
        sigma_star = np.append(sigma_star, scale)
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


# 負担率の大きい方を1, 小さい方を0に値を振り直す
def hard_cluster(gamma):
    max_index = np.argmax(gamma, axis=1)
    min_index = np.argmin(gamma, axis=1)
    for i in range(len(gamma)):
        gamma[i][max_index[i]] = 1
        gamma[i][min_index[i]] = 0

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
    

def clustering(allocation=hard_cluster):
    K = 2
    N = 1000 * K
    pi = np.random.rand(K)
    mu = np.random.randn(K)
    sigma = np.abs(np.random.randn(K))
    X, mu_star, sigma_star = create_data(N, K)
    log_likelihoods = []
    for iter in range(1000):
        gf = gaussian(mu, sigma)
        gamma = estimate_posterior_likelihood(X, pi, gf)
        update_gamma = allocation(gamma)
        mu, sigma, pi = estimate_gmm_parameter(X, gamma)
        gf = gaussian(mu, sigma)
        log_likelihood = calc_log_likelihood(X, pi, gf)
        log_likelihoods.append(log_likelihood[0])

    np_log_likelihoods = np.array(log_likelihoods)
    x = np.linspace(0, 10, 1000)
    plt.plot(x, np_log_likelihoods)
    plt.show()

hard_likelihoods = clustering(hard_cluster)
stochastic_likelihoods = clustering(stochastic_cluster)