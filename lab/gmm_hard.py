import numpy as np
import sys
import matplotlib.pyplot as plt

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
# 変分ベイズではηを計算する式
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

# Q関数を最大にするようなパラメータを求めてπ、μ、σを更新
# 式7.44と問7-3を計算する
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
    # print(p_i.shape)
    log_p_i = np.log(p_i)
    log_p = log_p_i.sum(axis=0)

    # xのsumを求める
    p = p_i.sum(axis=0)

    return log_p
    

K = 2
N = 1000 * K
# π,μ,σの値を初期化
# πの制約条件を満たしている？←満たしていないが、更新すると満たすので無視でも良い
pi = np.random.rand(K)
mu = np.random.randn(K)
sigma = np.abs(np.random.randn(K))
X, mu_star, sigma_star = create_data(N, K)

y = []
for iter in range(1000):
    gf = gaussian(mu, sigma)
    gamma = estimate_posterior_likelihood(X, pi, gf)
    hard_gamma = hard_cluster(gamma)
    # print("gamma:\n", gamma)
    # print(gamma.shape)
    mu, sigma, pi = estimate_gmm_parameter(X, gamma)
    gf = gaussian(mu, sigma)
    log_likelihood = calc_log_likelihood(X, pi, gf)
    print("log_likelihood: ", log_likelihood)
    y.append(log_likelihood[0])

print("gamma:\n", gamma)
np_y = np.array(y)
x = np.linspace(0, 10, 1000)
plt.plot(x, np_y)
plt.show()






# ハード割り当て用の関数を作る
# 確率的EMアルゴリズム用の関数を作る
# データのseedを固定して何回か繰り返す
# パラメータの初期化のseedを固定して何回か繰り返す
# 繰り返し回数は1000回など固定する（収束するまでにしない）
# Q関数は完全データの対数尤度を求めて期待値を計算している？→代わりに式6.5を使って対数尤度を求める？















# 変分ベイズのGMM


