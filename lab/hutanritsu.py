import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.special import digamma, gamma

def create_data(N, K):
    loc = np.array([5.0, 10.0]) # 平均の初期値
    scale = np.array([1.0, 2.0]) # 標準偏差の初期値
    X, mu_star, sigma_star = [], [], []
    for i in range(K):
        X = np.append(X, np.random.normal(loc = loc[i], scale = scale[i], size = int(N / K)))
        mu_star = np.append(mu_star, loc[i])
        sigma_star = np.append(sigma_star, scale[i])
    return (X, mu_star, sigma_star)


# 負担率を計算
# ηを計算する式（式7.29参照）PRMLだと式10.46
# 式7.29第1項において、多変量では正規-ウィシャート分布を使うが、一変量の場合は正規-ガンマ分布を使う
def e_like_step(X):
    # 正規-ガンマ分布の期待値を求める(wiki参照)
    ex_T = kappa / xi
    ex_T_X = psi * kappa / xi
    ex_T_X_2 = (1 / beta) + (psi ** 2) * kappa / xi
    eta = []
    for i in range(len(X)):
        elm = ( (X[i]**2)*ex_T - 2*X[i]*ex_T_X + ex_T_X_2 ) / 2 + digamma(dir_param) - digamma(dir_param.sum(axis=0))
        eta.append(elm)
    eta = np.array(eta)
    r = []
    for i in range(len(eta)):
        a = eta[i] / eta[i].sum()
        r.append(a)
    r = np.array(r)
    return r


np.random.seed(5)
K = 2 # クラス数
N = 1000 * K # データ数
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
dir_param = np.array([1.0, 1.0])

r = e_like_step(X)
# print(r)

for i, elm in enumerate(r):
    print(elm)






