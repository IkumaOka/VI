import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, gamma, gammaln
from numpy.random import *


class VariationalGaussianMixture(object):


    def __init__(self, n_component=10, alpha0=1.):
        self.n_component = n_component
        self.alpha0 = alpha0


    def init_params(self, X):
        self.D = 2
        self.sample_size, self.ndim = X.shape
        self.alpha0 = np.ones(self.n_component) * self.alpha0
        self.m0 = np.zeros(self.ndim)
        self.W0 = np.eye(self.ndim)
        self.nu0 = self.ndim
        self.beta0 = 1.

        self.component_size = self.sample_size / self.n_component + np.zeros(self.n_component)
        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        # indices = np.random.choice(self.sample_size, self.n_component, replace=False)
        # self.m = X[indices].T
        self.m = (np.tile(self.m0, (self.n_component, 1)) + np.random.normal(size=(self.n_component, self.ndim))).T
        self.W = np.tile(self.W0, (self.n_component, 1, 1)).T
        self.nu = self.nu0 + self.component_size
        self.log_likelihoods = []
        self.rho = 1.0


    def get_params(self):
        return self.alpha, self.beta, self.m, self.W, self.nu


    def normal_fit(self, X, iter_max=100):
        self.init_params(X)
        for i in range(iter_max):
            params = np.hstack([array.flatten() for array in self.get_params()])
            r = self.e_like_step(X)
            self.m_like_step(X, r)
            a = self.calc_lower_bound(X)
            self.log_likelihoods.append(a)


    def stochastic_fit(self, X, iter_max=100):
        self.init_params(X)
        for i in range(iter_max):
            params = np.hstack([array.flatten() for array in self.get_params()])
            r = self.e_like_step(X)
            stochastic_r = self.stochastic_cluster(r, self.n_component)
            self.m_like_step(X, stochastic_r)
            a = self.calc_lower_bound(X)
            self.log_likelihoods.append(a)


    def svi_fit(self, X, iter_max=100):
        self.init_params(X)
        for i in range(iter_max):
            params = np.hstack([array.flatten() for array in self.get_params()])
            rand_X = X[np.random.choice(X.shape[0], 80), :]
            r = self.e_like_step(rand_X)
            alpha_, beta_, m_, W_, nu_ = self.m_like_step(rand_X, r)
            self.alpha = self.update_stochastic_param(alpha_, self.alpha, i)
            self.beta = self.update_stochastic_param(beta_, self.beta, i)
            self.m = self.update_stochastic_param(m_, self.m, i)
            self.W = self.update_stochastic_param(W_, self.W, i)
            self.nu = self.update_stochastic_param(nu_, self.nu, i)
            a = self.calc_lower_bound(X)
            self.log_likelihoods.append(a)


    def e_like_step(self, X):
        d = X[:, :, None] - self.m
        gauss = np.exp(
            -0.5 * self.ndim / self.beta
            - 0.5 * self.nu * np.sum(
                np.einsum('ijk,njk->nik', self.W, d) * d,
                axis=1)
        )
        gauss[np.isinf(gauss) == True] = 1.0e+10
        pi = np.exp(digamma(self.alpha) - digamma(self.alpha.sum()))
        Lambda = np.exp(digamma(self.nu - np.arange(self.ndim)[:, None]).sum(axis=0) + self.ndim * np.log(2) + np.linalg.slogdet(self.W.T)[1])
        r = pi * np.sqrt(Lambda) * gauss
        r[np.where(r == 0)] = 1.0e-300
        r /= np.sum(r, axis=-1, keepdims=True)
        r[np.isnan(r)] = 1. / self.n_component
        return r


    def m_like_step(self, X, r):
        self.component_size = r.sum(axis=0)
        self.component_size[np.where(self.component_size == 0)] = 1.0e-300
        # (t-1)の変数を保存しておく
        alpha_ = self.alpha
        beta_ = self.beta
        m_ = self.m
        W_  = self.W
        nu_ = self.nu
        Xm = X.T.dot(r) / self.component_size
        d = X[:, :, None] - Xm
        S = np.einsum('nik,njk->ijk', d, r[:, None, :] * d) / self.component_size
        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        self.m = (self.beta0 * self.m0[:, None] + self.component_size * Xm) / self.beta
        d = Xm - self.m0[:, None]
        self.W = np.linalg.inv(
            np.linalg.inv(self.W0)
            + (self.component_size * S).T
            + (self.beta0 * self.component_size * np.einsum('ik,jk->ijk', d, d) / (self.beta0 + self.component_size)).T).T
        self.nu = self.nu0 + self.component_size
        return alpha_, beta_, m_, W_, nu_


    def update_stochastic_param(self, param_, param, iter):
        self.rho = (self.rho + iter) ** (-0.9)
        update_param = (1 - self.rho) * param_ + self.rho * param
        return update_param


    def normal_cluster(self, r):
            return r


    def stochastic_cluster(self, r, n_component):
        clusters = list(range(0, n_component))
        stochastic_r = []
        for i in range(len(r)):
            p = np.random.choice(a=clusters, p=r[i])
            a = np.zeros(len(r[i]))
            a[p] = 1.0
            stochastic_r.append(a)
        stochastic_r = np.array(stochastic_r)

        return stochastic_r


    def classify(self, X):
        return np.argmax(self.e_like_step(X), 1)


    def calc_lower_bound(self, r):
        a = - (r * np.log(r)).sum()
        b = self.logC(self.alpha0)
        c = self.logC(self.alpha)
        d = (np.log(self.beta0) - np.log(self.beta.sum())) * self.D / 2
        # e = self.logB(self.W0, self.nu0)
        f = self.logB(self.W, self.nu)
        return - (r * np.log(r)).sum() + \
            self.logC(self.alpha0*np.ones(self.n_component)) - self.logC(self.alpha) +\
            self.D/2 * (self.n_component * np.log(self.beta0) - np.log(self.beta).sum()) + \
            self.n_component * self.logB(self.W0, self.nu0) - self.logB(self.W, self.nu).sum()


    def logC(self, alpha):
        return gammaln(alpha.sum()) - gammaln(alpha).sum()


    def logB(self, W, nu):
        Wshape = W.T.shape
        if len(Wshape) == 2:
            D, _ = Wshape
            arg_gamma = nu - np.arange(0, D, 1)
            return -nu/2 * np.log(np.linalg.det(W)) - D/2 * nu * np.log(2) - D* (D - 1)/4 * np.log(np.pi) - gammaln(arg_gamma/2).sum()
        else:
            # K, D, elm = Wshape
            # arg_gamma = nu - np.reshape(np.arange(0, D, 1), (1, elm))
            # arg_gamma = nu - np.arange(1, elm+1, 1)
            K, D, _ = Wshape
            arg_gamma = np.reshape(nu, (K, 1)) - np.reshape(np.arange(0, D, 1), (1, D))
            return -nu/2 * np.log(np.linalg.det(W.T)) - D/2 * nu * np.log(2) - D* (D - 1)/4 * np.log(np.pi) - gammaln(arg_gamma/2).sum(axis=1)


def svi_clustering(X):
    model = VariationalGaussianMixture(n_component=10, alpha0=0.01)
    model.svi_fit(X, iter_max=2000)
    labels = model.classify(X)
    log_likelihoods = model.log_likelihoods
    return log_likelihoods


def normal_clustering(X):
    model = VariationalGaussianMixture(n_component=10, alpha0=0.01)
    model.normal_fit(X, iter_max=2000)
    labels = model.classify(X)
    log_likelihoods = model.log_likelihoods
    return log_likelihoods


def stochastic_clustering(X):
    model = VariationalGaussianMixture(n_component=10, alpha0=0.01)
    model.stochastic_fit(X, iter_max=2000)
    labels = model.classify(X)
    log_likelihoods = model.log_likelihoods
    return log_likelihoods


def create_toy_data():
    mu = np.array([[5, 5], [20, 20], [50, 10]])
    sigma = np.array([[[3, 1], 
                       [1, 3]],
                       [[3, 1],
                       [1, 3]],
                       [[3, 1],
                       [1, 3]]])
    data = []
    for i in range(len(mu)):
        values = multivariate_normal(mu[i], sigma[i], 100)
        data.extend(values)
    data = np.array(data)
    return data


def main():
    # np.random.seed(48)
    np.random.seed(5)
    X = create_toy_data()
    print("svi now...")
    svi_loglikelihoods = svi_clustering(X)
    print("normal now...")
    normal_loglikelihoods = normal_clustering(X)
    print("stochastic now...")
    stochastic_loglikelihoods = stochastic_clustering(X)
    plt.plot(svi_loglikelihoods, color='#ffff00', linestyle='solid')
    plt.plot(normal_loglikelihoods, color='#2971e5', linestyle='solid')
    plt.plot(stochastic_loglikelihoods, color='#ed3b3b', linestyle='solid')
    plt.legend(['svi', 'normal', 'stochastic'])
    plt.show()


if __name__ == '__main__':
    main()