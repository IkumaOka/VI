# vi_stochastic2.pyからコピー
# 実験は1回だけversion

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, gamma
from numpy.random import *
from sklearn.datasets import load_iris


class VariationalGaussianMixture(object):

    def __init__(self, n_component=10, alpha0=1.):
        self.n_component = n_component
        self.alpha0 = alpha0

    def init_params(self, X):
        self.sample_size, self.ndim = X.shape
        self.alpha0 = np.ones(self.n_component) * self.alpha0
        self.m0 = np.zeros(self.ndim)
        self.W0 = np.eye(self.ndim)
        self.nu0 = self.ndim
        self.beta0 = 1.

        self.component_size = self.sample_size / self.n_component + np.zeros(self.n_component)
        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        indices = np.random.choice(self.sample_size, self.n_component, replace=False)
        self.m = X[indices].T
        self.W = np.tile(self.W0, (self.n_component, 1, 1)).T
        self.nu = self.nu0 + self.component_size
        self.log_likelihoods = []

    def get_params(self):
        return self.alpha, self.beta, self.m, self.W, self.nu

    def fit(self, X, iter_max=100):
        self.init_params(X)
        for i in range(iter_max):
            params = np.hstack([array.flatten() for array in self.get_params()])
            r = self.e_like_step(X)
            stochastic_r = self.stochastic_cluster(r, self.n_component)
            print(stochastic_r)
            self.m_like_step(X, stochastic_r)
            a = self.calc_loglikelihood(X)
            self.log_likelihoods.append(a)
            if np.allclose(params, np.hstack([array.ravel() for array in self.get_params()])):
                break
        else:
            print("parameters may not have converged")

    def e_like_step(self, X):
        d = X[:, :, None] - self.m
        gauss = np.exp(
            -0.5 * self.ndim / self.beta
            - 0.5 * self.nu * np.sum(
                np.einsum('ijk,njk->nik', self.W, d) * d,
                axis=1)
        )
        pi = np.exp(digamma(self.alpha) - digamma(self.alpha.sum()))
        Lambda = np.exp(digamma(self.nu - np.arange(self.ndim)[:, None]).sum(axis=0) + self.ndim * np.log(2) + np.linalg.slogdet(self.W.T)[1])
        r = pi * np.sqrt(Lambda) * gauss
        r /= np.sum(r, axis=-1, keepdims=True)
        r[np.isnan(r)] = 1. / self.n_component
        return r

    def m_like_step(self, X, r):
        self.component_size = r.sum(axis=0)
        self.component_size[np.where(self.component_size == 0)] = 1.0e-300
        # print(self.component_size)
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

    def predict_proba(self, X):
        covs = self.nu * self.W
        precisions = np.linalg.inv(covs.T).T
        d = X[:, :, None] - self.m
        exponents = np.sum(np.einsum('nik,ijk->njk', d, precisions) * d, axis=1)
        gausses = np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(covs.T).T * (2 * np.pi) ** self.ndim)
        gausses *= (self.alpha0 + self.component_size) / (self.n_component * self.alpha0 + self.sample_size)
        return np.sum(gausses, axis=-1)

    def classify(self, X):
        return np.argmax(self.e_like_step(X), 1)

    def student_t(self, X):
        nu = self.nu + 1 - self.ndim
        L = nu * self.beta * self.W / (1 + self.beta)
        d = X[:, :, None] - self.m
        maha_sq = np.sum(np.einsum('nik,ijk->njk', d, L) * d, axis=1)
        return (
            gamma(0.5 * (nu + self.ndim))
            * np.sqrt(np.linalg.det(L.T))
            * (1 + maha_sq / nu) ** (-0.5 * (nu + self.ndim))
            / (gamma(0.5 * nu) * (nu * np.pi) ** (0.5 * self.ndim)))

    def predict_dist(self, X):
        return (self.alpha * self.student_t(X)).sum(axis=-1) / self.alpha.sum()

    def calc_loglikelihood(self, X):
        d = X[:, :, None] - self.m
        gauss = np.exp(
            -0.5 * self.ndim / self.beta
            - 0.5 * self.nu * np.sum(
                np.einsum('ijk,njk->nik', self.W, d) * d,
                axis=1)
        )
        ave_alpha = self.alpha / np.mean(self.alpha)
        # print(ave_alpha)
        p_i = ave_alpha * gauss
        # print(p_i)
        p_1 = np.sum(p_i, axis=1)
        p_2 = np.sum(p_1)
        log_likelihood = np.log(p_2)
        # print(log_likelihood)
        return log_likelihood


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
    np.random.seed(10)
    # X = create_toy_data()
    iris = load_iris()
    model = VariationalGaussianMixture(n_component=10, alpha0=0.01)
    iter_max = [1000]
    for iter in iter_max:
        model.fit(iris.data, iter_max=iter)
        labels = model.classify(iris.data)
        print(labels)
        x_test, y_test = np.meshgrid(
            np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
        X_test = np.array([x_test, y_test]).reshape(2, -1).transpose()
        plt.scatter(iris.data[:, 0], iris.data[:, 1], c=labels, cmap=cm.get_cmap())
        plt.title(iter)
        save_name = str(iter) + '.png'
        # plt.savefig(save_name, dpi=500)
        plt.show()
        plt.plot(model.log_likelihoods, color='#4169e1', linestyle='solid')
        plt.show()


if __name__ == '__main__':
    main()