class BayesianGaussianMixtureModel:

    def __init__(self, K, D, alpha0, beta0, nu0, m0, W0):
        self.K = K
        self.D = D
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.nu0 = nu0
        self.m0 = m0
        self.W0 = W0

        self.alpha = None
        self.beta = None
        self.nu = None
        self.m = None
        self.W = None
        self.lower_bound = None

    def _init_params(self, X, random_state=None):
        N, D = X.shape
        rnd = np.random.RandomState(seed=random_state)

        self.alpha = (self.alpha0 + N / self.K) * np.ones(self.K)
        self.beta = (self.beta0 + N / self.K) * np.ones(self.K)
        self.nu = (self.nu0 + N / self.K) * np.ones(self.K)
        self.m = X[rnd.randint(low=0, high=N, size=self.K)]
        self.W = np.tile(np.diag(1.0/np.var(X, axis=0)), (self.K, 1, 1))

    def _e_like_step(self, X):
        '''
        Method for calculating the array corresponding to responsibility.

        Parameters
        ----------
        X : 2D numpy array
            2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.

        Returns
        ----------
        r : 2D numpy array
            2D numpy array representing responsibility of each component for each sample in X, 
            where r[n, k] = $r_{n, k}$.

        '''
        N, _ = np.shape(X)

        tpi = np.exp( digamma(self.alpha) - digamma(self.alpha.sum()) )

        arg_digamma = np.reshape(self.nu, (self.K, 1)) - np.reshape(np.arange(0, self.D, 1), (1, self.D))
        tlam = np.exp( digamma(arg_digamma/2).sum(axis=1)  + self.D * np.log(2) + np.log(np.linalg.det(self.W)) )

        diff = np.reshape(X, (N, 1, self.D) ) - np.reshape(self.m, (1, self.K, self.D) )
        exponent = self.D / self.beta + self.nu * np.einsum("nkj,nkj->nk", np.einsum("nki,kij->nkj", diff, self.W), diff)

        exponent_subtracted = exponent - np.reshape(exponent.min(axis=1), (N, 1))
        rho = tpi*np.sqrt(tlam)*np.exp( -0.5 * exponent_subtracted )
        r = rho/np.reshape(rho.sum(axis=1), (N, 1))

        return r


    def _m_like_step(self, X, r):
        N, _ = np.shape(X)
        n_samples_in_component = r.sum(axis=0)
        barx = r.T @ X / np.reshape(n_samples_in_component, (self.K, 1))
        diff = np.reshape(X, (N, 1, self.D) ) - np.reshape(barx, (1, self.K, self.D) )
        S = np.einsum("nki,nkj->kij", np.einsum("nk,nki->nki", r, diff), diff) / np.reshape(n_samples_in_component, (self.K, 1, 1))

        self.alpha = self.alpha0 + n_samples_in_component
        self.beta = self.beta0 + n_samples_in_component
        self.nu = self.nu0 + n_samples_in_component
        self.m = (self.m0 * self.beta0 + barx * np.reshape(n_samples_in_component, (self.K, 1)))/np.reshape(self.beta, (self.K, 1))

        diff2 = barx - self.m0
        Winv = np.reshape(np.linalg.inv( self.W0 ), (1, self.D, self.D)) + \
            S * np.reshape(n_samples_in_component, (self.K, 1, 1)) + \
            np.reshape( self.beta0 * n_samples_in_component / (self.beta0 + n_samples_in_component), (self.K, 1, 1)) * np.einsum("ki,kj->kij",diff2,diff2) 
        self.W = np.linalg.inv(Winv)

    def _calc_lower_bound(self, r):
        return - (r * np.log(r)).sum() + \
            logC(self.alpha0*np.ones(self.K)) - logC(self.alpha) +\
            self.D/2 * (self.K * np.log(self.beta0) - np.log(self.beta).sum()) + \
            self.K * logB(self.W0, self.nu0) - logB(self.W, self.nu).sum()


    def fit(self, X, max_iter=1e3, tol=1e-4, random_state=None, disp_message=False):
        self._init_params(X, random_state=random_state)
        r = self._e_like_step(X)
        lower_bound = self._calc_lower_bound(r)

        for i in range(max_iter):
            self._m_like_step(X, r)
            r = self._e_like_step(X)

            lower_bound_prev = lower_bound
            lower_bound = self._calc_lower_bound(r)

            if abs(lower_bound - lower_bound_prev) < tol:
                break

        self.lower_bound = lower_bound

        if disp_message:
            print(f"n_iter : {i}")
            print(f"convergend : {i < max_iter}")
            print(f"lower bound : {lower_bound}")
            print(f"Change in the variational lower bound : {lower_bound - lower_bound_prev}")


    def _predict_joint_proba(self, X):
        L = np.reshape( (self.nu + 1 - self.D)*self.beta/(1 + self.beta), (self.K, 1,1) ) * self.W
        tmp = np.zeros((len(X), self.K))
        for k in range(self.K):
            tmp[:,k] = multi_student_t(X, self.m[k], L[k], self.nu[k] + 1 - self.D)
        return tmp * np.reshape(self.alpha/(self.alpha.sum()), (1, self.K))

    def calc_prob_density(self, X):
        joint_proba = self._predict_joint_proba(X)
        return joint_proba.sum(axis=1)

    def predict_proba(self, X):
        joint_proba = self._predict_joint_proba(X)
        return joint_proba / joint_proba.sum(axis=1).reshape(-1, 1)

    def predict(self, X):
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)