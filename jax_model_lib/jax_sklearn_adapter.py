import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from jax_model_lib import jax_vlMAP, adaptive_prior

class JaxSparseRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        tau0=1.0,
        max_iters=300,
        lr_pre=0.01,
        mb_size=50,
        invlam=False,
        logprox=False,
        track=False
    ):
        self.tau0 = tau0
        self.max_iters = max_iters
        self.lr_pre = lr_pre
        self.mb_size = mb_size
        self.invlam = invlam
        self.logprox = logprox
        self.track = track

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        P = X.shape[1]
        lam_prior_vars = {'log_gamma': np.zeros(P)}

        self.model_ = jax_vlMAP(
            X=X,
            y=y,
            lam_prior=adaptive_prior,
            lam_prior_vars=lam_prior_vars,
            lik='normal',
            tau0=self.tau0,
            mb_size=self.mb_size,
            invlam=self.invlam,
            logprox=self.logprox,
            track=self.track
        )
        self.model_.fit(max_iters=self.max_iters, lr_pre=self.lr_pre, verbose=False)
        return self

    def predict(self, X):
        return np.array(self.model_.predictive(np.asarray(X)).mean())

    def score(self, X, y):
        y_pred = self.predict(X)
        y = np.asarray(y)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def get_coefficients(self):
        return np.array(self.model_.beta)
