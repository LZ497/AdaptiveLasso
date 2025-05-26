import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from jax_model_lib import jax_vlMAP, adaptive_prior, group_prior, hier_prior

class JaxSparseRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        tau0=1.0,
        max_iters=300,
        lr_pre=0.01,
        mb_size=50,
        invlam=False,
        logprox=False,
        track=False,
        lam_prior='adaptive',
        lam_prior_vars=None,
        lik='normal',
        custom_prox=None,
        custom_nll_grad=None,
        custom_prior=None
    ):
        self.tau0 = tau0
        self.max_iters = max_iters
        self.lr_pre = lr_pre
        self.mb_size = mb_size
        self.invlam = invlam
        self.logprox = logprox
        self.track = track
        self.lam_prior = lam_prior
        self.lam_prior_vars = lam_prior_vars
        self.lik = lik
        self.custom_prox = custom_prox
        self.custom_nll_grad = custom_nll_grad
        self.custom_prior = custom_prior

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        P = X.shape[1]

        prior_map = {
            'adaptive': adaptive_prior,
            'group': group_prior,
            'hier': hier_prior
        }
        prior_func = prior_map[self.lam_prior]
        
        if self.lam_prior_vars is None:
            if self.lam_prior == 'group':
                G = int(np.ceil(np.sqrt(P)))  # P must be known here
                lam_prior_vars = {
                    'log_gamma': np.zeros(G),
                    'groups': np.repeat(np.arange(G), P // G + 1)[:P]
                }
            elif self.lam_prior == 'hier':
                G = P  
                lam_prior_vars = {
                    'log_gamma': np.zeros(G)
                }
            else:
                lam_prior_vars = {}
        else:
            lam_prior_vars = self.lam_prior_vars

        if 'groups' in lam_prior_vars:
            lam_prior_vars['groups'] = np.asarray(lam_prior_vars['groups']).astype(int)

        self.model_ = jax_vlMAP(
            X=X,
            y=y,
            lam_prior=prior_func,
            lam_prior_vars=lam_prior_vars,
            lik=self.lik,
            tau0=self.tau0,
            mb_size=self.mb_size,
            invlam=self.invlam,
            logprox=self.logprox,
            track=self.track,
            custom_prox=self.custom_prox,
            custom_nll_grad=self.custom_nll_grad,
            custom_prior=self.custom_prior
        )
        self.model_.fit(max_iters=self.max_iters, lr_pre=self.lr_pre, verbose=False)
        return self

    def predict(self, X):
        if self.lik == 'cauchy':
            # For Cauchy, use clipped location parameter directly instead of mean()
            X_arr = np.asarray(X)
            beta = self.model_.vv['beta']
            beta0 = self.model_.vv['beta0']
            # Calculate predictions and clip to avoid extreme values
            preds = X_arr @ beta + beta0
            return np.clip(np.array(preds), -10, 10)
        else:
            # Use normal prediction for other likelihoods
            return np.array(self.model_.predictive(np.asarray(X)).mean())

    def score(self, X, y):
        y_pred = self.predict(X)
        y = np.asarray(y)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def get_coefficients(self):
        return np.array(self.model_.beta)
