import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from adaptivelasso import jax_vlMAP, adaptive_prior, group_prior, hier_prior

class MAPAdaptiveLasso(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        tau0=1.0,
        max_iters=300,
        lr_pre=0.01,
        mb_size=50,
        invlam=False,
        logprox=True,
        track=False,
        lam_prior='adaptive',
        lam_prior_vars=None,
        lik='normal',
        custom_prox=None,
        custom_nll_grad=None,
        custom_prior=None,
        expand='none',  # new argument: 'none', 'intr', 'quad'
        cv=None,
        random_state=None,
        tol=1e-4,
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
        self.expand = expand
        self.cv = cv
        self.random_state = random_state
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        P = X.shape[1]

        if self.cv is None:
            # Standard fit
            prior_map = {
                'adaptive': adaptive_prior,
                'group': group_prior,
                'hier': hier_prior
            }
            prior_func = prior_map[self.lam_prior]

            # Handle expand logic
            expand = self.expand
            if expand not in {'none', 'intr', 'quad'}:
                raise ValueError("expand must be one of {'none', 'intr', 'quad'}")
            if self.lam_prior == 'hier' and expand not in {'intr', 'quad'}:
                print("Warning: lam_prior='hier' requires expand to be 'intr' or 'quad'. Setting expand='quad'.")
                expand = 'quad'

            intr = (expand == 'intr')
            quad = (expand == 'quad')

            # Group prior: require groups to be set
            if self.lam_prior == 'group':
                if self.lam_prior_vars is None or 'groups' not in self.lam_prior_vars:
                    raise ValueError("For lam_prior='group', you must provide 'groups' in lam_prior_vars.")

            # Set lam_prior_vars defaults if needed (except for group, which must be provided)
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
            
            if self.logprox and self.invlam:
                raise ValueError("Invalid configuration: logprox=True and invlam=True is disallowed.")

            if 'log_gamma' in lam_prior_vars:
                lam_prior_vars['log_gamma'] = np.clip(lam_prior_vars['log_gamma'], -10.0, 10.0)

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
                custom_prior=self.custom_prior,
                intr=intr,
                quad=quad
            )
            self.model_.fit(max_iters=self.max_iters, lr_pre=self.lr_pre, verbose=False)
            return self

        # --- Cross-validation logic ---
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        best_score = -np.inf
        best_model = None

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit a model on this fold
            model = MAPAdaptiveLasso(
                tau0=self.tau0,
                max_iters=self.max_iters,
                lr_pre=self.lr_pre,
                mb_size=self.mb_size,
                invlam=self.invlam,
                logprox=self.logprox,
                track=self.track,
                lam_prior=self.lam_prior,
                lam_prior_vars=self.lam_prior_vars,
                lik=self.lik,
                custom_prox=self.custom_prox,
                custom_nll_grad=self.custom_nll_grad,
                custom_prior=self.custom_prior,
                expand=self.expand,
                cv=None,  # Don't recurse!
                random_state=self.random_state,
                tol=self.tol
            )
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            if score > best_score:
                best_score = score
                best_model = model

        # Use the best model found
        self.model_ = best_model.model_
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


def grid_search_adaptivelasso(X, y, param_grid, n_splits=5, scoring='r2', random_state=42, verbose=True):
    """
    Grid search cross-validation for MAPAdaptiveLasso.
    
    Parameters:
        X, y: data
        param_grid: dict, e.g. {'tau0': [0.1, 1.0], 'lr_pre': [0.01, 0.1]}
        n_splits: int, number of CV folds
        scoring: 'r2' or 'neg_rmse'
        random_state: int
        verbose: bool
    Returns:
        best_params: dict of best parameters
        best_score: float
        all_results: list of dicts for each param set
        best_model: fitted MAPAdaptiveLasso instance
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = list(ParameterGrid(param_grid))
    all_results = []
    best_score = -np.inf if scoring == 'r2' else np.inf
    best_params = None
    best_model = None

    for params in grid:
        scores = []
        rmses = []
        if verbose:
            print(f"Testing params: {params}")
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            pipe = Pipeline([
                ('scale', StandardScaler()),
                ('alasso', MAPAdaptiveLasso(**params))
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            r2 = pipe.score(X_val, y_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(r2)
            rmses.append(rmse)
        mean_score = np.mean(scores)
        mean_rmse = np.mean(rmses)
        result = {
            **params,
            'mean_r2': mean_score,
            'std_r2': np.std(scores),
            'mean_rmse': mean_rmse,
            'std_rmse': np.std(rmses)
        }
        all_results.append(result)
        if (scoring == 'r2' and mean_score > best_score) or (scoring == 'neg_rmse' and mean_rmse < best_score):
            best_score = mean_score if scoring == 'r2' else mean_rmse
            best_params = params

    # Fit best model on all data
    best_model = Pipeline([
        ('scale', StandardScaler()),
        ('alasso', MAPAdaptiveLasso(**best_params))
    ])
    best_model.fit(X, y)
    return best_params, best_score, all_results, best_model
