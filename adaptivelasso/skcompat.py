import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from adaptivelasso import jax_vlMAP, adaptive_prior, group_prior, hier_prior
from sklearn.metrics import roc_auc_score, log_loss, f1_score, r2_score, mean_squared_error

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
        scoring: str, one of ['r2', 'neg_rmse', 'roc_auc', 'neg_log_loss', 'f1']
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

    # Initialize best_score
    if scoring in ['r2', 'f1', 'roc_auc']:
        best_score = -np.inf
    else:  # neg_rmse, neg_log_loss
        best_score = np.inf

    best_params = None
    best_model = None

    for params in grid:
        fold_scores = []
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

            # Predictions
            y_pred = pipe.predict(X_val)

            # For classification metrics, need probabilities
            y_pred_prob = None
            if params.get('lik', 'normal') == 'bernoulli':
                beta = pipe.named_steps['alasso'].get_coefficients()
                intercept = 0.0  # If you have an intercept in MAPAdaptiveLasso, adjust here
                linear_term = X_val @ beta + intercept
                prob = 1 / (1 + np.exp(-linear_term))
                y_pred_prob = prob

            # Compute score
            if scoring == 'r2':
                score = r2_score(y_val, y_pred)
            elif scoring == 'neg_rmse':
                score = -mean_squared_error(y_val, y_pred, squared=False)
            elif scoring == 'roc_auc':
                if y_pred_prob is None:
                    raise ValueError("roc_auc requires lik='bernoulli'")
                score = roc_auc_score(y_val, y_pred_prob)
            elif scoring == 'neg_log_loss':
                if y_pred_prob is None:
                    raise ValueError("neg_log_loss requires lik='bernoulli'")
                score = -log_loss(y_val, y_pred_prob)
            elif scoring == 'f1':
                if y_pred_prob is None:
                    raise ValueError("f1 requires lik='bernoulli'")
                y_pred_label = (y_pred_prob >= 0.5).astype(int)
                score = f1_score(y_val, y_pred_label)
            else:
                raise ValueError(f"Unsupported scoring metric: {scoring}")

            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        result = {**params, 'mean_score': mean_score, 'std_score': np.std(fold_scores)}
        all_results.append(result)

        # Update best model
        if ((scoring in ['r2', 'f1', 'roc_auc'] and mean_score > best_score) or
            (scoring in ['neg_rmse', 'neg_log_loss'] and mean_score < best_score)):
            best_score = mean_score
            best_params = params

    # Fit best model on all data
    best_model = Pipeline([
        ('scale', StandardScaler()),
        ('alasso', MAPAdaptiveLasso(**best_params))
    ])
    best_model.fit(X, y)

    return best_params, best_score, all_results, best_model


class MAPAdaptiveLassoCV(BaseEstimator, RegressorMixin):
    """
    Cross-validated MAPAdaptiveLasso with grid search over hyperparameters.
    """
    def __init__(
        self,
        param_grid=None,
        n_splits=5,
        scoring='r2',
        random_state=42,
        verbose=True
    ):
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose
        self.best_params_ = None
        self.best_score_ = None
        self.all_results_ = None
        self.best_model_ = None

    def fit(self, X, y):
        if self.param_grid is None:
            raise ValueError("param_grid must be specified for MAPAdaptiveLassoCV.")
        results = grid_search_adaptivelasso(
            X, y,
            param_grid=self.param_grid,
            n_splits=self.n_splits,
            scoring=self.scoring,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self.best_params_, self.best_score_, self.all_results_, self.best_model_ = results
        return self

    def predict(self, X):
        if self.best_model_ is None:
            raise RuntimeError("You must fit the model before predicting.")
        return self.best_model_.predict(X)

    def score(self, X, y):
        if self.best_model_ is None:
            raise RuntimeError("You must fit the model before scoring.")
        return self.best_model_.score(X, y)

    def get_coefficients(self):
        if self.best_model_ is None:
            raise RuntimeError("You must fit the model before getting coefficients.")
        # Assumes the last step is 'alasso'
        return self.best_model_.named_steps['alasso'].get_coefficients()
