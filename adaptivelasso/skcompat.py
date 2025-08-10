import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from adaptivelasso import jax_vlMAP, adaptive_prior, group_prior, hier_prior
from joblib import Parallel, delayed
from adaptivelasso.utils import report_model_results

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


def grid_search_adaptivelasso(X, y, param_grid, n_splits=5, scoring=None, random_state=42, verbose=True):
    """
    Grid search for MAPAdaptiveLasso with appropriate metrics for different likelihoods.
    
    Parameters
    ----------
    X : array-like
        Features matrix
    y : array-like
        Target vector
    param_grid : dict
        Grid of parameters to search
    n_splits : int, default=5
        Number of cross-validation folds
    scoring : str or None, default=None
        Scoring metric to use. If None, will be auto-selected based on likelihood
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Whether to print progress messages
        
    Returns
    -------
    tuple
        (best_params, best_score, all_results, best_model)
    """
    # Auto-determine scoring based on likelihood if not provided
    lik_values = param_grid.get('lik', ['normal'])
    if isinstance(lik_values, str):
        lik_values = [lik_values]
    lik = lik_values[0]  # get first lik if list
    
    if scoring is None:
        if lik == 'normal':
            scoring = 'r2'
        elif lik == 'bernoulli':
            scoring = 'acc'
        elif lik in ['poisson', 'nb', 'zinb']:
            scoring = 'neg_rmse'
        else:
            scoring = 'r2'  # fallback

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = list(ParameterGrid(param_grid))

    def evaluate_params(params):
        metrics_list = []
        if verbose:
            print(f"Testing params: {params}")
        
        try:
            current_lik = params.get('lik', 'normal')
            
            for train_idx, val_idx in kf.split(X):
                # Handle pandas DataFrames/Series or numpy arrays
                if hasattr(X, "iloc"):
                    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                else:
                    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                # Convert pandas to numpy if needed
                if hasattr(X_train_cv, 'to_numpy'):
                    X_train_cv = X_train_cv.to_numpy()
                    X_val_cv = X_val_cv.to_numpy()
                if hasattr(y_train_cv, 'to_numpy'):
                    y_train_cv = y_train_cv.to_numpy()
                    y_val_cv = y_val_cv.to_numpy()
                
                # Create model without using a pipeline
                model = MAPAdaptiveLasso(**params)
                
                model.fit(X_train_cv, y_train_cv)
                y_pred = model.predict(X_val_cv)
                
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)) or np.any(np.abs(y_pred) > 1e10):
                    if verbose:
                        print("Invalid predictions for params:", params)
                    return None
                
                metrics = report_model_results(model, X_val_cv, y_val_cv, lik=current_lik)
                metrics_list.append(metrics)
                
            # Aggregate metrics based on likelihood type
            if not metrics_list:
                return None
            
            # Common metrics for all likelihoods
            mean_r2 = np.mean([m.get('R2', 0) for m in metrics_list])
            mean_rmse = np.mean([m.get('RMSE', float('inf')) for m in metrics_list])
            
            # Start with common result dictionary
            result = {
                **params,
                'mean_r2': mean_r2,
                'mean_rmse': mean_rmse,
            }
            
            # Add likelihood-specific metrics
            if current_lik == 'normal':
                result['mean_explained_variance'] = np.mean([m.get('explained_variance', 0) for m in metrics_list])
                result['mean_mae'] = np.mean([m.get('MAE', float('inf')) for m in metrics_list])
                
            elif current_lik == 'bernoulli':
                # For binary classification
                result['mean_accuracy'] = np.mean([m.get('accuracy', 0) for m in metrics_list])
                result['mean_auc'] = np.mean([m.get('AUC', 0) for m in metrics_list])
                result['mean_f1'] = np.mean([m.get('F1', 0) for m in metrics_list])
                
            elif current_lik in ['poisson', 'nb', 'zinb']:
                # For count data
                result['mean_mae'] = np.mean([m.get('MAE', float('inf')) for m in metrics_list])
                result['mean_poisson_deviance'] = np.mean([m.get('poisson_deviance', float('inf')) for m in metrics_list])
                result['mean_nb_deviance'] = np.mean([m.get('nb_deviance', float('inf')) for m in metrics_list])
                # Add zero-inflation metrics if available
                if 'zero_inflation' in metrics_list[0]:
                    result['mean_zero_inflation'] = np.mean([m.get('zero_inflation', 0) for m in metrics_list])
            
            return result
        except Exception as e:
            if verbose:
                print(f"Error with params {params}: {str(e)}")
            return None

    results = Parallel(n_jobs=-1)(delayed(evaluate_params)(params) for params in grid)
    all_results = [r for r in results if r is not None]
    
    if not all_results:
        raise ValueError("All parameter combinations failed! Try using 'expand': ['none'] only.")

    # Decide main_metric and whether to maximize or minimize based on scoring and likelihood
    if scoring == 'r2':
        main_metric = 'mean_r2'
        maximize = True
    elif scoring == 'neg_rmse':
        main_metric = 'mean_rmse'
        maximize = False
    elif scoring == 'acc':
        main_metric = 'mean_accuracy'
        maximize = True
    elif scoring == 'neg_mae':
        main_metric = 'mean_mae'
        maximize = False
    elif scoring == 'neg_poisson_deviance':
        main_metric = 'mean_poisson_deviance'
        maximize = False
    elif scoring == 'neg_nb_deviance':
        main_metric = 'mean_nb_deviance'
        maximize = False
    elif scoring == 'f1':
        main_metric = 'mean_f1'
        maximize = True
    elif scoring == 'auc':
        main_metric = 'mean_auc'
        maximize = True
    else:
        # Default to R2 for unknown scoring
        main_metric = 'mean_r2'
        maximize = True

    best_score = -np.inf if maximize else np.inf
    best_params = None

    for result in all_results:
        score = result.get(main_metric, None)
        if score is None:
            continue
        if (maximize and score > best_score) or (not maximize and score < best_score):
            best_score = score
            best_params = {k: result[k] for k in param_grid.keys()}

    # Fit best model on all data - no pipeline/scaler for safety with expand options
    if best_params is None:
        raise ValueError("Could not find best parameters. Check if any parameter combinations succeeded.")
        
    best_model = MAPAdaptiveLasso(**best_params)
    
    # Convert X and y to numpy if needed
    X_np = X.to_numpy() if hasattr(X, 'to_numpy') else X
    y_np = y.to_numpy() if hasattr(y, 'to_numpy') else y
    
    best_model.fit(X_np, y_np)
    return best_params, best_score, all_results, best_model


class MAPAdaptiveLassoCV(BaseEstimator, RegressorMixin):
    """
    Cross-validated MAPAdaptiveLasso with grid search over hyperparameters.
    
    Parameters
    ----------
    param_grid : dict
        Dictionary with parameters names as keys and lists of parameter settings to try as values.
    n_splits : int, default=5
        Number of folds for cross-validation.
    scoring : str or None, default=None
        Scoring metric to use for model selection. If None, will automatically select
        the appropriate metric based on the likelihood:
        - 'r2' for 'normal' likelihood
        - 'acc' for 'bernoulli' likelihood
        - 'neg_rmse' for 'poisson', 'nb', 'zinb' likelihoods
    random_state : int, default=42
        Random state for K-fold cross-validation.
    verbose : bool, default=True
        Whether to print progress during grid search.
    """
    def __init__(
        self,
        param_grid=None,
        n_splits=5,
        scoring=None,  # Changed from 'r2' to None for auto-selection
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
        # best_model_ is now directly a MAPAdaptiveLasso instance, not a pipeline
        return self.best_model_.get_coefficients()
