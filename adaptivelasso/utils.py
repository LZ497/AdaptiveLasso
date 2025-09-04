import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score, accuracy_score, roc_auc_score, f1_score,
    mean_poisson_deviance
)
from pprint import pprint



def random_sparsity(P, Pnz):
    """Return a binary mask of length P with Pnz nonzeros at random positions."""
    Pnz = int(np.asarray(Pnz).item())  # Ensure integer
    mask = np.zeros(P)
    mask[:Pnz] = 1
    np.random.shuffle(mask)
    return mask

def generate_data(
    lik, N, P, Pnz=None, beta=None, sigma=1.0, random_state=None,
    sparsity_type='random', beta_style='random', zero_prob=0.3, max_tries=10
):
    rng = np.random.default_rng(random_state)
    # --- Sparsity pattern ---
    if sparsity_type == 'random':
        if Pnz is None:
            if beta is not None:
                # If beta is provided, use its nonzero count as Pnz
                Pnz = np.count_nonzero(beta)
            else:
                Pnz = max(1, P // 5)
        Pnz = int(Pnz)  # Ensure integer
        nonzero = random_sparsity(P, Pnz)
        X = rng.normal(size=(N, P))
    else:
        raise NotImplementedError("Only 'random' sparsity_type is implemented in this version.")

    # --- Beta style ---
    for attempt in range(max_tries):
        if beta is None:
            if beta_style == 'random':
                beta_dense = rng.normal(size=P)
            elif beta_style == 'floor':
                beta_pre = rng.normal(size=P)
                beta_dense = np.sign(beta_pre) * (0.5 + np.abs(beta_pre))
            else:
                beta_dense = np.ones(P)
            beta_used = beta_dense * nonzero
        else:
            beta_used = beta

        mu = X @ beta_used

        # For some likelihoods, rescale beta for stability
        if lik in ['nb', 'bernoulli']:
            if np.max(np.abs(mu)) > 0:
                beta_used = 5 * beta_used / np.max(np.abs(mu))
                mu = X @ beta_used

        # --- Generate y according to likelihood ---
        if lik == 'normal':
            y = mu + sigma * rng.normal(size=N)
        elif lik == 'cauchy':
            y = mu + sigma * rng.standard_cauchy(size=N)
        elif lik == 'bernoulli':
            probs = expit(mu)
            y = rng.binomial(1, probs)
        elif lik == 'poisson':
            lam = np.exp(mu)
            y = rng.poisson(lam)
        elif lik == 'zinb':
            is_zero = rng.uniform(size=N) < zero_prob
            mu_nb = np.exp(mu)
            n = 1 / sigma**2
            p = 1 / (1 + mu_nb * sigma**2)
            y_nb = rng.negative_binomial(n, p, size=N)
            y = np.where(is_zero, 0, y_nb)
        elif lik == 'nb':
            mu_nb = np.exp(mu)
            n = 1 / sigma**2
            p = 1 / (1 + mu_nb * sigma**2)
            y = rng.negative_binomial(n, p, size=N)
        else:
            raise ValueError(f"Unknown likelihood: {lik}")

        # Ensure y is not all zeros or all the same value (for meaningful data)
        if np.sum(y != 0) >= len(y) // 10 and np.std(y) > 1e-6:
            break

    return X, y # Could consider return beta_used (the true coefficient vector used to generate the data) 

def report_model_results(model, X_test, y_test, lik):
    """
    Calculates, prints, and returns a dictionary of performance metrics for a given model.
    
    Parameters
    ----------
    model : scikit-learn compatible model
        The fitted model to evaluate.
    X_test : array-like
        Test features.
    y_test : array-like
        Test target values.
    lik : str
        The likelihood used by the model, which determines which metrics are relevant.
        
    Returns
    -------
    dict
        A dictionary containing calculated performance metrics.
    """
    metrics = {}

    # Get estimated coefficients
    if hasattr(model, 'coef_'):
        metrics['coefs'] = model.coef_
    elif hasattr(model, 'get_coefficients'):
        metrics['coefs'] = model.get_coefficients()
    else:
        metrics['coefs'] = None

    # Get model predictions
    y_pred = model.predict(X_test)

    # --- Calculate all relevant metrics ---

    # For 'normal' and other regression-like likelihoods
    try:
        metrics['R2'] = r2_score(y_test, y_pred)
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics['MAE'] = mean_absolute_error(y_test, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_test, y_pred)
    except Exception:
        metrics['R2'] = -np.inf
        metrics['RMSE'] = np.inf
        metrics['MAE'] = np.inf
        metrics['explained_variance'] = -np.inf

    # Likelihood-specific metrics
    if lik == 'bernoulli':
        try:
            y_pred_class = (y_pred > 0.5).astype(int)
            metrics['accuracy'] = accuracy_score(y_test, y_pred_class)
            metrics['f1'] = f1_score(y_test, y_pred_class)
            if len(np.unique(y_test)) > 1:
                 metrics['AUC'] = roc_auc_score(y_test, y_pred)
        except Exception:
            metrics['accuracy'] = 0
            metrics['f1'] = 0
            metrics['AUC'] = 0

    elif lik in ['poisson', 'nb', 'zinb']:
        try:
            y_pred_clipped = np.maximum(y_pred, 0)
            metrics['poisson_deviance'] = mean_poisson_deviance(y_test, y_pred_clipped)
            metrics['nb_deviance'] = metrics['poisson_deviance'] # Proxy for selection
        except Exception:
            metrics['poisson_deviance'] = np.inf
            metrics['nb_deviance'] = np.inf
    
    elif lik == 'cauchy':
        error = y_test - y_pred
        metrics['MAD'] = np.median(np.abs(error - np.median(error)))

    # --- Print the results for immediate feedback ---
    print("\n--- Model Performance Metrics ---")
    # Exclude the large coefficient array from the printed output for clarity
    metrics_to_print = {k: v for k, v in metrics.items() if k != 'coefs'}
    pprint(metrics_to_print)
    print("-----------------------------\n")

    return metrics