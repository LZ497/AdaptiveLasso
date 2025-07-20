import numpy as np
from scipy.special import expit
from sklearn.metrics import (
        mean_squared_error, r2_score, accuracy_score,
        roc_auc_score, log_loss, f1_score
    )

def random_sparsity(P, Pnz):
    """Return a binary mask of length P with Pnz nonzeros at random positions."""
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
            Pnz = max(1, P // 5)
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
    Evaluate and report model performance based on the specified likelihood type.

    Parameters
    ----------
    model : object
        Fitted model with either `predict()` or `predict_proba()` method,
        and optionally `coef_` or `get_coefficients()` for inspecting estimated coefficients.

    X_test : array-like
        Test feature matrix.

    y_test : array-like
        Ground truth target values.

    lik : str
        Likelihood type used during model training. Supported values:
        - 'normal'    : regression
        - 'bernoulli' : binary classification
        - 'cauchy'    : robust regression
        - 'zinb'      : zero-inflated count regression

    Returns
    -------
    None
        Prints evaluation metrics to stdout.
    """
    # Coefficients
    if hasattr(model, 'coef_'):
        est_coefs = model.coef_
    elif hasattr(model, 'get_coefficients'):
        est_coefs = model.get_coefficients()
    else:
        est_coefs = None
    print("Estimated coefficients:", est_coefs)

    # Predictions
    y_pred = model.predict(X_test)

    if lik == 'normal':
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
        error = y_test - y_pred
        print("Estimated error variance:", np.var(error))

    elif lik == 'bernoulli':
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
        y_label = (y_prob > 0.5).astype(int)
        acc = accuracy_score(y_test, y_label)
        auc = roc_auc_score(y_test, y_prob)
        logloss = log_loss(y_test, y_prob)
        f1 = f1_score(y_test, y_label)
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"LogLoss: {logloss:.4f}")
        print(f"F1 Score: {f1:.4f}")

    elif lik == 'cauchy':
        error = y_test - y_pred
        mad = np.median(np.abs(error - np.median(error)))
        print(f"MAD: {mad:.4f}")

    elif lik == 'zinb':
        percent_zeros = np.mean(y_pred == 0)
        print(f"Percent zeros in predictions: {percent_zeros:.2%}")
    
    elif lik in ['poisson', 'nb']:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = np.mean(np.abs(y_test - y_pred))
        mean_true = np.mean(y_test)
        var_true = np.var(y_test)
        dispersion = var_true / mean_true if mean_true > 0 else np.nan
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Mean of y: {mean_true:.2f}")
        print(f"Variance of y: {var_true:.2f}")
        print(f"Dispersion (Var/Mean): {dispersion:.2f}")