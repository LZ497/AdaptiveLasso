import numpy as np
from scipy.special import expit
from sklearn.metrics import r2_score, mean_squared_error


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
    # Estimated coefficients
    if hasattr(model, 'coef_'):
        est_coefs = model.coef_
    elif hasattr(model, 'get_coefficients'):
        est_coefs = model.get_coefficients()
    else:
        est_coefs = None
    print("Estimated coefficients:", est_coefs)

    # Predictions
    y_pred = model.predict(X_test)

    # R² and RMSE
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Distribution-specific reporting
    error = y_test - y_pred
    if lik == 'normal':
        mad = None
        print("Estimated error variance:", np.var(error))
    elif lik == 'cauchy':
        mad = np.median(np.abs(error - np.median(error)))
        print("Estimated error MAD:", mad)
    elif lik == 'zinb':
        mad = None
        percent_zeros = np.mean(y_pred == 0)
        print("Percent zeros in predictions:", percent_zeros)
    elif lik == 'bernoulli':
        mad = None
        acc = np.mean((y_pred > 0.5) == y_test)
        print(f"Classification accuracy: {acc:.4f}")
    elif lik == 'poisson':
        mad = None
        print(f"Mean predicted count: {np.mean(y_pred):.4f}")
    elif lik == 'nb':
        mad = None
        print(f"Mean predicted count: {np.mean(y_pred):.4f}")

    return {
        'coefs': est_coefs,
        'R2': r2,
        'RMSE': rmse,
        'MAD': mad
    }