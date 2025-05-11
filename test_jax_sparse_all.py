# Generate updated test script for all lam_prior and lik options, with corrections

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from jax_model_lib.jax_sklearn_adapter import JaxSparseRegressor

np.random.seed(0)
N, P = 100, 20
X = np.random.randn(N, P)
true_beta = np.zeros(P)
true_beta[:4] = [3.0, -2.0, 1.5, -1.0]
y = X @ true_beta + np.random.randn(N) * 0.5

# Classification and count variants
y_bin = (y > np.median(y)).astype(np.float32)
y_count = np.random.poisson(lam=np.clip(np.exp(X[:, 0]), 0, 20)).astype(np.float32)
y_real = y.astype(np.float32)

# Custom loss for 'none' case
def custom_mse(vv, Xb, yb):
    import jax.numpy as jnp
    residual = Xb @ vv['beta'] - yb
    loss = jnp.sum(residual ** 2)
    grad = {'beta': 2 * Xb.T @ residual}
    return loss, grad

# Prior var generator for group
def get_group_vars(P):
    G = int(np.ceil(np.sqrt(P)))
    return {
        'log_gamma': np.zeros(G),
        'groups': np.repeat(np.arange(G), P // G + 1)[:P]
    }

lam_prior_list = ['adaptive', 'group']
lik_map = {
    'normal': y_real,
    'bernoulli': y_bin,
    'poisson': y_count,
    'nb': y_count,
    'cauchy': y_real,
    'none': y_real
}

print("Testing JaxSparseRegressor across all lam_prior and lik combinations:")
for lam_prior in lam_prior_list:
    for lik_name, y_used in lik_map.items():
        try:
            print(f"lam_prior='{lam_prior}', lik='{lik_name}'")
            prior_vars = get_group_vars(P) if lam_prior == 'group' else None
            model = JaxSparseRegressor(
                lam_prior=lam_prior,
                lam_prior_vars=prior_vars,
                lik='normal' if lik_name == 'none' else lik_name,
                custom_nll_grad=custom_mse if lik_name == 'none' else None
            )
            pipe = Pipeline([
                ('scale', StandardScaler()),
                ('jax_model', model)
            ])
            pipe.fit(X, y_used)
            y_pred = pipe.predict(X)
            r2 = r2_score(y_used, y_pred)
            mse = mean_squared_error(y_used, y_pred)
            nonzero = np.sum(np.abs(pipe.named_steps['jax_model'].get_coefficients()) > 1e-8)
            print(f"  MSE={mse:.4f}, R²={r2:.4f}, Nonzeros={nonzero}")
        except Exception as e:
            print(f"  Failed with error: {str(e)}")

