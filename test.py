import jax
import jax.numpy as jnp
import numpy as np
from jax_model_lib.jax_sklearn_adapter import JaxSparseRegressor

print(jax.devices())

x = jnp.array([1.0, 2.0, 3.0])
print(jax.device_get(x))

# Minimal smoke test
X = np.random.randn(10, 3)
y = np.random.randn(10)
model = JaxSparseRegressor(lam_prior='adaptive', lik='normal', max_iters=2)
model.fit(X, y)
print("Test passed: model fit and predict works.")

# Test hier prior with minimal input
X = np.random.randn(5, 2)
y = np.random.randn(5)
model = JaxSparseRegressor(lam_prior='hier', lik='normal', max_iters=2)
model.fit(X, y)
print("Test passed: hier prior with small input works.")