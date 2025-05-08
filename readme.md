# jax_model_lib

`jax_model_lib` is a sparse Bayesian regression library implemented in JAX with support for adaptive and hierarchical priors. It includes an `sklearn`-compatible wrapper for integration into standard machine learning pipelines.

---

## Installation

1. Unzip the package into your project directory.

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Directory Structure

```
your_project/
├── jax_model_lib/
│   ├── __init__.py
│   ├── core.py
│   ├── priors.py
│   └── jax_sklearn_adapter.py
├── your_script.py
```

---

## Basic Usage

### Import and Build Model

```python
from jax_model_lib import jax_vlMAP, adaptive_prior
import numpy as np

X = np.random.randn(100, 10)
true_beta = np.zeros(10)
true_beta[:3] = [2.0, -1.5, 0.5]
y = X @ true_beta + np.random.randn(100) * 0.5

lam_prior_vars = {'log_gamma': np.zeros(10)}

model = jax_vlMAP(
    X=X,
    y=y,
    lam_prior=adaptive_prior,
    lam_prior_vars=lam_prior_vars,
    lik='normal',
    tau0=1.0,
    mb_size=20
)

model.fit(max_iters=300, lr_pre=0.01)
y_pred = model.predictive(X).mean()
```

---

## Scikit-Learn Pipeline Integration

### Example

```python
from jax_sklearn_adapter import JaxSparseRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('jax_model', JaxSparseRegressor())
])

pipe.fit(X, y)
y_pred = pipe.predict(X)
r2 = pipe.score(X, y)
```

---

## Notes

- `adaptive_prior`, `group_prior`, and `hier_prior` are available in `jax_model_lib.priors`.
- For Apple Silicon (M1/M2), use `tensorflow-macos==2.12.0` and `tensorflow-metal==0.7.0` with `tensorflow-probability==0.20.1`.
- All computation is done via JAX; training is gradient-based MAP optimization with sparsity-inducing priors.

---

## Requirements

See `requirements.txt` for full list. Key packages:

- `jax`, `jaxlib`
- `tensorflow-probability`
- `scikit-learn`
- `numpy`, `scipy`, `tqdm`, `matplotlib`
