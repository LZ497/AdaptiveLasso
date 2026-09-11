# AdaptiveLasso

`adaptivelasso` is a sparse Bayesian regression library implemented with JAX. It fits adaptive lasso-style MAP estimators and exposes a scikit-learn compatible estimator for use in pipelines and model-selection workflows.

The package supports Gaussian, Bernoulli, Poisson, Negative Binomial, Cauchy, and zero-inflated Negative Binomial likelihoods, along with adaptive, group, and hierarchical priors over penalty coefficients.

## Installation

From a local checkout:

```bash
python -m pip install -e .
```

To install the dependencies used by the example scripts:

```bash
python -m pip install -r requirements.txt
```

JAX installation can be platform-specific for GPU/TPU use. The default package metadata installs the standard CPU-compatible `jax` package. For accelerator builds, install the appropriate JAX extra from the official JAX installation instructions before installing this package.

## Basic Usage

```python
import numpy as np
from adaptivelasso.skcompat import MAPAdaptiveLasso

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 10))
true_beta = np.zeros(10)
true_beta[:3] = [2.0, -1.5, 0.5]
y = X @ true_beta + rng.normal(scale=0.5, size=100)

model = MAPAdaptiveLasso(
    lam_prior="adaptive",
    lik="normal",
    tau0=1.0,
    max_iters=300,
    lr_pre=0.01,
    mb_size=20,
    random_state=0,
)

model.fit(X, y)
y_pred = model.predict(X)
beta = model.get_coefficients()
```

## Scikit-Learn Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adaptivelasso.skcompat import MAPAdaptiveLasso

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", MAPAdaptiveLasso(lik="normal", random_state=0)),
])

pipe.fit(X, y)
y_pred = pipe.predict(X)
r2 = pipe.score(X, y)
beta = pipe.named_steps["model"].get_coefficients()
```

For heavy-tailed likelihoods such as `"cauchy"`, consider `RobustScaler` instead of `StandardScaler`.

`score(X, y)` follows scikit-learn's "higher is better" convention. It returns R2 for Gaussian/Cauchy regression, accuracy for Bernoulli models, negative mean Poisson deviance for Poisson models, and mean predictive log likelihood for Negative Binomial and zero-inflated Negative Binomial models.

## Feature Expansion

`expand` controls optional feature expansion:

- `"none"`: use the input design matrix as-is.
- `"intr"`: add pairwise interaction terms.
- `"quad"`: add pairwise interactions and quadratic terms.

For `lam_prior="hier"`, use `"intr"` or `"quad"` so the hierarchical prior has interaction terms to structure.

## Regularization Selection

The core model exposes `lambda_max()` and `auto_tau0_grid()` for likelihood-aware regularization grids:

```python
model = MAPAdaptiveLasso(lik="poisson", max_iters=0, random_state=0)
model.fit(X, y)
tau_grid = model.model_.auto_tau0_grid(npts=10)
```

For grid search:

```python
from adaptivelasso.skcompat import grid_search_adaptivelasso

param_grid = {
    "tau0": [0.1, 1.0, 10.0],
    "lr_pre": [0.001, 0.01],
    "lik": ["normal"],
}

best_params, best_score, all_results, best_model = grid_search_adaptivelasso(
    X,
    y,
    param_grid,
    n_splits=5,
    random_state=0,
)
```

## Main Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `tau0` | `1.0` | Global regularization strength. |
| `max_iters` | `300` | Maximum optimizer iterations. |
| `lr_pre` | `0.01` | Learning-rate scale for preconditioned updates. |
| `mb_size` | `50` | Mini-batch size. SVRG is used when `mb_size < N`. |
| `snapshot_interval` | `10` | Iterations between SVRG snapshot updates. |
| `logprox` | `True` | Use the log-barrier proximal formulation for positive penalty coefficients. |
| `invlam` | `False` | Use inverse-lambda proximal mode. Cannot be combined with `logprox=True`. |
| `lam_prior` | `"adaptive"` | One of `"adaptive"`, `"group"`, or `"hier"`. |
| `lam_prior_vars` | `None` | Prior-specific variables such as `groups` or `log_gamma`. |
| `lik` | `"normal"` | One of `"normal"`, `"bernoulli"`, `"poisson"`, `"nb"`, `"cauchy"`, or `"zinb"`. |
| `expand` | `"none"` | One of `"none"`, `"intr"`, or `"quad"`. |
| `track` | `False` | Store optimizer diagnostics for plotting. |
| `random_state` | `None` | Random seed for minibatches and early-stop splits. |
| `tol` | `1e-4` | Cost-change tolerance for early stopping. |

For `lam_prior="group"`, provide integer group assignments:

```python
groups = np.array([0, 0, 1, 1, 2, 2])
model = MAPAdaptiveLasso(
    lam_prior="group",
    lam_prior_vars={"log_gamma": np.zeros(3), "groups": groups},
)
```

## Diagnostics

When `track=True`, the fitted core model stores optimizer traces and can plot them:

```python
model = MAPAdaptiveLasso(track=True, random_state=0)
model.fit(X, y)
model.model_.plot("diagnostics.pdf")
```

## Project Layout

```text
adaptivelasso/
  core.py          # JAX optimizer and likelihoods
  priors.py        # adaptive, group, and hierarchical lambda priors
  skcompat.py      # scikit-learn compatible estimators and grid search
  utils.py         # synthetic data and reporting helpers
  quartic_roots.py # proximal helper for inverse-lambda mode
examples/
  Usage.ipynb
  testing_script.py
experiments/
  hourly_compare.py
  hourly_tune.py
  hourly_smoke.py
paper/
  paper.md
  paper.bib
```

The `experiments/hourly_*.py` scripts reproduce the UCI Bike Sharing experiments discussed in the paper. They expect the Bike Sharing data files to be available locally.
