# AdaptiveLasso

Sparse Bayesian regression with JAX and sklearn interface.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

See `compare_loss_adaptiveLoss.ipynb` for examples and test cases.

# adaptivelasso

`adaptivelasso` is a sparse Bayesian regression library implemented in JAX with support for adaptive and hierarchical priors. It includes an `skcompat`-compatible wrapper for integration into standard machine learning pipelines.



## Installation

1. Unzip the package into your project directory.

2. Install required dependencies:

```bash
pip install -r requirements.txt
```



## Directory Structure

```
your_project/
├── adaptivelasso/
│   ├── __init__.py
│   ├── core.py
│   ├── priors.py
│   └── skcompat.py
├── your_script.py
```



## Basic Usage

### Import and Build Model

```python
from adaptivelasso.skcompat import MAPAdaptiveLasso
import numpy as np

X = np.random.randn(100, 10)
true_beta = np.zeros(10)
true_beta[:3] = [2.0, -1.5, 0.5]
y = X @ true_beta + np.random.randn(100) * 0.5

lam_prior_vars = {'log_gamma': np.zeros(10)}

model = MAPAdaptiveLasso(
    X=X,
    y=y,
    lam_prior='adaptive',
    lam_prior_vars=lam_prior_vars,
    lik='normal',
    tau0=1.0,
    mb_size=20
)

model.fit(max_iters=300, lr_pre=0.01)
y_pred = model.predictive(X).mean()
# Retrieve fitted coefficients
beta = model.get_coefficients()
```


## Scikit-Learn Pipeline Integration

### Example

```python
from adaptivelasso.skcompat import MAPAdaptiveLasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('jax_model', MAPAdaptiveLasso())
])

pipe.fit(X, y)
y_pred = pipe.predict(X)
r2 = pipe.score(X, y)
# Retrieve fitted coefficients
beta = pipe.named_steps['jax_model'].get_coefficients()
```

> **Note:** For heavy-tailed likelihoods like `'cauchy'`, use `RobustScaler` instead of `StandardScaler` for better stability.

## Model Evaluation

To evaluate model performance on a test set, use the utility function `report_model_results`:

```python
from adaptivelasso.utils import report_model_results
report_model_results(model, X_test, y_test, lik='normal')
```

### Output by Likelihood Type:

* `lik='normal'`: prints RMSE, R², MAE, and estimated error variance
* `lik='bernoulli'`: prints Accuracy, AUC, LogLoss, F1 Score
* `lik='poisson'` / `lik='nb'`: prints RMSE, MAE, mean, variance, and dispersion (Var/Mean)
* `lik='cauchy'`: prints MAD (Median Absolute Deviation)
* `lik='zinb'`: prints proportion of predicted zeros

The function automatically selects relevant metrics based on the modeling likelihood.


## Parameters


### MAPAdaptiveLasso

| Parameter         | Type          | Default      | Description |
|-------------------|---------------|--------------|-------------|
| `tau0`            | float         | 1.0          | Initial noise precision (inverse variance) of the likelihood. |
| `max_iters`       | int           | 300          | Maximum number of training iterations. |
| `lr_pre`          | float         | 0.01         | Learning rate for the preconditioned gradient updates. |
| `mb_size`         | int or None   | None         | Mini-batch size for stochastic updates. When `mb_size` is a positive integer and N > `mb_size`, SVRG is automatically enabled. If `None`, full-batch gradient descent is used. |
| `snapshot_interval` | int         | 10           | Number of iterations between SVRG snapshot updates. Effective only when `mb_size` is set and N > `mb_size`. Ignored during full-batch training. |
| `invlam`          | bool          | False        | If True, use inverse-λ formulation in optimization. |
| `logprox`         | bool          | True         | Whether to apply log-transform before applying proximal update. |
| `track`           | bool          | False        | If True, track and store training metrics for analysis. |
| `lam_prior`       | str           | 'adaptive'   | Type of prior for λ. Options: `'adaptive'`, `'group'`, `'hier'`. |
| `lam_prior_vars`  | dict or None  | None         | Variables required by selected prior (e.g. `'log_gamma'`, `'groups'`). **If `lam_prior='group'`, you must provide `'groups'` explicitly.** |
| `lik`             | str           | 'normal'     | Likelihood type. Options: `'normal'`, `'bernoulli'`, `'poisson'`, `'nb'`, `'cauchy'`, `'zinb'`. |
| `custom_prox`     | callable or None | None      | Custom proximal operator. Overrides default one if provided. |
| `custom_nll_grad` | callable or None | None      | Custom negative log-likelihood and its gradient. Used when `lik='none'`. |
| `custom_prior`    | callable or None | None      | Custom prior penalty and its gradient. Overrides default one from `lam_prior`. |
| `expand`          | str           | 'none'       | Feature expansion mode. Options: `'none'` (no expansion), `'intr'` (add interaction terms), `'quad'` (add interaction and quadratic terms). If `lam_prior='hier'`, only `'intr'` or `'quad'` are allowed (defaults to `'quad'` with a warning if not set). |
| `cv`              | int or None   | None         | Number of cross-validation folds. If set, cross-validation is performed and the best model is selected. |
| `random_state`    | int or None   | None         | Random seed for reproducibility. |
| `tol`             | float         | 1e-4         | Convergence tolerance for early stopping. |

---

> **Note on λ Penalty Mode Combinations**  
> Only one of `logprox` or `invlam` can be set to `True`. The fourth combination is invalid and explicitly blocked.

| `logprox` | `invlam` | Penalty Formulation| Description |
|-----------|----------|--------------------|-------------|
| `False`   | `False`  | lambda * abs(beta) | Standard Lasso: lambda is a free parameter, jointly optimized with beta. |
| `True`    | `False`  | lambda * abs(beta) + log(lambda) | MAP with log-prior: lambda jointly optimized with log-barrier. |
| `False`   | `True`   | abs(beta) / lambda| Adaptive Lasso: lambda fixed from initial beta estimate and not updated.   |
| `True`    | `True`   | abs(beta) / lambda - log(lambda)| Inverse-lambda MAP form (not implemented; disallowed in code).|

---

> **Note on `lam_prior='group'`**  
> You must provide a `'groups'` array in `lam_prior_vars` that assigns each feature to a group. No default grouping is performed.

> **Note on `expand`**  
> Controls feature expansion:  
> - `'none'`: No expansion (default)  
> - `'intr'`: Adds interaction terms  
> - `'quad'`: Adds interaction and quadratic terms  
> If `lam_prior='hier'`, only `'intr'` or `'quad'` are allowed. If not set, defaults to `'quad'` with a warning.



## Methods

- `fit(X, y)`: Fit the model to data.
- `predict(X)`: Predict target values for new data.
- `score(X, y)`: Return R² score.
- `get_coefficients()`: Return fitted regression coefficients.
- `plot_cost()`: Visualize training objective over iterations (if `track=True`).
- `plot()`: Save full diagnostic plots including final beta bar chart (if `track=True`).



## Attributes

| Attribute     | Description |
|---------------|-------------|
| `model_`      | The fitted `jax_vlMAP` model instance. |
| `beta`        | Fitted regression coefficients. |
| `costs`       | List of objective function values over iterations (if `track=True`). |



## Cross-Validation and Grid Search

- The `cv` parameter enables built-in K-fold cross-validation (like `LassoCV`).
- For full hyperparameter tuning, use the provided `grid_search_adaptivelasso` function:
    ```python
    from adaptivelasso.skcompat import grid_search_adaptivelasso
    param_grid = {'tau0': [0.1, 1.0], 'lr_pre': [0.01, 0.1]}
    best_params, best_score, all_results, best_model = grid_search_adaptivelasso(X, y, param_grid, n_splits=5)
    ```



## Visualization

When `track=True`, calling `.plot()` on the fitted model will output a multi-panel training diagnostic figure including:

- Objective cost over iterations
- Optimality gap (log-scale)
- Coefficient evolution (`beta`) and sparsity ratio
- Lambda dynamics (`lam`) with log-scale
- Auxiliary parameters (e.g., `beta0`, `sigma2`, `log_gamma`)
- Final beta bar plot showing selected features and their importance

## SVRG Optimization

This library supports Stochastic Variance Reduced Gradient (SVRG) for improved optimization stability and convergence. When `mb_size` is set and N > `mb_size`, SVRG is automatically activated.  
During training:
- Every svrg_every = ceil(N / mb_size) iterations, a snapshot of the model parameters is stored.
- Full gradients (`g0`) are computed over the entire dataset.
- Gradients for mini-batches are corrected using the SVRG control variate formula:

```python
sd[v] = -(N/mb_size * (grad_nll[v] - grad_nll0[v]) + g0[v] + grad_prior[v])
```
This improves convergence, especially in settings with high-variance stochastic gradients.  
- Use snapshot_interval to control how frequently snapshots are taken.  
- If mb_size is None, full-batch gradient descent is used instead.

## Notes

- `adaptive_prior`, `group_prior`, and `hier_prior` are available in `jax_model_lib.priors`.
- For Apple Silicon (M1/M2), use `tensorflow-macos==2.12.0` and `tensorflow-metal==0.7.0` with `tensorflow-probability==0.20.1`.
- All computation is done via JAX; training is gradient-based MAP optimization with sparsity-inducing priors.


## Data Generation

- Use `generate_data` from [adaptivelasso.utils](http://_vscodecontentref_/3) to create synthetic datasets for all supported likelihoods (`'normal'`, `'cauchy'`, `'bernoulli'`, `'poisson'`, `'zinb'`, `'nb'`).
- Returns [X, y, beta](http://_vscodecontentref_/4) (where [beta](http://_vscodecontentref_/5) is the true coefficient vector).



## Requirements

See `requirements.txt` for full list. Key packages:

- `jax`, `jaxlib`
- `tensorflow-probability`
- `scikit-learn`
- `numpy`, `scipy`, `tqdm`, `matplotlib`
