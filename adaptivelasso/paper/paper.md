---
title: 'adaptivelasso: A JAX-based Python Library for Adaptive and Structured Sparse Regression'
tags:
  - Python
  - machine learning
  - sparse regression
  - bayesian statistics
  - JAX
  - adaptive lasso
authors:
  - name: Nathan Wycoff
    affiliation: 1
affiliations:
 - name: Department of Mathematics and Statistics, University of Massachusetts, Amherst
   index: 1
 - name: Georgetown University
   index: 2
date: 11 December 2025
bibliography: paper.bib
---

# Summary

In the era of high-dimensional data, model interpretability is often as crucial as predictive performance. Augmenting a smooth cost function with an $l_{1}$ penalty allows analysts to efficiently conduct estimation and variable selection simultaneously. However, the standard $l_{1}$ penalty suffers from bias, where nonzero parameters are underestimated in magnitude. To address this, `adaptivelasso` is a Python library that implements a "one-algorithm-fits-all" framework for debiased variable selection with general sparsity structures [@Wycoff2024].

Based on the theoretical framework proposed by @Wycoff2024, `adaptivelasso` treats penalty coefficients as additional decision variables to be learned in a Maximum a Posteriori (MAP) manner. The library leverages **JAX** [@Bradbury2018] for automatic differentiation and hardware acceleration, offering a **Scikit-Learn** compatible interface that supports a wide range of non-Gaussian likelihoods and complex sparsity structures which are typically difficult to optimize using standard coordinate descent methods [@Wycoff2024].

# Statement of Need

While widely used tools like `glmnet` [@Friedman2010] and `scikit-learn` provide efficient implementations for standard Lasso and Elastic Net problems, they face limitations when dealing with complex, real-world modeling scenarios:

1.  **Limited Likelihood Support:** Standard packages often restrict users to Gaussian, Binomial, or Poisson families. Researchers working with heavy-tailed data (requiring Cauchy likelihoods) or overdispersed count data (requiring Negative Binomial models) often lack access to efficient sparse estimators [@Wycoff2024].
2.  **Bias in Estimation:** The standard $l_{1}$ penalty induces shrinkage that biases coefficient estimates towards zero. While the Adaptive Lasso mitigates this, it typically requires a two-step procedure that is difficult to tune for complex likelihoods [@Zou2006].
3.  **Structured Sparsity Complexity:** Implementing structured sparsity, such as Group Lasso or Hierarchical interactions (e.g., ensuring interaction terms are zero if main effects are zero), often requires specialized solvers or variable duplication strategies that are computationally expensive [@Wycoff2024].

`adaptivelasso` addresses these needs by implementing a joint optimization procedure using proximal gradient methods [@Wycoff2024]. It enables users to deploy sophisticated, bias-corrected sparse regression models through a familiar API.

# Mathematics

The core optimization problem solved by `adaptivelasso` differs from standard penalized regression. Instead of a fixed penalty parameter, we optimize the model parameters $\beta$ and the penalty coefficients $\lambda$ jointly. The objective function is defined as [@Wycoff2024]:

$$
\min_{\beta \in \mathbb{R}^P, \lambda \in \mathbb{R}_+^P} \mathcal{L}(\beta) + \sum_{p=1}^{P} [\tau \lambda_p |\beta_p| - \log \lambda_p] - \log p_\lambda(\lambda)
$$

where $\mathcal{L}(\beta)$ is the negative log-likelihood, $\tau$ is a global regularization strength, and the term $\tau \lambda_p |\beta_p| - \log \lambda_p$ represents a variable-coefficient $l_{1}$ penalty derived from a MAP-Bayesian perspective. This formulation allows the penalty strength to adapt locally to each coefficient, reducing bias for large parameters while maintaining sparsity [@Wycoff2024].

Although $\tau \lambda_p |\beta_p|$ is nonconvex once $\beta$ and $\lambda$ are optimized jointly, convergence guarantees for proximal splitting methods extend to such nonconvex composite objectives [@LiPong2015], which supports the proximal-gradient approach `adaptivelasso` uses to fit this objective.

# Comparison and Features

`adaptivelasso` supports a versatile set of likelihood functions and prior structures. A key advantage is its ability to handle non-standard distributions that are not supported by the standard `LassoCV` estimator in Scikit-Learn.

| Distribution / Feature | `adaptivelasso` | `sklearn.linear_model.LassoCV` |
| :--- | :---: | :---: |
| Gaussian (Normal) | ✅ | ✅ |
| Bernoulli (Binary) | ✅ | ❌$^1$ |
| Poisson (Count) | ✅ | ❌$^2$ |
| Negative Binomial | ✅ | ❌ |
| Cauchy (Robust) | ✅ | ❌ |
| Zero-Inflated NB (ZINB) | ✅ | ❌ |
| **Adaptive Penalty Learning** | ✅ | ❌ |
| **Group/Hierarchical Priors** | ✅ | ❌ |

$^1$ *Requires `LogisticRegression(penalty='l1')`.*
$^2$ *Requires `PoissonRegressor` (no built-in CV path).*

Key features include:
* **Adaptive Priors:** Supports Independent, Group, and Overlapping Group (Hierarchical) priors for structured sparsity [@Wycoff2024].
* **SVRG Optimization:** Automatically enables Stochastic Variance Reduced Gradient (SVRG) optimization for large datasets to ensure fast and stable convergence [@Wycoff2024].
* **Automatic Regularization Scaling:** `auto_tau0_grid()` derives a likelihood-normalized grid of regularization strengths from each model's null-model gradient, removing the need to manually search vastly different $\tau$ scales across likelihoods (which can span several orders of magnitude between, e.g., Gaussian and Poisson models).
* **Robustness:** Includes robust regression capabilities via the Cauchy likelihood, suitable for data with outliers [@Wycoff2024].

# Empirical Illustration

To illustrate the practical benefit of likelihood flexibility, we compare `adaptivelasso` against `LassoCV` on the UCI Bike Sharing hourly dataset [@FanaeeT2014], predicting `casual` rentals — a right-skewed count variable with a substantial fraction of true zeros, for which the Gaussian assumption underlying `LassoCV` is a poor fit. Both models use the same interaction-expanded feature set, with `adaptivelasso`'s regularization strength selected via `auto_tau0_grid()`.

| Model | Likelihood | RMSE $\downarrow$ | CRPS $\downarrow$ | 90% Coverage |
| :--- | :--- | :---: | :---: | :---: |
| `LassoCV` | Gaussian | 19.86 | 10.05 | 0.91 |
| `adaptivelasso` | Gaussian | 19.84 | 10.06 | 0.91 |
| `adaptivelasso` | Poisson | 15.90 | 7.16 | 0.67 |
| `adaptivelasso` | Negative Binomial | 16.46 | 6.64 | 0.90 |

The first two rows isolate the cost of `adaptivelasso`'s joint adaptive-penalty optimization from the benefit of likelihood choice: constrained to the same Gaussian assumption `LassoCV` is built on, `adaptivelasso` matches its accuracy and calibration essentially exactly (RMSE 19.84 vs. 19.86, CRPS 10.06 vs. 10.05, both at 0.91 coverage) — the added flexibility costs nothing when the Gaussian assumption actually holds. The remaining rows isolate the benefit of likelihood choice, something `LassoCV` cannot represent at all: matching a Negative Binomial likelihood to this overdispersed count target reduces RMSE by 17% and CRPS by 34% relative to the Gaussian baseline while remaining well-calibrated (0.90 coverage); Poisson achieves a similar RMSE reduction (20%) but its 90% intervals cover only 67% of held-out observations, reflecting the equidispersion assumption's failure on this data. This is precisely the gap `adaptivelasso` is designed to close: a package restricted to Poisson alone would trade accuracy for miscalibrated uncertainty, and one restricted to Gaussian (like `LassoCV`) cannot access either likelihood's accuracy gain in the first place. This benchmark is reproducible via `hourly_compare.py` in the project repository; a broader empirical evaluation is given in @Wycoff2024.

# Acknowledgements

The authors gratefully acknowledge funding from the Massive Data Institute, the Tech and Public Policy Grant Program, and NSF/NGA grant #2428033, as well as data from the United Nations High Commissioner for Refugees.

# References
