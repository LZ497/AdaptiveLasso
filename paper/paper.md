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
  # TODO: Replace placeholder names and complete affiliations before submission.
  - name: Name 1
  - name: Name 2
  - name: Nathan Wycoff
    affiliation: 2
affiliations:
 - name: Department of Mathematics and Statistics, University of Massachusetts, Amherst
   index: 2
date: 11 December 2025
bibliography: paper.bib
---

# Summary

`adaptivelasso` is a Python library for sparse regression when researchers need to select interpretable predictors while accounting for count responses, heavy-tailed observations, or relationships among variables. It combines six likelihoods with adaptive, group, and hierarchical penalty priors in a shared JAX implementation [@Bradbury2018]. Researchers can change the observation model and sparsity structure while retaining the same fitting, prediction, and model-selection workflow.

The library builds on the joint estimation of regression coefficients and penalty weights developed by @Wycoff2024. Our software contributions make combinations of observation models and sparsity structures usable within a common analysis workflow: variance-reduced optimization with expanded features, likelihood-aware regularization search, and repeatable model selection. A scikit-learn-compatible interface [@Pedregosa2011] connects these capabilities to established Python workflows.

# Statement of Need

Sparse regression is useful when prediction must be accompanied by a manageable set of explanatory variables. The Lasso [@Tibshirani1996] performs selection through a common penalty, while the Adaptive Lasso [@Zou2006] allows different amounts of shrinkage for different coefficients. Applied workflows often also need to accommodate overdispersed counts, outliers, or related predictors. Groups and interactions introduce further choices about which variables should be encouraged to enter a model together [@YuanLin2006; @Bien2013].

Combining these requirements creates a software problem: changing a likelihood or sparsity structure can require a different solver, tuning scale, feature representation, and evaluation procedure. `adaptivelasso` targets researchers comparing such combinations within one analysis, and methods researchers extending the underlying objective. Its built-in Gaussian, Bernoulli, Poisson, Negative Binomial, Cauchy, and zero-inflated Negative Binomial likelihoods share independent adaptive, group, and hierarchical penalty components. Structure is encouraged through priors on penalty weights; it does not impose hard group-selection or interaction-heredity constraints.

# Software Design

`adaptivelasso` separates likelihoods and sparsity priors from a shared optimization engine, allowing researchers to change the observation model or structural assumptions without implementing a new solver for each combination. Building on the joint proximal MAP framework of @Wycoff2024, the engine uses JAX differentiation for smooth likelihood and prior terms and a joint proximal update for regression coefficients and penalty weights. We extended this architecture with custom likelihood-gradient, prior, and proximal hooks so that new model components can reuse the fitting machinery. This modularity supports methodological experimentation alongside the built-in regression models.

To support interaction models on larger datasets, we integrated SVRG with on-demand feature expansion. Training inputs remain in their original representation; interaction and quadratic features are constructed for each minibatch. Full-data reference gradients are accumulated over chunks using the same transformation, and prediction applies the corresponding expansion to user inputs. Coordinating these paths allows variance reduction and expanded models to work together while avoiding storage of the entire expanded training matrix. The trade-off is repeated feature construction, and model parameters still grow with the expanded feature count. This design targets large in-memory analyses.

We also developed likelihood-aware regularization search to make model comparison practical. A common absolute penalty grid can miss useful sparsity levels when likelihood scales differ. Our search uses a reference scale derived from coefficient gradients at an intercept-only initialization, with likelihood-specific scale or dispersion estimates. This provides a data-dependent search range rather than a guaranteed sparsification threshold. Scikit-learn-compatible cross-validation and refitting connect this search to standard preprocessing workflows [@Pedregosa2011].

# Comparison and Features

`adaptivelasso` supports a versatile set of likelihood functions and prior structures. We compare it with established sparse regression tools in the Python and R ecosystems: scikit-learn's `LassoCV` [@Pedregosa2011]; R's `glmnet` [@Friedman2010], the standard coordinate-descent solver for penalized GLMs; `ncvreg` [@BrehenyHuang2011], which targets the same estimation-bias problem as the Adaptive Lasso but through nonconvex (MCP/SCAD) penalties rather than jointly-learned weights; its group-selection counterpart `grpreg` [@BrehenyHuang2009]; and `pyglmnet` [@Jas2020], a pure-Python elastic-net GLM library.

Joint estimation requires the optimizer to maintain and update both regression coefficients and penalty weights, including their coupled proximal step. This requirement goes beyond choosing fixed penalty factors in a conventional penalized-regression estimator. A dedicated fitting engine supports that joint state and its structural priors, while the scikit-learn interface reuses established preprocessing and model-selection tools.

The table distinguishes jointly learned penalty weights and soft priors on those weights from fixed penalty factors and group-selection penalties. The scikit-learn column refers specifically to `LassoCV`; related estimators are identified in the notes.

| Distribution / Feature | `adaptivelasso` | `LassoCV` | `glmnet` | `ncvreg` | `grpreg` | `pyglmnet` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Gaussian (Normal) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bernoulli (Binary) | ✅ | ❌$^1$ | ✅ | ✅ | ✅ | ✅ |
| Poisson (Count) | ✅ | ❌$^2$ | ✅ | ✅ | ✅ | ✅ |
| Negative Binomial | ✅ | ❌ | ✅$^6$ | ❌ | ❌ | ❌ |
| Cauchy (Robust) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Zero-Inflated NB (ZINB) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Adaptive Penalty Learning** | ✅ | ❌ | ❌$^3$ | ❌$^4$ | ❌$^4$ | ❌ |
| **Soft Group/Hierarchical Priors on Penalty Weights** | ✅ | ❌ | ❌ | ❌ | ❌$^5$ | ❌$^5$ |

$^1$ *Requires `LogisticRegression(penalty='l1')`.*
$^2$ *Requires `PoissonRegressor` (no built-in CV path).*
$^3$ *`glmnet` accepts a user-supplied, fixed `penalty.factor` per coefficient (enabling a manual two-step adaptive lasso), but does not jointly learn it with $\beta$.*
$^4$ *Debiasing is instead achieved via a nonconvex penalty shape (MCP/SCAD), a fixed-form alternative to jointly learning a per-coefficient weight.*
$^5$ *`grpreg` supports group-selection and bi-level penalties, allowing selection both between and within groups; `pyglmnet` supports group lasso. These penalties differ from the soft group/hierarchical priors on learned penalty weights used here (see the [grpreg documentation](https://pbreheny.github.io/grpreg/reference/grpreg.html) and [pyglmnet documentation](https://github.com/glm-tools/pyglmnet)).*

$^6$ *`glmnet` supports Negative Binomial regression through a family object such as `MASS::negative.binomial(theta = ...)`, with dispersion specified by the user. In `adaptivelasso`, dispersion is estimated during fitting (see the [glmnet family documentation](https://glmnet.stanford.edu/articles/glmnetFamily.html)).*

# Research Workflow Illustration

The repository includes an end-to-end analysis of hourly casual rentals in the UCI Bike Sharing dataset [@FanaeeT2014]. This count response illustrates why likelihood choice should be available within a common sparse-regression workflow. The experiment scripts provide seeded train/validation/test splits, training-only preprocessing, likelihood-specific regularization scaling, validation-based selection, and refitting before test evaluation.

The following results illustrate that workflow with interaction features. RMSE measures point-prediction error; the continuous ranked probability score (CRPS) assesses predictive distributions, with lower values preferred for both metrics. For `LassoCV`, the experiment constructs a Gaussian predictive distribution using the training residual scale.

| Model | Likelihood | RMSE | CRPS |
| :--- | :--- | ---: | ---: |
| `LassoCV` | Gaussian | 19.86 | 10.05 |
| `adaptivelasso` | Gaussian | 19.84 | 10.06 |
| `adaptivelasso` | Poisson | 15.90 | 7.16 |
| `adaptivelasso` | Negative Binomial | 16.46 | 6.64 |

The Gaussian rows have similar scores, while the count-model rows demonstrate the practical value of changing the observation model within the same software. This single dataset does not establish a general advantage of adaptive penalties or separate their effect from likelihood choice. The workflow is implemented in `experiments/hourly_compare.py`, with shared preprocessing and evaluation in `experiments/hourly_common.py`; methodological comparisons are discussed by @Wycoff2024.

# AI Usage Disclosure

OpenAI Codex assisted with manuscript drafting and revision, inspection of the repository's source code and development history, and checking software-comparison claims against official documentation.

<!-- Disclosure remains incomplete. Before submission, the authors must add model/version
information, disclose any additional AI use in software or documentation development,
and describe the human review, editing, and validation actually performed. Do not assert
that these steps are complete until confirmed by the authors.
Other submission TODOs: complete author metadata, add automated correctness tests and CI,
and provide data acquisition instructions, exact tuning flags, and archived results for
the retained hourly benchmark table. -->

# Acknowledgements

The authors gratefully acknowledge funding from the Massive Data Institute, the Tech and Public Policy Grant Program, and NSF/NGA grant #2428033, as well as data from the United Nations High Commissioner for Refugees.

# References
