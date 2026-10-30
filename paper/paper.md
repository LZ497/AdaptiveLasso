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
  - name: Linpei Zhang
    equal-contrib: true
    affiliation: 1
  - name: Jui-Ting Chang
    equal-contrib: true
    affiliation: 1
  - name: Nathan Wycoff
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: Independent Researcher
   index: 1
 - name: Department of Mathematics and Statistics, University of Massachusetts, Amherst
   index: 2
date: 30 October 2026
bibliography: paper.bib
---

# Summary

In data-rich research settings, predictive accuracy alone is often insufficient: researchers also need to understand which variables contribute to a model. Variable selection addresses this need by identifying a smaller set of predictors from a large collection, among them sparsity-inducing penalties, such as the Lasso [@Tibshirani1996], which perform selection as part of model fitting. Researchers may also have information about relationships among predictors. For example, several measurements may describe a common process and be expected to enter a model together [@YuanLin2006], or an interaction may be more interpretable when its corresponding main effects are also selected [@Bien2013]. Incorporating these relationships becomes more challenging when responses are counts, observations are heavy-tailed, or interaction terms greatly enlarge the feature space.

Specialized solvers can exploit a particular model and sparsity structure efficiently, but exploring different combinations can require substantial implementation work. @Wycoff2024 develop a proximal optimization framework that jointly estimates regression coefficients and their penalty weights, with priors on those weights encouraging the desired sparsity structure. This provides a common statistical foundation for combining observation models and structural assumptions without deriving a separate structured proximal operator for each prior, but the accompanying code is research-grade, and not optimized for fitting into existing workflows.

`adaptivelasso` develops this framework into a Python library for fitting and comparing sparse regression models. Implemented with JAX [@Bradbury2018], it provides Gaussian, Bernoulli, Poisson, Negative Binomial, Cauchy, and zero-inflated Negative Binomial likelihoods alongside adaptive, group, and hierarchical penalty priors. Its software contributions include coordinating variance-reduced minibatch optimization with on-demand interaction features and introducing likelihood-aware regularization search. A scikit-learn-compatible interface [@Pedregosa2011] connects the fitting engine to familiar preprocessing and model-selection workflows, helping researchers explore alternative model specifications within a single computational framework.

# Statement of Need

Sparse regression is useful when prediction must be accompanied by a manageable set of explanatory variables. The Lasso [@Tibshirani1996] performs selection through a common penalty, while the Adaptive Lasso [@Zou2006] allows different amounts of shrinkage for different coefficients. Applied workflows often also need to accommodate overdispersed counts, outliers, or related predictors. Groups and interactions introduce further choices about which variables should be encouraged to enter a model together [@YuanLin2006; @Bien2013].

Combining these requirements creates a software problem: changing a likelihood or sparsity structure can require a different solver, tuning scale, feature representation, and evaluation procedure. `adaptivelasso` targets researchers comparing such combinations within one analysis, and methods researchers extending the underlying objective. Its built-in Gaussian, Bernoulli, Poisson, Negative Binomial, Cauchy, and zero-inflated Negative Binomial likelihoods share independent adaptive, group, and hierarchical penalty components. Structure is encouraged through priors on penalty weights; it does not impose hard group-selection or interaction-heredity constraints.

# State of the Field

`adaptivelasso` supports a variety of likelihood functions and prior structures. We compare it with established sparse regression tools in the Python and R ecosystems: scikit-learn's `LassoCV` [@Pedregosa2011]; R's `glmnet` [@Friedman2010], the standard coordinate-descent solver for penalized GLMs; `ncvreg` [@BrehenyHuang2011], which targets the same estimation-bias problem as the Adaptive Lasso but through nonconvex (MCP/SCAD) penalties rather than jointly-learned weights; its group-selection counterpart `grpreg` [@BrehenyHuang2009]; and `pyglmnet` [@Jas2020], a pure-Python elastic-net GLM library.

Joint estimation requires the optimizer to maintain and update both regression coefficients and penalty weights, including their coupled proximal step, which requires different control structure than traditional penalties. A dedicated fitting engine supports that joint state and its structural priors, while the scikit-learn interface reuses established preprocessing and model-selection tools.

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

# Software Design

`adaptivelasso` is designed around a recurring research task, where the response distribution and sparsity assumptions vary. Its software contribution is an integrated fitting engine that coordinates model components, expanded feature representations, and regularization search. The architecture separates likelihood and prior specification from optimization, using JAX differentiation for smooth terms and the joint coefficient-and-penalty proximal update of @Wycoff2024 for the nonsmooth penalty to facilitate applications of the software. Custom likelihood-gradient and prior interfaces provide extension points for methods researchers. This separation allows new model specifications to reuse the optimizer and feature-handling infrastructure, reducing the implementation work needed to explore alternative sparse models.

A central engineering contribution is coordinating interaction-feature construction with variance-reduced optimization. Expanding all observations in advance can make interaction models expensive to store, even when the original data fit comfortably in memory. We integrated on-demand expansion with both SVRG minibatch updates and chunked full-data reference gradients, and applied the corresponding transformation during prediction. Researchers can therefore supply the original predictors throughout the workflow while fitting models with interaction or quadratic terms. Repeated feature construction trades computation for a smaller expanded training-matrix workspace, a design feature that allows out-of-memory computation for large datasets.

A second contribution addresses the practical cost of moving between likelihoods. Since useful penalty scales can differ substantially, we developed a regularization search based on coefficient gradients at an intercept-only initialization, with likelihood-specific scale or dispersion estimates. In the example workflow, the selected relative penalty is rescaled before refitting on the full training set. Scikit-learn-compatible estimator interfaces connect the fitting engine to existing preprocessing and model-selection tools [@Pedregosa2011], allowing researchers to reuse familiar analysis components alongside the package's specialized optimization.


# Research Impact Statement

The code for @Wycoff2024 has been refactored to this new standard, and we show just one example here. This repository includes an end-to-end analysis of hourly casual rentals in the UCI Bike Sharing dataset [@FanaeeT2014]. This count response illustrates why likelihood choice should be available within a common sparse-regression workflow. The experiment scripts provide seeded train/validation/test splits, training-only preprocessing, likelihood-specific regularization scaling, validation-based selection, and refitting before test evaluation.

The following results illustrate that workflow with interaction features. RMSE measures point-prediction error; the continuous ranked probability score (CRPS) assesses predictive distributions, with lower values preferred for both metrics. For `LassoCV`, the experiment constructs a Gaussian predictive distribution using the training residual scale.

| Model | Likelihood | RMSE | CRPS |
| :--- | :--- | ---: | ---: |
| `LassoCV` | Gaussian | 19.86 | 10.05 |
| `adaptivelasso` | Gaussian | 19.84 | 10.06 |
| `adaptivelasso` | Poisson | 15.90 | 7.16 |
| `adaptivelasso` | Negative Binomial | 16.46 | 6.64 |

The Gaussian rows have similar scores, while the count-model rows demonstrate the practical value of changing the observation model within the same software. This single dataset does not establish a general advantage of adaptive penalties or separate their effect from likelihood choice. The workflow is implemented in `experiments/hourly_compare.py`, with shared preprocessing and evaluation in `experiments/hourly_common.py`; more methodological comparisons are given in @Wycoff2024.

# AI Usage Disclosure

We used AI tools, including Claude Code with Opus 5.5, for code drafting, refactoring, and drafting the report.
We made the core design decisions, and checked and edited the code and text afterwards for accuracy and take full responsibility for correctness of the resulting products.

# Acknowledgements

NW gratefully acknowledges funding from the Massive Data Institute, the Tech and Public Policy Grant Program, and NSF/NGA grant #2428033.
The computational resources for this work were provided by the UMass Amherst’s partnership with the Unity Research Computing Platform, a multi-institutional cluster led by the University of Massachusetts and the University of Rhode Island.

# References
