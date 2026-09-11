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

In this data-rich era, where datasets are often large both in terms of the number of records ("large $N$") and the number of recorded features ("large $P$"), it is often insufficient to merely produce a model with adequate predictive performance.
It is also critical that the behavior of this model be interpretable. 
One strategy for this is to conduct *variable selection*, which involves presenting all $P$ variables to an algorithm but requiring that it choose only a few of these to use in its final model.
Though variable selection can be achieved through many means, in this article we are concerned with *sparsity-inducing penalties*: penalty functions that have a gradient singularity at the origin, allowing variable selection to be conducted as a byproduct of the optimization.
The most popular such penalty is the Lasso [@Tibshirani1996].

Oftentimes, we have information about the relationship between variables that we'd like our method to exploit. 
For instance, some columns of the dataset may encode interaction effects between two or more variables, and we'd only like to select the interaction if we've also selected the main effects [@Bien2013].
Alternatively, subject matter expertise may tell us that certain groups of variables are related and should be selected jointly or not at all [@YuanLin2006].
Expectations of even more sophisticated sparsity structures have been proposed in the form of, e.g., the graphical Lasso [@Friedman2008].
We will not be able here to provide an adequate review of all exciting work which has been conducted in this enormous field. 

In order to devise the most computationally efficient optimizer possible for any of these approaches, it is necessary to exploit the specific structure found in the model's cost function as well as the type of sparsity desired. 
However, this approach makes prototyping complex combinations of models and sparsity structures a labor-intensive process.
This was the motivation for our companion article [@Wycoff2024], in which we introduced an optimization method capable of handling generic combinations of models and sparsity structures, so long as both can be described via a differentiable cost function. 
The core idea is to motivate an adaptive lasso as a Bayesian MAP estimator.
We implement this using a novel proximal gradient method which fits simultaneously the model parameters and the penalty coefficients, with the sparsity structure being driven by a prior density on the penalty coefficients. 

In the present article, we present a wrapper of our method compatible with the **scikit-learn** [@Pedregosa2011] pipeline and based on **JAX** [@Bradbury2018], which allows for straightforward deployment on accelerators and CPUs alike.
This enables inference on a wide variety of problems involving large datasets via a variance-reduced incremental proximal gradient method. 
We hope our code will be useful to practitioners who wish to quickly prototype in settings where all three of the data, model, and sparsity are challenging for classical coordinate-descent methods. 


# Statement of Need

The authors of many existing tools for sparse inference have done excellent work in providing optimized algorithms for specific model classes. 
`glmnet` [@Friedman2010] provides a coordinate-descent optimizer which can be extremely efficient when only a few variables are active, and related coordinate-descent methods have also been implemented for a greater variety of sparsifying penalties by `ncvreg` [@BrehenyHuang2011]. 
Nevertheless, the computational efficiency of these methods is not universal: they require datasets to fit in memory and can be quite slow when many correlated variables are in the model (i.e., if only a moderate sparsity level is required).
In addition, shrinkage directly with the absolute value function produces bias, in that nonzero parameters have smaller magnitudes than is suggested directly by the data, and forces a trade-off between variable selection and estimation accuracy. 
The Adaptive Lasso [@Zou2006] addresses this last point by varying the penalty strengths for each variable in a pre-processing step. 
However, it's not clear how to achieve this efficiently in complex models with non-Gaussian likelihoods and nonlinearities.
Furthermore, proximal gradient methods allow for quite general smooth likelihoods to be deployed, but rely on the derivation of a proximal operator associated with any desired sparsity structure, which may or may not be tractable. 

The present work therefore allows the analyst to work with a much more general class of problems and sparsity structures than the existing literature while avoiding the bias of plain $\ell_1$ regularization, all while using mini-batch SVRG optimization to scale to large in-memory datasets.
In addressing these needs, `adaptivelasso` enables users to deploy sophisticated, bias-corrected sparse regression models through the familiar scikit-learn API.

<!-- While widely used tools like `glmnet`  and `scikit-learn` provide efficient implementations for standard Lasso and Elastic Net problems, they face limitations when dealing with complex, real-world modeling scenarios:

1.  **Limited Likelihood Support:** Standard packages often restrict users to Gaussian, Binomial, or Poisson families. Researchers working with heavy-tailed data (requiring Cauchy likelihoods) or overdispersed count data (requiring Negative Binomial models) often lack access to efficient sparse estimators [@Wycoff2024].
2.  **Bias in Estimation:** The standard $l_{1}$ penalty induces shrinkage that biases coefficient estimates towards zero. While the Adaptive Lasso mitigates this, it typically requires a two-step procedure that is difficult to tune for complex likelihoods .
3.  **Structured Sparsity Complexity:** Implementing structured sparsity, such as Group Lasso or Hierarchical interactions (e.g., ensuring interaction terms are zero if main effects are zero), often requires specialized solvers or variable duplication strategies that are computationally expensive [@Wycoff2024]. -->


# Mathematics

The underlying optimization problem involves both model parameters $\beta\in\mathbb{R}^P$ and an associated penalty coefficient $\lambda_p\in\mathbb{R}_+$ for each $p\in\{1,\ldots,P\}$.
The mathematical program is expressible as:

$$
\underset{\beta \in \mathbb{R}^P, \lambda \in \mathbb{R}_+^P}{\textrm{argmin}}
 \mathcal{L}(\beta) + \sum_{p=1}^{P} [\tau \lambda_p |\beta_p| - \log \lambda_p] - \log P_\lambda(\lambda)
$$

where $\mathcal{L}(\beta)$ is a cost function defining the model, such as a negative log-likelihood, $\tau$ is a global regularization strength, and the term $\tau \lambda_p |\beta_p| - \log \lambda_p$ represents a variable-coefficient $l_{1}$ penalty derived from a MAP-Bayesian perspective.
Finally, the term $\log P_\lambda$ refers to the prior log density of $\lambda$, which is where the desired structural aspects of sparsity are defined.
For instance, we could penalize the deviation of all $\lambda_p$ belonging to a group of variables from their mean, which would incentivize all variables in that group to be jointly zero or nonzero.
This framework incorporates sparsity structure in a ``soft" manner, encouraging it without mandating it. 
This allows for the possibility that data will overwhelm a misspecified sparsity structure, unlike in the classical group lasso setting.
<!-- This formulation allows the penalty strength to adapt locally to each coefficient, reducing bias for large parameters while maintaining sparsity [@Wycoff2024]. -->

# Software Design

The core joint $(\beta, \lambda)$ proximal MAP optimization follows the framework introduced in our companion paper [@Wycoff2024]. Our contribution here is packaging this into `adaptivelasso`: a scikit-learn-compatible `BaseEstimator` that lets the optimizer be dropped into standard pipelines and cross-validation tooling.
Two engineering decisions were central. First, feature expansion (`intr`/`quad`) is applied just-in-time per minibatch rather than materializing the full expanded design matrix, keeping memory use independent of the expansion order.
Second, since different likelihoods require the regularization strength $\tau$ to vary by several orders of magnitude to reach comparable sparsity, we added `auto_tau0_grid()`, which derives a likelihood-normalized grid from each model's null-model gradient rather than requiring users to search likelihood-specific scales by hand.

# Comparison and Features

`adaptivelasso` supports a versatile set of likelihood functions and prior structures. A key advantage is its ability to handle non-standard distributions that are not supported by the standard `LassoCV` estimator in scikit-learn.

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
* **SVRG Optimization:** Automatically enables Stochastic Variance Reduced Gradient (SVRG) optimization for large in-memory datasets to ensure fast and stable convergence [@Wycoff2024].
* **Automatic Regularization Scaling:** `auto_tau0_grid()` derives a likelihood-normalized grid of regularization strengths from each model's null-model gradient, removing the need to manually search vastly different $\tau$ scales across likelihoods (which can span several orders of magnitude between, e.g., Gaussian and Poisson models).
* **Robustness:** Includes robust regression capabilities via the Cauchy likelihood, suitable for data with outliers [@Wycoff2024].

# Empirical Illustration

We now deploy our method on the classical UCI Bike Sharing dataset [@FanaeeT2014], using data at the hourly level to emphasize the impact of non-Gaussianity. 
The number of `casual` renters is an integer-valued count variable with a significant number of zeros, making conditional normality a poor assumption. 
We compare our method under various likelihoods to the LassoCV method from scikit-learn.
For both models, we allow for interaction terms, and select our regularization strength $\tau$ using the `auto_tau0_grid()` method from our package.

| Model | Likelihood | RMSE $\downarrow$ | CRPS $\downarrow$ |
| :--- | :--- | :---: | :---: | 
| `LassoCV` | Gaussian | 19.86 | 10.05 | 
| `adaptivelasso` | Gaussian | 19.84 | 10.06 | 
| `adaptivelasso` | Poisson | 15.90 | 7.16 | 
| `adaptivelasso` | Negative Binomial | 16.46 | 6.64 | 

<!-- The first two rows isolate the cost of `adaptivelasso`'s joint adaptive-penalty optimization from the benefit of likelihood choice: constrained to the same Gaussian assumption `LassoCV` is built on, `adaptivelasso` matches its accuracy and calibration essentially exactly (RMSE 19.84 vs. 19.86, CRPS 10.06 vs. 10.05, both at 0.91 coverage) — the added flexibility costs nothing when the Gaussian assumption actually holds. -->

The first row gives the baseline performance of LassoCV, which uses a Gaussian likelihood, and measures performance in terms of RMSE and CRPS.
The second row gives the performance of our method using the same settings as LassoCV, which has comparable performance, indicating that adaptation is not especially important for this benchmark. 
However, the next two rows show that count models can lead to large improvements in both RMSE and CRPS, with the negative binomial giving slightly better uncertainty estimates (as measured by CRPS) while the Poisson provides slightly better point accuracy in terms of RMSE.
See `experiments/hourly_compare.py` in the project repository to reproduce this example, and see @Wycoff2024 for significantly expanded numerical comparisons.

# Acknowledgements

The authors gratefully acknowledge funding from the Massive Data Institute, the Tech and Public Policy Grant Program, and NSF/NGA grant #2428033, as well as data from the United Nations High Commissioner for Refugees.

# References
