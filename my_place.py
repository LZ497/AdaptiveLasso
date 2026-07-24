#!/usr/bin/env python3
"""
Compare MAPAdaptiveLasso vs. standard LassoCV on the bike-sharing dataset (day.csv),
using QUADRATIC feature expansion (main effects + all pairwise interactions + squares).

Both models receive the *same* pre-expanded, standardized 464-column design matrix, so
this is a controlled test of the shrinkage method (adaptive per-coefficient penalty vs.
a single global lasso penalty), not a test of who got more features.

Why pre-expand instead of MAPAdaptiveLasso(expand='quad')?
  The library's native expand='quad' path crashes in stochastic/SVRG mode: snapshot_update()
  (core.py) evaluates the full-data gradient on the *un-expanded* self.X against the expanded
  beta -> shape mismatch. Pre-expanding + expand='none' avoids that bug and is also the fairer
  controlled comparison.

MAPAdaptiveLasso is noisy (its minibatch/early-stop sampling is not seeded by random_state),
so its final fit is repeated N_REPEATS times and metrics are reported as mean +/- std.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score, median_absolute_error,
)
from adaptivelasso.skcompat import MAPAdaptiveLasso
import tensorflow_probability.substrates.jax as tfp
import properscoring as ps
from scipy import stats

tfpd = tfp.distributions

N_REPEATS = 20  # repeated AdaptiveLasso fits to average out optimizer stochasticity


def quad_expand(Xs):
    """[X, all pairwise interactions (i<j), X^2] -- matches core.expand_X_quad ordering."""
    n, Pu = Xs.shape
    inter = [Xs[:, i] * Xs[:, j] for i in range(Pu) for j in range(i + 1, Pu)]
    inter = np.stack(inter, axis=1) if inter else np.zeros((n, 0))
    return np.concatenate([Xs, inter, Xs * Xs], axis=1)


def load_and_preprocess_data(data_path):
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['instant', 'dteday', 'casual', 'registered', 'cnt'])
    y = df['cnt']
    X = pd.get_dummies(X, columns=['season', 'mnth', 'weekday', 'weathersit'], drop_first=True)
    return X, y


def train_test_preparation(X, y, random_state=0):
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return (X_train.to_numpy(dtype=float), X_test.to_numpy(dtype=float),
            y_train.to_numpy(dtype=float), y_test.to_numpy(dtype=float))


# Converged config for the pre-expanded (464-dim) problem.
# logprox=True -> the -log(lam) barrier is handled analytically by the log-prox operator
# (keeps lam > 0 by construction). random_state is set per-call: fits are now reproducible,
# so the repeated final fits below vary the seed to sample the holdout-split distribution.
MAPA_BASE_PARAMS = dict(
    max_iters=10000, lr_pre=0.1, mb_size=32, es_patience=int(1e8),
    invlam=False, logprox=True, lam_prior='adaptive', expand='none',
    lik='normal', tol=1e-4,
)


def select_tau0_by_cv(X_train, y_train, tau0_grid, n_splits=5, random_state=123):
    """Pick tau0 minimizing mean CV RMSE (mirrors LassoCV's alpha selection)."""
    print(f"\n--- Selecting MAPAdaptiveLasso tau0 by {n_splits}-fold CV "
          f"over {len(tau0_grid)} values (diverged folds -> high RMSE, auto-rejected) ---")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mean_rmse = np.full(len(tau0_grid), np.nan)
    for i, tau0 in enumerate(tau0_grid):
        fold_rmse = []
        for tr, va in kf.split(X_train):
            model = MAPAdaptiveLasso(tau0=tau0, random_state=0, **MAPA_BASE_PARAMS)
            model.fit(X_train[tr], y_train[tr])
            fold_rmse.append(np.sqrt(mean_squared_error(y_train[va], model.predict(X_train[va]))))
        mean_rmse[i] = np.mean(fold_rmse)
        print(f"  tau0={tau0:10.4g}   CV RMSE={mean_rmse[i]:.4f}")
    best = float(tau0_grid[int(np.nanargmin(mean_rmse))])
    print(f"Selected tau0 = {best:.4g}  (CV RMSE={np.nanmin(mean_rmse):.4f})")
    return best


def metrics_for(y_true, y_pred, dist):
    mu = np.asarray(dist.mean())
    sig = np.broadcast_to(np.asarray(dist.stddev()), mu.shape)
    return {
        'RMSE':            np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE':             mean_absolute_error(y_true, y_pred),
        'MedAE':           median_absolute_error(y_true, y_pred),
        'R2':              r2_score(y_true, y_pred),
        'ExplainedVar':    explained_variance_score(y_true, y_pred),
        'CRPS':            float(np.mean(ps.crps_gaussian(y_true, mu=mu, sig=sig))),
        'MeanLogPredDens': float(np.mean(np.asarray(dist.log_prob(y_true)))),
    }


HIGHER_BETTER = {
    'RMSE': False, 'MAE': False, 'MedAE': False,
    'R2': True, 'ExplainedVar': True, 'CRPS': False, 'MeanLogPredDens': True,
}


def main():
    X, y = load_and_preprocess_data('./day.csv')
    X_train, X_test, y_train, y_test = train_test_preparation(X, y, random_state=0)

    # Scale raw features, quad-expand, then scale the expanded columns (fair to LassoCV's
    # single global penalty). Both models see this identical matrix.
    sc1 = StandardScaler(); X_train = sc1.fit_transform(X_train); X_test = sc1.transform(X_test)
    Xtr = quad_expand(X_train); Xte = quad_expand(X_test)
    sc2 = StandardScaler(); Xtr = sc2.fit_transform(Xtr); Xte = sc2.transform(Xte)
    print(f"Quad-expanded design matrix: {Xtr.shape[1]} features, {Xtr.shape[0]} train rows")

    # --- MAPAdaptiveLasso: CV-select tau0 once, then repeat final fit to average noise ---
    tau0_grid = np.logspace(-2, 0, 5)
    best_tau0 = select_tau0_by_cv(Xtr, y_train, tau0_grid, n_splits=5)
    print(f"\n--- Fitting MAPAdaptiveLasso {N_REPEATS}x (tau0={best_tau0:.4g}) to quantify noise ---")
    mapa_runs, mapa_preds = [], []
    for r in range(N_REPEATS):
        m = MAPAdaptiveLasso(tau0=best_tau0, track=False, random_state=r, **MAPA_BASE_PARAMS)
        m.fit(Xtr, y_train)
        core = m.model_
        pred = np.asarray(m.predict(Xte))
        met = metrics_for(y_test, pred, core._predictive(Xte, core.vv))
        mapa_runs.append(met); mapa_preds.append(pred)
        print(f"  run {r+1:2d}/{N_REPEATS}: R2_test={met['R2']:.4f}  RMSE={met['RMSE']:.2f}", flush=True)

    # --- LassoCV baseline (deterministic; fit once) ---
    print("\n--- Training standard LassoCV model ---")
    lasso = LassoCV(cv=5, random_state=123, max_iter=20000).fit(Xtr, y_train)
    print(f"LassoCV best alpha = {lasso.alpha_:.4f}")
    preds_lasso = lasso.predict(Xte)
    rmse_train_lasso = np.sqrt(mean_squared_error(y_train, lasso.predict(Xtr)))
    dist_lasso = tfpd.Normal(loc=preds_lasso, scale=rmse_train_lasso)
    m_lasso = metrics_for(y_test, preds_lasso, dist_lasso)

    keys = list(m_lasso.keys())
    mapa_vals = {k: np.array([mr[k] for mr in mapa_runs]) for k in keys}

    # --- Report: metric table with one-sample significance vs the LassoCV constant ---
    # p_1samp: one-sided t-test that E[AdaptiveLasso metric] beats LassoCV, over optimizer noise.
    # %beat: fraction of the N_REPEATS runs that individually beat LassoCV.
    print("\n" + "=" * 100)
    print(f"  QUAD-EXPANDED TEST COMPARISON:  MAPAdaptiveLasso (n={N_REPEATS})  vs  LassoCV")
    print("=" * 100)
    print(f"{'Metric':<16}{'AdaptiveLasso mean±std':>26}{'LassoCV':>12}{'Winner':>16}{'p_1samp':>11}{'%beat':>8}")
    print("-" * 100)
    wins = {'AdaptiveLasso': 0, 'LassoCV': 0, 'tie': 0}
    for k in keys:
        vals, l = mapa_vals[k], m_lasso[k]
        a = float(vals.mean())
        better = (vals > l) if HIGHER_BETTER[k] else (vals < l)
        winner = ('AdaptiveLasso' if (a > l) == HIGHER_BETTER[k] else 'LassoCV')
        wins[winner] += 1
        # one-sided one-sample t-test toward "better"
        t = stats.ttest_1samp(vals, l)
        mean_is_better = (a > l) == HIGHER_BETTER[k]
        p_one = (t.pvalue / 2) if mean_is_better else (1 - t.pvalue / 2)
        note = '↑' if HIGHER_BETTER[k] else '↓'
        print(f"{k:<16}{a:>14.4f} ± {vals.std():<7.4f}{l:>12.4f}{winner:>16}"
              f"{p_one:>11.4f}{100*better.mean():>7.0f}%  {note}")
    print("-" * 100)
    print(f"Metric wins ->  AdaptiveLasso: {wins['AdaptiveLasso']}   "
          f"LassoCV: {wins['LassoCV']}   ties: {wins['tie']}")
    overall = ('MAPAdaptiveLasso' if wins['AdaptiveLasso'] > wins['LassoCV']
               else 'LassoCV' if wins['LassoCV'] > wins['AdaptiveLasso'] else 'TIE')
    print(f">>> OVERALL WINNER (metric count): {overall} <<<")

    # --- Paired per-observation significance on the test set (ensemble-mean prediction) ---
    # Tests whether AdaptiveLasso's per-point errors are significantly below LassoCV's.
    pred_ens = np.mean(mapa_preds, axis=0)
    print("\n" + "-" * 100)
    print(f"Paired per-observation test on the {len(y_test)} test points "
          f"(AdaptiveLasso ensemble mean of {N_REPEATS} fits vs LassoCV):")
    for name, err_a, err_l in [
        ('squared error', (y_test - pred_ens) ** 2, (y_test - preds_lasso) ** 2),
        ('absolute error', np.abs(y_test - pred_ens), np.abs(y_test - preds_lasso)),
    ]:
        # one-sided: is LassoCV error > AdaptiveLasso error (i.e. AdaptiveLasso better)?
        tp = stats.ttest_rel(err_l, err_a, alternative='greater').pvalue
        wp = stats.wilcoxon(err_l, err_a, alternative='greater').pvalue
        mean_impr = 100 * (err_l.mean() - err_a.mean()) / err_l.mean()
        print(f"  {name:<15} mean improvement {mean_impr:+5.1f}%   "
              f"paired t p={tp:.4f}   Wilcoxon p={wp:.4f}")
    print("=" * 100)


if __name__ == '__main__':
    main()
