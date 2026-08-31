"""Final hour.csv/casual benchmark: LassoCV baseline vs MAPAdaptiveLasso
{normal, poisson, nb} x {none, intr, quad}, SVRG (minibatch) mode.

Per adaptive-lasso model: tau0 grid = FRACS * lambda_max(fit split), select by predictive
deviance on the validation split, refit on the full train at frac * lambda_max(full train)
(scale-matched: the same absolute tau0 is relatively stronger on fewer rows), score on test.

Optimizer flags per likelihood come from hourly_flags.json (produced by the tuning stage).

Usage: python -m experiments.hourly_compare [flags_json] [results_csv] [targets...]
  targets: any of {lasso, normal, poisson, nb}; default all (enables one SLURM job per piece).
"""
import json
import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

from adaptivelasso.skcompat import MAPAdaptiveLasso
try:
    from .hourly_common import (load_hourly, make_splits, expand_np, eval_adalasso, eval_lasso)
except ImportError:
    from hourly_common import (load_hourly, make_splits, expand_np, eval_adalasso, eval_lasso)

FLAGS_JSON = sys.argv[1] if len(sys.argv) > 1 else 'hourly_flags.json'
CSV = sys.argv[2] if len(sys.argv) > 2 else 'hourly_compare_results.csv'
TARGETS = sys.argv[3:] if len(sys.argv) > 3 else ['lasso', 'normal', 'poisson', 'nb']

with open(FLAGS_JSON) as f:
    FLAGS = json.load(f)   # per-lik: {lr, mb, iters, snap, fracs: [...]}

X, y = load_hourly()
sp = make_splits(X, y)
YMAX = sp['y_fit'].max()


def save(row):
    pd.DataFrame([row]).to_csv(CSV, mode='a', header=not os.path.exists(CSV), index=False)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


def build(lik, expand, Xf, yf, fl, seed=0):
    m = MAPAdaptiveLasso(
        tau0=1.0, max_iters=0, lr_pre=fl['lr'], mb_size=fl['mb'], expand=expand, lik=lik,
        logprox=fl.get('logprox', True), es_patience=int(1e8), n_es=0, random_state=seed,
        snapshot_interval=fl['snap'], prog=False)
    m.fit(Xf, yf)
    return m


# ---------------- LassoCV baselines ----------------
for expand in ['none', 'intr', 'quad'] if 'lasso' in TARGETS else []:
    t0 = time.time()
    Xf = expand_np(sp['X_train'], expand)
    Xt = expand_np(sp['X_test_full'], expand)
    lasso = LassoCV(cv=5, n_jobs=-1, random_state=0, max_iter=50000).fit(Xf, sp['y_train'])
    res = eval_lasso(lasso, Xt, sp['y_test'], sp['y_train'], Xf)
    save(dict(method='LassoCV', lik='normal', expand=expand, frac=np.nan,
              alpha=float(lasso.alpha_), wall_s=time.time() - t0, err='', **res))

# ---------------- MAPAdaptiveLasso ----------------
for lik in [l for l in ['normal', 'poisson', 'nb'] if l in TARGETS]:
    fl = FLAGS[lik]
    for expand in ['none', 'intr', 'quad']:
        t0 = time.time()
        try:
            lam_max_fit = build(lik, expand, sp['X_fit'], sp['y_fit'], fl).model_.lambda_max()
            devs = {}
            for frac in fl['fracs']:
                m = build(lik, expand, sp['X_fit'], sp['y_fit'], fl)
                m.model_.set_tau0(frac * lam_max_fit)
                m.model_.fit(max_iters=fl['iters'], lr_pre=fl['lr'], verbose=False,
                             ada=True, tol=1e-10)
                vres = eval_adalasso(m.model_, sp['X_val'], sp['y_val'], YMAX, lik)
                # select on held-out predictive NLL: proper scoring rule, unlike deviance
                # (whose per-model saturated term lets blown-up high-dispersion fits win)
                devs[frac] = vres['nll']
                print(f"  [{lik}/{expand}] frac={frac:.3g} val_nll={devs[frac]:.5g} "
                      f"val_rmse={vres['rmse']:.1f} nnz={vres['nnz']}", flush=True)
            best_frac = min(devs, key=devs.get)
            # refit on the full training set, scale-matched to its own lambda_max
            mf = build(lik, expand, sp['X_train'], sp['y_train'], fl)
            lam_max_full = mf.model_.lambda_max()
            mf.model_.set_tau0(best_frac * lam_max_full)
            mf.model_.fit(max_iters=fl['iters'], lr_pre=fl['lr'], verbose=False,
                          ada=True, tol=1e-10)
            res = eval_adalasso(mf.model_, sp['X_test_full'], sp['y_test'],
                                sp['y_train'].max(), lik)
            save(dict(method='AdaLasso', lik=lik, expand=expand, frac=best_frac,
                      alpha=np.nan, wall_s=time.time() - t0, err='', **res))
        except Exception as e:
            traceback.print_exc()
            save(dict(method='AdaLasso', lik=lik, expand=expand, frac=np.nan, alpha=np.nan,
                      wall_s=time.time() - t0, err=repr(e), rmse=np.nan, mae=np.nan,
                      medae=np.nan, crps=np.nan, cov90=np.nan, deviance=np.nan, nnz=-1))

print("COMPARE DONE")
