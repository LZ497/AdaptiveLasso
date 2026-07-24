"""Stage-1 optimizer tuning on hour.csv/casual: for each likelihood, sweep lr_pre x mb_size
(SVRG minibatch) at two tau0 strengths (frac of lambda_max), expand='quad' (the hardest case).
Writes one CSV row per fit as it completes.

Usage: python hourly_tune.py [csv_out] [lik ...]
"""
import os
import sys
import time
import traceback
import numpy as np
import pandas as pd

from adaptivelasso.skcompat import MAPAdaptiveLasso
from hourly_common import load_hourly, make_splits, eval_adalasso

CSV = sys.argv[1] if len(sys.argv) > 1 else 'hourly_tune_results.csv'
LIKS = sys.argv[2:] if len(sys.argv) > 2 else ['normal', 'poisson', 'nb']

LR_GRID = {
    'normal': [0.01, 0.03, 0.1, 0.3],
    'poisson': [0.0003, 0.001, 0.003, 0.01],
    'nb': [0.0003, 0.001, 0.003, 0.01],
}
MB_GRID = [256, 1024, 4096]
FRACS = [0.05, 0.5]
MAX_ITERS = 3000
SNAP = 20
EXPAND = 'quad'

X, y = load_hourly()
sp = make_splits(X, y)
print(f"N_fit={sp['X_fit'].shape[0]} features={X.shape[1]}", flush=True)


def build(lik, expand, mb, seed=0):
    m = MAPAdaptiveLasso(
        tau0=1.0, max_iters=0, lr_pre=0.01, mb_size=mb, expand=expand, lik=lik,
        logprox=True, es_patience=int(1e8), n_es=0, random_state=seed,
        snapshot_interval=SNAP, prog=False)
    m.fit(sp['X_fit'], sp['y_fit'])  # max_iters=0: builds the core model, runs no iterations
    return m


def run_one(lik, mb, lr, frac, lam_max, iters=MAX_ITERS):
    t0 = time.time()
    m = build(lik, EXPAND, mb)
    core = m.model_
    core.set_tau0(frac * lam_max)
    try:
        core.fit(max_iters=iters, lr_pre=lr, verbose=False, ada=True, tol=1e-10)
        it_last = getattr(core, 'last_it', iters - 1)
        cost_last = core.costs[it_last]
        diverged = not np.isfinite(cost_last)
        # slope of the last 20% of the (finite) cost trace, per 100 iters, relative to |cost|
        cc = core.costs[:it_last + 1]
        cc = cc[np.isfinite(cc)]
        tail = cc[-max(2, len(cc) // 5):]
        slope = np.polyfit(np.arange(len(tail)), tail, 1)[0] * 100 / max(abs(tail.mean()), 1e-9)
        res = eval_adalasso(core, sp['X_val'], sp['y_val'], sp['y_fit'].max(), lik)
        err = ''
    except Exception as e:
        traceback.print_exc()
        it_last, cost_last, diverged, slope, err = -1, np.nan, True, np.nan, repr(e)
        res = dict(rmse=np.nan, mae=np.nan, medae=np.nan, crps=np.nan, cov90=np.nan,
                   deviance=np.nan, nnz=-1)
    row = dict(lik=lik, expand=EXPAND, mb=mb, lr=lr, frac=frac, tau0=frac * lam_max,
               iters=iters, it_last=it_last, diverged=diverged, cost_last=cost_last,
               tail_slope_pct=slope, wall_s=time.time() - t0, err=err, **res)
    pd.DataFrame([row]).to_csv(CSV, mode='a', header=not os.path.exists(CSV), index=False)
    print(f"lik={lik} mb={mb} lr={lr} frac={frac}: rmse={res['rmse']:.1f} dev={res['deviance']:.4g} "
          f"nnz={res['nnz']} div={diverged} ({row['wall_s']:.0f}s)", flush=True)
    return row


for lik in LIKS:
    lam_max = build(lik, EXPAND, 1024).model_.lambda_max()
    print(f"### {lik}: lambda_max={lam_max:.4g}", flush=True)
    for mb in MB_GRID:
        for lr in LR_GRID[lik]:
            for frac in FRACS:
                run_one(lik, mb, lr, frac, lam_max)
print("TUNE STAGE 1 DONE")
