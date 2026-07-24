"""Smoke test: SVRG (mb_size < N) + native expansion on hour.csv/casual must run without
shape errors for every lik x expand combo, and lambda_max must work under expansion.
Short fits only — checks plumbing, not accuracy."""
import time
import numpy as np
from adaptivelasso.skcompat import MAPAdaptiveLasso
from hourly_common import load_hourly, make_splits, eval_adalasso

X, y = load_hourly()
sp = make_splits(X, y)
print(f"features={X.shape[1]}  N_fit={sp['X_fit'].shape[0]}  N_val={sp['X_val'].shape[0]}  "
      f"N_test={sp['X_test'].shape[0]}")
print(f"y_fit: mean={sp['y_fit'].mean():.1f} var/mean={sp['y_fit'].var()/sp['y_fit'].mean():.1f} "
      f"zeros={np.mean(sp['y_fit']==0):.3f} max={sp['y_fit'].max():.0f}")

for lik in ['normal', 'poisson', 'nb']:
    for expand in ['none', 'intr', 'quad']:
        t0 = time.time()
        m = MAPAdaptiveLasso(
            tau0=1.0, max_iters=60, lr_pre=0.01, mb_size=512, expand=expand, lik=lik,
            logprox=True, es_patience=int(1e8), n_es=0, random_state=0, snapshot_interval=20,
            prog=False)
        m.fit(sp['X_fit'], sp['y_fit'])
        lm = m.model_.lambda_max()
        res = eval_adalasso(m.model_, sp['X_val'], sp['y_val'], sp['y_fit'].max(), lik)
        mu_pred = m.predict(sp['X_val'][:5])  # exercises the skcompat expansion path
        print(f"[OK] lik={lik:8s} expand={expand:5s} P={m.model_.P:5d} lambda_max={lm:10.3g} "
              f"val_rmse={res['rmse']:9.2f} nnz={res['nnz']:4d} ({time.time()-t0:.1f}s) "
              f"pred_head={np.round(np.asarray(mu_pred[:2]),2)}")
print("SMOKE PASSED")
