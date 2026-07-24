"""Shared setup for the hour.csv (UCI bike sharing, hourly) experiments: preprocessing,
train/val/test splits, and likelihood-aware evaluation for MAPAdaptiveLasso vs LassoCV.

Target is `casual` (mean ~36, heavily skewed, real zeros at night) — the regime where a
count likelihood should beat a Gaussian, per the day.csv findings.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adaptivelasso.core import expand_X_int, expand_X_quad


def load_hourly(data_path='./hour.csv', target='casual'):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['instant', 'dteday', 'casual', 'registered', 'cnt'])
    y = df[target]
    X = pd.get_dummies(X, columns=['season', 'mnth', 'hr', 'weekday', 'weathersit'],
                       drop_first=True)
    return X, y


def make_splits(X, y, random_state=0, val_frac=0.2):
    """80/20 train/test, then a further val split off the train for tau0 selection.
    Returns scaled arrays (scaler fit on the fit-split only)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=random_state)

    def _np(a):
        return a.to_numpy(dtype=float) if hasattr(a, 'to_numpy') else np.asarray(a, dtype=float)

    scaler = StandardScaler()
    out = dict(
        X_fit=scaler.fit_transform(_np(X_fit)),
        X_val=scaler.transform(_np(X_val)),
        X_test=scaler.transform(_np(X_test)),
        y_fit=_np(y_fit), y_val=_np(y_val), y_test=_np(y_test),
        feature_names=list(X.columns),
    )
    # a second scaler over the full training set, for the final refit after selection
    scaler_full = StandardScaler()
    out['X_train'] = scaler_full.fit_transform(_np(X_train))
    out['X_test_full'] = scaler_full.transform(_np(X_test))
    out['y_train'] = _np(y_train)
    return out


def expand_np(X, expand):
    """Materialize the same expansion the core applies just-in-time (for LassoCV / eval)."""
    X = np.asarray(X)
    Pu = X.shape[1]
    if expand == 'none':
        return X
    ind1 = np.repeat(np.arange(Pu - 1), np.flip(np.arange(Pu - 1)) + 1)
    ind2 = np.concatenate([np.arange(p + 1, Pu) for p in range(Pu)])
    if expand == 'intr':
        return np.asarray(expand_X_int(X, ind1, ind2))
    if expand == 'quad':
        return np.asarray(expand_X_quad(X, ind1, ind2))
    raise ValueError(expand)


def linpred(model_, X_raw):
    """Linear predictor of a fitted jax_vlMAP with just-in-time expansion, in float64 numpy."""
    if getattr(model_, 'quad', False):
        Xe = np.asarray(expand_X_quad(X_raw, model_.ind1_exp, model_.ind2_exp))
    elif getattr(model_, 'intr', False):
        Xe = np.asarray(expand_X_int(X_raw, model_.ind1_exp, model_.ind2_exp))
    else:
        Xe = np.asarray(X_raw)
    return Xe @ np.asarray(model_.vv['beta']) + float(np.asarray(model_.vv['beta0']).ravel()[0])


def eval_adalasso(model_, X_raw, y, y_train_max, lik):
    """Metrics for a fitted jax_vlMAP. The count log-link can blow up on a diverged fit, so the
    linear predictor is capped at log(20*max(y_train)) before exponentiating — this only guards
    eval-time overflow; a genuinely blown-up fit still scores terribly."""
    eta = linpred(model_, X_raw)
    sigma2 = float(np.asarray(model_.vv['sigma2']).ravel()[0])
    if lik == 'normal':
        mu = eta
        sd = np.exp(sigma2 / 2) * np.ones_like(mu)  # sigma2_exp: log-variance
        lo, hi = stats.norm.interval(0.9, loc=mu, scale=sd)
        crps = crps_gaussian(y, mu, sd)
        dev = float(np.sum((y - mu) ** 2))
        nll = float(-np.mean(stats.norm.logpdf(y, loc=mu, scale=np.maximum(sd, 1e-12))))
    else:
        cap = np.log(20.0 * y_train_max)
        eta = np.minimum(eta, cap)
        # floor keeps scipy's poisson/nbinom in their domain when a diverged fit sends eta -> -inf
        mu = np.maximum(np.exp(eta), 1e-9)
        if lik == 'poisson':
            d = stats.poisson(mu)
        elif lik == 'nb':
            r = np.exp(-sigma2)  # sigma2 = -log(r) in sigma2_exp mode
            p = r / (r + mu)     # scipy nbinom: mean = n*(1-p)/p
            d = stats.nbinom(r, p)
        else:
            raise ValueError(lik)
        lo, hi = d.ppf(0.05), d.ppf(0.95)
        crps = crps_count_dist(y, d)
        dev = count_deviance(y, mu, lik, r=None if lik == 'poisson' else np.exp(-sigma2))
        # NB deviance uses the model's OWN r in its saturated term, so a blown-up huge-variance
        # fit can out-score a good sharp one; predictive NLL is proper and does not.
        nll = float(-np.mean(d.logpmf(np.round(y))))
    nnz = int(np.sum(np.asarray(model_.vv['beta']) != 0))
    return _metrics(y, mu, lo, hi, crps, dev, nnz, nll)


def eval_gaussian_preds(y, mu, sd):
    lo, hi = stats.norm.interval(0.9, loc=mu, scale=sd)
    crps = crps_gaussian(y, mu, sd)
    dev = float(np.sum((y - mu) ** 2))
    return y, mu, lo, hi, crps, dev


def _metrics(y, mu, lo, hi, crps, dev, nnz, nll=np.nan):
    return dict(
        rmse=float(np.sqrt(np.mean((y - mu) ** 2))),
        mae=float(np.mean(np.abs(y - mu))),
        medae=float(np.median(np.abs(y - mu))),
        crps=float(np.mean(crps)),
        cov90=float(np.mean((y >= lo) & (y <= hi))),
        deviance=float(dev),
        nll=float(nll),
        nnz=nnz,
    )


def eval_lasso(lasso, X, y, y_fit, X_fit):
    """LassoCV eval with a Normal predictive whose scale is the train RMSE (as in prior work)."""
    mu = lasso.predict(X)
    sd = float(np.sqrt(np.mean((y_fit - lasso.predict(X_fit)) ** 2)))
    lo, hi = stats.norm.interval(0.9, loc=mu, scale=sd)
    crps = crps_gaussian(y, mu, sd)
    dev = float(np.sum((y - mu) ** 2))
    nll = float(-np.mean(stats.norm.logpdf(y, loc=mu, scale=max(sd, 1e-12))))
    nnz = int(np.sum(lasso.coef_ != 0))
    return _metrics(y, mu, lo, hi, crps, dev, nnz, nll)


def crps_gaussian(y, mu, sd):
    sd = np.maximum(np.asarray(sd, dtype=float), 1e-12)
    z = (np.asarray(y) - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_count_dist(y, d, n_samples=400, seed=0):
    """Sample-based CRPS for a scipy count distribution: E|X-y| - 0.5 E|X-X'|."""
    rng = np.random.default_rng(seed)
    s = d.rvs(size=(n_samples,) + np.shape(y), random_state=rng).astype(float)  # (S, n)
    term1 = np.mean(np.abs(s - np.asarray(y)[None, :]), axis=0)
    s2 = rng.permuted(s, axis=0)
    term2 = 0.5 * np.mean(np.abs(s - s2), axis=0)
    return term1 - term2


def count_deviance(y, mu, lik, r=None):
    """Predictive deviance, 2*[ll(sat) - ll(mu)], for tau0 selection (lower is better)."""
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    ylog = np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu), 0.0)
    if lik == 'poisson':
        return float(2 * np.sum(ylog - (y - mu)))
    # NB2 deviance with dispersion r
    return float(2 * np.sum(ylog - (y + r) * np.log((y + r) / (mu + r))))
