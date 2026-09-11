# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`adaptivelasso` is a research library for sparse Bayesian regression: gradient-based MAP
estimation with sparsity-inducing (adaptive/group/hierarchical) priors, implemented in JAX
with a scikit-learn-compatible wrapper. The project is installable through `pyproject.toml`;
use an editable install for local development.

## Environment & commands

```bash
python -m pip install -e .             # core package
python -m pip install -r requirements.txt  # package plus example dependencies
python examples/testing_script.py      # end-to-end demo: MAPAdaptiveLasso vs LassoCV on day.csv (bike sharing)
```

- **No test suite / linter / CI.** `examples/testing_script.py` and `examples/my_place.py` are
  runnable demo/scratch scripts (top-level code, not `pytest`); `examples/Usage.ipynb` and
  `examples/compare_lasscv_adaptivelass.ipynb` are the worked examples. Validate changes by running
  relevant examples or experiment smoke scripts, not a test command.
- `quartic_roots.py` imports **numba** (`@jit`), which is declared as a package dependency because
  it is required on the `invlam=True` code path.
- JAX runs in float64 (`jax.config.update("jax_enable_x64", True)` in `core.py`). Keep this — the
  proximal/root-finding math depends on double precision.

## Running code on this HPC (SLURM + conda)

This repo lives on an HPC **login node** — do NOT run compute (JAX/Python) work directly on it.
Everything must run on a **compute node** via SLURM, after loading the conda module and activating
the `AdaptiveLasso` env (at `/work/pi_nwycoff_umass_edu/.conda/envs/AdaptiveLasso`).

Every run must chain the full setup in a **single** command, because each Bash tool call starts a
fresh shell (loaded modules / activated envs do not persist between calls):

```bash
srun --partition=cpu --cpus-per-task=4 --mem=16G --time=01:00:00 \
  bash -lc 'module load conda/latest && conda activate AdaptiveLasso &&
            cd /home/nwycoff_umass_edu/AdaptiveLasso && python examples/testing_script.py'
```

For several commands in a row, allocate once and reuse (avoids re-queueing each time):

```bash
salloc --partition=cpu --cpus-per-task=4 --mem=16G --time=04:00:00 --no-shell   # note the JOBID
srun --jobid=<JOBID> bash -lc 'module load conda/latest && conda activate AdaptiveLasso &&
                               cd /home/nwycoff_umass_edu/AdaptiveLasso && python <script>'
scancel <JOBID>   # release the node when done
```

For long/unattended runs, submit with `sbatch ... --wrap '<same chain>'` and poll `squeue -u $USER`.

**Gotchas:**
- `module` is an Lmod **shell function** → the login shell flag (`bash -lc`, note the `-l`) is required.
- **Never pipe `module load`** (e.g. `module load conda | head`): it runs in a subshell and the PATH
  changes are lost, so `conda` becomes "command not found". Always chain with `&&`.
- `cpu` is the default CPU partition; `cpu-preempt` is faster to schedule but jobs can be preempted.
  `jaxlib==0.4.20` here is a CPU build — use `--partition=gpu --gres=gpu:1` only with a CUDA jaxlib.

## Architecture

Three layers, bottom-up:

1. **`core.py` — `jax_vlMAP`**: the optimizer. This is the heart of the library and the file to read
   first. All optimization state lives in a single dict `self.vv` (`beta`, `lam`, `beta0`, `sigma2`,
   plus prior-specific vars like `log_gamma`). `fit()` runs proximal stochastic gradient descent:
   the smooth part (likelihood NLL + smooth prior terms) is handled by JAX autodiff
   (`eval_nll_grad_subset`, `eval_prior_grad`), and the non-smooth `|beta|` penalty is applied via a
   closed-form **joint proximal step on `(beta, lam)`** — `jax_apply_prox` / `jax_apply_prox_log` /
   `jax_apply_prox_inv` at the bottom of the file.

2. **`priors.py`**: prior penalty functions `adaptive_prior`, `group_prior`, `hier_prior`. Each has
   signature `(x, pv, mod)` where `x` is `lam`, `pv` is `vv`, and `mod` is the `jax_vlMAP` instance
   (so priors can read `mod.N`, `mod.Pu`, `mod.Pi`, `mod.Pq`). Passed into `jax_vlMAP` as the
   `lam_prior` callable.

3. **`skcompat.py`**: sklearn `BaseEstimator`/`RegressorMixin` wrappers — `MAPAdaptiveLasso`
   (main entry point, maps string `lam_prior`→prior func, handles `expand`, built-in `cv`),
   `MAPAdaptiveLassoCV`, and `grid_search_adaptivelasso` (joblib-parallel grid search).

`utils.py` has `generate_data` (synthetic data per likelihood) and `report_model_results` (metrics).
`__init__.py` re-exports `jax_vlMAP`, the three priors, and `quartic_roots`.

Top-level organization:
- `adaptivelasso/`: library code only.
- `examples/`: runnable demos and notebooks.
- `experiments/`: paper benchmark and tuning scripts, including `hourly_compare.py`.
- `paper/`: manuscript source and bibliography.

### Key concepts to keep straight

- **Three λ-penalty modes**, mutually constrained by `logprox` and `invlam` (both `True` is asserted
  invalid). `prox_mode` ∈ {`std`, `log`, `inv`} is derived in `__init__` and selects both the
  non-smooth prior term and which `jax_apply_prox_*` runs. `invlam` uses quartic root-finding
  (`quartic_roots.py`, numba) so that path is not JIT-able.
- **Likelihoods** (`lik`): `normal`, `cauchy`, `bernoulli`, `poisson`, `nb`, `zinb`, and `none`
  (deterministic, for use with `custom_nll_grad`). Each maps to a TFP distribution in `_predictive`.
  `cauchy` has special stability handling (min scale, clipped location) in both `_predictive` and
  `MAPAdaptiveLasso.predict`. `zinb` doubles the coefficient vector (`Pb = 2*P`).
- **SVRG**: enabled automatically when `mb_size < N` (`is_stochastic`). Snapshots every
  `snapshot_interval`; full gradient `g0` recomputed each epoch (`svrg_every = batches_per_epoch`).
  Full-batch when `mb_size is None` and `N < auto_N_sg` (5000).
- **Feature expansion** (`expand` / `intr` / `quad`): interaction and quadratic terms are built
  **just-in-time per minibatch** via `expand_X_int` / `expand_X_quad` (never materialized fully).
  `Pu` = original features, `P` = expanded, `Pi`/`Pq` = interaction/quadratic counts — these feed the
  `hier_prior`. The `hier` prior requires `expand` to be `intr` or `quad`.
- **Early stopping**: a held-out split (`N_es`) is carved out inside `set_Xy`; `best_vv` tracks the
  best params by held-out NLL with `es_patience`.

## Conventions

- `beta` = regression coefficients, `lam` = per-coefficient penalty strengths (also optimized, not a
  scalar), `beta0` = intercept, `sigma2` = noise variance (log-parameterized when `sigma2_exp`,
  which is the default). All are keys in `vv`.
- Custom hooks let you bypass the built-ins: `custom_nll_grad` (with `lik='none'`),
  `custom_prior`, `custom_prox`.
- `examples/testing_script.py` expects the bike-sharing `day.csv` file to be available locally.
