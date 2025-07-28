#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import copy
from time import time
import jax
import jax.numpy as jnp
from tqdm import tqdm
from jax import jit
import matplotlib.pyplot as plt
from functools import partial
from adaptivelasso.quartic_roots import quartic_roots
from tensorflow_probability.substrates import jax as tfp
tfpd = tfp.distributions

# al  _prox

jax.config.update("jax_enable_x64", True)

# Go from a design matrix to the design mat with interaction terms.
@jit
def expand_X_int(Xs, ind1_exp, ind2_exp):
    Xis = Xs[:, ind1_exp] * Xs[:, ind2_exp]
    Xhs = jnp.concatenate([Xs, Xis], axis=1)
    return Xhs

# Go from a design matrix to the design mat with interaction and quadratic terms.
@jit
def expand_X_quad(Xs, ind1_exp, ind2_exp):
    Xis = Xs[:, ind1_exp] * Xs[:, ind2_exp]
    Xqs = Xs*Xs
    #Xqs = Xqs[:, :-n_re]  # Remove quadratic terms for indicator functions
    Xhs = jnp.concatenate([Xs, Xis, Xqs], axis=1)
    return Xhs

@jit
def get_sd_svrg(grad_nll, grad_nll0, g0, grad_prior, mb_size, N):
    sd = {}
    for v in grad_nll:
        sd[v] = -(N/mb_size*(grad_nll[v] - grad_nll0[v]) + g0[v]  + grad_prior[v])
    return sd

@jit
def get_sd(grad, ss):
    sd = {}
    for v in grad:
        sd[v] = -grad[v]*ss[v]
    return(sd)


class jax_vlMAP:
    auto_N_sg = 5000

    def __init__(self, X, y, lam_prior, lam_prior_vars, tau0=1, lik='normal', mb_size = None, snapshot_interval=10,no_adapt = False, invlam = False, track = False, lam_accel = True, l2_coef = 0., dont_pen = None, do_jit = True, logprox = False, beta0_init = None, intr = False, quad = False, N_es = 1000, es_patience = 500, custom_prox=None, custom_prior=None, custom_nll_grad=None):
        self.custom_prox = custom_prox
        self.custom_prior = custom_prior
        self.custom_nll_grad = custom_nll_grad
        self.lik = lik
        self.tau0 = tau0
        #self.big_N = big_N
        assert tau0 > 0
        self.no_adapt = no_adapt
        self.lam_prior_vars = lam_prior_vars if lam_prior_vars is not None else {} 
        self.lam_prior = lam_prior
        self.invlam = invlam
        self.logprox = logprox
        self.track = track
        self.lam_accel = lam_accel
        self.l2_coef = l2_coef
        self.beta0_init = beta0_init 
        self.do_jit = do_jit
        self.snapshot_interval = snapshot_interval

        assert not (self.logprox and self.invlam), "logprox and invlam cannot both be True: incompatible modes of λ treatment."
        if self.invlam:
            self.prox_mode = 'inv'
        elif self.logprox:
            self.prox_mode = 'log'
        else:
            self.prox_mode = 'std'

        assert not (intr and quad)
        self.intr = intr
        self.quad = quad
        if self.quad or self.intr:
            Pu = X.shape[1]
            self.ind1_exp = jnp.repeat(jnp.arange(Pu-1), 1+jnp.flip(jnp.arange(Pu-1)))
            self.ind2_exp = jnp.concatenate([jnp.arange(p+1, Pu) for p in range(Pu)])

        self.N_es = min(N_es, X.shape[0]//10)
        self.es_patience = es_patience # in iterations
        self.set_Xy(X, y)

        # Stochastic gradient settings.
        if mb_size is None:
            if self.N >= jax_vlMAP.auto_N_sg:
                self.mb_size = jax_vlMAP.auto_N_sg
            else:
                self.mb_size = self.N
        else:
            self.mb_size = mb_size
        if self.mb_size >= self.N:
            self.mb_size = self.N
        if self.mb_size == 0:
            self.mb_size = 1

        self.is_stochastic = self.mb_size < self.N

        self.batches_per_epoch = int(np.ceil(self.N / self.mb_size))  # This is about ratio of dataset to mb_size.

        self.svrg_every = self.batches_per_epoch
        self.es_every = self.batches_per_epoch


        self.dont_pen = dont_pen
        if self.dont_pen is not None:
            self.do_pen = np.array(list(set(np.arange(self.Pb)).difference(dont_pen))).astype(int)
        else:
            self.do_pen = np.arange(self.Pb)

        # Params
        self.sigma2_exp = True

        self.lam_prior_vars_init = {}
        for v in self.lam_prior_vars:
            self.lam_prior_vars_init[v] = jnp.copy(self.lam_prior_vars[v])

        self.reset_vars()
        self._compile_costs_updates(do_jit = self.do_jit)

        # Adam parameters.
        self.beta_adam = 0.5
        self.eps_adam = 1e-7

        if self.lik in ['poisson','nb']:
            nbig = 5
            self.biginds = np.argpartition(self.y,-nbig)[-nbig:]
            self.hoardlike = True
        else:
            self.hoardlike = False

    def set_tau0(self, tau0):
        self.tau0 = tau0

    def set_Xy(self, X, y):
        X = np.array(X)
        y = np.asarray(y, dtype=np.float64)  # Always cast y to float
        N, self.Pu = X.shape
        ind_es = np.random.choice(N,self.N_es,replace=False)
        ind_train = np.setdiff1d(np.arange(N), ind_es)
        X_es = X[ind_es,:]
        if self.intr:
            X_es = expand_X_int(X_es, self.ind1_exp, self.ind2_exp)  
        if self.quad:
            X_es = expand_X_quad(X_es, self.ind1_exp, self.ind2_exp)  
        y_es = y[ind_es]
        X_train = X[ind_train,:]
        y_train = y[ind_train]

        self.N = X_train.shape[0]

        if self.quad:
            self.P = 2*self.Pu + int(scipy.special.binom(self.Pu, 2))
        elif self.intr:
            self.P = self.Pu + int(scipy.special.binom(self.Pu, 2))
        else:
            self.P = self.Pu

        if self.lik=='zinb':
            self.Pb=2*self.P
        else:
            self.Pb = self.P

        self.X = np.array(X_train)
        self.y = np.array(y_train, dtype=np.float64)  # Ensure float
        self.X_es = np.array(X_es)
        self.y_es = np.array(y_es, dtype=np.float64)  # Ensure float
        
        if self.lam_prior == 'hier' and not self.lam_prior_vars:
            self.lam_prior_vars = {
            'log_gamma': np.zeros(self.Pi)}
        
    # Define Pi and Pq for hierarchical prior logic
        num_interactions = int(scipy.special.binom(self.Pu, 2)) if self.quad or self.intr else 0
        self.Pi = num_interactions
        self.Pq = self.Pu if self.quad else 0

    def reset_vars(self):
        min_var = 1e-6
        # NOTE: This is indeed X.shape[0], NOT N. (Oh, is it still tho?)
        if self.lik in ['poisson','nb']:
            #beta0 = jnp.mean(jnp.log(self.y+0.5))
            beta0 = jnp.log(jnp.mean(self.y))
        elif self.lik=='zinb':
            iz = np.array(self.y)==0
            beta00 = jnp.mean(iz.astype(np.float64))
            beta0 = jnp.mean(jnp.log(self.y[~iz]+0.5))
        elif self.lik=='cauchy':
            mu_hat = jnp.median(self.y)
            sigma2 = jnp.median(jnp.abs(self.y - mu_hat))
            min_var = 1.0
            if sigma2 < min_var or not jnp.isfinite(sigma2):
                sigma2 = jnp.array(min_var)
            beta0 = mu_hat
            # If using sigma2_exp, take log after clamping
            if self.sigma2_exp:
                sigma2 = jnp.log(jnp.maximum(sigma2, min_var))
        else:
            beta0 = jnp.mean(np.array(self.y).astype(np.float64))

        if self.lik in ['poisson','bernoulli','nb']:
            sigma2 = jnp.array(1.)
        elif self.lik in ['zinb']:
            sigma2 = jnp.array(1.)
        else:
            sigma2 = jnp.var(self.y)
            if sigma2 < min_var:
                print(f"Variance too small ({sigma2}), setting to {min_var}")
                sigma2 = jnp.array(min_var)
            if self.sigma2_exp:
                sigma2 = jnp.log(jnp.maximum(sigma2, min_var))
            if not jnp.isfinite(sigma2) or sigma2 <= 0:
                print("Non-finite or non-positive sigma2 in reset_vars:", sigma2)
                sigma2 = jnp.array(1.0)

        if self.sigma2_exp:
            sigma2 = jnp.log(jnp.maximum(sigma2, min_var))

        beta = jnp.zeros(self.Pb)
        lam = 1.1 * jnp.ones_like(beta)
        if len(lam.shape) == 0:  # Ensure lam is at least 1-dimensional
            lam = lam.reshape([1])

        if self.dont_pen is not None:
            lam = lam.at[self.dont_pen].set(0.) 

        self.vv = {
                'beta':beta,
                'lam':lam,
                'beta0':beta0,
                'sigma2':sigma2}
        if self.lik=='zinb':
            self.vv['beta00']=beta00

        if self.lam_prior_vars is None:
            self.lam_prior_vars = {}  # Ensure it's a dictionary
        if self.lam_prior_vars is None and self.lam_prior == 'adaptive':
            lam_prior_vars = {}
        else:
            lam_prior_vars = self.lam_prior_vars if self.lam_prior_vars is not None else {}

        self.lam_prior_vars_init = {}
        for v in lam_prior_vars:
            if v == 'groups':
                continue  # Do not treat groups as a differentiable parameter
            self.lam_prior_vars_init[v] = jnp.copy(lam_prior_vars[v])
            self.lam_prior_vars[v] = jnp.copy(lam_prior_vars[v])
        self.vv = {**self.vv, **{k: v for k, v in self.lam_prior_vars.items() if k != 'groups'}}

        self.vv = _normalize_shapes(self.vv)
        self.last_beta = np.array(self.vv['beta'])

        self.best_vv = {}
        for v in self.vv:
            self.best_vv[v] = jnp.copy(self.vv[v])

    def _compile_costs_updates(self, do_jit = True):

        def eval_nll_subset(vv, X, y):
            nll = -jnp.sum(self._predictive(X, vv).log_prob(y))
            return nll

        if self.prox_mode == 'std':
                def eval_prior_nonsmooth(vv, tau0):
                    ll = vv['lam']
                    npd_ns = tau0 * jnp.sum(ll * jnp.abs(vv['beta']))
                    return npd_ns

        elif self.prox_mode == 'log':
            def eval_prior_nonsmooth(vv, tau0):
                ll = vv['lam']
                npd_ns = tau0 * jnp.sum(ll * jnp.abs(vv['beta']))
                llp_uni = -jnp.sum(jnp.log(vv['lam'][self.do_pen]))
                return npd_ns + llp_uni

        elif self.prox_mode == 'inv':
            def eval_prior_nonsmooth(vv, tau0):
                ll = 1.0 / vv['lam']
                npd_ns = tau0 * jnp.sum(ll * jnp.abs(vv['beta']))
                return npd_ns

        else:
            raise ValueError(f"Invalid prox_mode: {self.prox_mode}")

        def eval_prior(vv, tau0):
            # Beta prior
            llp_uni = -jnp.sum(jnp.log(vv['lam'][self.do_pen]))
            if self.logprox:
                lam_lap = 0.
            else:
                lam_lap = llp_uni
            if self.invlam:
                lam_lap *= -1

            beta_l2 = self.l2_coef*self.N*jnp.sum(jnp.square(vv['beta']))/2
            
            # Lam prior
            lprior_lam = self.lam_prior(vv['lam'], vv, self)

            # sigma2 "prior"
            if not self.lik in ['poisson','bernoulli']:
                if self.sigma2_exp:
                    lprior_sigma2 = jnp.array(0) # The Reference prior for OLS?
                else:
                    lprior_sigma2 = jnp.log(vv['sigma2']) # The Reference prior for OLS
            else:
                lprior_sigma2 = jnp.array(0)
            prior_smooth = lam_lap + lprior_lam + lprior_sigma2 + beta_l2

            npd_ns = eval_prior_nonsmooth(vv, tau0)
            
            return prior_smooth, npd_ns

        def adam_update_ss_nojit(v_adam, vhat_adam, grad, vv, it):
                #grad = gradlike
            for v in v_adam:
                v_adam[v] = self.beta_adam*v_adam[v] + (1-self.beta_adam)*jnp.square(grad[v])

            for v in v_adam:
                vhat_adam[v] = v_adam[v] / (1-jnp.power(self.beta_adam,it+1))

            ss = {}
            for v in vv:
                ss[v] = 1/jnp.sqrt(vhat_adam[v]+self.eps_adam)

            return v_adam, vhat_adam, ss

        eval_nll_grad_subset = jax.value_and_grad(eval_nll_subset)
        eval_prior_grad = jax.value_and_grad(eval_prior, has_aux = True)

        if do_jit:
            self.eval_nll_grad_subset = jit(eval_nll_grad_subset)
            self.eval_prior_grad = jit(eval_prior_grad)
            self.adam_update_ss = jit(adam_update_ss_nojit)
        else:
            self.eval_nll_grad_subset = jax.value_and_grad(eval_nll_subset)
            self.eval_prior_grad = jax.value_and_grad(eval_prior, has_aux = True)
            self.adam_update_ss = adam_update_ss_nojit

    def init_adam(self):
        return dict([(v,np.zeros_like(self.vv[v])) for v in self.vv]), dict([(v,np.zeros_like(self.vv[v])) for v in self.vv])

    def snapshot_update(self):
        self.snapshot_vv = copy.deepcopy(self.vv)
        _, self.g0 = self.eval_nll_grad_subset(self.snapshot_vv, self.X, self.y)

    def compute_grad_svrg(self, X_batch, y_batch, grad_prior):
        if self.custom_nll_grad:
            _, grad_nll = self.custom_nll_grad(self.vv, X_batch, y_batch)
            _, grad_nll0 = self.custom_nll_grad(self.snapshot_vv, X_batch, y_batch)
        else:
            _, grad_nll = self.eval_nll_grad_subset(self.vv, X_batch, y_batch)
            _, grad_nll0 = self.eval_nll_grad_subset(self.snapshot_vv, X_batch, y_batch)
        sd = get_sd_svrg(grad_nll, grad_nll0, self.g0, grad_prior, self.mb_size, self.N)
        return sd
    
    def get_vv(self, vv, sd, ss, lr, tau0):
        new_vv = {v: vv[v] + lr * ss[v] * sd[v] for v in vv}

        x0 = np.array(jax.device_get(new_vv['beta']))
        lam0 = np.array(jax.device_get(new_vv['lam']))
        sx = np.array(jax.device_get(tau0 * lr * ss['beta']))
        sl = np.array(jax.device_get(tau0 * lr * ss['lam']))

        if self.prox_mode == 'inv':
            new_beta, new_lam = jax_apply_prox_inv(x0, lam0, sx, sl)
        elif self.prox_mode == 'log':
            new_beta, new_lam = jax_apply_prox_log(x0, lam0, sx, sl, 1 / tau0)
        else:
            new_beta, new_lam = jax_apply_prox(x0, lam0, sx, sl)

        new_vv['beta'] = jnp.where(ss['beta'] > 0, new_beta, new_vv['beta'])
        new_vv['lam'] = jnp.where(ss['lam'] > 0, new_lam, new_vv['lam'])
        return new_vv
    
    def fit(self, max_iters=300, lr_pre=1e-2, verbose=True, debug=np.inf, hoardlarge=False, ada=False, warm_up=False, limit_lam_ss=False, prefit=False, tol=1e-4):

        if ada:
            lr = lr_pre
        else:
            lr = lr_pre / self.N

        es_num = int(np.ceil(max_iters / self.es_every))

        ss_ones = {}
        for v in self.vv:
            ss_ones[v] = np.ones_like(self.vv[v])

        if ada:
            self.v_adam = {}
            self.vhat_adam = {}
            for v in self.vv:
                self.v_adam[v] = np.zeros_like(self.vv[v])
                self.vhat_adam[v] = np.zeros_like(self.vv[v])


        self.costs = np.zeros(max_iters)*np.nan
        self.nll_es = np.zeros(es_num)*np.nan

        self.tracking = {}
        track_smol = ['beta','lam']+list(self.lam_prior_vars.keys())
        if self.track:
            for v in self.vv:
                    if (self.quad or self.intr) and v in track_smol:
                        self.tracking[v] = np.zeros((max_iters,self.Pu))
                    else:
                        self.tracking[v] = np.zeros((max_iters,)+self.vv[v].shape)

        for i in tqdm(range(max_iters), disable = not verbose, smoothing = 0.):
            
            if self.is_stochastic and i % self.snapshot_interval == 0:
                self.snapshot_update()

            if self.tracking:
                for v in self.vv:
                    if (self.quad or self.intr) and v in track_smol:
                        self.tracking[v][i] = self.vv[v][:self.Pu]
                    else:
                        self.tracking[v][i] = self.vv[v]

            if i % self.es_every==0:
                nll_es = -np.sum(self.predictive(self.X_es).log_prob(self.y_es))
                if i==0 or (nll_es < np.nanmin(self.nll_es)):
                    for v in self.vv:
                        self.best_vv[v] = jnp.copy(self.vv[v])
                self.nll_es[i//self.es_every] = nll_es
                best_it = np.nanargmin(self.nll_es) * self.es_every
                if i-best_it > self.es_patience:
                    print("ES stop!")
                    break

            if i % self.svrg_every==0:
                g0 = dict([(v, np.zeros_like(self.vv[v])) for v in self.vv])

                self.vv0 = {}
                for v in self.vv:
                    self.vv0[v] = jnp.copy(self.vv[v])

                for iti in tqdm(range(self.batches_per_epoch), leave=False, disable = not verbose):
                    samp_vr = np.arange(iti*self.mb_size,np.minimum((iti+1)*self.mb_size,self.N))
                    Xs_vr = self.X[samp_vr, :]
                    ys_vr = self.y[samp_vr]
                    if self.quad:
                        X_use_vr = expand_X_quad(Xs_vr, self.ind1_exp, self.ind2_exp)  # Create interaction terms just in time.
                    elif self.intr:
                        X_use_vr = expand_X_int(Xs_vr, self.ind1_exp, self.ind2_exp)  # Create interaction terms just in time.
                    else:
                        X_use_vr = Xs_vr  # Create interaction terms just in time.
                    _, grad_vr = self.eval_nll_grad_subset(self.vv0, X_use_vr, ys_vr)
                    for v in self.vv:
                        g0[v] += grad_vr[v]

            ind = np.random.choice(self.N,self.mb_size,replace=False)

            Xs = self.X[ind,:]
            ys = self.y[ind]
            #TODO: DRY violation.
            if self.quad:
                Xs_use = expand_X_quad(Xs, self.ind1_exp, self.ind2_exp)
            elif self.intr:
                Xs_use = expand_X_int(Xs, self.ind1_exp, self.ind2_exp)  
            else:
                Xs_use = Xs
            if self.custom_nll_grad:
                cost_nll, grad_nll = self.custom_nll_grad(self.vv, Xs_use, ys)
            else:
                cost_nll, grad_nll = self.eval_nll_grad_subset(self.vv, Xs_use, ys)
            if self.custom_prior:
                cost_prior, grad_from_prior = self.custom_prior(self.vv['lam'], self.vv, self)
                grad_prior = {}
                for v in self.vv:
                    if v in grad_from_prior:
                        grad_prior[v] = grad_from_prior[v]
                        if len(grad_prior[v].shape) == 0:  # Ensure gradients are at least 1-dimensional
                            grad_prior[v] = grad_prior[v].reshape([1])
                    else:
                        grad_prior[v] = jnp.zeros_like(self.vv[v])
            else:
                cost_prior, grad_prior = self.eval_prior_grad(self.vv, self.tau0)

            self.costs[i] = self.N / self.mb_size * cost_nll + sum(cost_prior)
            # print(f"Iteration {i}, Cost: {self.costs[i]}, NLL: {cost_nll}, Prior: {cost_prior}")
            if not np.isfinite(self.costs[i]):
                print("Infinite cost!")
                # import IPython; IPython.embed()
                break

            if i > 0 and abs(self.costs[i] - self.costs[i-1]) < tol:
                if verbose:
                    print(f"Converged at iteration {i} (|cost change| < tol={tol})")
                self.last_it = i
                break

            #sd = get_sd_svrg(grad_nll, grad_nll0, g0, grad_prior, self.mb_size, self.N)
            if self.is_stochastic and self.vv0 is not None and g0 is not None:
                sd = self.compute_grad_svrg(Xs_use, ys, grad_prior)
            else:
                for k in grad_nll:
                    grad_nll[k] += grad_prior[k]
                sd = grad_nll

            if ada:
                adam_grad = sd
                self.v_adam, self.vhat_adam, ss = self.adam_update_ss(self.v_adam, self.vhat_adam, adam_grad, self.vv, i)
            else:
                ss = ss_ones
            wu_its = 10
            if warm_up and i < wu_its:
                wu_rates = np.logspace(-8,np.log10(lr),num=wu_its)
                lr_use = wu_rates[i]
            else:
                lr_use = lr
            if prefit:
                pf_vars = ['beta','lam']+list(self.lam_prior_vars.keys())
                for v in pf_vars:
                    ss[v] *= 0
            next_vv = self.get_vv(self.vv, sd, ss, lr_use, self.tau0)
            self.vv = next_vv
            if 'sigma2' in self.vv and self.lik == 'cauchy':
                min_var = 1.0  # or 1e-3, but 1.0 is safer for Cauchy
                if not jnp.isfinite(self.vv['sigma2']).all() or (self.vv['sigma2'] <= 0).any():
                    self.vv['sigma2'] = jnp.array(min_var)
                self.vv['sigma2'] = jnp.maximum(self.vv['sigma2'], min_var)
            elif 'sigma2' in self.vv:
                min_var = 1e-6
                if not jnp.isfinite(self.vv['sigma2']).all() or (self.vv['sigma2'] <= 0).any():
                    self.vv['sigma2'] = jnp.array(min_var)
                if self.sigma2_exp:
                    self.vv['sigma2'] = jnp.maximum(self.vv['sigma2'], jnp.log(min_var))
                else:
                    self.vv['sigma2'] = jnp.maximum(self.vv['sigma2'], min_var)
            self.last_it = i

        self.beta = self.vv['beta']

    def _predictive(self, XX, vv):
        if self.sigma2_exp:
            stddev = jnp.exp(vv['sigma2']/2)
        else:
            stddev = jnp.sqrt(vv['sigma2'])

        if self.lik == 'zinb':
            preds_y = XX @ vv['beta'][:self.P] + vv['beta0']
            preds_z = XX @ vv['beta'][self.P:] + vv['beta00']
        else:
            preds = XX @ vv['beta'] + vv['beta0']

        if self.lik == 'normal':
            dist_pred = tfpd.Normal(preds, scale=stddev)
        elif self.lik == 'cauchy':
            # Ensure scale parameter is always large (crucial for Cauchy stability)
            stddev = jnp.maximum(stddev, 1.0)
            
            # Create a standard Cauchy with clipped location parameter
            loc = jnp.clip(preds, -10, 10)  # Clip to avoid extreme predictions
            dist_pred = tfpd.Cauchy(loc=loc, scale=stddev)
        elif self.lik == 'bernoulli':
            dist_pred = tfpd.Bernoulli(logits=preds)
        elif self.lik == 'poisson':
            dist_pred = tfpd.Poisson(log_rate=preds)
        elif self.lik == 'nb':
            dist_pred = tfpd.NegativeBinomial(total_count=1/jnp.square(stddev), logits=preds)
        elif self.lik=='zinb':
            dist_nz = tfpd.NegativeBinomial(total_count=1/jnp.square(stddev), logits = preds_y)
            dist_z = tfpd.Deterministic(loc=jnp.zeros_like(preds_y))
            pnz = tfpd.Categorical(logits=jnp.stack([preds_z,-preds_z]).T)
            dist_pred = tfpd.Mixture(pnz, components=[dist_z, dist_nz])
        elif self.lik == 'none':
            dist_pred = tfpd.Deterministic(loc=preds)
        else:
            raise Exception("Unrecognized Likelihood.")
        return dist_pred

    def big_nll(self, XX, yy):
        if XX.shape[0] <= self.mb_size:
            return self._predictive(XX, self.vv)
        else:
            nll = 0.
            bpt = int(np.ceil(XX.shape[0] / self.mb_size))
            for iti in tqdm(range(bpt), leave=False):
                samp_pred = np.arange(iti*self.mb_size,np.minimum((iti+1)*self.mb_size,XX.shape[0]))
                XXs = XX[samp_pred, :]
                yys = yy[samp_pred]
                # TODO: DRY violation.
                if self.quad:
                    XXs_use = expand_X_quad(XXs, self.ind1_exp, self.ind2_exp)  
                elif self.intr:
                    XXs_use = expand_X_int(XXs, self.ind1_exp, self.ind2_exp)  
                else:
                    XXs_use = XXs
                nll += -jnp.sum(self._predictive(XXs_use, self.vv).log_prob(yys))
        return nll

    def predictive(self, XX):
        return self._predictive(XX, self.vv)


    def plot(self, fname='fit.png'):
        if not self.track:
            return  # no plot if nothing tracked

        tracked_vars = list(self.tracking.keys())
        n_tracked = len(tracked_vars)
        nfigs = 3 + n_tracked + (1 if self.is_stochastic else 0)
        nrows, ncols = int(np.ceil(nfigs / 2)), 2

        fig = plt.figure(figsize=(ncols * 5, nrows * 2))
        plot_idx = 1

        # Plot cost
        ax = fig.add_subplot(nrows, ncols, plot_idx)
        ax.plot(self.costs[:self.last_it+1])
        ax.set_title('Cost')
        ax2 = ax.twinx()
        ax2.plot(self.es_every * np.arange(len(self.nll_es)), self.nll_es, color='orange')
        plot_idx += 1

        # Plot optimality gap
        ax = fig.add_subplot(nrows, ncols, plot_idx)
        gap = np.log10(1 + self.costs[:self.last_it] - np.min(self.costs[:self.last_it]))
        ax.plot(gap)
        ax.set_title('Optimality Gap (log(1+))')
        plot_idx += 1

        # Plot tracked variables
        for v in tracked_vars:
            ax = fig.add_subplot(nrows, ncols, plot_idx)
            ax.plot(self.tracking[v][:self.last_it+1])
            ax.set_title(v)

            if v == 'beta':
                sparsity = np.mean(self.tracking[v][:self.last_it] == 0, axis=1)
                ax2 = ax.twinx()
                ax2.plot(sparsity, color='green', linestyle='--')
                ax2.set_ylabel('Sparsity')

            if v == 'lam':
                ax.set_yscale('log')

            plot_idx += 1

        # Final beta bar plot (feature selection summary)
        ax = fig.add_subplot(nrows, ncols, plot_idx)
        final_beta = self.tracking['beta'][self.last_it]
        x = np.arange(len(final_beta))
        ax.bar(x, final_beta)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_title('Final Beta (Feature Selection)')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticklabels([str(i) for i in x], rotation=90)

        plt.tight_layout()
        plt.show()


def sto(x0,lam,sx):
    return jnp.maximum(0,jnp.abs(x0)-lam*sx)*jnp.sign(x0)

#@jit
def jax_apply_prox(x0, lam0, sx, sl):
    is_convex = sl*sx<1

    nc_sw = (lam0/sl) < (jnp.abs(x0)/sx)
    nonconv_sol = (1-nc_sw) * lam0 # + nc_sw*0

    c_sw = lam0<jnp.abs(x0)/sx
    rect_lam = jnp.maximum(0,lam0)
    conv_sol = jnp.minimum(jnp.maximum(0,lam0-sl*jnp.abs(x0)) / (1-sl*sx), rect_lam)

    lam_sol = jnp.where(is_convex, conv_sol, nonconv_sol)
    # TODO: replace with sto func below.
    x_sol = jnp.maximum(0,jnp.abs(x0)-lam_sol*sx)*jnp.sign(x0)

    return x_sol, lam_sol

def jax_prox_log_cost(x,lam,x0,lam0,sx,sl,c):
    return jnp.abs(x)*lam + jnp.square(x-x0)/(2*sx) + jnp.square(lam-lam0)/(2*sl) - c*jnp.log(lam)

def jax_apply_prox_log(x0, lam0, sx, sl, c):
    ## Check if x is going to be zero.
    rootdisc_iz = jnp.sqrt(jnp.square(lam0) + 4 * c * sl)
    lamizm = jnp.maximum(0,(lam0 - rootdisc_iz) / 2)
    lamizp = jnp.maximum(0,(lam0 + rootdisc_iz) / 2)

    costm = jax_prox_log_cost(0., lamizm, x0, lam0, sx, sl, c)
    costp = jax_prox_log_cost(0., lamizp, x0, lam0, sx, sl, c)
    mwins = costm<costp
    lamiz = jnp.where(mwins, lamizm, lamizp)

    xiz = sto(x0,lamiz,sx)
    zerocost = jnp.where(mwins, costm, costp)

    ## Get value if x is nonzero
    qa = sx-1/sl
    qb = lam0/sl - jnp.abs(x0)
    qc = c # Coefficient for -log(lam).
    disc = jnp.square(qb)-4*qa*qc
    discroot = jnp.sqrt(jnp.maximum(0,disc)) # Threshold negative to zero, anticipate inferior cost will weed it out.
    lm = jnp.maximum(0,(-qb - discroot) / (2*qa))
    lp = jnp.maximum(0,(-qb + discroot) / (2*qa))

    xm = sto(x0,lm,sx)
    xp = sto(x0,lp,sx)

    costm = jax_prox_log_cost(xm, lm, x0, lam0, sx, sl, c)
    costp = jax_prox_log_cost(xp, lp, x0, lam0, sx, sl, c)
    mwins = costm<costp
    x_sol = jnp.where(mwins, xm, xp)
    lam_sol = jnp.where(mwins, lm, lp)
    nzcost = jnp.where(mwins, costm, costp)

    ## Check if we return zero or nonzero value.
    xiszero = zerocost < nzcost
    x_sol = jnp.where(xiszero, xiz, x_sol)
    lam_sol = jnp.where(xiszero, lamiz, lam_sol)

    return x_sol, lam_sol

def jax_prox_inv_cost(x,lam,x0,lam0,sx,sl):
    return jnp.abs(x)/lam + jnp.square(x-x0)/(2*sx) + jnp.square(lam-lam0)/(2*sl)


## Can't jit because of the rootfinding.
def jax_apply_prox_inv(x0, lam0, sx, sl):
    p = np.stack([1/(sx*sl), -lam0/(sx*sl), jnp.zeros_like(x0), -jnp.abs(x0)/sx, jnp.ones_like(x0)]).T
    #p = jnp.stack([1/(sx*sl), -lam0/(sx*sl), jnp.zeros(P), -jnp.abs(x0)/sx, jnp.ones(P)]).T

    roots = quartic_roots(p)

    lamcand1 = roots[:,2].real
    xcand1 = jnp.maximum(0,jnp.abs(x0)-sx*lamcand1)*jnp.sign(x0)
    cost1 = jax_prox_inv_cost(xcand1, lamcand1, x0, lam0, sx, sl)
    lamcand2 = roots[:,3].real
    xcand2 = jnp.maximum(0,jnp.abs(x0)-sx*lamcand2)*jnp.sign(x0)
    cost2 = jax_prox_inv_cost(xcand2, lamcand2, x0, lam0, sx, sl)

    # TODO: Comparison seems unnecessary; always seems to be upper root (lower root seems to be inflection point).
    c1w = cost2>cost1
    lamcand = c1w*lamcand1 + (1-c1w)*lamcand2
    xcand = c1w*xcand1 + (1-c1w)*xcand2
    xiszero = (lam0*jnp.abs(x0))/sx < 1
    lamcand = xiszero*lam0 + (1-xiszero)*lamcand
    xcand *= (1-xiszero)*xcand

    return xcand, lamcand

def _normalize_shapes(vv):
    """Ensure all vv values are at least 1D arrays."""
    for k in vv:
        if hasattr(vv[k], 'shape') and len(vv[k].shape) == 0:
            vv[k] = vv[k].reshape([1])
    return vv
