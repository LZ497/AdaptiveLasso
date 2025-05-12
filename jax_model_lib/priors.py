#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  jax_hier_lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.14.2024
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import jax
tfpd = tfp.distributions
import numpy as np 

def adaptive_prior(x, pv, mod):
    lam_dist = tfpd.Cauchy(loc=0., scale=1.)
    lam_dens = -jnp.sum(lam_dist.log_prob(x) - jnp.log((1 - lam_dist.cdf(0))))
    return lam_dens

def group_prior(x, pv, mod):
    gamma = jnp.exp(pv['log_gamma'])

    if 'groups' in pv:
        groups_raw  = pv['groups']
    else:
        G = gamma.shape[0]
        P = x.shape[0]
        groups_raw  = jnp.tile(jnp.arange(G), P // G + 1)[:P]
    groups = jax.lax.stop_gradient(jnp.array(groups_raw).astype(jnp.int32))
    
    gamma_dist = tfpd.Cauchy(loc=0., scale=1.)
    gamma_dens = -jnp.sum(gamma_dist.log_prob(gamma) - jnp.log(1 - gamma_dist.cdf(0)))

    gamma_big = gamma[groups]
    lam_sd = 1. / jnp.sqrt(mod.N)
    lam_dist = tfpd.Normal(loc=gamma_big, scale=lam_sd)
    lam_dens = -jnp.sum(lam_dist.log_prob(x) - jnp.log(1 - lam_dist.cdf(0)))

    return lam_dens + gamma_dens

def hier_prior(x, pv, mod):
    gamma = jnp.exp(pv['log_gamma'])
    gamma_dist = tfpd.Cauchy(loc=0., scale=1.)
    gamma_dens = -jnp.sum(gamma_dist.log_prob(gamma)-jnp.log((1-gamma_dist.cdf(0))))

    lam_sd = 1/jnp.sqrt(mod.N)

    GAMMAt = make_gamma_mat(Pu).astype(int)
    temp = 1/jnp.sqrt(Pu)
    gam_meq = jnp.apply_along_axis(lambda x: jnp.sum(jax.nn.softmax(-x/temp) * x), 0, gamma[GAMMAt])
    meq_dist =  tfpd.Normal(loc=gam_meq, scale=lam_sd)

    lam_me = x[np.arange(Pu)]
    lam_q = x[np.arange(Pu+Pi,Pu+Pi+Pq)]
    me_dens = -jnp.sum(meq_dist.log_prob(lam_me)-jnp.log(1-meq_dist.cdf(0)))
    q_dens = -jnp.sum(meq_dist.log_prob(lam_q)-jnp.log(1-meq_dist.cdf(0)))

    lam_i = x[np.arange(Pu,Pu+Pi)]
    i_dist =  tfpd.Normal(loc=gamma, scale=lam_sd)
    i_dens = -jnp.sum(i_dist.log_prob(lam_i)-jnp.log(1-i_dist.cdf(0)))

    return gamma_dens + me_dens + q_dens + i_dens
