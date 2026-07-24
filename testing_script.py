#!/usr/bin/env python3
# filepath: /Users/u543860/AdaptiveLasso/testing_script.py

"""
Test script for MAPAdaptiveLassoCV on the bike sharing dataset.

This script demonstrates how to use MAPAdaptiveLassoCV for regression,
including data preprocessing, model training, and evaluation using both
CRPS and Log-Likelihood scoring rules.

Usage:
    python testing_script.py [--data_path DATA_PATH] [--random_state RANDOM_STATE]
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from adaptivelasso.skcompat import MAPAdaptiveLasso
import tensorflow_probability.substrates.jax as tfp
import properscoring as ps
import jax

tfpd = tfp.distributions

def load_and_preprocess_data(data_path):
    """Load and preprocess the bike sharing dataset."""
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['instant', 'dteday', 'casual', 'registered', 'cnt'])
    y = df['cnt']
    X = pd.get_dummies(X, columns=['season', 'mnth', 'weekday', 'weathersit'], drop_first=True)
    return X, y

def train_test_preparation(X, y, random_state=0):
    """Split data into train and test sets and prepare them."""
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return X_train.to_numpy(dtype=float), X_test.to_numpy(dtype=float), y_train.to_numpy(dtype=float), y_test.to_numpy(dtype=float)

def _get_predictive_distributions(mapa_model, lasso_model, X_train, y_train, X_test):
    """Helper function to get predictive distributions for both models."""
    # Get core MAPA model (handles both CV and non-CV instances)
    if hasattr(mapa_model, 'best_model_'):
        core_model = mapa_model.best_model_.model_
    else:
        core_model = mapa_model.model_
    
    # Get MAPA predictive distribution
    dist_pred_mapa = core_model._predictive(X_test, core_model.vv)
    
    # Construct LassoCV predictive distribution (always Normal)
    preds_lasso = lasso_model.predict(X_test)
    rmse_train_lasso = np.sqrt(mean_squared_error(y_train, lasso_model.predict(X_train)))
    dist_pred_lasso = tfpd.Normal(loc=preds_lasso, scale=rmse_train_lasso)
    
    return dist_pred_mapa, dist_pred_lasso, core_model.lik

def compare_models_on_crps(mapa_model, lasso_model, X_train, y_train, X_test, y_test):
    """Compares models based on the mean Continuous Ranked Probability Score (CRPS). Lower is better."""
    print("\n--- Comparing Models on Continuous Ranked Probability Score (CRPS) ---")
    
    dist_pred_mapa, dist_pred_lasso, mapa_lik = _get_predictive_distributions(
        mapa_model, lasso_model, X_train, y_train, X_test
    )

    # Calculate CRPS for MAPA model
    if mapa_lik in ['nb', 'poisson', 'zinb']:
        # For discrete distributions, use ensemble-based CRPS with samples
        n_samples = 1000
        # Create a JAX random key for sampling
        rng_key = jax.random.PRNGKey(42)
        samples_mapa = dist_pred_mapa.sample(n_samples, seed=rng_key).T  # Shape: (n_test, n_samples)
        crps_mapa_values = np.array([ps.crps_ensemble(y_test[i], samples_mapa[i]) 
                                      for i in range(len(y_test))])
    else:
        # For Normal distribution, use analytical CRPS
        crps_mapa_values = ps.crps_gaussian(y_test, 
                                            mu=np.array(dist_pred_mapa.mean()), 
                                            sig=np.array(dist_pred_mapa.stddev()))
    
    mean_crps_mapa = np.mean(crps_mapa_values)
    
    # Calculate CRPS for LassoCV (always Normal, use analytical formula)
    crps_lasso_values = ps.crps_gaussian(y_test, 
                                         mu=np.array(dist_pred_lasso.mean()), 
                                         sig=np.array(dist_pred_lasso.stddev()))
    mean_crps_lasso = np.mean(crps_lasso_values)

    print(f"MAPAdaptiveLasso ({mapa_lik}) Mean CRPS: {mean_crps_mapa:.2f}")
    print(f"LassoCV (Normal) Mean CRPS:      {mean_crps_lasso:.2f}")

    if mean_crps_mapa < mean_crps_lasso:
        print(f"\nConclusion (CRPS): The MAPAdaptiveLasso ({mapa_lik}) model provides a better probabilistic forecast.")
    else:
        print("\nConclusion (CRPS): The LassoCV (Normal) model provides a better probabilistic forecast.")

def compare_models_on_log_likelihood(mapa_model, lasso_model, X_train, y_train, X_test, y_test):
    """Compares models based on the total log predictive density (Log-Likelihood). Higher is better."""
    print("\n--- Comparing Models on Log Predictive Density (Log-Likelihood) ---")
    
    dist_pred_mapa, dist_pred_lasso, mapa_lik = _get_predictive_distributions(
        mapa_model, lasso_model, X_train, y_train, X_test
    )

    log_lik_mapa = np.sum(dist_pred_mapa.log_prob(y_test))
    log_lik_lasso = np.sum(dist_pred_lasso.log_prob(y_test))

    print(f"MAPAdaptiveLasso ({mapa_lik}) Total Log-Likelihood: {log_lik_mapa:.2f}")
    print(f"LassoCV (Normal) Total Log-Likelihood:      {log_lik_lasso:.2f}")

    if log_lik_mapa > log_lik_lasso:
        print(f"\nConclusion (Log-Likelihood): The MAPAdaptiveLasso ({mapa_lik}) model is a better fit for this data.")
    else:
        print("\nConclusion (Log-Likelihood): The LassoCV (Normal) model is a better fit for this data.")






"""Main function to run the script."""
data_path = './day.csv'
X, y = load_and_preprocess_data(data_path)
X_train, X_test, y_train, y_test = train_test_preparation(X, y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lams = np.logspace(-4,4,num=100)
L = len(lams)
me_is = np.nan*np.zeros(L)
me_oos = np.nan*np.zeros(L)
glm_is = np.nan*np.zeros(L)
glm_oos = np.nan*np.zeros(L)
for li,l in enumerate(lams):
    print("\n--- Training Best MAPAdaptiveLasso (NB) Model ---")
    best_mapa_params = {
        #'tau0': 0.01, 'max_iters': 1000, 'lr_pre': 0.001, 'mb_size': 32, 
        #'tau0': 0.01, 'max_iters': 1000, 'lr_pre': 0.001, 'mb_size': 32, 'es_patience' : 1e8,
        #'tau0': 0.01, 'max_iters': 2000, 'lr_pre': 0.001, 'mb_size': 32, 'es_patience' : 1e8,
        'tau0': l, 'max_iters': 2000, 'lr_pre': 0.001, 'mb_size': 32, 'es_patience' : 1e8,
        #'tau0': 1., 'max_iters': 1000, 'lr_pre': 0.001, 'mb_size': 32, 
        'invlam': False, 'logprox': False, 'lam_prior': 'adaptive', 'expand' : 'quad',
        'lik': 'nb', 'expand': 'none', 'tol': 0.0001
    }
    mapa_nb_model = MAPAdaptiveLasso(**best_mapa_params, random_state=123, track=True)
    mapa_nb_model.fit(X_train_scaled, y_train)
    #mapa_nb_model.model_.plot('traj.png')
    print("Training complete.")

    print("\n--- Training Standard LassoCV Model ---")
    lasso_cv_model = LassoCV(cv=5, random_state=123, max_iter=10000)
    lasso_cv_model.fit(X_train_scaled, y_train)
    print(f"Training complete. Best alpha found: {lasso_cv_model.alpha_:.4f}")

    fit_me = mapa_nb_model.predict(X_train_scaled)
    fit_glm = lasso_cv_model.predict(X_train_scaled)
    pred_me = mapa_nb_model.predict(X_test_scaled)
    pred_glm = lasso_cv_model.predict(X_test_scaled)

    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #for i in range(2):
    #    plt.subplot(1,2,i+1)
    #
    #    if i==0:
    #        plt.scatter(fit_me, y_train, label = 'me')
    #        plt.scatter(fit_glm, y_train, label = 'glm')
    #        plt.title("In sample")
    #    else:
    #        plt.scatter(pred_me, y_test, label = 'me')
    #        plt.scatter(pred_glm, y_test, label = 'glm')
    #        plt.title("OoS")
    #
    #    plt.plot([0,np.max(y_train)],[0,np.max(y_train)],color='black', linestyle='--')
    #    plt.legend()
    #
    #plt.savefig("pva.pdf")
    #plt.close()

    mapa_nb_model.model_.vv['beta']
    lasso_cv_model.coef_

    trans = lambda x: x
    #loss = np.abs
    loss = np.square

    me_is[li] = np.mean(loss(trans(fit_me) - trans(y_train)))
    glm_is[li] = np.mean(loss(trans(fit_glm) - trans(y_train)))
    print(me_is[li])
    print(glm_is[li])

    me_oos[li] = np.mean(loss(trans(pred_me) - trans(y_test)))
    glm_oos[li] = np.mean(loss(trans(pred_glm) - trans(y_test)))
    print(me_oos[li])
    print(glm_oos[li])

# --- Perform both comparisons on the test set ---
compare_models_on_crps(
    mapa_model=mapa_nb_model,
    lasso_model=lasso_cv_model,
    X_train=X_train_scaled,
    y_train=y_train,
    X_test=X_test_scaled,
    y_test=y_test
)

compare_models_on_log_likelihood(
    mapa_model=mapa_nb_model,
    lasso_model=lasso_cv_model,
    X_train=X_train_scaled,
    y_train=y_train,
    X_test=X_test_scaled,
    y_test=y_test
)

