#!/usr/bin/env python3
# filepath: /Users/u543860/AdaptiveLasso/testing_script.py

"""
Test script for MAPAdaptiveLassoCV on the bike sharing dataset.

This script demonstrates how to use MAPAdaptiveLassoCV for regression,
including data preprocessing, model training, and evaluation.

Usage:
    python testing_script.py [--data_path DATA_PATH] [--random_state RANDOM_STATE]
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from adaptivelasso.skcompat import MAPAdaptiveLassoCV, MAPAdaptiveLasso
from adaptivelasso.utils import report_model_results
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions

def load_and_preprocess_data(data_path):
    """Load and preprocess the bike sharing dataset."""
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['instant', 'dteday', 'casual', 'registered', 'cnt'])
    y = df['cnt']
    
    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=['season', 'mnth', 'weekday', 'weathersit'], drop_first=True)
    
    return X, y

def train_test_preparation(X, y, random_state=0):
    """Split data into train and test sets and prepare them."""
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Reset indices
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Fill missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())
    y_train = y_train.fillna(y_train.mean())
    y_test = y_test.fillna(y_test.mean())
    
    # Convert to float
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    # Convert to numpy arrays
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()
    
    return X_train_np, X_test_np, y_train_np, y_test_np

def train_and_evaluate_model(X_train, X_test, y_train, y_test, random_state=42):
    """Train and evaluate the MAPAdaptiveLassoCV model."""
    print("Creating model with parameter grid...")
    alasso_cv = MAPAdaptiveLassoCV(
        param_grid={
            'tau0': np.logspace(-3, 1, 100),  # Logarithmic scale for tau0
            'max_iters': [10000],
            'lr_pre': [0.001],
            'mb_size': [None],
            'invlam': [False],
            'logprox': [True],
            'lam_prior': ['adaptive'],
            'lik': ['nb'],
            'expand': ['none'],
            'tol': [1e-4]
        },
        n_splits=5, 
        scoring=None,  # Auto-select scoring based on likelihood
        random_state=random_state, 
        verbose=True
    )
    
    print("Fitting model (this may take a while)...")
    alasso_cv.fit(X_train, y_train)
    
    print("Evaluating model...")
    best_lik = alasso_cv.best_params_['lik']
    metrics = report_model_results(alasso_cv, X_test, y_test, lik=best_lik)
    
    print("\n--- Results ---")
    print("MAPAdaptiveLassoCV test metrics:", metrics)
    print("Best params:", alasso_cv.best_params_)
    
    return alasso_cv, metrics

data_path = './day.csv'
# Load and prepare data
X, y = load_and_preprocess_data(data_path)
X_train, X_test, y_train, y_test = train_test_preparation(X, y, random_state=0)

mu_X = np.mean(X_train, axis = 0)
sig_x = np.std(X_train, axis = 0)
X_train = (X_train - mu_X[np.newaxis,:]) / (1e-6+sig_x[np.newaxis,:])
X_test = (X_test - mu_X[np.newaxis,:]) / (1e-6+sig_x[np.newaxis,:])

# # Train and evaluate model with CV
# modelCV, metrics = train_and_evaluate_model(X_train, X_test, y_train, y_test, random_state=123)

# Best MAPAdaptiveLasso
best_mapa_params = {
    'tau0': 0.01, 
    'max_iters': 1000, 
    'lr_pre': 0.001, 
    'mb_size': 32, 
    'invlam': False, 
    'logprox': False, 
    'lam_prior': 'adaptive', 
    'lik': 'nb', 
    'expand': 'none', 
    'tol': 0.0001
}
mod = MAPAdaptiveLasso(**best_mapa_params, random_state=123)
mod.fit(X_train, y_train)
print(mod.score(X_test, y_test))
mod.model_.plot('temp.pdf')
model = mod.model_
np.corrcoef(mod.predict(X_train), y_train)
np.corrcoef(mod.predict(X_test), y_test)
np.square(np.corrcoef(mod.predict(X_test), y_test)[1,0])


def compare_models_on_log_likelihood(mapa_model, lasso_model, X_train, y_train, X_test, y_test):
    """
    Compares two models based on their total log predictive density on the test set.

    This is a principled way to compare models with different likelihood assumptions.
    """
    print("\n--- Comparing Models on Log Predictive Density ---")

    # 1. Get predictive distribution for the MAPAdaptiveLasso model
    # This uses the internal _predictive method
    dist_pred_nb = mapa_model.model_._predictive(X_test, mapa_model.model_.vv)
    log_lik = np.sum(dist_pred_nb.log_prob(y_test))
    print(f"MAPAdaptiveLasso Total Log-Likelihood: {log_lik:.2f}")

    # 2. Manually construct the predictive distribution for the LassoCV (Normal) model
    # The mean is the standard prediction
    preds_lasso = lasso_model.predict(X_test)
    # The scale (std dev) is the RMSE calculated on the TRAINING data
    rmse_train_lasso = np.sqrt(mean_squared_error(y_train, lasso_model.predict(X_train)))
    dist_pred_lasso = tfpd.Normal(loc=preds_lasso, scale=rmse_train_lasso)
    log_lik_lasso = np.sum(dist_pred_lasso.log_prob(y_test))
    print(f"LassoCV (Normal) Total Log-Likelihood: {log_lik_lasso:.2f}")

    if log_lik > log_lik_lasso:
        print("\nConclusion: The MAPAdaptiveLasso model is a better fit for this data.")
    else:
        print("\nConclusion: The LassoCV (Normal) model is a better fit for this data.")
        
# --- Train standard LassoCV ---
print("\n--- Training Standard LassoCV ---")
lasso_cv = LassoCV(cv=5, random_state=123, max_iter=10000, tol=1e-4)
# Use scaled X and ORIGINAL y
lasso_cv.fit(X_train, y_train)

# --- Compare the two models using the new function ---
compare_models_on_log_likelihood(
    mapa_model=mod,
    lasso_model=lasso_cv,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)