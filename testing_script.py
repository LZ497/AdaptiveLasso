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

from adaptivelasso.skcompat import MAPAdaptiveLassoCV, MAPAdaptiveLasso
from adaptivelasso.utils import report_model_results

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

# Train and evaluate model with CV
model, metrics = train_and_evaluate_model(X_train, X_test, y_train, y_test, random_state=123)

# # Test MAPAdaptiveLasso
# mod = MAPAdaptiveLasso(tau0=1e-1, track = True, max_iters = 10000, lr_pre = 1e0, mb_size = 50, random_state=123, ada=True, prog=True, tol = 1e-4)
# mod.fit(X_train, y_train)
# print(mod.score(X_test, y_test))
# mod.model_.plot('temp.pdf')
# model = mod.model_
# np.corrcoef(mod.predict(X_train), y_train)
# np.corrcoef(mod.predict(X_test), y_test)
# np.square(np.corrcoef(mod.predict(X_test), y_test)[1,0])

# # MAPAdaptiveLasso with scaler
# pipe_mapa = Pipeline([
#     ('mapa', MAPAdaptiveLasso(tau0=1e-1, track = True, max_iters = 10000, lr_pre = 1e0, mb_size = 50, random_state=123, ada=True, prog=True, tol = 1e-4))
# ])
# pipe_mapa.fit(X_train, y_train)

# report_model_results(pipe_mapa.named_steps['mapa'], X_test, y_test, lik='normal')

# pipe_mapa_nb = Pipeline([
#     ('mapa_nb', MAPAdaptiveLasso(tau0=1e-1, track = True, max_iters = 10000, lr_pre = 1e0, mb_size = 50, random_state=123, ada=True, prog=True, tol = 1e-4, lik='nb'))
# ])
# pipe_mapa_nb.fit(X_train, y_train)

# report_model_results(pipe_mapa_nb.named_steps['mapa_nb'], X_test, y_test, lik='nb')

# # Use a pipeline for scaling and LassoCV
# pipe_lassocv = Pipeline([
#     ('lasso', LassoCV(cv=5, random_state=123, max_iter=10000, tol=1e-4))
# ])
# pipe_lassocv.fit(X_train, y_train)

# report_model_results(pipe_lassocv.named_steps['lasso'], X_test, y_test, lik='normal')
