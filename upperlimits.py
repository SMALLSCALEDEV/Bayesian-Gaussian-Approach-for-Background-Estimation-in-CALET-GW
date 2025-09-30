"""
upperlimits.py
Compute Bayesian credible upper limits marginalising over background uncertainty.
"""

import numpy as np
from scipy.stats import poisson
from scipy.integrate import quad

def posterior_lambda(n_obs, bg_samples, lam_grid):
    """
    Posterior over signal counts lambda given Poisson data and uncertain background.
    """
    post = np.zeros_like(lam_grid, dtype=float)
    for b in bg_samples:
        post += poisson.pmf(n_obs, lam_grid + b)
    post /= np.trapz(post, lam_grid)
    return post

def credible_upper_limit(n_obs, bg_samples, lam_grid, cred=0.9):
    """
    Compute Bayesian credible upper limit.

    Returns
    -------
    lam_ul : float
        Credible upper limit on signal counts.
    """
    post = posterior_lambda(n_obs, bg_samples, lam_grid)
    cdf = np.cumsum(post) / np.sum(post)
    idx = np.searchsorted(cdf, cred)
    return lam_grid[idx]
