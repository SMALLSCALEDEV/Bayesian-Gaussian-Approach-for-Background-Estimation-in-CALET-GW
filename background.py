"""
background.py
Gaussian Process background modelling for CALET GW follow-ups.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

def gp_background_fit(times, counts, length_scale=10.0, nu=1.5, noise_level=1.0):
    """
    Fit a Gaussian Process to detector background counts.

    Parameters
    ----------
    times : array_like
        Time array.
    counts : array_like
        Count rates at each time.
    length_scale : float
        GP kernel correlation length.
    nu : float
        Smoothness parameter for Matern kernel.
    noise_level : float
        White noise variance.

    Returns
    -------
    gp : sklearn.gaussian_process.GaussianProcessRegressor
        Trained GP model.
    mu : np.ndarray
        Posterior mean of background.
    sigma : np.ndarray
        Posterior standard deviation.
    """
    kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=nu) \
             + WhiteKernel(noise_level=noise_level)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    gp.fit(times[:, None], counts)
    mu, sigma = gp.predict(times[:, None], return_std=True)
    return gp, mu, sigma
