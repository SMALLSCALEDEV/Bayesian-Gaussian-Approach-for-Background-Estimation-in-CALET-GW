"""
detection.py
Detection statistics using GP background models.
"""

import numpy as np

def snr_with_uncertainty(data, template, mu, sigma):
    """
    Compute SNR including GP background uncertainty.

    Parameters
    ----------
    data : np.ndarray
        Observed counts.
    template : np.ndarray
        Signal template.
    mu : np.ndarray
        GP posterior mean background.
    sigma : np.ndarray
        GP posterior std background.

    Returns
    -------
    rho : float
        Modified SNR statistic.
    """
    residual = data - mu
    var = sigma**2 + np.var(data - mu)
    numerator = np.dot(template, residual)
    denominator = np.sqrt(np.dot(template, template * var))
    return numerator / denominator
