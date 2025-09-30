"""
injections.py
Injection–recovery testing for GP background modelling.
"""

import numpy as np
from .detection import snr_with_uncertainty
from .background import gp_background_fit

def inject_signal(times, counts, signal_template, scale=1.0):
    """Add scaled signal template into counts."""
    injected = counts + scale * signal_template
    return injected

def run_injection_recovery(times, counts, signal_template, scales, snr_threshold=5.0):
    """
    Test recovery rate for injected signals of different scales.
    """
    gp, mu, sigma = gp_background_fit(times, counts)
    detections = []
    for s in scales:
        injected = inject_signal(times, counts, signal_template, s)
        rho = snr_with_uncertainty(injected, signal_template, mu, sigma)
        detections.append(rho > snr_threshold)
    return np.array(detections)
