"""
changepoints.py
Hybrid GP + change-point background modelling.
"""

import numpy as np
import ruptures as rpt
from .background import gp_background_fit

def detect_changepoints(times, counts, penalty=10):
    """
    Detect change-points in background counts using PELT.

    Returns
    -------
    cp_times : list
        Times of detected change-points.
    """
    algo = rpt.Pelt(model="rbf").fit(counts)
    cps = algo.predict(pen=penalty)
    return [times[i-1] for i in cps[:-1]]

def fit_segments(times, counts, penalty=10):
    """
    Fit separate GPs to each background segment.

    Returns
    -------
    segment_models : list
        List of (gp, mu, sigma) tuples for each segment.
    """
    cps = detect_changepoints(times, counts, penalty=penalty)
    segment_models = []
    start = 0
    for cp in cps + [len(times)]:
        t_seg, c_seg = times[start:cp], counts[start:cp]
        gp, mu, sigma = gp_background_fit(t_seg, c_seg)
        segment_models.append((gp, t_seg, mu, sigma))
        start = cp
    return segment_models
