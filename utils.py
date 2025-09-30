"""
utils.py
Shared utilities for CALET GW background modelling.
"""

import numpy as np

def make_time_series(n=1000, dt=0.1, seed=None):
    """Generate synthetic times for testing."""
    rng = np.random.default_rng(seed)
    times = np.arange(0, n*dt, dt)
    counts = rng.poisson(50, size=n)
    return times, counts
