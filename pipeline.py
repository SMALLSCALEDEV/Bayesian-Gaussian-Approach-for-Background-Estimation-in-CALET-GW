"""
pipeline.py
End-to-end pipeline for CALET GW follow-ups using GP background modelling.
"""

import numpy as np
from .background import gp_background_fit
from .changepoints import fit_segments
from .detection import snr_with_uncertainty
from .upperlimits import credible_upper_limit

def run_pipeline(times, counts, template, n_obs, lam_grid, use_changepoints=True):
    """
    High-level function to run GP background, detection, and upper-limit analysis.
    """
    if use_changepoints:
        models = fit_segments(times, counts)
        mu = np.concatenate([m[2] for m in models])
        sigma = np.concatenate([m[3] for m in models])
    else:
        _, mu, sigma = gp_background_fit(times, counts)

    rho = snr_with_uncertainty(counts, template, mu, sigma)
    lam_ul = credible_upper_limit(n_obs, sigma, lam_grid)

    return {"snr": rho, "upper_limit": lam_ul}
