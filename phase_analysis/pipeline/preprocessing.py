"""Background subtraction and data preprocessing."""

import numpy as np
from phase_analysis import HAS_GSASII, GSASIIpwd


def auto_background(trace, log_lam=4, opt=0):
    """
    Compute background using GSASII autoBkgCalc (arpls/iarpls).
    Falls back to rolling percentile if GSASII unavailable.
    """
    if HAS_GSASII:
        bkgdict = {
            'autoPrms': {
                'logLam': log_lam,
                'opt': opt,
            }
        }
        return GSASIIpwd.autoBkgCalc(bkgdict, trace)
    else:
        from scipy.ndimage import minimum_filter1d, uniform_filter1d
        bkg = minimum_filter1d(trace, size=max(3, len(trace) // 20))
        bkg = uniform_filter1d(bkg, size=max(3, len(trace) // 10))
        return bkg


def preprocess(timeslices, log_lam=4):
    """Background-subtract all timeslices using autoBkgCalc."""
    processed = np.zeros_like(timeslices, dtype=float)
    for i in range(timeslices.shape[0]):
        bkrd = auto_background(timeslices[i], log_lam=log_lam)
        processed[i] = np.clip(timeslices[i] - bkrd, 0, None)
    return processed
