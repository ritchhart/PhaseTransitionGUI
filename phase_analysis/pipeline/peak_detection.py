"""Peak detection — easily swappable backend."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from phase_analysis import HAS_STEP_DETECT, step_detect


def detect_peaks_in_trace(trace, two_theta, sigma=3, threshold=0.01,
                          use_step_detect=True, step_threshold=0.01):
    """
    Find peak positions in a 1D trace.

    Parameters
    ----------
    use_step_detect : bool
        True  → derivative + step_detect.find_steps (original method)
        False → scipy find_peaks fallback

    Returns
    -------
    indices : ndarray of int
        Indices into the two_theta array where peaks were found.
    """
    if use_step_detect and HAS_STEP_DETECT:
        gauss_filtered = gaussian_filter1d(trace, sigma, order=1)
        normalized = gauss_filtered / (np.max(np.abs(gauss_filtered)) + 1e-10)
        steps = step_detect.find_steps(normalized, step_threshold)
        return np.array(steps, dtype=int)
    else:
        smoothed = gaussian_filter1d(trace, sigma)
        height_thresh = threshold * np.max(smoothed)
        peaks, _ = find_peaks(
            smoothed,
            height=height_thresh,
            distance=int(sigma * 2),
            prominence=height_thresh * 0.5,
        )
        return peaks
