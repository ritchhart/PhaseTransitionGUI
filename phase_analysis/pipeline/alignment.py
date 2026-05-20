"""Cross-correlation alignment of reference patterns to data."""

import numpy as np
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d


def cross_correlate_shift(reference, target, max_shift=50):
    """Find optimal shift to align reference to target via cross-correlation."""
    corr = correlate(target, reference, mode='full')
    center = len(reference) - 1
    search_range = slice(center - max_shift, center + max_shift + 1)
    corr_window = corr[search_range]

    best_offset = np.argmax(corr_window) - max_shift
    best_score = corr_window[np.argmax(corr_window)]
    norm = np.sqrt(np.sum(reference**2) * np.sum(target**2))
    best_score = best_score / (norm + 1e-10)

    return best_offset, best_score


def align_reference_to_data(reference, data_slice, max_shift=50):
    """Shift a reference pattern to best match a data slice."""
    shift, score = cross_correlate_shift(reference, data_slice, max_shift)
    aligned = np.roll(reference, shift)

    if shift > 0:
        aligned[:shift] = 0
    elif shift < 0:
        aligned[shift:] = 0

    return aligned, shift, score


def prepare_references_for_mcr(candidate_profiles, data, segments=None,
                               max_shift=50, broaden_sigma=2):
    """
    Prepare reference profiles for MCR-ALS:
    1. Broaden (optional)
    2. Align to data mean
    3. Normalize

    Returns: (prepared_profiles, shifts_found)
    """
    prepared = []
    shifts_found = []
    target = np.mean(data, axis=0)

    for prof in candidate_profiles:
        ref = np.array(prof, dtype=float)
        if broaden_sigma > 0:
            ref = gaussian_filter1d(ref, sigma=broaden_sigma)

        aligned, shift, score = align_reference_to_data(ref, target, max_shift)
        aligned = aligned / (aligned.max() + 1e-10)
        prepared.append(aligned)
        shifts_found.append(shift)

    return prepared, shifts_found
