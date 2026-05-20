"""Phase transition detection and segment labeling."""

import numpy as np


def detect_transitions(data, weights, threshold_pct):
    """
    Detect phase transitions from time-evolving diffraction data.

    Parameters
    ----------
    data : ndarray, shape (n_times, n_channels)
    weights : dict with keys 'dissimilarity', 'intensity_change',
              'channel_change', 'rank_change'
    threshold_pct : int
        Percentile for the combined score threshold.

    Returns
    -------
    dict with indicator arrays, combined score, threshold, and transitions.
    """
    n_times, n_channels = data.shape

    dI_dt = np.diff(data, axis=0)
    dI_dt = np.vstack([dI_dt, dI_dt[-1:]])

    # Dissimilarity (cosine)
    dissimilarity = np.zeros(n_times)
    for t in range(1, n_times):
        a, b = data[t - 1], data[t]
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
        dissimilarity[t] = 1 - dot / norm

    # Intensity change
    intensity_change = np.sum(np.abs(dI_dt), axis=1)

    # Channel change count
    threshold_ch = np.percentile(np.abs(dI_dt), 90)
    channel_change = np.sum(
        np.abs(dI_dt) > threshold_ch, axis=1
    ).astype(float)

    # Rank change
    rank_change = np.zeros(n_times)
    for t in range(1, n_times):
        r1 = np.argsort(np.argsort(data[t - 1]))
        r2 = np.argsort(np.argsort(data[t]))
        rank_change[t] = np.sum(np.abs(r1 - r2))

    # Normalize each to [0, 1]
    def norm01(x):
        mx = np.max(x)
        return x / mx if mx > 0 else x

    dissimilarity = norm01(dissimilarity)
    intensity_change = norm01(intensity_change)
    channel_change = norm01(channel_change)
    rank_change = norm01(rank_change)

    # Weighted combination
    combined = (
        weights.get('dissimilarity', 1.0) * dissimilarity
        + weights.get('intensity_change', 1.0) * intensity_change
        + weights.get('channel_change', 1.0) * channel_change
        + weights.get('rank_change', 1.0) * rank_change
    )
    combined = norm01(combined)

    # Threshold and find transitions
    threshold = np.percentile(combined, threshold_pct)
    above = combined > threshold

    transitions = []
    in_region = False
    start = 0
    for t in range(n_times):
        if above[t] and not in_region:
            start = t
            in_region = True
        elif not above[t] and in_region:
            transitions.append((start + t) // 2)
            in_region = False
    if in_region:
        transitions.append((start + n_times - 1) // 2)

    return {
        'dI_dt': dI_dt,
        'dissimilarity': dissimilarity,
        'intensity_change': intensity_change,
        'channel_change': channel_change,
        'rank_change': rank_change,
        'combined_score': combined,
        'threshold': threshold,
        'transitions': transitions,
    }


def get_segment_labels(transitions, n_times):
    """Convert transition list to per-frame segment labels (0-indexed)."""
    labels = np.zeros(n_times, dtype=int)
    boundaries = [0] + sorted(transitions) + [n_times]
    for seg_idx in range(len(boundaries) - 1):
        labels[boundaries[seg_idx]:boundaries[seg_idx + 1]] = seg_idx
    return labels
