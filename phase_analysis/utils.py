"""Small utility functions used across the pipeline."""

import numpy as np


def d_to_two_theta(d_array, wavelength):
    """Convert d-spacings (Å) to 2θ (degrees) via Bragg's law."""
    d_array = np.asarray(d_array, dtype=float)

    if np.any(d_array <= 0):
        raise ValueError(
            f"d-spacings must be positive. Got min = {d_array.min():.4f} Å"
        )

    d_min = wavelength / 2.0
    if np.any(d_array < d_min):
        invalid_d = d_array[d_array < d_min]
        raise ValueError(
            f"Some d-spacings are too small for Bragg's Law at λ={wavelength} Å.\n"
            f"  Minimum measurable d-spacing: {d_min:.4f} Å (= λ/2)\n"
            f"  Problematic d-spacings: {invalid_d}\n"
            f"  Require d >= λ/2 so that sin(θ) <= 1.0"
        )

    sin_theta = wavelength / (2.0 * d_array)
    two_theta = np.rad2deg(2.0 * np.arcsin(sin_theta))
    return two_theta


def filter_reflections(data, max_index=4):
    """
    Remove reflections where any Miller index exceeds threshold.

    data : shape (4, N) — rows 0-2 are h,k,l; row 3 is peak values
    """
    mask = np.all(np.abs(data[:3]) <= max_index, axis=0)
    return data[:, mask]
