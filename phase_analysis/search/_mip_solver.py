"""MIP search-match solver (private implementation)."""

import numpy as np
from phase_analysis import HAS_MIP, Model, xsum, maximize, BINARY
from phase_analysis.pipeline.peak_detection import detect_peaks_in_trace


def mip_search_match(test_peaks, candidate_library, tolerance,
                     parsimony_weight=1.0, false_positive_weight=0.5,
                     coverage_bonus_weight=3.0):
    """
    MIP search-match: find optimal subset of candidates that explain
    observed peaks.
    """
    if not HAS_MIP:
        raise ImportError("python-mip required for Peak Match")

    n_test = len(test_peaks)
    if n_test == 0:
        return []

    candidates = list(candidate_library.keys())
    n_cand = len(candidates)

    align_score = np.zeros((n_test, n_cand))
    false_pos_count = np.zeros(n_cand)
    n_peaks_cand = np.zeros(n_cand)

    for j, name in enumerate(candidates):
        cand_peaks = np.array(candidate_library[name])
        n_peaks_cand[j] = len(cand_peaks)

        for i, tp in enumerate(test_peaks):
            dists = np.abs(cand_peaks - tp)
            min_dist = dists.min() if len(dists) > 0 else np.inf
            if min_dist <= tolerance:
                align_score[i, j] = 1.0 - (min_dist / tolerance) ** 2

        for cp in cand_peaks:
            if np.min(np.abs(test_peaks - cp)) > tolerance:
                false_pos_count[j] += 1

    frac_matched = 1.0 - false_pos_count / (n_peaks_cand + 1e-10)

    m = Model()
    m.verbose = 0

    z = [m.add_var(var_type=BINARY) for _ in range(n_cand)]
    y = [[m.add_var(var_type=BINARY) for _ in range(n_cand)]
         for _ in range(n_test)]

    for i in range(n_test):
        for j in range(n_cand):
            if align_score[i, j] > 0:
                m += y[i][j] <= z[j]
            else:
                m += y[i][j] == 0

    for i in range(n_test):
        m += xsum(y[i][j] for j in range(n_cand)) <= 1

    m.objective = maximize(
        xsum(
            (align_score[i, j] / (n_peaks_cand[j] + 1e-10)) * y[i][j]
            for i in range(n_test) for j in range(n_cand)
        )
        + coverage_bonus_weight * xsum(
            frac_matched[j] * z[j] for j in range(n_cand)
        )
        - parsimony_weight * xsum(z[j] for j in range(n_cand))
        - false_positive_weight * xsum(
            (false_pos_count[j] / (n_peaks_cand[j] + 1e-10)) * z[j]
            for j in range(n_cand)
        )
    )

    m.optimize()
    selected = [candidates[j] for j in range(n_cand) if z[j].x > 0.5]
    return selected


def mip_search_match_binned(data, two_theta, segment_labels,
                            candidate_library, bin_size=5,
                            tolerance=0.2, peak_sigma=3,
                            peak_threshold=0.01,
                            use_step_detect=True,
                            step_threshold=0.01,
                            parsimony_weight=1.0,
                            false_positive_weight=0.5,
                            coverage_bonus_weight=3.0,
                            progress_callback=None):
    """Run MIP search-match per bin within each segment."""
    unique_segments = np.unique(segment_labels)
    all_selections = []
    per_segment_results = {}

    total_bins = 0
    for seg_idx in unique_segments:
        n_seg = np.sum(segment_labels == seg_idx)
        total_bins += max(1, n_seg // bin_size)

    bin_counter = 0

    for seg_idx in unique_segments:
        seg_mask = segment_labels == seg_idx
        seg_times = np.where(seg_mask)[0]
        seg_data = data[seg_mask, :]
        n_seg = len(seg_times)

        n_bins = max(1, n_seg // bin_size)
        seg_selections = []

        for b in range(n_bins):
            start = b * bin_size
            end = (b + 1) * bin_size if b < n_bins - 1 else n_seg
            bin_avg = np.mean(seg_data[start:end, :], axis=0)
            bin_center_time = seg_times[(start + min(end, n_seg) - 1) // 2]

            peak_indices = detect_peaks_in_trace(
                bin_avg, two_theta, sigma=peak_sigma,
                threshold=peak_threshold,
                use_step_detect=use_step_detect,
                step_threshold=step_threshold)
            test_peaks = two_theta[peak_indices]

            if len(test_peaks) > 0:
                selected = mip_search_match(
                    test_peaks, candidate_library, tolerance,
                    parsimony_weight, false_positive_weight,
                    coverage_bonus_weight)
            else:
                selected = []

            all_selections.append((bin_center_time, selected))
            seg_selections.append(selected)
            bin_counter += 1

            if progress_callback:
                progress_callback(bin_counter, total_bins)

        per_segment_results[seg_idx] = {
            'selections': seg_selections,
            'n_bins': n_bins,
            'times': seg_times,
        }

    return all_selections, per_segment_results
