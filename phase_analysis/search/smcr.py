"""Sparse MCR-ALS with candidate screening and per-segment fitting."""

import numpy as np
from scipy.optimize import nnls
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d


class SparseMCR_ALS:
    """MCR-ALS with sparsity enforcement and candidate screening."""

    def __init__(self, two_theta):
        self.two_theta = two_theta
        self.n_channels = len(two_theta)

        self.S_candidates = None
        self.candidate_names = None
        self.n_candidates = 0

        self.C = None
        self.S = None
        self.active_mask = None
        self.residuals = None
        self.convergence_history = []
        self.segment_results = {}
        self.S_per_segment = {}

    # ─── Loading ──────────────────────────────────────────────

    def load_candidates(self, candidate_profiles, names=None,
                        broaden_sigma=2):
        """
        Load candidate spectral profiles (pre-simulated powder patterns).
        These should already be on the correct two_theta grid.
        """
        self.n_candidates = len(candidate_profiles)
        self.S_candidates = np.zeros((self.n_channels, self.n_candidates))

        for i, prof in enumerate(candidate_profiles):
            s = np.array(prof, dtype=float)
            if broaden_sigma > 0:
                s = gaussian_filter1d(s, sigma=broaden_sigma)
            s = np.clip(s, 0, None)
            s = s / (np.max(s) + 1e-10)
            self.S_candidates[:, i] = s

        if names is None:
            names = [f'Candidate_{i}' for i in range(self.n_candidates)]
        self.candidate_names = list(names)

    # ─── Correlation helpers ──────────────────────────────────

    def _normalized_similarity(self, ref_a_norm, ref_b_norm, max_shift):
        """Shift-tolerant normalized similarity between two profiles."""
        corr = correlate(ref_a_norm, ref_b_norm, mode='full')
        center = len(ref_a_norm) - 1
        lo = max(0, center - max_shift)
        hi = min(len(corr), center + max_shift + 1)
        corr_window = corr[lo:hi]

        best_idx = np.argmax(corr_window)
        best_shift = best_idx - (center - lo)

        if best_shift >= 0:
            a_region = ref_a_norm[best_shift:]
            b_region = ref_b_norm[:len(a_region)]
        else:
            b_region = ref_b_norm[-best_shift:]
            a_region = ref_a_norm[:len(b_region)]

        min_len = min(len(a_region), len(b_region))
        a_region = a_region[:min_len]
        b_region = b_region[:min_len]

        norm_a = np.linalg.norm(a_region) + 1e-10
        norm_b = np.linalg.norm(b_region) + 1e-10
        return np.dot(a_region, b_region) / (norm_a * norm_b)

    def _cross_correlate(self, reference, target, max_shift):
        """Cross-correlation with bounded shift."""
        corr = correlate(target, reference, mode='full')
        center = len(reference) - 1
        lo = max(0, center - max_shift)
        hi = min(len(corr), center + max_shift + 1)
        corr_window = corr[lo:hi]

        best_idx = np.argmax(corr_window)
        best_offset = best_idx - (center - lo)
        norm = np.sqrt(np.sum(reference**2) * np.sum(target**2)) + 1e-10
        return best_offset, corr_window[best_idx] / norm

    # ─── Screening ────────────────────────────────────────────

    def screen_candidates(self, data, max_shift=50,
                          correlation_threshold=0.3,
                          min_time_presence=5,
                          dedup_threshold=0.95,
                          progress_callback=None):
        """
        Pre-screen which candidates could plausibly be in the data.
        Returns: (plausible_indices, best_correlations, presence_counts)
        """
        n_times = data.shape[0]
        presence_count = np.zeros(self.n_candidates)
        best_correlations = np.zeros(self.n_candidates)

        step = max(1, n_times // 40)
        time_indices = list(range(0, n_times, step))

        for i in range(self.n_candidates):
            ref = self.S_candidates[:, i]
            corrs_this = []
            for t in time_indices:
                target = data[t, :]
                _, score = self._cross_correlate(ref, target, max_shift)
                corrs_this.append(score)
                if score > correlation_threshold:
                    presence_count[i] += step
            best_correlations[i] = max(corrs_this)
            if progress_callback:
                progress_callback(i + 1, self.n_candidates)

        # Filter by presence threshold
        passes_threshold = presence_count >= min_time_presence
        passing_indices = np.where(passes_threshold)[0]

        # Deduplicate: sort by best correlation descending
        sorted_passing = passing_indices[
            np.argsort(best_correlations[passing_indices])[::-1]
        ]

        accepted = []
        for idx in sorted_passing:
            ref_i = self.S_candidates[:, idx]
            ref_i_norm = ref_i / (np.linalg.norm(ref_i) + 1e-10)
            is_duplicate = False
            for accepted_idx in accepted:
                ref_j = self.S_candidates[:, accepted_idx]
                ref_j_norm = ref_j / (np.linalg.norm(ref_j) + 1e-10)
                similarity = self._normalized_similarity(
                    ref_i_norm, ref_j_norm, max_shift)
                if similarity > dedup_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                accepted.append(idx)

        return np.array(accepted), best_correlations, presence_count

    def select_candidates(self, indices):
        """Reduce working set to selected candidates."""
        self.S = self.S_candidates[:, indices].copy()
        self.candidate_names = [self.candidate_names[i] for i in indices]
        self.n_candidates = len(indices)
        self.active_mask = np.ones(self.n_candidates, dtype=bool)

    # ─── Solvers ──────────────────────────────────────────────

    def _solve_C_sparse(self, data, S, max_components_per_trace=None,
                        sparsity_method='iterative_threshold',
                        l1_alpha=0.01):
        """Solve for C (concentrations) with sparsity constraint."""
        n_times = data.shape[0]
        n_comp = S.shape[1]
        C = np.zeros((n_times, n_comp))

        for t in range(n_times):
            target = data[t, :]

            if sparsity_method == 'hard_threshold':
                c, _ = nnls(S, target)
                if (max_components_per_trace is not None
                        and np.sum(c > 0) > max_components_per_trace):
                    sorted_idx = np.argsort(c)[::-1]
                    keep = sorted_idx[:max_components_per_trace]
                    mask = np.zeros(n_comp, dtype=bool)
                    mask[keep] = True
                    c_refined, _ = nnls(S[:, mask], target)
                    c = np.zeros(n_comp)
                    c[mask] = c_refined
                C[t, :] = c

            elif sparsity_method == 'iterative_threshold':
                active = np.ones(n_comp, dtype=bool)
                c, _ = nnls(S, target)
                while (max_components_per_trace is not None
                       and np.sum(c > 0) > max_components_per_trace):
                    contributions = c * np.max(S, axis=0)
                    nonzero_mask = c > 0
                    if not np.any(nonzero_mask):
                        break
                    weakest = np.where(nonzero_mask)[0][
                        np.argmin(contributions[nonzero_mask])
                    ]
                    active[weakest] = False
                    c = np.zeros(n_comp)
                    if np.any(active):
                        c_sub, _ = nnls(S[:, active], target)
                        c[active] = c_sub
                C[t, :] = c

            elif sparsity_method == 'l1':
                c, _ = nnls(S, target)
                threshold = l1_alpha * np.max(c)
                c[c < threshold] = 0
                surviving = c > 0
                if np.any(surviving):
                    if (max_components_per_trace
                            and np.sum(surviving) > max_components_per_trace):
                        top_k = np.argsort(c)[::-1][:max_components_per_trace]
                        surviving = np.zeros(n_comp, dtype=bool)
                        surviving[top_k] = True
                    c_refined, _ = nnls(S[:, surviving], target)
                    c = np.zeros(n_comp)
                    c[surviving] = c_refined
                C[t, :] = c

        return C

    # ─── Per-segment fitting ──────────────────────────────────

    def fit_per_segment_binned(self, data, segment_labels,
                               bin_size=5,
                               max_components_per_trace=4,
                               max_components_per_segment=6,
                               fix_known_spectra=True,
                               spectral_smoothness=0.5,
                               max_iter=100, tol=1e-4,
                               progress_callback=None):
        """
        Fit MCR-ALS per segment using binned data.
        Returns global R² score.
        """
        unique_segments = np.unique(segment_labels)
        n_segments = len(unique_segments)
        n_times, n_channels = data.shape
        n_comp = self.S.shape[1]

        self.C = np.zeros((n_times, n_comp))
        self.S_per_segment = {}
        self.segment_results = {}

        for seg_num, seg_idx in enumerate(unique_segments):
            seg_mask = segment_labels == seg_idx
            seg_times = np.where(seg_mask)[0]
            seg_data = data[seg_mask, :]
            n_seg = len(seg_times)

            # Bin the segment
            n_bins = max(1, n_seg // bin_size)
            binned_data = np.zeros((n_bins, n_channels))
            bin_membership = []
            for b in range(n_bins):
                start = b * bin_size
                end = (b + 1) * bin_size if b < n_bins - 1 else n_seg
                binned_data[b, :] = np.mean(seg_data[start:end, :], axis=0)
                bin_membership.append(seg_times[start:end])

            # ALS iterations
            S_seg = self.S.copy()
            fixed_mask = (np.ones(n_comp, dtype=bool) if fix_known_spectra
                          else np.zeros(n_comp, dtype=bool))
            C_binned = np.zeros((n_bins, n_comp))
            prev_norm = np.inf

            for iteration in range(max_iter):
                # C step
                C_binned = self._solve_C_sparse(
                    binned_data, S_seg,
                    max_components_per_trace=max_components_per_trace,
                    sparsity_method='iterative_threshold')

                # Segment-level sparsity
                if max_components_per_segment is not None:
                    seg_contributions = np.sum(C_binned, axis=0)
                    n_active_seg = np.sum(seg_contributions > 0)
                    if n_active_seg > max_components_per_segment:
                        top_k = np.argsort(seg_contributions)[::-1][
                            :max_components_per_segment]
                        kill_mask = np.ones(n_comp, dtype=bool)
                        kill_mask[top_k] = False
                        C_binned[:, kill_mask] = 0

                # S step (free components only)
                S_new = S_seg.copy()
                free_idx = np.where(~fixed_mask)[0]
                if len(free_idx) > 0 and np.any(C_binned[:, free_idx] > 0):
                    if np.any(fixed_mask):
                        data_residual = (
                            binned_data
                            - C_binned[:, fixed_mask] @ S_seg[:, fixed_mask].T
                        )
                    else:
                        data_residual = binned_data.copy()

                    C_free = C_binned[:, free_idx]
                    active_rows = np.any(C_free > 0, axis=1)
                    if np.sum(active_rows) >= 1:
                        for ch in range(n_channels):
                            s_ch, _ = nnls(
                                C_free[active_rows],
                                data_residual[active_rows, ch])
                            S_new[ch, free_idx] = s_ch

                # Smoothness regularization
                if spectral_smoothness > 0:
                    for i in free_idx:
                        S_new[:, i] = gaussian_filter1d(
                            S_new[:, i], sigma=spectral_smoothness)

                # Normalize S, scale C
                for i in range(n_comp):
                    s_max = np.max(S_new[:, i])
                    if s_max > 1e-10 and not fixed_mask[i]:
                        C_binned[:, i] *= s_max
                        S_new[:, i] /= s_max

                S_seg = S_new

                # Convergence check
                res = binned_data - C_binned @ S_seg.T
                res_norm = np.linalg.norm(res)
                rel_change = abs(prev_norm - res_norm) / (prev_norm + 1e-10)
                if rel_change < tol and iteration > 5:
                    break
                prev_norm = res_norm

            # Map bins back to original time resolution
            for b in range(n_bins):
                for t in bin_membership[b]:
                    self.C[t, :] = C_binned[b, :]

            self.S_per_segment[seg_idx] = S_seg.copy()
            active_in_seg = np.any(C_binned > 0, axis=0)
            r2_seg = 1 - (res_norm**2) / (
                np.linalg.norm(binned_data)**2 + 1e-10)

            self.segment_results[seg_idx] = {
                'S': S_seg.copy(),
                'C_binned': C_binned.copy(),
                'bin_membership': bin_membership,
                'binned_data': binned_data,
                'active_components': np.where(active_in_seg)[0],
                'r2': r2_seg,
                'n_iterations': iteration + 1,
            }

            if progress_callback:
                progress_callback(seg_num + 1, n_segments)

        # Global residual
        reconstruction = np.zeros_like(data)
        for seg_idx in unique_segments:
            seg_mask = segment_labels == seg_idx
            if seg_idx in self.S_per_segment:
                reconstruction[seg_mask] = (
                    self.C[seg_mask] @ self.S_per_segment[seg_idx].T)

        self.residuals = data - reconstruction
        r2_global = 1 - (
            np.linalg.norm(self.residuals)**2 / np.linalg.norm(data)**2
        )
        return r2_global

    # ─── Accessors ────────────────────────────────────────────

    def get_component_contribution(self, idx):
        return np.outer(self.C[:, idx], self.S[:, idx])

    def get_active_components(self):
        if self.active_mask is None:
            active = np.any(self.C > 0, axis=0)
        else:
            active = self.active_mask
        indices = np.where(active)[0]
        names = [self.candidate_names[i] for i in indices]
        return indices, names
