"""Sparse MCR-ALS solver (private implementation)."""
import numpy as np
from scipy.optimize import nnls
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
from phase_analysis.pipeline.alignment import align_reference_to_data


class SparseMCR_ALS_broaden:
    """MCR-ALS with sparsity enforcement and candidate screening."""

    def __init__(self, two_theta):
        self.two_theta = two_theta
        self.n_channels = len(two_theta)
        self.S_candidates = None
        self.S_raw_all = None  # Raw (unbroadened) candidate spectra
        self.candidate_names = None
        self.n_candidates = 0
        self.C = None
        self.S = None
        self.S_raw = None  # Raw spectra for selected candidates
        self.active_mask = None
        self.residuals = None
        self.convergence_history = []
        self.segment_results = {}
        self.S_per_segment = {}

    def load_candidates(self, candidate_profiles, names=None, broaden_sigma=2):
        self.n_candidates = len(candidate_profiles)
        self.S_candidates = np.zeros((self.n_channels, self.n_candidates))
        self.S_raw_all = np.zeros((self.n_channels, self.n_candidates))

        for i, prof in enumerate(candidate_profiles):
            s = np.array(prof, dtype=float)
            s = np.clip(s, 0, None)
            s_norm = s / (np.max(s) + 1e-10)
            # Store raw (pre-broadening) normalized profile
            self.S_raw_all[:, i] = s_norm

            if broaden_sigma > 0:
                s = gaussian_filter1d(s, sigma=broaden_sigma)
                s = np.clip(s, 0, None)
                s = s / (np.max(s) + 1e-10)
                self.S_candidates[:, i] = s
            else:
                self.S_candidates[:, i] = s_norm

        if names is None:
            names = [f'Candidate_{i}' for i in range(self.n_candidates)]
        self.candidate_names = list(names)

    def _normalized_similarity(self, ref_a_norm, ref_b_norm, max_shift):
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
        corr = correlate(target, reference, mode='full')
        center = len(reference) - 1
        lo = max(0, center - max_shift)
        hi = min(len(corr), center + max_shift + 1)
        corr_window = corr[lo:hi]
        best_idx = np.argmax(corr_window)
        best_offset = best_idx - (center - lo)
        norm = np.sqrt(np.sum(reference**2) * np.sum(target**2)) + 1e-10
        return best_offset, corr_window[best_idx] / norm

    def screen_candidates(self, data, max_shift=50,
                          correlation_threshold=0.3,
                          min_time_presence=5,
                          dedup_threshold=0.95,
                          progress_callback=None):
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

        passes_threshold = presence_count >= min_time_presence
        passing_indices = np.where(passes_threshold)[0]
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
        self.S = self.S_candidates[:, indices].copy()
        self.S_raw = self.S_raw_all[:, indices].copy()
        self.candidate_names = [self.candidate_names[i] for i in indices]
        self.n_candidates = len(indices)
        self.active_mask = np.ones(self.n_candidates, dtype=bool)

    def _solve_C_sparse(self, data, S, max_components_per_trace=None,
                        sparsity_method='iterative_threshold',
                        l1_alpha=0.01):
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

    def _align_references_to_segment(self, S, seg_data, max_shift):
        """
        Re-align each column of S to the segment-local mean.

        Parameters
        ----------
        S : ndarray, shape (n_channels, n_comp)
            Current reference spectra (will be copied, not mutated).
        seg_data : ndarray, shape (n_seg_times, n_channels)
            Data for this segment.
        max_shift : int
            Maximum allowed shift in channels.

        Returns
        -------
        S_aligned : ndarray, shape (n_channels, n_comp)
            Re-aligned references.
        shifts : list of int
            Shift applied to each component.
        """
        n_comp = S.shape[1]
        S_aligned = S.copy()
        shifts = []
        seg_target = np.mean(seg_data, axis=0)

        for i in range(n_comp):
            ref = S[:, i]
            if np.max(ref) < 1e-10:
                shifts.append(0)
                continue
            aligned, shift, score = align_reference_to_data(
                ref, seg_target, max_shift)
            aligned = aligned / (np.max(aligned) + 1e-10)
            S_aligned[:, i] = aligned
            shifts.append(shift)

        return S_aligned, shifts

    def _optimize_component_broadening(self, binned_data, C_binned, S_current,
                                       S_raw_seg, component_idx,
                                       max_sigma=5.0, n_steps=11):
        """
        Find optimal per-component Gaussian broadening for a single component.

        Removes component i's contribution from the data, then tests different
        broadening sigmas applied to the raw (unbroadened) reference for that
        component. For each trial sigma, re-solves the single-component
        concentration via non-negative projection and evaluates the residual.

        Parameters
        ----------
        binned_data : ndarray, shape (n_bins, n_channels)
            Binned observed data for this segment.
        C_binned : ndarray, shape (n_bins, n_comp)
            Current concentration matrix for this segment.
        S_current : ndarray, shape (n_channels, n_comp)
            Current spectra matrix for this segment.
        S_raw_seg : ndarray, shape (n_channels, n_comp)
            Raw (unbroadened, but aligned) reference spectra for this segment.
        component_idx : int
            Index of the component to optimize broadening for.
        max_sigma : float
            Maximum Gaussian sigma (in channels) to try.
        n_steps : int
            Number of evenly-spaced sigma values to evaluate in [0, max_sigma].

        Returns
        -------
        best_sigma : float
            Optimal broadening sigma for this component.
        best_spec : ndarray, shape (n_channels,)
            The optimally broadened (and normalized) spectrum.
        best_c : ndarray, shape (n_bins,)
            Re-solved concentrations for this component with the optimal broadening.
        """
        i = component_idx
        n_bins = binned_data.shape[0]
        raw_spec = S_raw_seg[:, i]

        # If raw spectrum is empty, nothing to do
        if np.max(raw_spec) < 1e-10:
            return 0.0, S_current[:, i].copy(), C_binned[:, i].copy()

        # If component has no contribution currently, still try broadening
        # (it might become active with better broadening)

        # Compute data residual without component i's contribution
        other_mask = np.ones(S_current.shape[1], dtype=bool)
        other_mask[i] = False
        residual_without_i = binned_data - C_binned[:, other_mask] @ S_current[:, other_mask].T

        sigmas = np.linspace(0, max_sigma, n_steps)
        best_sigma = 0.0
        best_rnorm = np.inf
        best_spec = S_current[:, i].copy()
        best_c = C_binned[:, i].copy()

        for sigma in sigmas:
            if sigma > 0:
                broadened = gaussian_filter1d(raw_spec, sigma=sigma)
            else:
                broadened = raw_spec.copy()

            broadened = np.clip(broadened, 0, None)
            bmax = np.max(broadened)
            if bmax < 1e-10:
                continue
            broadened = broadened / bmax

            # Single-component NNLS reduces to non-negative projection:
            #   min || residual_without_i[t] - c_t * broadened ||^2,  c_t >= 0
            #   solution: c_t = max(0, dot(broadened, residual) / dot(broadened, broadened))
            norm_sq = np.dot(broadened, broadened) + 1e-10
            c_trial = np.zeros(n_bins)
            for t in range(n_bins):
                dot_val = np.dot(broadened, residual_without_i[t, :])
                c_trial[t] = max(0.0, dot_val / norm_sq)

            # Evaluate residual with this broadening
            res = residual_without_i - np.outer(c_trial, broadened)
            rnorm = np.linalg.norm(res)

            if rnorm < best_rnorm:
                best_rnorm = rnorm
                best_sigma = sigma
                best_spec = broadened.copy()
                best_c = c_trial.copy()

        return best_sigma, best_spec, best_c

    def fit_per_segment_binned(self, data, segment_labels,
                               bin_size=5,
                               max_components_per_trace=4,
                               max_components_per_segment=6,
                               fix_known_spectra=True,
                               spectral_smoothness=0.5,
                               align_per_segment=False,
                               max_shift=50,
                               max_broaden_sigma=0.0,
                               broaden_n_steps=11,
                               broaden_start_iter=3,
                               broaden_interval=3,
                               max_iter=100, tol=1e-4,
                               progress_callback=None):
        """
        Fit MCR-ALS per segment using binned data.

        Parameters
        ----------
        align_per_segment : bool
            If True, re-align reference spectra to each segment's local
            mean before ALS iterations. Handles thermal drift and phases
            that only appear in certain segments.
        max_shift : int
            Maximum shift (in channels) allowed during per-segment alignment.
            Only used when align_per_segment=True.
        max_broaden_sigma : float
            Maximum per-component Gaussian broadening sigma (in channels).
            Each component independently gets [0, max_broaden_sigma] broadening
            optimized to minimize the segment residual. Set to 0 to disable.
        broaden_n_steps : int
            Number of sigma values to evaluate per component (resolution of
            the broadening grid search).
        broaden_start_iter : int
            ALS iteration at which to begin broadening optimization
            (allows concentrations to stabilize first).
        broaden_interval : int
            Perform broadening optimization every N iterations.

        Returns
        -------
        r2_global : float
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

            # Start with global references
            S_seg = self.S.copy()
            S_raw_seg = self.S_raw.copy()

            # Per-segment alignment: re-align to this segment's data
            seg_shifts = None
            if align_per_segment:
                S_seg, seg_shifts = self._align_references_to_segment(
                    S_seg, seg_data, max_shift)
                # Also align the raw references so broadening uses aligned versions
                S_raw_seg, _ = self._align_references_to_segment(
                    S_raw_seg, seg_data, max_shift)

            fixed_mask = (np.ones(n_comp, dtype=bool) if fix_known_spectra
                          else np.zeros(n_comp, dtype=bool))

            C_binned = np.zeros((n_bins, n_comp))
            component_sigmas = np.zeros(n_comp)  # Per-component broadening
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

                # Per-component broadening optimization
                # Only for fixed components (their spectra are otherwise locked)
                # Applied after sparsity so we only broaden active components
                if (max_broaden_sigma > 0
                        and iteration >= broaden_start_iter
                        and iteration % broaden_interval == 0):
                    for i in range(n_comp):
                        # Only optimize broadening for fixed components
                        if not fixed_mask[i]:
                            continue
                        # Skip inactive components (but allow reactivation)
                        # A component might benefit from broadening even if
                        # currently zero, so we check raw spec validity instead
                        if np.max(S_raw_seg[:, i]) < 1e-10:
                            continue

                        sigma_i, spec_i, c_i = self._optimize_component_broadening(
                            binned_data, C_binned, S_seg, S_raw_seg, i,
                            max_sigma=max_broaden_sigma,
                            n_steps=broaden_n_steps)

                        S_seg[:, i] = spec_i
                        C_binned[:, i] = c_i
                        component_sigmas[i] = sigma_i

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

                # Smoothness (free components only)
                if spectral_smoothness > 0:
                    for i in free_idx:
                        S_new[:, i] = gaussian_filter1d(
                            S_new[:, i], sigma=spectral_smoothness)

                # Normalize
                for i in range(n_comp):
                    s_max = np.max(S_new[:, i])
                    if s_max > 1e-10 and not fixed_mask[i]:
                        C_binned[:, i] *= s_max
                        S_new[:, i] /= s_max

                S_seg = S_new

                # Convergence
                res = binned_data - C_binned @ S_seg.T
                res_norm = np.linalg.norm(res)
                rel_change = abs(prev_norm - res_norm) / (prev_norm + 1e-10)
                if rel_change < tol and iteration > 5:
                    break
                prev_norm = res_norm

            # Map bins back
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
                'alignment_shifts': seg_shifts,
                'component_broaden_sigmas': component_sigmas.copy(),
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
