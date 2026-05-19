# ============================================================
# PhaseTransitionsGUIv2.py — Integrated Phase Analysis GUI
# SMCR + Peak Match (MIP) pipelines
# ============================================================
import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, find_peaks
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from collections import Counter

# Optional GSASII imports
try:
    from GSASII import GSASIIscriptable as G2sc
    from GSASII import GSASIIpwd
    HAS_GSASII = True
except ImportError:
    HAS_GSASII = False

# Optional MIP import
try:
    from mip import Model, xsum, maximize, BINARY
    HAS_MIP = True
except ImportError:
    HAS_MIP = False

# Optional step_detect import — swap this back easily
try:
    import step_detect
    HAS_STEP_DETECT = True
except ImportError:
    HAS_STEP_DETECT = False


# ─── Utility functions ───────────────────────────────────────

def d_to_two_theta(d_array, wavelength):
    d_array = np.asarray(d_array, dtype=float)
    if np.any(d_array <= 0):
        raise ValueError(
            f"d-spacings must be positive. Got min = {d_array.min():.4f} Ang"
        )
    d_min = wavelength / 2.0
    if np.any(d_array < d_min):
        invalid_d = d_array[d_array < d_min]
        raise ValueError(
            f"Some d-spacings are too small for Bragg's Law at wavelength={wavelength} Ang.\n"
            f"  Minimum measurable d-spacing: {d_min:.4f} Ang (= lambda/2)\n"
            f"  Problematic d-spacings: {invalid_d}\n"
            f"  Require d >= lambda/2 so that sin(theta) <= 1.0"
        )
    sin_theta = wavelength / (2.0 * d_array)
    two_theta = np.rad2deg(2.0 * np.arcsin(sin_theta))
    return two_theta

def filter_reflections(data, max_index=5):
    """
    Remove reflections where any Miller index exceeds threshold.
    
    data : shape (4, N) — rows 0-2 are h,k,l; row 3 is peak values
    """
    mask = np.all(np.abs(data[:3]) <= max_index, axis=0)
    return data[:, mask]

# ============================================================
# PEAK DETECTION — easily swappable
# ============================================================
def detect_peaks_in_trace(trace, two_theta, sigma=3, threshold=0.01,
                          use_step_detect=True, step_threshold=0.01):
    """
    Find peak positions in a 1D trace.
    
    Set use_step_detect=True to use the original step_detect.find_steps
    Set use_step_detect=False to fall back to scipy find_peaks.
    
    Returns: indices into two_theta array
    """
    if use_step_detect and HAS_STEP_DETECT:
        # Original method: derivative + step detection
        gauss_filtered = gaussian_filter1d(trace, sigma, order=1)
        normalized = gauss_filtered / (np.max(np.abs(gauss_filtered)) + 1e-10)
        steps = step_detect.find_steps(normalized, step_threshold)
        return np.array(steps, dtype=int)
    else:
        # Scipy fallback
        smoothed = gaussian_filter1d(trace, sigma)
        height_thresh = threshold * np.max(smoothed)
        peaks, _ = find_peaks(smoothed, height=height_thresh,
                              distance=int(sigma * 2),
                              prominence=height_thresh * 0.5)
        return peaks


# ============================================================
# BACKGROUNDING — MIP-style (GSASII autoBkgCalc)
# ============================================================
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


# ============================================================
# GSASII POWDER SIMULATION
# ============================================================
def gsas2powdersim(cifpath, outpath, twotheta, instrument_file):
    """
    Simulate powder pattern from CIF using GSASII.
    Produces an .xy file with x-axis matching input two_theta spacing.
    """
    if not HAS_GSASII:
        raise ImportError("GSASII required for powder simulation")

    gpx = G2sc.G2Project(newgpx='my_simulation.gpx')

    filename = os.path.basename(cifpath)
    savefilename = os.path.join(outpath, filename + ".xy")

    phase = gpx.add_phase(phasefile=cifpath, phasename='MyPhase')
    hist = gpx.add_simulated_powder_histogram(
        histname='Simulation',
        iparams=instrument_file,
        Tmin=np.min(twotheta),
        Tmax=np.max(twotheta),
        Tstep=(np.max(twotheta) - np.min(twotheta)) / (len(twotheta) - 1),
        phases=[phase]
    )
    gpx.do_refinements([{}])

    x, y = hist.getdata('X'), hist.getdata('Ycalc')
    np.savetxt(savefilename, np.column_stack([x, y]),
               header="2theta intensity", fmt="%.6f")
    return x, y


def build_powder_cache_from_cifs(cif_folder, output_folder, twotheta,
                                  instrument_file, progress_callback=None):
    """
    Batch-simulate all CIFs in a folder → cached .xy powder patterns.
    Returns dict {name: profile_array} ready for SMCR.
    """
    if not HAS_GSASII:
        return {}, []

    os.makedirs(output_folder, exist_ok=True)
    cif_files = sorted([f for f in os.listdir(cif_folder) if f.endswith('.cif')])
    profiles = {}
    names = []

    for i, cif in enumerate(cif_files):
        cifpath = os.path.join(cif_folder, cif)
        try:
            x, y = gsas2powdersim(cifpath, output_folder, twotheta,
                                   instrument_file)
            basename = cif[:-4]
            profiles[basename] = y
            names.append(basename)
        except Exception:
            print("FAILED POWDER SIM")
            pass

        if progress_callback:
            progress_callback(i + 1, len(cif_files))

    return profiles, names


def load_precomputed_powders(powder_folder):
    """
    Load pre-computed powder patterns from .xy files (the original approach).
    Each file: 2 columns (2theta, intensity), 1 header line.
    
    Returns: candidate_profiles (list of arrays), candidate_names (list of str)
    """
    candidate_profiles = []
    candidate_names = []

    for file in sorted(os.listdir(powder_folder)):
        if file.startswith('.'):
            continue
        filepath = os.path.join(powder_folder, file)
        try:
            df = np.genfromtxt(filepath, skip_header=1, unpack=True)
            candidate_profiles.append(df[1])
            candidate_names.append(file)
        except Exception:
            pass

    return candidate_profiles, candidate_names


# ============================================================
# REFERENCE ALIGNMENT (from original pipeline)
# ============================================================
def cross_correlate_shift(reference, target, max_shift=50):
    """Find optimal shift to align reference to target."""
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


def prepare_references_for_mcr(candidate_profiles, data, segments,
                                max_shift=50, broaden_sigma=2):
    """
    Prepare reference profiles for MCR-ALS:
    1. Broaden (optional)
    2. Align to data
    3. Normalize
    """
    prepared = []
    shifts_found = []
    target = np.mean(data, axis=0)

    for i, prof in enumerate(candidate_profiles):
        ref = np.array(prof, dtype=float)
        if broaden_sigma > 0:
            ref = gaussian_filter1d(ref, sigma=broaden_sigma)
        aligned, shift, score = align_reference_to_data(ref, target, max_shift)
        aligned = aligned / (aligned.max() + 1e-10)
        prepared.append(aligned)
        shifts_found.append(shift)

    return prepared, shifts_found


# ============================================================
# CANDIDATE LIBRARY (for MIP peak matching)
# ============================================================
def load_candidate_library(peak_folder):
    """Load peak position CSVs for MIP matching."""
    candidate_library = {}
    csv_files = glob.glob(os.path.join(peak_folder, "*.csv"))

    for filepath in csv_files:
        name = os.path.splitext(os.path.basename(filepath))[0]
        try:
            peaks = np.loadtxt(filepath, delimiter=",", ndmin=1)
            if peaks.ndim > 1:
                peaks = peaks[:, 0]
            peaks = peaks[~np.isnan(peaks)]
            if len(peaks) > 0:
                candidate_library[name] = peaks
        except ValueError:
            try:
                peaks = np.loadtxt(filepath, delimiter=",",
                                   skiprows=1, ndmin=1)
                if peaks.ndim > 1:
                    peaks = peaks[:, 0]
                peaks = peaks[~np.isnan(peaks)]
                if len(peaks) > 0:
                    candidate_library[name] = peaks
            except Exception:
                pass

    return candidate_library


# ============================================================
# SPARSE MCR-ALS CLASS
# ============================================================
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

    def load_candidates(self, candidate_profiles, names=None,
                        broaden_sigma=2):
        """
        Load candidate spectral profiles (pre-simulated powder patterns).
        These should already be on the correct two_theta grid from gsas2powdersim.
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

    def _normalized_similarity(self, ref_a_norm, ref_b_norm, max_shift):
        """Shift-tolerant normalized similarity."""
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
        norm = (np.sqrt(np.sum(reference**2) * np.sum(target**2)) + 1e-10)
        return best_offset, corr_window[best_idx] / norm

    def screen_candidates(self, data, max_shift=50,
                          correlation_threshold=0.3,
                          min_time_presence=5,
                          dedup_threshold=0.95,
                          progress_callback=None):
        """
        Pre-screen which candidates could plausibly be in the data.
        Returns: plausible_indices, best_correlations, presence_counts
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
        passes_threshold = (presence_count >= min_time_presence)
        passing_indices = np.where(passes_threshold)[0]

        # Deduplicate: sort by best correlation descending
        sorted_passing = passing_indices[
            np.argsort(best_correlations[passing_indices])[::-1]]
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

    def _solve_C_sparse(self, data, S, max_components_per_trace=None,
                        sparsity_method='iterative_threshold',
                        l1_alpha=0.01):
        """Solve for C with sparsity constraint."""
        n_times = data.shape[0]
        n_comp = S.shape[1]
        C = np.zeros((n_times, n_comp))

        for t in range(n_times):
            target = data[t, :]

            if sparsity_method == 'hard_threshold':
                c, _ = nnls(S, target)
                if (max_components_per_trace is not None and
                        np.sum(c > 0) > max_components_per_trace):
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
                while (max_components_per_trace is not None and
                       np.sum(c > 0) > max_components_per_trace):
                    contributions = c * np.max(S, axis=0)
                    nonzero_mask = c > 0
                    if not np.any(nonzero_mask):
                        break
                    weakest = np.where(nonzero_mask)[0][
                        np.argmin(contributions[nonzero_mask])]
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
                    if (max_components_per_trace and
                            np.sum(surviving) > max_components_per_trace):
                        top_k = np.argsort(c)[::-1][:max_components_per_trace]
                        surviving = np.zeros(n_comp, dtype=bool)
                        surviving[top_k] = True
                    c_refined, _ = nnls(S[:, surviving], target)
                    c = np.zeros(n_comp)
                    c[surviving] = c_refined
                C[t, :] = c

        return C

    def _solve_S_method(self, data, C, fixed_mask=None):
        """Solve for S given C with optional fixed components."""
        n_channels = data.shape[1]
        n_comp = C.shape[1]
        S_new = (self.S.copy() if self.S is not None
                 else np.zeros((n_channels, n_comp)))

        if fixed_mask is None:
            fixed_mask = np.zeros(n_comp, dtype=bool)

        free_idx = np.where(~fixed_mask)[0]
        if len(free_idx) == 0:
            return S_new

        if np.any(fixed_mask):
            data_residual = data - C[:, fixed_mask] @ S_new[:, fixed_mask].T
        else:
            data_residual = data.copy()

        C_free = C[:, free_idx]
        for ch in range(n_channels):
            s_ch, _ = nnls(C_free, data_residual[:, ch])
            S_new[ch, free_idx] = s_ch

        return S_new

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
        Handles all cases: bin_size=1 through bin_size=segment_length.
        Segments smaller than bin_size become their own single bin.
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

            # Bin the segment — segments < bin_size get 1 bin
            n_bins = max(1, n_seg // bin_size)
            binned_data = np.zeros((n_bins, n_channels))
            bin_membership = []

            for b in range(n_bins):
                start = b * bin_size
                end = ((b + 1) * bin_size if b < n_bins - 1 else n_seg)
                binned_data[b, :] = np.mean(seg_data[start:end, :], axis=0)
                bin_membership.append(seg_times[start:end])

            # ALS on binned data
            S_seg = self.S.copy()
            fixed_mask = (np.ones(n_comp, dtype=bool) if fix_known_spectra
                          else np.zeros(n_comp, dtype=bool))
            C_binned = np.zeros((n_bins, n_comp))
            prev_norm = np.inf

            for iteration in range(max_iter):
                C_binned = self._solve_C_sparse(
                    binned_data, S_seg,
                    max_components_per_trace=max_components_per_trace,
                    sparsity_method='iterative_threshold')

                if max_components_per_segment is not None:
                    seg_contributions = np.sum(C_binned, axis=0)
                    n_active_seg = np.sum(seg_contributions > 0)
                    if n_active_seg > max_components_per_segment:
                        top_k = np.argsort(seg_contributions)[::-1][
                            :max_components_per_segment]
                        kill_mask = np.ones(n_comp, dtype=bool)
                        kill_mask[top_k] = False
                        C_binned[:, kill_mask] = 0

                S_new = S_seg.copy()
                free_idx = np.where(~fixed_mask)[0]
                if (len(free_idx) > 0 and
                        np.any(C_binned[:, free_idx] > 0)):
                    if np.any(fixed_mask):
                        data_residual = (binned_data -
                                         C_binned[:, fixed_mask] @
                                         S_seg[:, fixed_mask].T)
                    else:
                        data_residual = binned_data.copy()
                    C_free = C_binned[:, free_idx]
                    active_rows = np.any(C_free > 0, axis=1)
                    if np.sum(active_rows) >= 1:
                        for ch in range(n_channels):
                            s_ch, _ = nnls(C_free[active_rows],
                                           data_residual[active_rows, ch])
                            S_new[ch, free_idx] = s_ch

                if spectral_smoothness > 0:
                    for i in free_idx:
                        S_new[:, i] = gaussian_filter1d(
                            S_new[:, i], sigma=spectral_smoothness)

                for i in range(n_comp):
                    s_max = np.max(S_new[:, i])
                    if s_max > 1e-10 and not fixed_mask[i]:
                        C_binned[:, i] *= s_max
                        S_new[:, i] /= s_max

                S_seg = S_new
                res = binned_data - C_binned @ S_seg.T
                res_norm = np.linalg.norm(res)
                rel_change = abs(prev_norm - res_norm) / (prev_norm + 1e-10)
                if rel_change < tol and iteration > 5:
                    break
                prev_norm = res_norm

            # Map back to original time resolution
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
        r2_global = 1 - (np.linalg.norm(self.residuals)**2 /
                         (np.linalg.norm(data)**2))
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


# ============================================================
# MIP SEARCH-MATCH
# ============================================================
def mip_search_match(test_peaks, candidate_library, tolerance,
                     parsimony_weight=1.0, false_positive_weight=0.5):
    """MIP search-match: find optimal subset of candidates."""
    if not HAS_MIP:
        raise ImportError("python-mip required for Peak Match")

    n_test = len(test_peaks)
    if n_test == 0:
        return []

    candidates = list(candidate_library.keys())
    n_cand = len(candidates)

    align_score = np.zeros((n_test, n_cand))
    false_pos_count = np.zeros(n_cand)

    for j, name in enumerate(candidates):
        cand_peaks = np.array(candidate_library[name])
        for i, tp in enumerate(test_peaks):
            dists = np.abs(cand_peaks - tp)
            min_dist = dists.min() if len(dists) > 0 else np.inf
            if min_dist <= tolerance:
                align_score[i, j] = 1.0 - (min_dist / tolerance) ** 2
        for cp in cand_peaks:
            if np.min(np.abs(test_peaks - cp)) > tolerance:
                false_pos_count[j] += 1

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
        xsum(align_score[i, j] * y[i][j]
             for i in range(n_test) for j in range(n_cand))
        - parsimony_weight * xsum(z[j] for j in range(n_cand))
        - false_positive_weight * xsum(
            false_pos_count[j] * z[j] for j in range(n_cand))
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
                            progress_callback=None):
    """
    Run MIP search-match per bin within each segment.
    Bins smaller than bin_size (tail of segment) are their own bin.
    """
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

            # Find peaks
            peak_indices = detect_peaks_in_trace(
                bin_avg, two_theta, sigma=peak_sigma,
                threshold=peak_threshold,
                use_step_detect=use_step_detect,
                step_threshold=step_threshold)
            test_peaks = two_theta[peak_indices]

            if len(test_peaks) > 0:
                selected = mip_search_match(
                    test_peaks, candidate_library, tolerance,
                    parsimony_weight, false_positive_weight)
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


# ============================================================
# SEGMENTATION
# ============================================================
def detect_transitions(data, weights, threshold_pct):
    """Detect phase transitions from time-evolving data."""
    n_times, n_channels = data.shape

    dI_dt = np.diff(data, axis=0)
    dI_dt = np.vstack([dI_dt, dI_dt[-1:]])

    dissimilarity = np.zeros(n_times)
    for t in range(1, n_times):
        a, b = data[t - 1], data[t]
        dot = np.dot(a, b)
        norm = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        dissimilarity[t] = 1 - dot / norm

    intensity_change = np.sum(np.abs(dI_dt), axis=1)
    threshold_ch = np.percentile(np.abs(dI_dt), 90)
    channel_change = np.sum(
        np.abs(dI_dt) > threshold_ch, axis=1).astype(float)

    rank_change = np.zeros(n_times)
    for t in range(1, n_times):
        r1 = np.argsort(np.argsort(data[t - 1]))
        r2 = np.argsort(np.argsort(data[t]))
        rank_change[t] = np.sum(np.abs(r1 - r2))

    def norm01(x):
        mx = np.max(x)
        return x / mx if mx > 0 else x

    dissimilarity = norm01(dissimilarity)
    intensity_change = norm01(intensity_change)
    channel_change = norm01(channel_change)
    rank_change = norm01(rank_change)

    combined = (weights.get('dissimilarity', 1.0) * dissimilarity +
                weights.get('intensity_change', 1.0) * intensity_change +
                weights.get('channel_change', 1.0) * channel_change +
                weights.get('rank_change', 1.0) * rank_change)
    combined = norm01(combined)

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
    """Convert transition list to per-frame segment labels."""
    labels = np.zeros(n_times, dtype=int)
    boundaries = [0] + sorted(transitions) + [n_times]
    for seg_idx in range(len(boundaries) - 1):
        labels[boundaries[seg_idx]:boundaries[seg_idx + 1]] = seg_idx
    return labels


# ============================================================
# RESULTS WINDOW
# ============================================================
class ResultsWindow:
    """Second window for search/match results."""

    def __init__(self, parent, title="Search Results"):
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("1400x900")

        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig = Figure(figsize=(14, 9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="")
        ttk.Label(main_frame, textvariable=self.status_var,
                  font=('Consolas', 9)).pack(fill=tk.X, pady=2)

    def show_smcr_results(self, smcr, data, two_theta, transitions,
                          segment_labels):
        """Display SMCR fit summary."""
        self.fig.clear()
        import matplotlib.cm as cm

        unique_segments = np.unique(segment_labels)
        n_comp = smcr.C.shape[1]

        active_mask = np.any(smcr.C > 0, axis=0)
        active_idx = np.where(active_mask)[0]
        active_names = [smcr.candidate_names[i] for i in active_idx]
        n_active = len(active_idx)

        if n_active == 0:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No active components found",
                    ha='center', va='center', fontsize=14)
            self.canvas.draw()
            return

        gs = GridSpec(3, 1, figure=self.fig,
                      height_ratios=[5, 1.5, 2], hspace=0.35)

        # Activity map
        ax1 = self.fig.add_subplot(gs[0])
        colors = [cm.tab10(i / max(n_active, 1)) for i in range(n_active)]

        for seg_idx in unique_segments:
            if seg_idx not in smcr.segment_results:
                continue
            seg_res = smcr.segment_results[seg_idx]
            seg_mask = segment_labels == seg_idx
            seg_times = np.where(seg_mask)[0]
            t_start, t_end = seg_times[0], seg_times[-1]

            for y_pos, comp_i in enumerate(active_idx):
                if comp_i in seg_res['active_components']:
                    mean_c = np.mean(smcr.C[seg_mask, comp_i])
                    max_c = np.max(smcr.C[:, comp_i]) + 1e-10
                    alpha = 0.3 + 0.7 * (mean_c / max_c)
                    ax1.barh(y_pos, t_end - t_start, left=t_start,
                             height=0.8, alpha=alpha, color=colors[y_pos],
                             edgecolor=colors[y_pos], linewidth=0.5)

        ax1.set_yticks(range(n_active))
        ax1.set_yticklabels(active_names, fontsize=9)
        ax1.set_xlabel('Time step')
        ax1.set_title(f'Component Activity ({n_active} phases found)')
        for tr in transitions:
            ax1.axvline(tr, color='gray', alpha=0.6, linestyle='--', lw=0.8)
        ax1.set_xlim(0, data.shape[0])
        ax1.grid(True, alpha=0.2, axis='x')

        # R² per segment
        ax2 = self.fig.add_subplot(gs[1])
        seg_r2, seg_centers, seg_widths = [], [], []
        for seg_idx in unique_segments:
            if seg_idx not in smcr.segment_results:
                continue
            seg_mask = segment_labels == seg_idx
            seg_times = np.where(seg_mask)[0]
            seg_r2.append(smcr.segment_results[seg_idx]['r2'])
            seg_centers.append((seg_times[0] + seg_times[-1]) / 2)
            seg_widths.append(seg_times[-1] - seg_times[0])

        ax2.bar(seg_centers, seg_r2, width=seg_widths,
                alpha=0.6, color='steelblue', edgecolor='navy', linewidth=0.5)
        ax2.set_ylabel('R²')
        ax2.set_xlabel('Time step')
        ax2.set_title('Fit Quality per Segment')
        if seg_r2:
            ax2.set_ylim(min(0.8, min(seg_r2) - 0.05), 1.0)
        ax2.axhline(0.95, color='green', linestyle=':', alpha=0.5)
        ax2.axhline(0.90, color='orange', linestyle=':', alpha=0.5)
        for tr in transitions:
            ax2.axvline(tr, color='gray', alpha=0.4, linestyle='--', lw=0.8)
        ax2.set_xlim(0, data.shape[0])
        ax2.grid(True, alpha=0.2)

        # Residual heatmap
        ax3 = self.fig.add_subplot(gs[2])
        res_vmax = np.percentile(np.abs(smcr.residuals), 97)
        ax3.imshow(smcr.residuals.T, aspect='auto', cmap='RdBu_r',
                   vmin=-res_vmax, vmax=res_vmax,
                   extent=[0, data.shape[0], two_theta[-1], two_theta[0]],
                   origin='upper')
        ax3.set_xlabel('Time step')
        ax3.set_ylabel('2θ (°)')
        ax3.set_title('Residual')
        for tr in transitions:
            ax3.axvline(tr, color='lime', alpha=0.5, linestyle='--', lw=0.8)

        self.fig.tight_layout()
        self.canvas.draw()

        r2_global = 1 - (np.linalg.norm(smcr.residuals)**2 /
                         (np.linalg.norm(data)**2))
        self.status_var.set(
            f"SMCR Complete — {n_active} active phases, "
            f"Global R² = {r2_global:.5f}")

    def show_mip_results(self, selections, data, two_theta, transitions,
                         segment_labels, bin_size=5):
        """Display MIP peak-match results as timeline heatmap."""
        self.fig.clear()
        import matplotlib.cm as cm
        
        
        all_candidates = [c for _, sel in selections for c in sel]
        if not all_candidates:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No candidates matched",
                    ha='center', va='center', fontsize=14)
            self.canvas.draw()
            return

        freq = Counter(all_candidates)
        unique_candidates = [c for c, _ in freq.most_common()]
        n_time = data.shape[0]
        n_cand = len(unique_candidates)

        # Build presence matrix using actual bin width
        presence = np.zeros((n_cand, n_time), dtype=int)
        half_bin = max(1, bin_size // 2)

        for time_idx, sel in selections:
            for name in sel:
                row = unique_candidates.index(name)
                # Fill the actual bin width centered on the bin center
                t_start = max(0, time_idx - half_bin)
                t_end = min(n_time, time_idx + half_bin + 1)
                presence[row, t_start:t_end] = 1
                    
        gs = GridSpec(2, 1, figure=self.fig,
                      height_ratios=[1, max(n_cand, 4)], hspace=0.08)


        # Segment bar
        ax_seg = self.fig.add_subplot(gs[0])
        unique_segs = np.unique(segment_labels)
        for i, seg_idx in enumerate(unique_segs):
            seg_mask = segment_labels == seg_idx
            seg_times = np.where(seg_mask)[0]
            start, end = seg_times[0], seg_times[-1]
            ax_seg.axvspan(start - 0.5, end + 0.5,
                           color=cm.Set3(i / max(len(unique_segs), 1)),
                           alpha=0.7)
            mid = (start + end) / 2
            ax_seg.text(mid, 0.5, f"Seg {i + 1}", ha='center',
                        va='center', fontsize=8, fontweight='bold')
        ax_seg.set_xlim(-0.5, n_time - 0.5)
        ax_seg.set_ylim(0, 1)
        ax_seg.set_yticks([])
        ax_seg.set_title("Phase Composition Over Time (Peak Match)")

        # Heatmap
        ax_heat = self.fig.add_subplot(gs[1])
        cmap_hm = ListedColormap(['#f0f0f0', '#2c7bb6'])
        ax_heat.imshow(presence, aspect='auto', cmap=cmap_hm,
                       interpolation='none', origin='upper',
                       extent=[-0.5, n_time - 0.5, n_cand - 0.5, -0.5])
        for tr in transitions:
            ax_heat.axvline(tr, color='red', linewidth=1.5,
                            linestyle='--', alpha=0.7)
        ax_heat.set_yticks(range(n_cand))
        ax_heat.set_yticklabels(unique_candidates, fontsize=9)
        ax_heat.set_xlabel("Time Step")
        ax_heat.set_ylabel("Candidate Phase")
        for i in range(n_cand - 1):
            ax_heat.axhline(i + 0.5, color='white', linewidth=0.5)

        self.fig.tight_layout()
        self.canvas.draw()
        self.status_var.set(
            f"Peak Match Complete — {n_cand} unique phases across "
            f"{len(selections)} bins")


# ============================================================
# MAIN GUI
# ============================================================
class TransitionAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Phase Transition Analysis")
        self.root.geometry("1600x1050")

        self.data = None
        self.two_theta = None
        self.timeslices = None
        self.n_times = 0
        self.n_channels = 0
        self.transitions = []
        self.segment_labels = None
        self.cif_folder = None

        # Candidate data
        self.candidate_library = {}        # MIP: {name: peak_positions}
        self.candidate_profiles = []       # SMCR: list of profile arrays
        self.candidate_names = []          # SMCR: names matching profiles
        self.powder_cache_folder = None    # path to cached .xy simulations

        self._build_ui()

    def _build_ui(self):
        # DOUBLED control panel width
        self.controls_frame = ttk.Frame(self.root, width=600)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.controls_frame.pack_propagate(False)

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_controls()

        self.fig = Figure(figsize=(10, 9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.axes = []

    def _make_instrument_file(self):
        wavelength = self.wavelength_var.get()
        instprm_content = f"""#GSAS-II instrument parameter file;
Type:PXC;Bank:1
Lam:{wavelength};Zero:0.0;Polariz.:0.7;Azimuth:0.0
U:2.0;V:-2.0;W:5.0;X:0.0;Y:0.0;Z:0.0;SH/L:0.002
"""
        outpath = os.path.join(os.getcwd(), 'Instrument.instprm')
        with open(outpath, 'w') as f:
            f.write(instprm_content)
        self.inst_file_var.set(outpath)
        self.status_var.set(f"Instrument file created: {outpath}")

    def _load_cif_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing CIF files")
        if not folder:
            return
        cif_files = [f for f in os.listdir(folder) if f.lower().endswith('.cif')]
        if not cif_files:
            self.status_var.set("No .cif files found in selected folder.")
            return
        self.cif_folder = folder
        self.cif_folder_var.set(f"{os.path.basename(folder)} ({len(cif_files)} CIFs)")
        self.status_var.set(f"Loaded CIF folder: {folder} — {len(cif_files)} files found")
            
    def _build_controls(self):
        frame = self.controls_frame

        # ── Data Loading ──
        lf_data = ttk.LabelFrame(frame, text="Data Loading")
        lf_data.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(lf_data, text="Load Data Folder",
                   command=self._load_data).pack(fill=tk.X, padx=5, pady=2)

        bg_frame = ttk.Frame(lf_data)
        bg_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(bg_frame, text="Background logλ:").pack(side=tk.LEFT)
        self.bg_lambda_var = tk.IntVar(value=4)
        ttk.Spinbox(bg_frame, from_=1, to=10, width=5,
                    textvariable=self.bg_lambda_var).pack(side=tk.RIGHT)

        # Wavelength
        wl_frame = ttk.Frame(lf_data)
        wl_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(wl_frame, text="Wavelength λ (Å):").pack(side=tk.LEFT)
        self.wavelength_var = tk.DoubleVar(value=0.25450)
        ttk.Entry(wl_frame, textvariable=self.wavelength_var,
                  width=10).pack(side=tk.RIGHT)

        # ── Candidate Patterns ──
        lf_cand = ttk.LabelFrame(frame, text="Candidate Patterns")
        lf_cand.pack(fill=tk.X, padx=5, pady=5)

        # For SMCR: load pre-computed powder .xy files
        ttk.Button(lf_cand, text="Load Powder Profiles (.xy folder)",
                   command=self._load_powder_profiles).pack(
                       fill=tk.X, padx=5, pady=2)


        # ── Load CIF folder (shared by simulate & peak-list) ──
        cif_frame = ttk.Frame(lf_cand)
        cif_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(cif_frame, text="Load CIF Folder",
                   command=self._load_cif_folder).pack(side=tk.LEFT, padx=2)
        self.cif_folder_var = tk.StringVar(value="No CIF folder loaded")
        ttk.Label(cif_frame, textvariable=self.cif_folder_var,
                  font=('Consolas', 8)).pack(side=tk.LEFT, padx=5)

        # Action buttons (require CIF folder to be loaded first)
        if HAS_GSASII:
            ttk.Button(lf_cand, text="Simulate Powder Profiles (GSASII)",
                       command=self._simulate_from_cifs).pack(fill=tk.X, padx=5, pady=2)
            ttk.Button(lf_cand, text="Generate Peak Lists (for Peak Match)",
                       command=self._generate_peak_lists).pack(fill=tk.X, padx=5, pady=2)

        # Instrument file (for GSASII simulation)
        inst_frame = ttk.Frame(lf_cand)
        
        inst_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(inst_frame, text="Instrument file:").pack(side=tk.LEFT)
        self.inst_file_var = tk.StringVar(value="")
        ttk.Entry(inst_frame, textvariable=self.inst_file_var,
                  width=25).pack(side=tk.LEFT, padx=2)
        ttk.Button(inst_frame, text="...",
                   command=self._select_inst_file, width=3).pack(side=tk.LEFT)

        self.cand_status_var = tk.StringVar(value="No candidates loaded")
        ttk.Label(lf_cand, textvariable=self.cand_status_var,
                  font=('Consolas', 8), wraplength=580).pack(
                      fill=tk.X, padx=5, pady=2)

        ttk.Button(inst_frame, text="Make",
           command=self._make_instrument_file, width=5).pack(side=tk.LEFT, padx=2)
        
        # ── Segmentation Parameters ──
        lf_seg = ttk.LabelFrame(frame, text="Segmentation Parameters")
        lf_seg.pack(fill=tk.X, padx=5, pady=5)

        thresh_frame = ttk.Frame(lf_seg)
        thresh_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(thresh_frame, text="Threshold percentile:").pack(
            side=tk.LEFT)
        self.threshold_var = tk.IntVar(value=90)
        self.thresh_label = ttk.Label(thresh_frame, text="90")
        self.thresh_label.pack(side=tk.RIGHT)
        ttk.Scale(thresh_frame, from_=50, to=99,
                  variable=self.threshold_var, orient=tk.HORIZONTAL,
                  command=self._on_param_change).pack(
                      side=tk.RIGHT, fill=tk.X, expand=True)

        self.weight_vars = {}
        weight_names = ['dissimilarity', 'intensity_change',
                        'channel_change', 'rank_change']
        for wname in weight_names:
            wf = ttk.Frame(lf_seg)
            wf.pack(fill=tk.X, padx=5, pady=1)
            ttk.Label(wf, text=f"{wname}:").pack(side=tk.LEFT)
            var = tk.DoubleVar(value=1.0)
            lbl = ttk.Label(wf, text="1.0")
            lbl.pack(side=tk.RIGHT)
            var._label = lbl
            ttk.Scale(wf, from_=0, to=3.0, variable=var,
                      orient=tk.HORIZONTAL,
                      command=self._on_param_change).pack(
                          side=tk.RIGHT, fill=tk.X, expand=True)
            self.weight_vars[wname] = var

        ttk.Button(lf_seg, text="Update Segmentation",
                   command=self._update_plots).pack(fill=tk.X, padx=5, pady=5)

        # ── Search / Match ──
        lf_match = ttk.LabelFrame(frame, text="Search / Match")
        lf_match.pack(fill=tk.X, padx=5, pady=5)

        # Method choice
        method_frame = ttk.Frame(lf_match)
        method_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="SMCR")
        ttk.Radiobutton(method_frame, text="SMCR",
                        variable=self.method_var,
                        value="SMCR").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(method_frame, text="Peak Match",
                        variable=self.method_var,
                        value="Peak Match").pack(side=tk.LEFT, padx=10)

        # Bin size (shared)
        bin_frame = ttk.Frame(lf_match)
        bin_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(bin_frame, text="Bin size (frames):").pack(side=tk.LEFT)
        self.bin_size_var = tk.IntVar(value=5)
        ttk.Spinbox(bin_frame, from_=1, to=100, width=6,
                    textvariable=self.bin_size_var).pack(side=tk.RIGHT)

        # SMCR options
        lf_smcr = ttk.LabelFrame(lf_match, text="SMCR Options")
        lf_smcr.pack(fill=tk.X, padx=5, pady=2)

        sf1 = ttk.Frame(lf_smcr)
        sf1.pack(fill=tk.X, padx=3, pady=1)
        ttk.Label(sf1, text="Max components/trace:").pack(side=tk.LEFT)
        self.max_comp_trace_var = tk.IntVar(value=4)
        ttk.Spinbox(sf1, from_=1, to=20, width=5,
                    textvariable=self.max_comp_trace_var).pack(side=tk.RIGHT)

        sf2 = ttk.Frame(lf_smcr)
        sf2.pack(fill=tk.X, padx=3, pady=1)
        ttk.Label(sf2, text="Max components/segment:").pack(side=tk.LEFT)
        self.max_comp_seg_var = tk.IntVar(value=6)
        ttk.Spinbox(sf2, from_=1, to=30, width=5,
                    textvariable=self.max_comp_seg_var).pack(side=tk.RIGHT)

        sf3 = ttk.Frame(lf_smcr)
        sf3.pack(fill=tk.X, padx=3, pady=1)
        ttk.Label(sf3, text="Broaden sigma (ch):").pack(side=tk.LEFT)
        self.broaden_var = tk.DoubleVar(value=2.0)
        ttk.Spinbox(sf3, from_=0, to=10, width=5, increment=0.5,
                    textvariable=self.broaden_var).pack(side=tk.RIGHT)

        sf4 = ttk.Frame(lf_smcr)
        sf4.pack(fill=tk.X, padx=3, pady=1)
        ttk.Label(sf4, text="Correlation threshold:").pack(side=tk.LEFT)
        self.corr_thresh_var = tk.DoubleVar(value=0.3)
        ttk.Spinbox(sf4, from_=0.1, to=0.9, width=5, increment=0.05,
                    textvariable=self.corr_thresh_var).pack(side=tk.RIGHT)

        sf5 = ttk.Frame(lf_smcr)
        sf5.pack(fill=tk.X, padx=3, pady=1)
        ttk.Label(sf5, text="Max shift (channels):").pack(side=tk.LEFT)
        self.max_shift_var = tk.IntVar(value=50)
        ttk.Spinbox(sf5, from_=0, to=200, width=5,
                    textvariable=self.max_shift_var).pack(side=tk.RIGHT)

        # MIP options
        lf_mip = ttk.LabelFrame(lf_match, text="Peak Match Options")
        lf_mip.pack(fill=tk.X, padx=5, pady=2)

        mf1 = ttk.Frame(lf_mip)
        mf1.pack(fill=tk.X, padx=3, pady=1)
        ttk.Label(mf1, text="2θ tolerance (°):").pack(side=tk.LEFT)
        self.tolerance_var = tk.DoubleVar(value=0.2)
        ttk.Spinbox(mf1, from_=0.01, to=1.0, width=6, increment=0.05,
                    textvariable=self.tolerance_var).pack(side=tk.RIGHT)

        mf2 = ttk.Frame(lf_mip)
        mf2.pack(fill=tk.X, padx=3, pady=1)
        ttk.Label(mf2, text="Parsimony weight:").pack(side=tk.LEFT)
        self.parsimony_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(mf2, from_=0, to=5.0, width=6, increment=0.1,
                    textvariable=self.parsimony_var).pack(side=tk.RIGHT)

        mf3 = ttk.Frame(lf_mip)
        mf3.pack(fill=tk.X, padx=3, pady=1)
        ttk.Label(mf3, text="False positive weight:").pack(side=tk.LEFT)
        self.fp_weight_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(mf3, from_=0, to=5.0, width=6, increment=0.1,
                    textvariable=self.fp_weight_var).pack(side=tk.RIGHT)

        # Peak detection method toggle
        mf4 = ttk.Frame(lf_mip)
        mf4.pack(fill=tk.X, padx=3, pady=1)
        self.use_step_detect_var = tk.BooleanVar(value=HAS_STEP_DETECT)
        ttk.Checkbutton(mf4, text="Use step_detect (uncheck=scipy peaks)",
                        variable=self.use_step_detect_var).pack(
                            side=tk.LEFT)

        # Run button
        ttk.Button(lf_match, text="▶  Run Search / Match",
                   command=self._run_search).pack(fill=tk.X, padx=5, pady=8)

        # ── Status ──
        self.status_var = tk.StringVar(value="Load data to begin")
        ttk.Label(frame, textvariable=self.status_var,
                  font=('Consolas', 9), wraplength=580,
                  justify=tk.LEFT).pack(fill=tk.X, padx=5, pady=5)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=2)

    # ──────────────────────────────────────────────────────────
    # LOADING
    # ──────────────────────────────────────────────────────────

    def _load_data(self):
        folder = filedialog.askdirectory(title="Select data folder")
        if not folder:
            return
        try:
            self.status_var.set("Loading data...")
            self.root.update()

            approved_extensions = {'.xy', '.xye'}

            time_files = sorted([
                f for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))
                and not f.startswith('.')
                and os.path.splitext(f)[1].lower() in approved_extensions
            ])

            if not time_files:
                self.status_var.set("Error: No valid files found (.xy or .xye)")
                return
            df = np.genfromtxt(os.path.join(folder, time_files[0]),
                               skip_header=3, unpack=True)
            self.two_theta = df[0]
            timeslices = []
            for file in time_files:
                filepath = os.path.join(folder, file)
                df = np.genfromtxt(filepath, skip_header=3, unpack=True)
                timeslices.append(df[1])
            self.timeslices = np.array(timeslices)[:-3]
            self.n_times, self.n_channels = self.timeslices.shape
            # MIP-style backgrounding
            log_lam = self.bg_lambda_var.get()
            self.status_var.set(
                f"Background subtracting {self.n_times} frames...")
            self.root.update()
            self.data = preprocess(self.timeslices, log_lam=log_lam)
            self.status_var.set(
                f"Loaded: {self.n_times} frames × {self.n_channels} ch | "
                f"2θ: {self.two_theta[0]:.2f}°–{self.two_theta[-1]:.2f}°")
            self._update_plots()
        except Exception as e:
            self.status_var.set(f"Error loading: {str(e)[:200]}")

    def _select_inst_file(self):
        path = filedialog.askopenfilename(
            title="Select GSASII instrument parameter file",
            filetypes=[("Instrument files", "*.instprm *.prm"),
                       ("All files", "*.*")])
        if path:
            self.inst_file_var.set(path)

    def _load_powder_profiles(self):
        """Load pre-computed powder .xy files from powder_cache/."""
        folder = os.path.join(os.getcwd(), 'powder_cache')
        if not os.path.isdir(folder):
            messagebox.showwarning("Not Found",
                                   "powder_cache/ not found. Simulate from CIFs first.")
            return

        profiles, names = load_precomputed_powders(folder)
        self.candidate_profiles = profiles
        self.candidate_names = names
        self.powder_cache_folder = folder
        self.cand_status_var.set(
            f"SMCR: {len(profiles)} powder profiles loaded from powder_cache/")

    def _simulate_from_cifs(self):
        """Simulate powder patterns from CIF folder, save to powder_cache/."""
        if not HAS_GSASII:
            messagebox.showerror("Error", "GSASII not available")
            return
        if self.two_theta is None:
            messagebox.showwarning("No Data", "Load data first (need 2θ grid)")
            return

        inst_file = self.inst_file_var.get()
        if not inst_file or not os.path.isfile(inst_file):
            messagebox.showwarning("Missing",
                                   "Select or create an instrument parameter file first")
            return

        if self.cif_folder is None:
            self.status_var.set("Please load a CIF folder first.")
            return
        cif_folder = self.cif_folder

        # Auto output to powder_cache in working directory
        out_folder = os.path.join(os.getcwd(), 'powder_cache')
        os.makedirs(out_folder, exist_ok=True)

        self.status_var.set("Simulating powder patterns from CIFs...")
        self.root.update()

        def sim_progress(current, total):
            self.progress_var.set(current / total * 100)
            self.root.update_idletasks()

        profiles_dict, names = build_powder_cache_from_cifs(
            cif_folder, out_folder, self.two_theta,
            inst_file, progress_callback=sim_progress)

        self.candidate_profiles = list(profiles_dict.values())
        self.candidate_names = names
        self.powder_cache_folder = out_folder

        self.progress_var.set(100)
        self.cand_status_var.set(
            f"SMCR: Simulated {len(names)} patterns → powder_cache/")

    def _generate_peak_lists(self):
        """Generate candidate peak CSVs from CIF folder into peak_cache/."""
        if not HAS_GSASII:
            messagebox.showerror("Error", "GSASII not available")
            return

        if self.cif_folder is None:
            self.status_var.set("Please load a CIF folder first.")
            return
        cif_folder = self.cif_folder

        wavelength = self.wavelength_var.get()
        tt_max = (self.two_theta[-1] if self.two_theta is not None else 12.0)

        peak_folder = os.path.join(os.getcwd(), 'peak_cache')
        os.makedirs(peak_folder, exist_ok=True)

        self.status_var.set("Generating peak lists from CIFs...")
        self.root.update()

        count = 0
        cif_files = sorted([f for f in os.listdir(cif_folder) if f.endswith('.cif')])

        for i, cif in enumerate(cif_files):
            try:
                gpx = G2sc.G2Project(newgpx='simulation.gpx')
                phase = gpx.add_phase(os.path.join(cif_folder, cif),
                                      phasename='MyPhase')
                cell = phase.get_cell()
                cell_tup = (cell['length_a'], cell['length_b'], cell['length_c'],
                            cell['angle_alpha'], cell['angle_beta'], cell['angle_gamma'])
                refls = G2sc.GenerateReflections(
                    phase.data['General']['SGData']['SpGrp'],
                    cell_tup, TTmax=tt_max, wave=wavelength)
                basename = cif[:-4]
                peaks_2th = d_to_two_theta(
                    filter_reflections(np.array(refls).T)[3], wavelength)
                if len(peaks_2th) > 0:
                    np.savetxt(os.path.join(peak_folder, basename + '.csv'),
                               peaks_2th, delimiter=',')
                    count += 1
            except Exception:
                pass

            self.progress_var.set((i + 1) / len(cif_files) * 100)
            self.root.update_idletasks()

        # Now load the generated library
        self.candidate_library = load_candidate_library(peak_folder)
        self.progress_var.set(100)
        self.cand_status_var.set(
            f"MIP: Generated {count} peak lists → peak_cache/")

    # ──────────────────────────────────────────────────────────
    # SEGMENTATION
    # ──────────────────────────────────────────────────────────

    def _on_param_change(self, *args):
        self.thresh_label.config(text=f"{self.threshold_var.get()}")
        for name, var in self.weight_vars.items():
            if hasattr(var, '_label'):
                var._label.config(text=f"{var.get():.1f}")

    def _update_plots(self):
        if self.data is None:
            self.status_var.set("No data loaded")
            return
        self.status_var.set("Computing segmentation...")
        self.root.update()
        weights = {name: var.get() for name, var in self.weight_vars.items()}
        threshold_pct = self.threshold_var.get()
        results = detect_transitions(self.data, weights, threshold_pct)
        self.transitions = results['transitions']
        transitions_sorted = sorted(self.transitions)
        self.segment_labels = get_segment_labels(
            transitions_sorted, self.n_times)
        # Plot
        self.fig.clear()
        self.axes = self.fig.subplots(3, 1)
        t_axis = np.arange(self.n_times)
        # Plot 1: dI/dt map
        ax = self.axes[0]
        dI_dt = results['dI_dt']
        vlim = np.percentile(np.abs(dI_dt), 99)
        ax.imshow(dI_dt.T, aspect='auto', cmap='RdBu_r',
                  vmin=-vlim, vmax=vlim,
                  extent=[0, self.n_times - 1,
                          self.two_theta[-1], self.two_theta[0]],
                  origin='upper')
        for tr in transitions_sorted:
            ax.axvline(tr, color='lime', alpha=0.7, linestyle='-', lw=1.2)
        ax.set_xlabel('Time step')
        ax.set_ylabel('2θ (°)')
        ax.set_title('dI/dt map')
        # Plot 2: Data with transitions
        ax = self.axes[1]
        vmax = np.percentile(self.data, 99)
        ax.imshow(self.data, aspect='auto', cmap='bone_r', vmin=0, vmax=vmax,
                  extent=[self.two_theta[0], self.two_theta[-1],
                          self.n_times, 0], origin='upper')
        for tr in transitions_sorted:
            ax.axhline(tr, color='lime', alpha=0.7, linestyle='-', lw=1.2)
        ax.set_xlabel('2θ (°)')
        ax.set_ylabel('Time step')
        ax.set_title('Data with transitions')
        # Plot 3: Indicators with hatched pattern fills
        ax = self.axes[2]
        indicator_styles = [
            {'data': results['dissimilarity'],   'label': 'Dissimilarity',
             'color': '#e41a1c', 'hatch': '///'},
            {'data': results['intensity_change'],'label': '|dI/dt|',
             'color': '#377eb8', 'hatch': '\\\\\\'},
            {'data': results['channel_change'],  'label': '# channels',
             'color': '#4daf4a', 'hatch': '|||'},
            {'data': results['rank_change'],     'label': 'Rank change',
             'color': '#984ea3', 'hatch': '---'},
        ]
        for style in indicator_styles:
            ax.fill_between(t_axis, style['data'],
                            facecolor='none', edgecolor=style['color'],
                            hatch=style['hatch'], linewidth=0.0,
                            label=style['label'])
            # Thin colored outline along the top of each fill
            ax.plot(t_axis, style['data'], color=style['color'],
                    lw=0.8, alpha=0.7)
        ax.plot(t_axis, results['combined_score'], 'k-', lw=1.5,
                alpha=0.9, label='Combined')
        ax.axhline(results['threshold'], color='orange', linestyle='-',
                   lw=2, label=f"Threshold (P{threshold_pct})")
        for tr in transitions_sorted:
            ax.axvline(tr, color='red', alpha=0.5, linestyle='--')
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Score')
        ax.set_title('Phase Transition Indicators')
        ax.legend(loc='upper right', fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()
        n_segs = len(np.unique(self.segment_labels))
        self.status_var.set(
            f"{len(transitions_sorted)} transitions → {n_segs} segments")

    # ──────────────────────────────────────────────────────────
    # SEARCH / MATCH
    # ──────────────────────────────────────────────────────────

    def _run_search(self):
        if self.data is None:
            messagebox.showwarning("No Data", "Load data first.")
            return
        if self.segment_labels is None:
            self._update_plots()

        method = self.method_var.get()
        if method == "SMCR":
            self._run_smcr()
        elif method == "Peak Match":
            self._run_peak_match()

    def _run_smcr(self):
        """Run Sparse MCR-ALS using pre-computed powder profiles."""
        if not self.candidate_profiles:
            messagebox.showwarning(
                "No Profiles",
                "Load powder profiles (.xy folder) or simulate from CIFs first.")
            return

        self.status_var.set("SMCR: preparing references...")
        self.progress_var.set(0)
        self.root.update()

        # Prepare references with alignment (original approach)
        max_shift = self.max_shift_var.get()
        broaden_sigma = self.broaden_var.get()

        prepared_refs, ref_shifts = prepare_references_for_mcr(
            self.candidate_profiles, self.data, None,
            max_shift=max_shift,
            broaden_sigma=broaden_sigma)

        # Initialize SMCR
        smcr = SparseMCR_ALS(self.two_theta)
        smcr.load_candidates(
            prepared_refs,
            names=self.candidate_names,
            broaden_sigma=0)  # already broadened in prepare step

        # Screen candidates
        self.status_var.set("SMCR: screening candidates...")
        self.root.update()

        def screen_progress(current, total):
            self.progress_var.set(current / total * 30)
            self.root.update_idletasks()

        plausible_idx, best_corrs, presence = smcr.screen_candidates(
            self.data,
            max_shift=max_shift,
            correlation_threshold=self.corr_thresh_var.get(),
            min_time_presence=5,
            dedup_threshold=0.95,
            progress_callback=screen_progress)

        if len(plausible_idx) == 0:
            self.status_var.set(
                "No plausible candidates. Lower correlation threshold.")
            return

        smcr.select_candidates(plausible_idx)

        # Fit per segment binned
        self.status_var.set(
            f"SMCR: fitting {len(plausible_idx)} candidates...")
        self.root.update()

        def fit_progress(current, total):
            self.progress_var.set(30 + current / total * 70)
            self.root.update_idletasks()

        r2_global = smcr.fit_per_segment_binned(
            self.data,
            self.segment_labels,
            bin_size=self.bin_size_var.get(),
            max_components_per_trace=self.max_comp_trace_var.get(),
            max_components_per_segment=self.max_comp_seg_var.get(),
            fix_known_spectra=True,
            spectral_smoothness=0.5,
            max_iter=100, tol=1e-4,
            progress_callback=fit_progress)

        self.progress_var.set(100)
        self.status_var.set(f"SMCR complete — Global R² = {r2_global:.5f}")

        # Results window
        results_win = ResultsWindow(self.root, "SMCR Results")
        results_win.show_smcr_results(
            smcr, self.data, self.two_theta,
            self.transitions, self.segment_labels)

    def _run_peak_match(self):
        """Run MIP peak-match pipeline."""
        if not HAS_MIP:
            messagebox.showerror("Missing Dependency",
                                 "python-mip required. Install: pip install mip")
            return

        # Auto-load from peak_cache if not already loaded
        if not self.candidate_library:
            peak_folder = os.path.join(os.getcwd(), 'peak_cache')
            if os.path.isdir(peak_folder):
                self.candidate_library = load_candidate_library(peak_folder)
            else:
                messagebox.showwarning("No Peak Lists",
                                       "Generate peak lists from CIFs first.")
                return

        self.status_var.set("Peak Match: running MIP search...")
        self.progress_var.set(0)
        self.root.update()

        # Filter library to data 2θ range
        tt_min, tt_max = self.two_theta[0], self.two_theta[-1]
        filtered_library = {}
        for name, peaks in self.candidate_library.items():
            in_range = peaks[(peaks >= tt_min) & (peaks <= tt_max)]
            if len(in_range) > 0:
                filtered_library[name] = in_range

        if not filtered_library:
            self.status_var.set("No candidates have peaks in data range.")
            return

        def match_progress(current, total):
            self.progress_var.set(current / total * 100)
            self.root.update_idletasks()

        selections, per_seg_results = mip_search_match_binned(
            self.data, self.two_theta, self.segment_labels,
            filtered_library,
            bin_size=self.bin_size_var.get(),
            tolerance=self.tolerance_var.get(),
            peak_sigma=3,
            peak_threshold=0.01,
            use_step_detect=self.use_step_detect_var.get(),
            step_threshold=0.01,
            parsimony_weight=self.parsimony_var.get(),
            false_positive_weight=self.fp_weight_var.get(),
            progress_callback=match_progress)

        self.progress_var.set(100)
        n_unique = len(set(c for _, sel in selections for c in sel))
        self.status_var.set(
            f"Peak Match complete — {n_unique} unique phases")

        results_win = ResultsWindow(self.root, "Peak Match Results")
        results_win.show_mip_results(
            selections, self.data, self.two_theta,
            self.transitions, self.segment_labels, bin_size=self.bin_size_var.get())



# ============================================================
import sys

def get_terminal_position():
    """
    Attempt to get terminal position cross-platform.
    Returns (x, y) of terminal center, or None on failure.
    """

    # --- Linux / X11 ---
    if sys.platform.startswith('linux'):
        try:
            import subprocess
            win_id = subprocess.check_output(
                ['xdotool', 'getactivewindow'], stderr=subprocess.DEVNULL
            ).decode().strip()

            out = subprocess.check_output(
                ['xdotool', 'getwindowgeometry', '--shell', win_id],
                stderr=subprocess.DEVNULL
            ).decode()

            p = {k: int(v) for k, v in
                 (line.split('=') for line in out.strip().split('\n'))}

            return p['X'] + p['WIDTH'] // 2, p['Y'] + p['HEIGHT'] // 2
        except Exception:
            pass

    # --- macOS ---
    elif sys.platform == 'darwin':
        try:
            import subprocess
            script = '''
            tell application "System Events"
                set frontApp to name of first application process whose frontmost is true
            end tell
            tell application frontApp
                set b to bounds of front window
                return item 1 of b & "," & item 2 of b & "," & item 3 of b & "," & item 4 of b
            end tell
            '''
            out = subprocess.check_output(
                ['osascript', '-e', script], stderr=subprocess.DEVNULL
            ).decode().strip()

            x1, y1, x2, y2 = map(int, out.split(','))
            return (x1 + x2) // 2, (y1 + y2) // 2
        except Exception:
            pass

    # --- Windows ---
    elif sys.platform == 'win32':
        try:
            import ctypes
            import ctypes.wintypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            cx = (rect.left + rect.right)  // 2
            cy = (rect.top  + rect.bottom) // 2
            return cx, cy
        except Exception:
            pass

    return None  # All methods failed
# ============================================================
if __name__ == "__main__":
    # Capture terminal position BEFORE tk.Tk() steals focus
    pos = get_terminal_position()

    root = tk.Tk()
    app = TransitionAnalysisGUI(root)

    # Center window on the same screen as the terminal
    win_w, win_h = 2000, 1400  # adjust to your preferred startup size
    root.update_idletasks()

    if pos:
        cx, cy = pos
        x = cx - win_w // 2
        y = cy - win_h // 2
    else:
        x = (root.winfo_screenwidth()  - win_w) // 2
        y = (root.winfo_screenheight() - win_h) // 2

    root.geometry(f"{win_w}x{win_h}+{x}+{y}")

    root.mainloop()
