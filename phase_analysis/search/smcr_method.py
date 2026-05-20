"""Sparse MCR-ALS search method."""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm

from phase_analysis.search import register
from phase_analysis.search.base import SearchMethod, SearchResult, Parameter
from phase_analysis.cache import CandidateCache
from phase_analysis.pipeline.alignment import prepare_references_for_mcr
from phase_analysis.search._smcr_solver import SparseMCR_ALS


class SMCRResult(SearchResult):
    """SMCR-specific result with custom plotting and serialization."""

    def to_dict(self) -> dict:
        """Include SMCR-specific metrics in saved output."""
        d = super().to_dict()

        smcr = self.raw.get('smcr_object')
        segment_labels = self.raw.get('segment_labels')

        if smcr is not None:
            d['r2_global'] = self.raw.get('r2_global')

            # Per-segment breakdown
            seg_details = {}
            for seg_idx, seg_res in smcr.segment_results.items():
                seg_details[int(seg_idx)] = {
                    'r2': float(seg_res['r2']),
                    'n_iterations': seg_res['n_iterations'],
                    'active_components': [
                        smcr.candidate_names[i]
                        for i in seg_res['active_components']
                        if i < len(smcr.candidate_names)
                    ],
                }
            d['segments'] = seg_details

            # Per-component mean concentration
            if smcr.C is not None:
                active_mask = np.any(smcr.C > 0, axis=0)
                d['mean_concentrations'] = {
                    smcr.candidate_names[i]: float(np.mean(smcr.C[:, i]))
                    for i in range(len(smcr.candidate_names))
                    if active_mask[i]
                }

        return d

    def plot(self, fig: Figure, transitions=None, **kwargs):
        fig.clear()

        smcr = self.raw['smcr_object']
        data = self.raw['data']
        two_theta = self.raw['two_theta']
        segment_labels = self.raw['segment_labels']
        transitions = transitions or []

        unique_segments = np.unique(segment_labels)
        active_mask = np.any(smcr.C > 0, axis=0)
        active_idx = np.where(active_mask)[0]
        active_names = [smcr.candidate_names[i] for i in active_idx]
        n_active = len(active_idx)

        if n_active == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No active components found",
                    ha='center', va='center', fontsize=14)
            return

        gs = GridSpec(3, 1, figure=fig,
                      height_ratios=[5, 1.5, 2], hspace=0.35)
        colors = [cm.tab10(i / max(n_active, 1)) for i in range(n_active)]

        # Activity map
        ax1 = fig.add_subplot(gs[0])
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
        ax1.set_title(f'Component Activity ({n_active} phases)')
        ax1.set_xlim(0, data.shape[0])
        for tr in transitions:
            ax1.axvline(tr, color='gray', alpha=0.6, linestyle='--', lw=0.8)

        # R² per segment
        ax2 = fig.add_subplot(gs[1])
        for seg_idx in unique_segments:
            if seg_idx not in smcr.segment_results:
                continue
            seg_mask = segment_labels == seg_idx
            seg_times = np.where(seg_mask)[0]
            r2 = smcr.segment_results[seg_idx]['r2']
            center = (seg_times[0] + seg_times[-1]) / 2
            width = seg_times[-1] - seg_times[0]
            ax2.bar(center, r2, width=width, alpha=0.6, color='steelblue')

        ax2.set_ylabel('R²')
        ax2.set_xlabel('Time step')
        ax2.set_xlim(0, data.shape[0])
        ax2.axhline(0.95, color='green', linestyle=':', alpha=0.5)
        ax2.axhline(0.90, color='orange', linestyle=':', alpha=0.5)

        # Residual
        ax3 = fig.add_subplot(gs[2])
        res_vmax = np.percentile(np.abs(smcr.residuals), 97)
        ax3.imshow(smcr.residuals.T, aspect='auto', cmap='RdBu_r',
                   vmin=-res_vmax, vmax=res_vmax,
                   extent=[0, data.shape[0], two_theta[-1], two_theta[0]],
                   origin='upper')
        ax3.set_xlabel('Time step')
        ax3.set_ylabel('2θ (°)')

        fig.tight_layout()


@register
class SMCRSearch(SearchMethod):

    @property
    def name(self):
        return "smcr"

    @property
    def display_name(self):
        return "Sparse MCR-ALS"

    @property
    def requires_powder_cache(self):
        return True

    @property
    def requires_peak_cache(self):
        return False

    def parameters(self) -> list[Parameter]:
        return [
            Parameter('bin_size', 'Bin size (frames)', int, 5, 1, 100, 1),
            Parameter('max_comp_trace', 'Max components/trace', int, 4, 1, 20),
            Parameter('max_comp_segment', 'Max components/segment', int, 6, 1, 30),
            Parameter('broaden_sigma', 'Broaden σ (ch)', float, 2.0, 0, 10, 0.5),
            Parameter('corr_threshold', 'Correlation threshold', float, 0.3, 0.1, 0.9, 0.05),
            Parameter('max_shift', 'Max shift (channels)', int, 50, 0, 200),
        ]

    def run(self, data, two_theta, segment_labels, cache, params,
            progress_callback=None):

        profiles = list(cache.powder_profiles.values())
        names = list(cache.powder_profiles.keys())

        max_shift = params['max_shift']
        prepared_refs, _ = prepare_references_for_mcr(
            profiles, data, None,
            max_shift=max_shift,
            broaden_sigma=params['broaden_sigma'])

        smcr = SparseMCR_ALS(two_theta)
        smcr.load_candidates(prepared_refs, names=names, broaden_sigma=0)

        def _screen_progress(cur, tot):
            if progress_callback:
                progress_callback(cur, tot * 3)

        plausible_idx, _, _ = smcr.screen_candidates(
            data, max_shift=max_shift,
            correlation_threshold=params['corr_threshold'],
            min_time_presence=5, dedup_threshold=0.96,
            progress_callback=_screen_progress)

        if len(plausible_idx) == 0:
            return SMCRResult(
                method_name=self.display_name,
                summary="No plausible candidates found.",
                phases_found=[],
                params_used=params)

        smcr.select_candidates(plausible_idx)

        n_cand = len(plausible_idx)

        def _fit_progress(cur, tot):
            if progress_callback:
                progress_callback(n_cand + cur, n_cand + tot)

        r2 = smcr.fit_per_segment_binned(
            data, segment_labels,
            bin_size=params['bin_size'],
            max_components_per_trace=params['max_comp_trace'],
            max_components_per_segment=params['max_comp_segment'],
            fix_known_spectra=True, spectral_smoothness=0.5,
            max_iter=100, tol=1e-4,
            progress_callback=_fit_progress)

        active_mask = np.any(smcr.C > 0, axis=0)
        active_names = [smcr.candidate_names[i]
                        for i in np.where(active_mask)[0]]

        return SMCRResult(
            method_name=self.display_name,
            summary=f"{len(active_names)} active phases, Global R² = {r2:.5f}",
            phases_found=active_names,
            confidence={name: float(np.mean(smcr.C[:, i]))
                        for i, name in enumerate(smcr.candidate_names)
                        if active_mask[i]},
            params_used=params,
            raw={
                'smcr_object': smcr,
                'r2_global': r2,
                'data': data,
                'two_theta': two_theta,
                'segment_labels': segment_labels,
            },
        )
