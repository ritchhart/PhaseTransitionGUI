"""Sparse MCR-ALS search method."""
import os
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from phase_analysis.search import register
from phase_analysis.search.base import SearchMethod, SearchResult, Parameter
from phase_analysis.cache import CandidateCache
from phase_analysis.pipeline.alignment import prepare_references_for_mcr
from phase_analysis.search._smcr_solver_broaden import SparseMCR_ALS_broaden
ACTIVITY_THRESHOLD = 1e-6


class SMCRResultBroaden(SearchResult):
    """SMCR-specific result with custom plotting and serialization."""
    def to_dict(self) -> dict:
        d = super().to_dict()
        smcr = self.raw.get('smcr_object')
        if smcr is not None:
            d['r2_global'] = self.raw.get('r2_global')
            seg_details = {}
            for seg_idx, seg_res in smcr.segment_results.items():
                detail = {
                    'r2': float(seg_res['r2']),
                    'n_iterations': seg_res['n_iterations'],
                    'active_components': [
                        smcr.candidate_names[i]
                        for i in seg_res['active_components']
                        if i < len(smcr.candidate_names)
                    ],
                }
                if seg_res.get('alignment_shifts') is not None:
                    detail['alignment_shifts'] = {
                        smcr.candidate_names[i]: int(shift)
                        for i, shift in enumerate(seg_res['alignment_shifts'])
                        if i < len(smcr.candidate_names)
                    }
                if seg_res.get('component_broaden_sigmas') is not None:
                    sigmas = seg_res['component_broaden_sigmas']
                    broaden_dict = {
                        smcr.candidate_names[i]: float(sigmas[i])
                        for i in range(len(smcr.candidate_names))
                        if i < len(sigmas) and sigmas[i] > 0
                    }
                    if broaden_dict:
                        detail['component_broadening'] = broaden_dict
                seg_details[int(seg_idx)] = detail
            d['segments'] = seg_details
            if smcr.C is not None:
                active_mask = np.any(smcr.C > ACTIVITY_THRESHOLD * np.max(smcr.C), axis=0)
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
        active_mask = np.any(smcr.C > ACTIVITY_THRESHOLD * np.max(smcr.C), axis=0)
        active_idx = np.where(active_mask)[0]
        active_names = [smcr.candidate_names[i] for i in active_idx]
        n_active = len(active_idx)

        if n_active == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No active components found",
                    ha='center', va='center', fontsize=14)
            return

        valid_segments = [s for s in unique_segments if s in smcr.segment_results]
        n_seg_plots = len(valid_segments)

        # Build a consistent color map
        comp_colors = {i: cm.tab10(k / max(n_active, 1))
                       for k, i in enumerate(active_idx)}

        # --- Main figure: Activity map, R² bar, Residual heatmap only ---
        gs = GridSpec(3, 1, figure=fig,
                      height_ratios=[4, 1.2, 1.8],
                      hspace=0.55)

        # --- Activity map ---
        ax1 = fig.add_subplot(gs[0, 0])
        for seg_idx in unique_segments:
            if seg_idx not in smcr.segment_results:
                continue
            seg_res = smcr.segment_results[seg_idx]
            seg_mask = segment_labels == seg_idx
            seg_times = np.where(seg_mask)[0]
            for y_pos, comp_i in enumerate(active_idx):
                bin_membership = seg_res['bin_membership']
                C_binned = seg_res['C_binned']
                for b, times in enumerate(bin_membership):
                    if C_binned[b, comp_i] > 1e-8:
                        max_c = np.max(smcr.C[:, comp_i]) + 1e-10
                        alpha = 0.3 + 0.7 * (C_binned[b, comp_i] / max_c)
                        ax1.barh(y_pos, times[-1] - times[0], left=times[0],
                                height=0.8, alpha=alpha, color=comp_colors[comp_i],
                                edgecolor='none', linewidth=0)
        ax1.set_yticks(range(n_active))
        ax1.set_yticklabels(active_names, fontsize=9)
        ax1.set_xlabel('Time step')
        ax1.set_title(f'Component Activity ({n_active} phases)')
        ax1.set_xlim(0, data.shape[0])
        for tr in transitions:
            ax1.axvline(tr, color='gray', alpha=0.6, linestyle='--', lw=0.8)

        # --- R² per segment ---
        ax2 = fig.add_subplot(gs[1, 0])
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

        # --- Residual heatmap ---
        ax3 = fig.add_subplot(gs[2, 0])
        res_vmax = np.percentile(np.abs(smcr.residuals), 97)
        ax3.imshow(smcr.residuals.T, aspect='auto', cmap='RdBu_r',
                   vmin=-res_vmax, vmax=res_vmax,
                   extent=[0, data.shape[0], two_theta[-1], two_theta[0]],
                   origin='upper')
        ax3.set_xlabel('Time step')
        ax3.set_ylabel('2θ (°)')

        fig.tight_layout()

        # --- Component decomposition: separate figure, single column, 4:1 aspect ---
        if n_seg_plots > 0:
            fig_width = 12.0  # inches
            subplot_height = fig_width / 4.0  # 4:1 aspect ratio
            total_height = subplot_height * n_seg_plots + 0.5 * (n_seg_plots - 1)

            comp_fig = Figure(figsize=(fig_width, total_height), dpi=150)
            comp_gs = GridSpec(n_seg_plots, 1, figure=comp_fig,
                               hspace=0.4)

            for plot_idx, seg_idx in enumerate(valid_segments):
                seg_res = smcr.segment_results[seg_idx]
                ax = comp_fig.add_subplot(comp_gs[plot_idx, 0])

                binned_data = seg_res['binned_data']
                C_binned = seg_res['C_binned']
                S_seg = seg_res['S']
                n_bins = binned_data.shape[0]
                mid_bin = n_bins // 2
                observed = binned_data[mid_bin, :]
                concentrations = C_binned[mid_bin, :]

                comp_sigmas = seg_res.get('component_broaden_sigmas')

                total_fit = np.zeros_like(observed)
                for comp_i in active_idx:
                    contribution = concentrations[comp_i] * S_seg[:, comp_i]
                    total_fit += contribution
                    if np.max(np.abs(contribution)) < 1e-10:
                        continue
                    label = smcr.candidate_names[comp_i]
                    if comp_sigmas is not None and comp_i < len(comp_sigmas):
                        sigma_val = comp_sigmas[comp_i]
                        if sigma_val > 0:
                            label += f' (σ={sigma_val:.1f})'
                    color = comp_colors[comp_i]
                    ax.fill_between(two_theta, 0, contribution,
                                    alpha=0.35, color=color, linewidth=0)
                    ax.plot(two_theta, contribution, color=color,
                            lw=0.7, alpha=0.8, label=label)

                ax.plot(two_theta, observed, 'k-', lw=1.0, label='Observed')
                ax.plot(two_theta, total_fit, color='red', lw=1.0,
                        ls='-', label='Total fit')

                residual = observed - total_fit
                res_offset = -np.max(observed) * 0.05
                ax.plot(two_theta, residual + res_offset, color='gray',
                        lw=0.6, ls='-', label='Residual')
                ax.axhline(res_offset, color='gray', lw=0.3, ls=':')

                ax.set_xlabel('2θ (°)', fontsize=10)
                ax.set_ylabel('Intensity', fontsize=10)
                ax.tick_params(labelsize=9)
                r2_seg = seg_res['r2']
                ax.set_title(f'Segment {int(seg_idx)} · bin {mid_bin+1}/{n_bins} · '
                             f'R²={r2_seg:.3f}', fontsize=11)

                n_active_in_seg = len(seg_res['active_components'])
                if n_active_in_seg <= 8:
                    ax.legend(fontsize=8, loc='upper right', framealpha=0.7,
                              handlelength=1.2, labelspacing=0.3)
                ax.set_xlim(two_theta[0], two_theta[-1])
                y_max = max(np.max(observed), np.max(total_fit)) * 1.1
                y_min = min(res_offset + np.min(residual), 0) * 1.2
                ax.set_ylim(y_min, y_max)

            comp_fig.tight_layout()

            # Save to components folder (cross-platform)
            resutls_dir = os.path.join(os.getcwd(), 'results')
            os.makedirs(resutls_dir, exist_ok=True)
            components_dir = os.path.join(resutls_dir, 'components')
            os.makedirs(components_dir, exist_ok=True)
            file_name = kwargs.get('file_name', 'component_decomposition')
            output_path = os.path.join(components_dir, file_name + '.jpg')
            comp_fig.savefig(output_path, format='jpg', dpi=300,
                             bbox_inches='tight', pil_kwargs={'quality': 92})


@register
class SMCRSearchBroaden(SearchMethod):
    @property
    def name(self):
        return "smcrbroaden"

    @property
    def display_name(self):
        return "Sparse MCR-ALS Broaden"

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
            Parameter('broaden_sigma', 'Initial broaden σ (ch)', float, 2.0, 0, 10, 0.5,
                      tooltip="Global broadening applied during reference preparation."),
            Parameter('max_broaden_sigma', 'Per-component max σ (ch)', float, 7.0, 0, 10, 0.5,
                      tooltip="Maximum additional per-component broadening optimized during fit. "
                              "Each reference spectrum independently gets 0 to this value. "
                              "Set to 0 to disable."),
            Parameter('corr_threshold', 'Correlation threshold', float, 0.3, 0.1, 0.9, 0.05),
            Parameter('max_shift', 'Max shift (channels)', int, 50, 0, 200),
            Parameter('align_per_segment', 'Align per segment', bool, False,
                      tooltip="Re-align references to each segment's local mean. "
                              "Helps with thermal drift or phases appearing only in some segments."),
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

        smcr = SparseMCR_ALS_broaden(two_theta)
        smcr.load_candidates(prepared_refs, names=names, broaden_sigma=0)

        # Screen
        def _screen_progress(cur, tot):
            if progress_callback:
                progress_callback(cur, tot * 3)

        plausible_idx, _, _ = smcr.screen_candidates(
            data, max_shift=max_shift,
            correlation_threshold=params['corr_threshold'],
            min_time_presence=5, dedup_threshold=0.96,
            progress_callback=_screen_progress)

        if len(plausible_idx) == 0:
            return SMCRResultBroaden(
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
            fix_known_spectra=True,
            spectral_smoothness=0.5,
            align_per_segment=params.get('align_per_segment', False),
            max_shift=max_shift,
            max_broaden_sigma=params.get('max_broaden_sigma', 0.0),
            broaden_n_steps=11,
            broaden_start_iter=3,
            broaden_interval=3,
            max_iter=100, tol=1e-4,
            progress_callback=_fit_progress)

        active_mask = np.any(smcr.C > ACTIVITY_THRESHOLD * np.max(smcr.C), axis=0)
        active_names = [smcr.candidate_names[i]
                        for i in np.where(active_mask)[0]]

        max_per_seg = max(
            len(seg_res['active_components'])
            for seg_res in smcr.segment_results.values()
        ) if smcr.segment_results else 0

        return SMCRResultBroaden(
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