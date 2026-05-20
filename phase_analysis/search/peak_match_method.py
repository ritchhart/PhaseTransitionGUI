"""MIP peak-match search method."""

import numpy as np
from collections import Counter
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.cm as cm

from phase_analysis.search import register
from phase_analysis.search.base import SearchMethod, SearchResult, Parameter
from phase_analysis.cache import CandidateCache
from phase_analysis.search._mip_solver import mip_search_match_binned
from phase_analysis import HAS_MIP, HAS_STEP_DETECT


class PeakMatchResult(SearchResult):
    """MIP-specific result with timeline heatmap plotting and serialization."""

    def to_dict(self) -> dict:
        """Include per-segment phase lists and bin-level detail."""
        d = super().to_dict()

        selections = self.raw.get('selections', [])
        segment_labels = self.raw.get('segment_labels')
        bin_size = self.raw.get('bin_size', 5)
        top_per_seg = self.raw.get('top_per_seg', 5)

        # Per-bin selections (time_index → phases)
        d['bin_selections'] = [
            {'time_index': int(t), 'phases': sel}
            for t, sel in selections
        ]

        # Per-segment top phases
        if segment_labels is not None:
            n_time = len(segment_labels)
            unique_segs = np.unique(segment_labels)

            # Spread selections to per-time
            per_time = [[] for _ in range(n_time)]
            half_bin = max(1, bin_size // 2)
            for time_idx, sel in selections:
                for t in range(max(0, time_idx - half_bin),
                               min(n_time, time_idx + half_bin + 1)):
                    per_time[t] = list(set(per_time[t] + sel))

            # Segment boundaries
            segments = [0]
            for i in range(len(unique_segs) - 1):
                seg_times = np.where(segment_labels == unique_segs[i])[0]
                segments.append(int(seg_times[-1] + 1))
            segments.append(n_time)

            seg_summaries = {}
            for i in range(len(segments) - 1):
                start, end = segments[i], segments[i + 1]
                counts = Counter(
                    c for sel in per_time[start:end] for c in sel)
                length = end - start
                seg_summaries[f"segment_{i+1}"] = {
                    'time_range': [start, end],
                    'top_phases': [
                        {'name': name, 'presence_fraction': round(count / length, 3)}
                        for name, count in counts.most_common(top_per_seg)
                    ],
                    'total_unique_phases': len(counts),
                }
            d['segments'] = seg_summaries

        return d

    def plot(self, fig: Figure, transitions=None, **kwargs):
        fig.clear()

        selections = self.raw['selections']
        data = self.raw['data']
        segment_labels = self.raw['segment_labels']
        bin_size = self.raw['bin_size']
        top_per_seg = self.raw.get('top_per_seg', 5)

        n_time = data.shape[0]

        per_time = [[] for _ in range(n_time)]
        half_bin = max(1, bin_size // 2)
        for time_idx, sel in selections:
            for t in range(max(0, time_idx - half_bin),
                           min(n_time, time_idx + half_bin + 1)):
                per_time[t] = list(set(per_time[t] + sel))

        unique_segs = np.unique(segment_labels)
        segments = [0]
        for i in range(len(unique_segs) - 1):
            seg_times = np.where(segment_labels == unique_segs[i])[0]
            segments.append(seg_times[-1] + 1)
        segments.append(n_time)
        n_seg = len(segments) - 1

        seg_top = {}
        for i in range(n_seg):
            start, end = segments[i], segments[i + 1]
            counts = Counter(
                c for sel in per_time[start:end] for c in sel)
            seg_top[i] = [c for c, _ in counts.most_common(top_per_seg)]

        show_candidates = []
        for i in range(n_seg):
            for c in seg_top[i]:
                if c not in show_candidates:
                    show_candidates.append(c)
        n_cand = len(show_candidates)

        if n_cand == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No candidates matched",
                    ha='center', va='center', fontsize=14)
            return

        presence = np.zeros((n_cand, n_time))
        for t, sel in enumerate(per_time):
            for name in sel:
                if name in show_candidates:
                    presence[show_candidates.index(name), t] = 1

        highlight = np.zeros_like(presence)
        for i in range(n_seg):
            start, end = segments[i], segments[i + 1]
            for name in seg_top[i]:
                if name in show_candidates:
                    row = show_candidates.index(name)
                    highlight[row, start:end] = presence[row, start:end]

        display = np.zeros_like(presence)
        display[presence > 0] = 1
        display[highlight > 0] = 2

        gs = GridSpec(2, 1, figure=fig,
                      height_ratios=[1, max(n_cand, 4)], hspace=0.08)

        ax_seg = fig.add_subplot(gs[0])
        seg_colors = cm.Set3(np.linspace(0, 1, max(n_seg, 1)))
        for i in range(n_seg):
            start, end = segments[i], segments[i + 1]
            ax_seg.axvspan(start - 0.5, end - 0.5,
                           color=seg_colors[i], alpha=0.7)
            ax_seg.text((start + end) / 2, 0.5, f"Seg {i+1}",
                        ha='center', va='center', fontsize=8)
        ax_seg.set_xlim(-0.5, n_time - 0.5)
        ax_seg.set_ylim(0, 1)
        ax_seg.set_yticks([])
        ax_seg.set_title(f"Top {top_per_seg} Phases Per Segment")

        ax_heat = fig.add_subplot(gs[1])
        cmap_hm = ListedColormap(['#f5f5f5', '#a6cee3', '#1f4e79'])
        ax_heat.imshow(display, aspect='auto', cmap=cmap_hm,
                       interpolation='none', origin='upper',
                       extent=[-0.5, n_time - 0.5, n_cand - 0.5, -0.5],
                       vmin=0, vmax=2)
        ax_heat.set_yticks(range(n_cand))
        ax_heat.set_yticklabels(show_candidates, fontsize=8)
        ax_heat.set_xlabel("Time Step")

        legend_elements = [
            Patch(facecolor='#f5f5f5', edgecolor='gray', label='Absent'),
            Patch(facecolor='#a6cee3', label='Present'),
            Patch(facecolor='#1f4e79', label='Top in segment'),
        ]
        ax_heat.legend(handles=legend_elements, loc='lower right', fontsize=8)
        fig.tight_layout()


@register
class PeakMatchSearch(SearchMethod):

    @property
    def name(self):
        return "peak_match"

    @property
    def display_name(self):
        return "MIP Peak Match"

    @property
    def requires_powder_cache(self):
        return False

    @property
    def requires_peak_cache(self):
        return True

    def parameters(self) -> list[Parameter]:
        return [
            Parameter('bin_size', 'Bin size (frames)', int, 5, 1, 100),
            Parameter('tolerance', '2θ tolerance (°)', float, 0.2, 0.01, 1.0, 0.05),
            Parameter('parsimony_weight', 'Parsimony weight', float, 1.0, 0, 5, 0.1),
            Parameter('fp_weight', 'False positive weight', float, 0.5, 0, 5, 0.1),
            Parameter('coverage_bonus', 'Coverage bonus weight', float, 3.0, 0, 10, 0.5),
            Parameter('top_per_seg', 'Top phases per segment', int, 5, 1, 20),
            Parameter('use_step_detect', 'Use step_detect', bool, HAS_STEP_DETECT),
        ]

    def run(self, data, two_theta, segment_labels, cache, params,
            progress_callback=None):

        if not HAS_MIP:
            return PeakMatchResult(
                method_name=self.display_name,
                summary="ERROR: python-mip not installed",
                phases_found=[],
                params_used=params)

        selections, per_seg = mip_search_match_binned(
            data, two_theta, segment_labels,
            cache.peak_positions,
            bin_size=params['bin_size'],
            tolerance=params['tolerance'],
            parsimony_weight=params['parsimony_weight'],
            false_positive_weight=params['fp_weight'],
            coverage_bonus_weight=params['coverage_bonus'],
            use_step_detect=params.get('use_step_detect', True),
            progress_callback=progress_callback)

        all_phases = list(set(c for _, sel in selections for c in sel))

        return PeakMatchResult(
            method_name=self.display_name,
            summary=f"{len(all_phases)} unique phases identified",
            phases_found=all_phases,
            params_used=params,
            raw={
                'selections': selections,
                'per_segment_results': per_seg,
                'data': data,
                'segment_labels': segment_labels,
                'bin_size': params['bin_size'],
                'top_per_seg': params['top_per_seg'],
            },
        )
