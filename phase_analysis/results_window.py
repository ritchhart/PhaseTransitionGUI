"""Results display window for SMCR and MIP search-match outputs."""

import tkinter as tk
from tkinter import ttk
import numpy as np
from collections import Counter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.cm as cm


class ResultsWindow:
    """Second window for search/match results visualization."""

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

    # ─── SMCR results ────────────────────────────────────────

    def show_smcr_results(self, smcr, data, two_theta, transitions,
                          segment_labels):
        """Display SMCR fit summary: activity map, R² bars, residual."""
        self.fig.clear()

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

        # --- Activity map ---
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

        # --- R² per segment ---
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

        # --- Residual heatmap ---
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

        r2_global = 1 - (np.linalg.norm(smcr.residuals)**2
                         / (np.linalg.norm(data)**2))
        self.status_var.set(
            f"SMCR Complete — {n_active} active phases, "
            f"Global R² = {r2_global:.5f}")

    # ─── MIP results ─────────────────────────────────────────

    def show_mip_results(self, selections, data, two_theta, transitions,
                         segment_labels, bin_size=5, top_per_seg=5):
        """
        Display MIP peak-match results as timeline heatmap with
        per-segment top-N highlighting.
        """
        self.fig.clear()

        n_time = data.shape[0]

        # Spread bin selections across time
        per_time_selections = [[] for _ in range(n_time)]
        half_bin = max(1, bin_size // 2)
        for time_idx, sel in selections:
            t_start = max(0, time_idx - half_bin)
            t_end = min(n_time, time_idx + half_bin + 1)
            for t in range(t_start, t_end):
                per_time_selections[t] = list(
                    set(per_time_selections[t] + sel))

        # Compute segment boundaries
        unique_segs = np.unique(segment_labels)
        segments = [0]
        for i in range(len(unique_segs) - 1):
            seg_mask = segment_labels == unique_segs[i]
            seg_times = np.where(seg_mask)[0]
            segments.append(seg_times[-1] + 1)
        segments.append(n_time)
        n_seg = len(segments) - 1

        # Top candidates per segment
        seg_top = {}
        for i in range(n_seg):
            start, end = segments[i], segments[i + 1]
            counts = Counter(
                c for sel in per_time_selections[start:end] for c in sel)
            seg_top[i] = [c for c, _ in counts.most_common(top_per_seg)]

        # Union of all per-segment tops, ordered by first appearance
        seen = []
        for i in range(n_seg):
            for c in seg_top[i]:
                if c not in seen:
                    seen.append(c)
        show_candidates = seen
        n_cand = len(show_candidates)

        if n_cand == 0:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No candidates matched",
                    ha='center', va='center', fontsize=14)
            self.canvas.draw()
            return

        # Build presence/highlight matrices
        presence = np.zeros((n_cand, n_time), dtype=float)
        for t, sel in enumerate(per_time_selections):
            for name in sel:
                if name in show_candidates:
                    presence[show_candidates.index(name), t] = 1

        highlight = np.zeros((n_cand, n_time), dtype=int)
        for i in range(n_seg):
            start, end = segments[i], segments[i + 1]
            for name in seg_top[i]:
                if name in show_candidates:
                    row = show_candidates.index(name)
                    for t in range(start, min(end, n_time)):
                        if presence[row, t] > 0:
                            highlight[row, t] = 1

        # Display matrix: 0=absent, 1=present, 2=present & top
        display = np.zeros_like(presence)
        display[presence > 0] = 1
        display[highlight > 0] = 2

        # --- Plot ---
        gs = GridSpec(2, 1, figure=self.fig,
                      height_ratios=[1, max(n_cand, 4)], hspace=0.08)

        # Segment bar
        ax_seg = self.fig.add_subplot(gs[0])
        segment_colors = cm.Set3(np.linspace(0, 1, max(n_seg, 1)))
        for i in range(n_seg):
            start, end = segments[i], segments[i + 1]
            ax_seg.axvspan(start - 0.5, end - 0.5,
                           color=segment_colors[i], alpha=0.7)
            mid = (start + end) / 2
            ax_seg.text(mid, 0.5, f"Seg {i + 1}", ha='center',
                        va='center', fontsize=8, fontweight='bold')
        ax_seg.set_xlim(-0.5, n_time - 0.5)
        ax_seg.set_ylim(0, 1)
        ax_seg.set_yticks([])
        ax_seg.set_title(
            f"Top {top_per_seg} Phases Per Segment (Peak Match)",
            fontsize=12)

        # Heatmap
        ax_heat = self.fig.add_subplot(gs[1])
        cmap_hm = ListedColormap(['#f5f5f5', '#a6cee3', '#1f4e79'])
        ax_heat.imshow(display, aspect='auto', cmap=cmap_hm,
                       interpolation='none', origin='upper',
                       extent=[-0.5, n_time - 0.5, n_cand - 0.5, -0.5],
                       vmin=0, vmax=2)

        for s in segments[1:-1]:
            ax_heat.axvline(s - 0.5, color='red', linewidth=1.5,
                            linestyle='--', alpha=0.7)
        ax_heat.set_yticks(range(n_cand))
        ax_heat.set_yticklabels(show_candidates, fontsize=8)
        ax_heat.set_xlabel("Time Step")
        ax_heat.set_ylabel("Candidate Phase")
        for i in range(n_cand - 1):
            ax_heat.axhline(i + 0.5, color='white', linewidth=0.5)

        legend_elements = [
            Patch(facecolor='#f5f5f5', edgecolor='gray', label='Absent'),
            Patch(facecolor='#a6cee3', label='Present'),
            Patch(facecolor='#1f4e79', label='Present & top in segment'),
        ]
        ax_heat.legend(handles=legend_elements, loc='lower right', fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

        # Summary
        summary_lines = []
        for i in range(n_seg):
            start, end = segments[i], segments[i + 1]
            tops = ", ".join(seg_top[i]) if seg_top[i] else "(none)"
            summary_lines.append(f"Seg {i+1} [{start}:{end}]: {tops}")
        self.status_var.set(
            f"Peak Match Complete — {n_cand} unique phases | "
            + " | ".join(summary_lines))
