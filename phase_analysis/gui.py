"""Main GUI application — thin orchestrator over pipeline + search registry."""

import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from phase_analysis import HAS_GSASII, G2sc
from phase_analysis.utils import d_to_two_theta, filter_reflections
from phase_analysis.pipeline.preprocessing import preprocess
from phase_analysis.pipeline.simulation import (
    build_powder_cache_from_cifs,
    load_candidate_library,
)
from phase_analysis.pipeline.segmentation import detect_transitions, get_segment_labels
from phase_analysis.cache import CacheManager, CandidateCache
from phase_analysis.search import available_methods
from phase_analysis.search.base import SearchMethod, SearchResult, Parameter
from phase_analysis.results_window import ResultsWindow


class TransitionAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Phase Transition Analysis")
        self.root.geometry("1600x1050")

        # ── Data state ──
        self.data = None
        self.two_theta = None
        self.timeslices = None
        self.n_times = 0
        self.n_channels = 0
        self.transitions = []
        self.segment_labels = None
        self.data_folder_name = None

        # ── Cache ──
        self.cache_manager = CacheManager()
        self.cache = CandidateCache()

        # ── CIF folder (for generating caches) ──
        self.cif_folder = None

        # ── Search methods (auto-discovered from registry) ──
        self.search_methods: dict[str, SearchMethod] = {}
        for method_cls in available_methods():
            instance = method_cls()
            self.search_methods[instance.name] = instance

        # ── Parameter widgets: (method_name, param_name) → tk variable ──
        self.param_widgets: dict[tuple[str, str], tk.Variable] = {}

        # ── Build UI ──
        self._build_ui()

    # ══════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════

    def _build_ui(self):
        # Left panel: controls
        self.controls_frame = ttk.Frame(self.root, width=600)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.controls_frame.pack_propagate(False)

        # Right panel: segmentation plots
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_controls()

        # Main figure (segmentation display)
        self.fig = Figure(figsize=(10, 9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.axes = []

    def _build_controls(self):
        frame = self.controls_frame

        # ── Data Loading ──────────────────────────────────────
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

        wl_frame = ttk.Frame(lf_data)
        wl_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(wl_frame, text="Wavelength λ (Å):").pack(side=tk.LEFT)
        self.wavelength_var = tk.DoubleVar(value=0.25450)
        ttk.Entry(wl_frame, textvariable=self.wavelength_var,
                  width=10).pack(side=tk.RIGHT)

        # ── Candidate Cache ───────────────────────────────────
        lf_cand = ttk.LabelFrame(frame, text="Candidate Patterns")
        lf_cand.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(lf_cand, text="Load Cached Profiles (powder + peak)",
                   command=self._load_caches).pack(fill=tk.X, padx=5, pady=2)

        # CIF folder
        cif_frame = ttk.Frame(lf_cand)
        cif_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(cif_frame, text="Load CIF Folder",
                   command=self._load_cif_folder).pack(side=tk.LEFT, padx=2)
        self.cif_folder_var = tk.StringVar(value="No CIF folder loaded")
        ttk.Label(cif_frame, textvariable=self.cif_folder_var,
                  font=('Consolas', 8)).pack(side=tk.LEFT, padx=5)

        # GSASII actions (only if available)
        if HAS_GSASII:
            ttk.Button(lf_cand, text="Simulate Powder Profiles (GSASII)",
                       command=self._simulate_from_cifs).pack(
                           fill=tk.X, padx=5, pady=2)
            ttk.Button(lf_cand, text="Generate Peak Lists (for Peak Match)",
                       command=self._generate_peak_lists).pack(
                           fill=tk.X, padx=5, pady=2)

        # Instrument file
        inst_frame = ttk.Frame(lf_cand)
        inst_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(inst_frame, text="Instrument file:").pack(side=tk.LEFT)
        self.inst_file_var = tk.StringVar(value="")
        ttk.Entry(inst_frame, textvariable=self.inst_file_var,
                  width=25).pack(side=tk.LEFT, padx=2)
        ttk.Button(inst_frame, text="...",
                   command=self._select_inst_file, width=3).pack(side=tk.LEFT)
        ttk.Button(inst_frame, text="Make",
                   command=self._make_instrument_file, width=5).pack(
                       side=tk.LEFT, padx=2)

        self.cand_status_var = tk.StringVar(value="No candidates loaded")
        ttk.Label(lf_cand, textvariable=self.cand_status_var,
                  font=('Consolas', 8), wraplength=580).pack(
                      fill=tk.X, padx=5, pady=2)

        # ── Segmentation Parameters ──────────────────────────
        lf_seg = ttk.LabelFrame(frame, text="Segmentation Parameters")
        lf_seg.pack(fill=tk.X, padx=5, pady=5)

        thresh_frame = ttk.Frame(lf_seg)
        thresh_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(thresh_frame, text="Threshold percentile:").pack(side=tk.LEFT)
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

        # ── Search / Match ────────────────────────────────────
        lf_match = ttk.LabelFrame(frame, text="Search / Match")
        lf_match.pack(fill=tk.X, padx=5, pady=5)

        # Method selector (populated from registry)
        method_frame = ttk.Frame(lf_match)
        method_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
        method_names = [m.display_name for m in self.search_methods.values()]
        self.method_var = tk.StringVar(
            value=method_names[0] if method_names else "")
        self.method_combo = ttk.Combobox(
            method_frame, textvariable=self.method_var,
            values=method_names, state='readonly', width=25)
        self.method_combo.pack(side=tk.LEFT, padx=5)
        self.method_combo.bind('<<ComboboxSelected>>', self._on_method_change)

        # Candidate name filter
        filter_frame = ttk.Frame(lf_match)
        filter_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(filter_frame, text="Filter candidates:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar(value="")
        ttk.Entry(filter_frame, textvariable=self.filter_var,
                  width=35).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Label(lf_match,
                  text="(space/comma separated; name must contain ≥1, case-sensitive)",
                  font=('Consolas', 7)).pack(fill=tk.X, padx=5)

        # Dynamic parameter area (rebuilt when method changes)
        self.param_container = ttk.Frame(lf_match)
        self.param_container.pack(fill=tk.X, padx=5, pady=2)
        self._build_method_params()

        # Run button
        ttk.Button(lf_match, text="▶  Run Search / Match",
                   command=self._run_search).pack(fill=tk.X, padx=5, pady=8)

        # ── Status ────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Load data to begin")
        ttk.Label(frame, textvariable=self.status_var,
                  font=('Consolas', 9), wraplength=580,
                  justify=tk.LEFT).pack(fill=tk.X, padx=5, pady=5)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=2)

    # ──────────────────────────────────────────────────────────
    # DYNAMIC PARAMETER PANEL
    # ──────────────────────────────────────────────────────────

    def _get_selected_method(self) -> SearchMethod | None:
        """Look up the currently selected method instance by display_name."""
        display = self.method_var.get()
        for method in self.search_methods.values():
            if method.display_name == display:
                return method
        return None

    def _on_method_change(self, event=None):
        """Rebuild parameter widgets when method selection changes."""
        self._build_method_params()

    def _build_method_params(self):
        """Destroy old widgets, build new ones from selected method's parameters()."""
        # Clear existing
        for child in self.param_container.winfo_children():
            child.destroy()
        self.param_widgets.clear()

        method = self._get_selected_method()
        if method is None:
            return

        for param in method.parameters():
            row = ttk.Frame(self.param_container)
            row.pack(fill=tk.X, padx=3, pady=1)
            ttk.Label(row, text=f"{param.label}:").pack(side=tk.LEFT)

            if param.type == bool:
                var = tk.BooleanVar(value=param.default)
                ttk.Checkbutton(row, variable=var).pack(side=tk.RIGHT)

            elif param.type == int:
                var = tk.IntVar(value=param.default)
                kwargs = {'width': 6, 'textvariable': var}
                if param.min_val is not None:
                    kwargs['from_'] = param.min_val
                if param.max_val is not None:
                    kwargs['to'] = param.max_val
                if param.step is not None:
                    kwargs['increment'] = param.step
                ttk.Spinbox(row, **kwargs).pack(side=tk.RIGHT)

            elif param.type == float:
                var = tk.DoubleVar(value=param.default)
                kwargs = {'width': 6, 'textvariable': var}
                if param.min_val is not None:
                    kwargs['from_'] = param.min_val
                if param.max_val is not None:
                    kwargs['to'] = param.max_val
                if param.step is not None:
                    kwargs['increment'] = param.step
                ttk.Spinbox(row, **kwargs).pack(side=tk.RIGHT)

            else:
                # Fallback: string entry
                var = tk.StringVar(value=str(param.default))
                ttk.Entry(row, textvariable=var, width=10).pack(side=tk.RIGHT)

            self.param_widgets[(method.name, param.name)] = var

    def _gather_params(self, method: SearchMethod) -> dict:
        """Read current widget values into a dict keyed by param name."""
        params = {}
        for param in method.parameters():
            key = (method.name, param.name)
            widget_var = self.param_widgets.get(key)
            if widget_var is not None:
                params[param.name] = widget_var.get()
            else:
                params[param.name] = param.default
        return params

    # ──────────────────────────────────────────────────────────
    # FILTER HELPERS
    # ──────────────────────────────────────────────────────────

    def _parse_filter_words(self) -> list[str]:
        text = self.filter_var.get().strip()
        if not text:
            return []
        words = [w.strip() for w in text.replace(',', ' ').split()]
        return [w for w in words if w]

    # ──────────────────────────────────────────────────────────
    # DATA LOADING
    # ──────────────────────────────────────────────────────────

    def _load_data(self):
        folder = filedialog.askdirectory(title="Select data folder")
        if not folder:
            return
        try:
            p = Path(folder)
            name = p.name
            if name == "xye":
                parent_name = p.parent.name
                if parent_name:
                    name = parent_name

            self.data_folder_name = name
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

    # ──────────────────────────────────────────────────────────
    # CACHE MANAGEMENT
    # ──────────────────────────────────────────────────────────

    def _load_caches(self):
        """Load both powder and peak caches from working directory."""
        self.cache = self.cache_manager.load()
        status = self.cache_manager.status_string(self.cache)

        if not self.cache.has_powder and not self.cache.has_peaks:
            self.status_var.set("Error: No caches found.")
            messagebox.showwarning(
                "Cache Not Found",
                "No powder_cache/ or peak_cache/ folder found.\n\n"
                "Please either:\n"
                "• Simulate powder patterns from CIFs first, or\n"
                "• Generate peak lists from CIFs first.")
        else:
            self.cand_status_var.set(status)

    def _load_cif_folder(self):
        folder = filedialog.askdirectory(
            title="Select folder containing CIF files")
        if not folder:
            return
        cif_files = [f for f in os.listdir(folder)
                     if f.lower().endswith('.cif')]
        if not cif_files:
            self.status_var.set("No .cif files found in selected folder.")
            return
        self.cif_folder = folder
        self.cif_folder_var.set(
            f"{os.path.basename(folder)} ({len(cif_files)} CIFs)")
        self.status_var.set(
            f"Loaded CIF folder: {folder} — {len(cif_files)} files found")

    def _select_inst_file(self):
        path = filedialog.askopenfilename(
            title="Select GSASII instrument parameter file",
            filetypes=[("Instrument files", "*.instprm *.prm"),
                       ("All files", "*.*")])
        if path:
            self.inst_file_var.set(path)

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

    def _simulate_from_cifs(self):
        """Simulate powder patterns from CIF folder → powder_cache/."""
        if not HAS_GSASII:
            messagebox.showerror("Error", "GSASII not available")
            return
        if self.two_theta is None:
            messagebox.showwarning("No Data", "Load data first (need 2θ grid)")
            return
        inst_file = self.inst_file_var.get()
        if not inst_file or not os.path.isfile(inst_file):
            messagebox.showwarning(
                "Missing", "Select or create an instrument parameter file first")
            return
        if self.cif_folder is None:
            self.status_var.set("Please load a CIF folder first.")
            return

        out_folder = self.cache_manager.powder_folder
        os.makedirs(out_folder, exist_ok=True)

        self.status_var.set("Simulating powder patterns from CIFs...")
        self.root.update()

        def progress(current, total):
            self.progress_var.set(current / total * 100)
            self.root.update_idletasks()

        profiles_dict, names = build_powder_cache_from_cifs(
            self.cif_folder, out_folder, self.two_theta,
            inst_file, progress_callback=progress)

        self.progress_var.set(100)

        # Reload cache to pick up new files
        self.cache = self.cache_manager.load()
        self.cand_status_var.set(
            f"Simulated {len(names)} patterns → powder_cache/ | "
            + self.cache_manager.status_string(self.cache))

    def _generate_peak_lists(self):
        """Generate candidate peak CSVs from CIF folder → peak_cache/."""
        if not HAS_GSASII:
            messagebox.showerror("Error", "GSASII not available")
            return
        if self.cif_folder is None:
            self.status_var.set("Please load a CIF folder first.")
            return

        wavelength = self.wavelength_var.get()
        tt_max = (self.two_theta[-1] if self.two_theta is not None else 12.0)
        peak_folder = self.cache_manager.peak_folder
        os.makedirs(peak_folder, exist_ok=True)

        self.status_var.set("Generating peak lists from CIFs...")
        self.root.update()

        count = 0
        cif_files = sorted([f for f in os.listdir(self.cif_folder)
                            if f.endswith('.cif')])

        for i, cif in enumerate(cif_files):
            try:
                gpx = G2sc.G2Project(newgpx='simulation.gpx')
                phase = gpx.add_phase(
                    os.path.join(self.cif_folder, cif), phasename='MyPhase')
                cell = phase.get_cell()
                cell_tup = (
                    cell['length_a'], cell['length_b'], cell['length_c'],
                    cell['angle_alpha'], cell['angle_beta'], cell['angle_gamma'])
                refls = G2sc.GenerateReflections(
                    phase.data['General']['SGData']['SpGrp'],
                    cell_tup, TTmax=tt_max, wave=wavelength)
                basename = cif[:-4]
                peaks_2th = d_to_two_theta(
                    filter_reflections(np.array(refls).T)[3], wavelength)
                if len(peaks_2th) > 0:
                    np.savetxt(
                        os.path.join(peak_folder, basename + '.csv'),
                        peaks_2th, delimiter=',')
                    count += 1
            except Exception:
                pass
            self.progress_var.set((i + 1) / len(cif_files) * 100)
            self.root.update_idletasks()

        self.progress_var.set(100)

        # Reload cache
        self.cache = self.cache_manager.load()
        self.cand_status_var.set(
            f"Generated {count} peak lists → peak_cache/ | "
            + self.cache_manager.status_string(self.cache))

    # ──────────────────────────────────────────────────────────
    # SEGMENTATION (main window display)
    # ──────────────────────────────────────────────────────────

    def _on_param_change(self, *args):
        """Update slider value labels."""
        self.thresh_label.config(text=f"{self.threshold_var.get()}")
        for name, var in self.weight_vars.items():
            if hasattr(var, '_label'):
                var._label.config(text=f"{var.get():.1f}")

    def _update_plots(self):
        """Compute segmentation and draw 3-panel figure in main window."""
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

        # ── Draw ──
        self.fig.clear()
        self.axes = self.fig.subplots(3, 1)
        t_axis = np.arange(self.n_times)

        # Panel 1: dI/dt heatmap
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

        # Panel 2: data with transitions
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

        # Panel 3: indicator curves
        ax = self.axes[2]
        indicator_styles = [
            {'data': results['dissimilarity'], 'label': 'Dissimilarity',
             'color': '#e41a1c', 'hatch': '///'},
            {'data': results['intensity_change'], 'label': '|dI/dt|',
             'color': '#377eb8', 'hatch': '\\\\\\'},
            {'data': results['channel_change'], 'label': '# channels',
             'color': '#4daf4a', 'hatch': '|||'},
            {'data': results['rank_change'], 'label': 'Rank change',
             'color': '#984ea3', 'hatch': '---'},
        ]
        for style in indicator_styles:
            ax.fill_between(t_axis, style['data'],
                            facecolor='none', edgecolor=style['color'],
                            hatch=style['hatch'], linewidth=0.0,
                            label=style['label'])
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
    # SEARCH DISPATCH (generic for any registered method)
    # ──────────────────────────────────────────────────────────

    def _run_search(self):
        """Run whichever search method is currently selected, then auto-save."""
        if self.data is None:
            messagebox.showwarning("No Data", "Load data first.")
            return
        if self.segment_labels is None:
            self._update_plots()

        method = self._get_selected_method()
        if method is None:
            messagebox.showerror("Error", "No search method selected.")
            return

        # Validate
        ok, msg = method.validate(self.cache)
        if not ok:
            messagebox.showwarning("Cannot Run", msg)
            return

        # Filter cache
        filter_words = self._parse_filter_words()
        filtered_cache = self.cache.filter_by_names(filter_words)
        if method.requires_peak_cache:
            filtered_cache = filtered_cache.filter_peaks_by_range(
                self.two_theta[0], self.two_theta[-1])

        if method.requires_powder_cache and not filtered_cache.has_powder:
            messagebox.showwarning(
                "No Candidates After Filter",
                f"No powder profiles match filter: '{self.filter_var.get()}'")
            return
        if method.requires_peak_cache and not filtered_cache.has_peaks:
            messagebox.showwarning(
                "No Candidates After Filter",
                f"No peak lists match filter: '{self.filter_var.get()}'")
            return

        # Gather params
        params = self._gather_params(method)

        # Progress
        self.progress_var.set(0)
        self.status_var.set(f"Running {method.display_name}...")
        self.root.update()

        def progress(current, total):
            self.progress_var.set(current / total * 100)
            self.root.update_idletasks()

        # Execute
        result: SearchResult = method.run(
            data=self.data,
            two_theta=self.two_theta,
            segment_labels=self.segment_labels,
            cache=filtered_cache,
            params=params,
            progress_callback=progress,
        )

        self.progress_var.set(100)

        # ── Auto-save results ──
        saved_path = self._save_result(result)

        # ── Display ──
        if saved_path:
            self.status_var.set(f"{result.summary}  •  Saved → {os.path.basename(saved_path)}")
        else:
            self.status_var.set(result.summary)

        results_win = ResultsWindow(self.root, result.method_name)
        result.plot(results_win.fig, transitions=self.transitions)
        results_win.canvas.draw()
        results_win.status_var.set(result.summary)

    def _save_result(self, result: SearchResult) -> str | None:
        """Auto-save search result to results/ folder. Returns filepath or None."""
        try:
            output_dir = os.path.join(os.getcwd(), 'results')
            os.makedirs(output_dir, exist_ok=True)

            # Use data folder name if available, else "unknown"
            folder_name = self.data_folder_name or "unknown_data"

            filepath = SearchResult.generate_filepath(
                output_dir, folder_name, result.method_name)

            actual_path = result.save(filepath)
            return actual_path

        except Exception as e:
            print(f"Warning: Failed to save results: {e}")
            return None
