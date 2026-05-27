"""
GSASII powder simulation, cache building, and candidate library loading.
"""

import os
import glob
import numpy as np
from phase_analysis import HAS_GSASII, G2sc


# ─── Powder simulation ────────────────────────────────────────

def gsas2powdersim(cifpath, outpath, twotheta, instrument_file, default_xtal_um=10):
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
        phases=[phase],
    )

    gpx.link_histogram_phase(hist, phase)
    phase.setSampleProfile(hist, 'size', 'isotropic', default_xtal_um)  # 10 μm crystallites

    gpx.do_refinements([{}])
    x, y = hist.getdata('X'), hist.getdata('Ycalc')
    np.savetxt(savefilename, np.column_stack([x, y]),
               header="2theta intensity", fmt="%.6f")
    return x, y


def build_powder_cache_from_cifs(cif_folder, output_folder, twotheta,
                                 instrument_file, progress_callback=None):
    """
    Batch-simulate all CIFs in a folder → cached .xy powder patterns.
    Returns dict {name: profile_array} and list of names.
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
            print(f"FAILED POWDER SIM: {cif}")
        if progress_callback:
            progress_callback(i + 1, len(cif_files))

    return profiles, names


# ─── Loading pre-computed data ────────────────────────────────

def load_precomputed_powders(powder_folder):
    """
    Load pre-computed powder patterns from .xy files.
    Each file: 2 columns (2theta, intensity), 1 header line.

    Returns: (candidate_profiles, candidate_names)
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


def load_candidate_library(peak_folder):
    """
    Load peak position CSVs for MIP matching.
    Returns dict {name: peak_positions_array}.
    """
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
