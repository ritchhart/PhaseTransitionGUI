"""Unified cache access for search methods."""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from phase_analysis.pipeline.simulation import (
    load_precomputed_powders,
    load_candidate_library,
)


@dataclass
class CandidateCache:
    """
    Immutable snapshot of available candidate data.
    Search methods receive this — they never touch the filesystem directly.
    """
    # Powder profiles: name → intensity array (on the data's 2θ grid)
    powder_profiles: dict[str, np.ndarray] = field(default_factory=dict)

    # Peak positions: name → array of 2θ positions
    peak_positions: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def has_powder(self) -> bool:
        return len(self.powder_profiles) > 0

    @property
    def has_peaks(self) -> bool:
        return len(self.peak_positions) > 0

    def filter_by_names(self, words: list[str]) -> "CandidateCache":
        """Return a new cache containing only names matching any word."""
        if not words:
            return self

        def matches(name):
            return any(w in name for w in words)

        return CandidateCache(
            powder_profiles={k: v for k, v in self.powder_profiles.items()
                            if matches(k)},
            peak_positions={k: v for k, v in self.peak_positions.items()
                          if matches(k)},
        )

    def filter_peaks_by_range(self, tt_min: float, tt_max: float) -> "CandidateCache":
        """Return cache with peak positions clipped to 2θ range."""
        filtered_peaks = {}
        for name, peaks in self.peak_positions.items():
            in_range = peaks[(peaks >= tt_min) & (peaks <= tt_max)]
            if len(in_range) > 0:
                filtered_peaks[name] = in_range

        return CandidateCache(
            powder_profiles=self.powder_profiles,
            peak_positions=filtered_peaks,
        )


class CacheManager:
    """
    Handles loading/building caches from disk.
    The GUI talks to this; search methods only see CandidateCache.
    """

    def __init__(self, working_dir: Optional[str] = None):
        self.working_dir = working_dir or os.getcwd()
        self._powder_folder = os.path.join(self.working_dir, 'powder_cache')
        self._peak_folder = os.path.join(self.working_dir, 'peak_cache')

    @property
    def powder_folder(self):
        return self._powder_folder

    @property
    def peak_folder(self):
        return self._peak_folder

    def load(self) -> CandidateCache:
        """Load whatever caches exist on disk."""
        powder = {}
        peaks = {}

        if os.path.isdir(self._powder_folder):
            profiles, names = load_precomputed_powders(self._powder_folder)
            powder = dict(zip(names, profiles))

        if os.path.isdir(self._peak_folder):
            peaks = load_candidate_library(self._peak_folder)

        return CandidateCache(powder_profiles=powder, peak_positions=peaks)

    def status_string(self, cache: CandidateCache) -> str:
        parts = []
        if cache.has_powder:
            parts.append(f"{len(cache.powder_profiles)} powder profiles")
        if cache.has_peaks:
            parts.append(f"{len(cache.peak_positions)} peak lists")
        return " + ".join(parts) if parts else "No candidates loaded"
