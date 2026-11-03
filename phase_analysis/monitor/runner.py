"""Headless re-implementation of the GUI's "Load Data" -> "Update
Segmentation" -> "Run Search" chain, for a single pass over whatever
files currently exist in a data folder.

Segmentation and search are holistic (they look at the whole time series
at once), so there's no meaningful way to analyze "just the new frame" —
every run reprocesses the full, current set of files. The only part of
the pipeline that *is* purely per-frame is background subtraction, so
that's memoized across runs (see PreprocessCache) to keep a growing
dataset from re-paying for frames it already background-subtracted.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from phase_analysis.pipeline.preprocessing import auto_background
from phase_analysis.pipeline.segmentation import detect_transitions, get_segment_labels
from phase_analysis.cache import CacheManager
from phase_analysis.search import get_method
from phase_analysis.search.base import SearchResult
from phase_analysis.metadata import read_temperatures
from phase_analysis.monitor.config import MonitorConfig


def list_data_files(data_folder: str, extensions) -> list[str]:
    exts = {e.lower() for e in extensions}
    return sorted(
        f for f in os.listdir(data_folder)
        if os.path.isfile(os.path.join(data_folder, f))
        and not f.startswith('.')
        and os.path.splitext(f)[1].lower() in exts
    )


class PreprocessCache:
    """Memoizes per-file background subtraction, keyed by (mtime, size, log_lam)
    so an unchanged file is never reprocessed across monitor runs."""

    def __init__(self):
        self._store: dict[str, tuple] = {}  # path -> (key, two_theta, processed)

    def get_or_compute(self, path: str, log_lam: int):
        st = os.stat(path)
        key = (st.st_mtime_ns, st.st_size, log_lam)
        cached = self._store.get(path)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]

        raw = np.genfromtxt(path, skip_header=3, unpack=True)
        two_theta, intensity = raw[0], raw[1]
        bkrd = auto_background(intensity, log_lam=log_lam)
        processed = np.clip(intensity - bkrd, 0, None)
        self._store[path] = (key, two_theta, processed)
        return two_theta, processed

    def prune(self, live_paths) -> None:
        """Drop entries for files that no longer exist, so the cache doesn't
        grow unbounded if files get renamed/removed over a long-running
        monitor session."""
        live = set(live_paths)
        for stale in [p for p in self._store if p not in live]:
            del self._store[stale]


@dataclass
class RunResult:
    timestamp: float
    data_folder: str
    files: list[str]
    temperatures: list
    data: np.ndarray
    two_theta: np.ndarray
    segmentation: dict
    segment_labels: np.ndarray
    method_name: str
    method_display_name: str
    search_result: Optional[SearchResult] = None
    search_error: Optional[str] = None
    skipped_files: list = field(default_factory=list)

    @property
    def n_times(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]


def run_once(config: MonitorConfig, preprocess_cache: PreprocessCache) -> Optional[RunResult]:
    """Process every matching file currently in config.data_folder. Returns
    None if there's nothing to process (empty folder)."""

    filenames = list_data_files(config.data_folder, config.file_extensions)
    if not filenames:
        return None

    paths = [os.path.join(config.data_folder, f) for f in filenames]
    preprocess_cache.prune(paths)

    two_theta = None
    rows = []
    kept_files = []
    skipped = []
    for path in paths:
        try:
            tt, processed = preprocess_cache.get_or_compute(path, config.bg_lambda)
        except Exception as exc:
            skipped.append((path, str(exc)))
            continue
        if two_theta is None:
            two_theta = tt
        elif len(tt) != len(two_theta):
            skipped.append((path, f"2theta grid length {len(tt)} != {len(two_theta)}"))
            continue
        rows.append(processed)
        kept_files.append(path)

    if not rows:
        return None

    data = np.array(rows)
    n_times = data.shape[0]

    seg = detect_transitions(data, config.segmentation_weights, config.threshold_percentile)
    segment_labels = get_segment_labels(sorted(seg['transitions']), n_times)

    temperatures = read_temperatures(kept_files)

    cache_manager = CacheManager(
        powder_folder=config.resolved_powder_folder(),
        peak_folder=config.resolved_peak_folder())
    candidate_cache = cache_manager.load()
    if config.candidate_filter:
        candidate_cache = candidate_cache.filter_by_names(config.candidate_filter)

    search_result = None
    search_error = None
    method_display_name = config.search_method
    try:
        method_cls = get_method(config.search_method)
    except KeyError:
        search_error = f"Unknown search_method '{config.search_method}'"
    else:
        method = method_cls()
        method_display_name = method.display_name
        ok, msg = method.validate(candidate_cache)
        if not ok:
            search_error = msg
        else:
            filtered = candidate_cache
            if method.requires_peak_cache:
                filtered = filtered.filter_peaks_by_range(two_theta[0], two_theta[-1])
            params = {p.name: config.search_params.get(p.name, p.default)
                      for p in method.parameters()}
            try:
                search_result = method.run(
                    data=data, two_theta=two_theta,
                    segment_labels=segment_labels, cache=filtered, params=params)
            except Exception as exc:
                search_error = f"{type(exc).__name__}: {exc}"

    return RunResult(
        timestamp=time.time(),
        data_folder=config.data_folder,
        files=[os.path.basename(p) for p in kept_files],
        temperatures=temperatures,
        data=data,
        two_theta=two_theta,
        segmentation=seg,
        segment_labels=segment_labels,
        method_name=config.search_method,
        method_display_name=method_display_name,
        search_result=search_result,
        search_error=search_error,
        skipped_files=skipped,
    )
