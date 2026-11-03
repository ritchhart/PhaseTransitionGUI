"""JSON-file configuration for the live monitor. Keep this dependency-free
(stdlib json rather than YAML/TOML) so the monitor needs nothing beyond
what the GUI already requires.
"""

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MonitorConfig:
    # ── Required ──
    data_folder: str
    output_html: str

    # ── Pattern library ──
    # Either point pattern_folder at a directory with powder/peak
    # subfolders (names configurable below), or set powder_folder /
    # peak_folder directly — those win if both are given.
    pattern_folder: Optional[str] = None
    powder_subfolder: str = "powder_cache"
    peak_subfolder: str = "peak_cache"
    powder_folder: Optional[str] = None
    peak_folder: Optional[str] = None

    # ── Data loading ──
    file_extensions: list = field(default_factory=lambda: [".xy", ".xye"])
    bg_lambda: int = 4

    # ── Segmentation (mirrors the GUI's "Segmentation Parameters" panel) ──
    threshold_percentile: int = 90
    segmentation_weights: dict = field(default_factory=lambda: {
        'dissimilarity': 1.0,
        'intensity_change': 1.0,
        'channel_change': 1.0,
        'rank_change': 1.0,
    })

    # ── Search / match (mirrors the GUI's "Search / Match" panel) ──
    search_method: str = "smcr"
    search_params: dict = field(default_factory=dict)
    candidate_filter: list = field(default_factory=list)

    # ── Monitor loop ──
    poll_interval_seconds: float = 5.0
    quiet_seconds: float = 15.0
    html_refresh_seconds: Optional[int] = 30  # None disables <meta refresh>

    title: Optional[str] = None

    def resolved_powder_folder(self) -> Optional[str]:
        if self.powder_folder:
            return self.powder_folder
        if self.pattern_folder:
            return os.path.join(self.pattern_folder, self.powder_subfolder)
        return None

    def resolved_peak_folder(self) -> Optional[str]:
        if self.peak_folder:
            return self.peak_folder
        if self.pattern_folder:
            return os.path.join(self.pattern_folder, self.peak_subfolder)
        return None

    @classmethod
    def load(cls, path: str) -> "MonitorConfig":
        with open(path) as f:
            raw: dict[str, Any] = json.load(f)

        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"Unknown key(s) in {path}: {sorted(unknown)}. "
                f"Valid keys: {sorted(known)}")

        if 'segmentation_weights' in raw:
            merged = dict(cls.__dataclass_fields__['segmentation_weights']
                          .default_factory())
            merged.update(raw['segmentation_weights'])
            raw['segmentation_weights'] = merged

        config = cls(**raw)

        if not os.path.isdir(config.data_folder):
            raise ValueError(f"data_folder does not exist: {config.data_folder}")

        return config
