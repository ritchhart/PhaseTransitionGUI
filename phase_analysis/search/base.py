"""Base classes for search/match methods."""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from phase_analysis.cache import CandidateCache


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Only serialize small arrays; skip large data matrices
            if obj.size <= 1000:
                return obj.tolist()
            else:
                return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


@dataclass
class Parameter:
    """Describes one tunable parameter for a search method."""
    name: str
    label: str
    type: type
    default: Any
    min_val: Any = None
    max_val: Any = None
    step: Any = None
    tooltip: str = ""


@dataclass
class SearchResult:
    """
    Standard output container from any search method.
    Carries both raw data (for programmatic use) and knows how to plot/save.
    """
    # Identity
    method_name: str
    summary: str

    # Raw output — may contain non-serializable objects (solver instances, etc.)
    # Used for plotting and programmatic access; NOT saved to disk.
    raw: dict = field(default_factory=dict)

    # Structured fields — always serializable
    phases_found: list[str] = field(default_factory=list)
    phase_presence: np.ndarray = None       # (n_phases, n_time)
    confidence: dict[str, float] = field(default_factory=dict)
    params_used: dict[str, Any] = field(default_factory=dict)

    def plot(self, fig: Figure, **kwargs) -> None:
        """Render results onto a matplotlib Figure. Override in subclass."""
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"No plot implemented for {self.method_name}",
                ha='center', va='center', transform=ax.transAxes)

    def to_dict(self) -> dict:
        """
        Return a JSON-serializable summary of results.
        Override in subclasses to add method-specific fields.
        Does NOT include raw (which may contain large arrays / objects).
        """
        d = {
            'method': self.method_name,
            'summary': self.summary,
            'phases_found': self.phases_found,
            'confidence': self.confidence,
            'params_used': self.params_used,
            'n_phases': len(self.phases_found),
        }
        if self.phase_presence is not None:
            d['phase_presence'] = self.phase_presence
        return d

    def save(self, filepath: str) -> str:
        """
        Save serializable results to JSON.

        Parameters
        ----------
        filepath : str
            Output path. If it ends in '/' or is a directory,
            an auto-generated filename is appended.

        Returns
        -------
        The actual filepath written.
        """
        # If filepath is a directory, generate a filename
        if os.path.isdir(filepath) or filepath.endswith(os.sep):
            os.makedirs(filepath, exist_ok=True)
            filename = self._auto_filename()
            filepath = os.path.join(filepath, filename)

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        data = self.to_dict()
        data['saved_at'] = datetime.now().isoformat()

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)

        return filepath

    def _auto_filename(self) -> str:
        """Generate a default filename from method name + timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_method = self.method_name.lower().replace(' ', '_').replace('-', '_')
        return f"{safe_method}_{timestamp}.json"

    @staticmethod
    def generate_filepath(output_dir: str, data_folder_name: str,
                          method_name: str) -> str:
        """
        Build a descriptive filepath:
          output_dir/data_folder_name__method__YYYYMMDD_HHMMSS.json
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_method = method_name.lower().replace(' ', '_').replace('-', '_')
        safe_folder = os.path.basename(data_folder_name.rstrip(os.sep))
        # Remove problematic characters
        safe_folder = "".join(
            c if c.isalnum() or c in ('_', '-') else '_'
            for c in safe_folder
        )
        filename = f"{safe_folder}__{safe_method}__{timestamp}.json"
        return os.path.join(output_dir, filename)


class SearchMethod(ABC):
    """
    Base class for all search/match algorithms.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        ...

    @property
    @abstractmethod
    def requires_powder_cache(self) -> bool:
        ...

    @property
    @abstractmethod
    def requires_peak_cache(self) -> bool:
        ...

    @abstractmethod
    def parameters(self) -> list[Parameter]:
        ...

    @abstractmethod
    def run(self,
            data: np.ndarray,
            two_theta: np.ndarray,
            segment_labels: np.ndarray,
            cache: CandidateCache,
            params: dict[str, Any],
            progress_callback=None) -> SearchResult:
        ...

    def validate(self, cache: CandidateCache) -> tuple[bool, str]:
        if self.requires_powder_cache and not cache.has_powder:
            return False, f"{self.display_name} requires powder profiles."
        if self.requires_peak_cache and not cache.has_peaks:
            return False, f"{self.display_name} requires peak position lists."
        return True, ""
