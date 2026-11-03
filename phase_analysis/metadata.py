"""Extract acquisition temperature from sidecar .tif.metadata files.

Beamline acquisition software (qxrd) writes one .tif.metadata file per
raw frame, living next to the .tif it describes — typically one directory
above the xye/ folder that the integrated 1D patterns get written into.
The sample temperature (°C) is stashed in userComment4 as a quoted,
comma-separated pair like userComment4="94.4,0.0"; we only need the first
number.
"""

import re
from pathlib import Path
from typing import Optional

_USER_COMMENT4_RE = re.compile(r'^userComment4\s*=\s*"?\s*([-+0-9.eE]+)', re.MULTILINE)


def find_metadata_file(xye_path) -> Optional[Path]:
    """Locate the .tif.metadata sidecar for a given .xye file, if any."""
    xye_path = Path(xye_path)
    stem = xye_path.stem  # "A2_850C_d1000-00000" from ".../xye/A2_850C_d1000-00000.xye"
    name = stem + ".tif.metadata"
    for candidate in (xye_path.with_name(name), xye_path.parent.parent / name):
        if candidate.is_file():
            return candidate
    return None


def read_temperature_c(xye_path) -> Optional[float]:
    """Read the sample temperature (°C) from the sidecar metadata, if present."""
    meta_path = find_metadata_file(xye_path)
    if meta_path is None:
        return None
    try:
        text = meta_path.read_text(errors='replace')
    except OSError:
        return None
    match = _USER_COMMENT4_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def read_temperatures(xye_paths) -> list:
    """Temperature (°C) per file, in order; None where unavailable."""
    return [read_temperature_c(p) for p in xye_paths]
