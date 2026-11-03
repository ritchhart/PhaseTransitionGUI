GUI for analyzing timeseries of powder diffraction files using GSASii backend for crystal data handling and using ML analysis methods

Search methods are modular and just need to implement base.py parent class

Current pipeline assumes a few things about data structure eg beamline folders of .xy files

## Setup

```
pip install -r requirements.txt
```

That's enough to run the GUI and both search methods against a pre-built
pattern library (`powder_cache/` + `peak_cache/`, or any folder laid out
the same way). GSAS-II is only needed to *generate* that library from CIFs
(or for its background-fit routine instead of the built-in fallback) — it's
a heavy, slow-to-install dependency, kept out of the main requirements:

```
pip install -r requirements-gsas.txt
```

Everything that touches GSAS-II is gated behind `HAS_GSASII` and degrades
gracefully without it (see `phase_analysis/__init__.py`).

## Running the GUI

```
python -m phase_analysis
```

## Live analysis monitor

`phase_analysis/monitor` reruns the GUI's load → segment → search pipeline
headlessly against a data folder, as a background process, and writes the
result to a static HTML file you refresh in a browser — no second GUI to
keep in sync, no server to run.

```
python -m phase_analysis.monitor path/to/config.json
```

See `monitor_config.example.json` for every option; the required ones are
`data_folder` and `output_html`. Point `pattern_folder` at a directory with
powder/peak subfolders (`powder_subfolder` / `peak_subfolder`, defaulting
to `powder_cache` / `peak_cache`), or set `powder_folder` / `peak_folder`
directly for an unconventional layout. `search_method` / `search_params`
/ `candidate_filter` mirror the GUI's "Search / Match" panel, and
`segmentation_weights` / `threshold_percentile` mirror "Segmentation
Parameters".

**How it handles acquisition outrunning analysis:** segmentation and
search look at the whole time series at once, so there's no such thing as
analyzing "just the new frame" — every run reprocesses every file
currently in `data_folder`. The monitor never queues a run per new file
(that would just build an ever-growing backlog of stale runs); instead it
waits for the folder to go quiet for `quiet_seconds`, runs once against
that snapshot, and immediately checks again when done. If acquisition
outpaced that run, the next snapshot already reflects wherever things
landed, and gets processed once. Per-frame background subtraction (the
one part of the pipeline that *isn't* holistic) is memoized across runs,
so a growing dataset doesn't keep re-paying for frames it already
background-subtracted.

The HTML includes an optional `<meta refresh>` (`html_refresh_seconds`,
`null` to disable) — either way, the file is rewritten atomically on each
run, so a manual refresh is always safe.

## Sample temperature from acquisition metadata

If a `.xye` file has a sidecar `<name>.tif.metadata` next to it (or one
directory up — `qxrd` writes those alongside the raw `.tif`, one level
above the `xye/` folder the integrated patterns land in), the pipeline
reads the sample temperature from `userComment4`, e.g.
`userComment4="94.4,0.0"` → 94.4 °C. See `phase_analysis/metadata.py`.
The live monitor plots this alongside the segmentation view; if no
metadata is found the plot just falls back to frame index.
