"""Headless, config-driven live monitor: watches a data folder, reruns the
GUI's analysis pipeline as new frames arrive, and writes a static HTML
report. See README.md for usage; run with:

    python -m phase_analysis.monitor path/to/config.json
"""
