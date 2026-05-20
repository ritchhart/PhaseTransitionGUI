"""
PhaseAnalysis — Integrated Phase Transition Analysis
SMCR + Peak Match (MIP) pipelines
"""

# Optional dependency flags (checked once at import time)
try:
    from GSASII import GSASIIscriptable as G2sc
    from GSASII import GSASIIpwd
    HAS_GSASII = True
except ImportError:
    G2sc = None
    GSASIIpwd = None
    HAS_GSASII = False

try:
    from mip import Model, xsum, maximize, BINARY
    HAS_MIP = True
except ImportError:
    Model = xsum = maximize = BINARY = None
    HAS_MIP = False

try:
    import step_detect
    HAS_STEP_DETECT = True
except ImportError:
    step_detect = None
    HAS_STEP_DETECT = False
