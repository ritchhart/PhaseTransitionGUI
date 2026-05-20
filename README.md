GUI for analyzing timeseries of powder diffraction files using GSASii backend for crystal data handling and using ML analysis methods

Run
python -m phase_analysis

Search methods are modular and just need to implement base.py parent class

Current pipeline assumes a few things about data structure eg beamline folders of .xy files
