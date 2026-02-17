# General description

These tools aims at processing CROCO model outputs. It contains:

- \_\_init\_\_.py: initialisation file
- anova.py: to perform ANOVA decomposition
- bulk_flux.py: offline version of the bulk_flux.F of CROCO, inclusing COARE3 algorythme only
- convection.py: specific diagnostics for NBQ, (free) convection simulations
- convert_as_cmems.py: Convert native NEMO model output to a CMEMS-like dataset (i.e. regular grid, convection names and units), to be used in CROCO preprocessing tools.
- energy.py: energy diagnostics (e.g. kinetic energy, spectra).
- grid.py: define grids with xgcm ; vertical interpolation ; add metrics to xarray datasets.
- location_uncertainty.py: time-filter a xarray dataset with stochastic estimates of residual from filtering following Location Uncertainty framework.
- plot.py: for plots.


### \_\_init\_\_.py

### anova.py

### bulk_flux.py

### convection.py

### convert_as_cmems.py

### energy.py

### grid.py

### location_uncertainty.py

### plot.py
