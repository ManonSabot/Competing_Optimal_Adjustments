# TractLSM (a Tractable simplified Land Surface Model)

The **TractLSM** is a LSM framework that integrates a suite of competing plant
optimality principles that operate at different functional levels and over
various timescales.

The model is organised as follows:

```bash
TractLSM
├── run_homogeneous_surface.py
├── run_utils.py
├── CH2OCoupler
│   ├── ProfitMax.py
│   ├── USO.py
├── SPAC
│   ├── canatm.py
│   ├── hydraulics.py
│   ├── leaf.py
│   ├── soil.py
├── TraitOptimisation
│   ├── optimise_JV.py
│   ├── optimise_kmax.py
└── Utils
    ├── build_final_forcings.py
    ├── built_in_plots.py
    ├── calculate_solar_geometry.py
    ├── constants_and_conversions.py
    ├── default_params.py
    ├── drivers_site_level.py
    ├── general_utils.py
    └── weather_generator.py
```

&nbsp;

`run_homogeneous_surface.py` is where the forcing is read, the main routines
called, and the output written. `run_utils.py` contains functions to support
these actions.

&nbsp;

The `CH2OCoupler/` is where you can find the `ProfitMax.py` approach, which is
derived/adapted from the work of
[Sperry et al. (2017)](https://doi.org/10.1111/pce.12852).
A more standard flux coupling method can be found in `USO.py`; it uses the
[Medlyn et al. (2011)](https://doi.org/10.1111/j.1365-2486.2010.02375.x) model.

&nbsp;

The model's biogeophysical routines can be found in the `SPAC/` repository,
ranging from micrometeorology (`canatm.py`) to plant hydraulics
(`hydraulics.py`).

&nbsp;

`TraitOptimisation/` contains a routine (`optimise_JV.py`) that optimises leaf
nitrogen allocation to photosynthetic compounds, thus optimising
V<sub>cmax25</sub> and J<sub>max25</sub> on timescales of days, as defined by
the user in the `irun.txt` file.
It also contains a routine (`optimise_kmax.py`) which calculates maximum
hydraulic conductance (k<sub>max</sub>) based on background climate at
site-level, and assuming coordination between the photosynthetic and hydraulic
traits. This routine can be used to calculate the input parameter
k<sub>max</sub>, needed to run the **ProfitMax** coupler, when it is not known.

&nbsp;

All support routines (automating the format of the input files, etc.) can be
found in `Utils/`.

&nbsp;

Manon Sabot: [m.e.b.sabot@gmail.com](mailto:m.e.b.sabot@gmail.com?subject=[Competing_Optimisations_Code]%20Source%20Han%20Sans)
