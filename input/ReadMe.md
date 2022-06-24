`input/atm/` contains *in situ* half-hourly meteorological data and atmospheric
[CO<sub>2</sub>] that were recorded, gap-filled, and aggregated to 30-minute
timesteps as per [Yang et al. (2020)](https://doi.org/10.5194/bg-17-265-2020)
and [Mu et al. (2021)](https://doi.org/10.5194/hess-25-447-2021).

&nbsp;

`input/canopy/` contains the plant area index based on diffuse canopy
transmittance and monthly leaf litter production, extended from
[Duursma et al. (2016)](http://dx.doi.org/10.4225/35/563159f223739).

&nbsp;

`input/obs/` contains the following datasets:

* From [Gimeno et al. (2016)](http://dx.doi.org/10.4225/35/55b6e313444ff):
    EucFACE_water_potential_all_trees_2012_2014;
    EucFACE_water_potential_dominant_trees_2012_2013.

* From [Gimeno et al. (2018)](http://doi.org/10.4225/35/5ab9bd1e2f4fb):
    EucFACE_interception_2012_2014;
    EucFACE_sapflow_2012_2014;
    EucFACE_underET_2012_2014.

* Extended from
[Ellsworth et al. (2017)](http://doi.org/10.4225/35/57ec5d4a2b78e) and
[Wujeska‐Klause et al. (2019)](https://doi.org/10.1111/gcb.14555):
    EucFACE_photo_capacity_nitrogen_dominant_trees_2013_2020.

* Observations of soil moisture and soil texture at depth:
    EucFACE_sm_gap_filled;
    EucFACE_sm_neutron;
    EucFACE_soiltext.

&nbsp;

`input/params/` contains text files which summarise a series of calculated or
calibrated model parameters at each of EucFACE's rings.

&nbsp;

`input/projects/` contains the model driving files.

&nbsp;

`input/soil/` contains observation-based estimates of root-zone soil moisture
at EucFACE's rings.

&nbsp;

`C_budget_model_params.csv` contains the parameters of a simple data-derived,
carbon budget framework specific to EucFACE
([Extended Data figure 7 in Jiang et al. (2020)](https://www.nature.com/articles/s41586-020-2128-9/figures/10)).

`site_params.csv` summarises the site- or ring-specific model parameters used
to run the model.

&nbsp;

Manon Sabot: [m.e.b.sabot@gmail.com](mailto:m.e.b.sabot@gmail.com?subject=[Competing_Optimisations_Data]%20Source%20Han%20Sans)
