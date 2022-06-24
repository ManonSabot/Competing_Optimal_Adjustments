try:
    from hydraulics import f, Weibull_params, hydraulics
    from canatm import vpsat, slope_vpsat, LH_water_vapour, psychometric
    from canatm import conductances, canopy_intercept
    from canatm import emissivity, net_radiation, absorbed_radiation_2_leaves
    from leaf import leaf_temperature, leaf_energy_balance
    from leaf import calc_photosynthesis, rubisco_limit
    from soil import wetness, water_potential, soil_evap

except (ImportError, ModuleNotFoundError):
    from TractLSM.SPAC.hydraulics import f, Weibull_params, hydraulics
    from TractLSM.SPAC.canatm import vpsat, slope_vpsat, LH_water_vapour
    from TractLSM.SPAC.canatm import psychometric, conductances
    from TractLSM.SPAC.canatm import canopy_intercept
    from TractLSM.SPAC.canatm import emissivity, net_radiation
    from TractLSM.SPAC.canatm import absorbed_radiation_2_leaves
    from TractLSM.SPAC.leaf import leaf_temperature, leaf_energy_balance
    from TractLSM.SPAC.leaf import calc_photosynthesis, rubisco_limit
    from TractLSM.SPAC.soil import wetness, water_potential, soil_evap
