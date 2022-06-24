all: convert_files_format preparation simulations analyses plots

convert_files_format:
	- src/preparation/convert_npy_2_csv.py
	rm -f input/obs_test/*.npy
	rm -f input/canopy_test/*.npy

preparation:
	#src/estimate_model_parameters.sh

simulations:
	#src/timescales_of_optimality.sh
	#src/alt_params_forcings.sh -R timescales -v sw
	#src/alt_params_forcings.sh -R timescales -v Vcmax
	#src/alt_params_forcings.sh -R timescales -v LAI

analyses:
	src/analysis/test_differences.py -R timescales
	src/analysis/performance_scores.py -R timescales
	- src/analysis/N_photocapacities_drivers.py -R timescales

plots:
	src/plots/environmental_drivers.py -R timescales
	src/plots/effect_sizes.py -R timescales
	src/plots/performance.py -R timescales
	src/plots/N_photocapacities.py -R timescales
	src/plots/surface_fluxes_PLC.py -R timescales
	src/plots/soil_moisture.py -R timescales
	src/plots/surface_fluxes_PLC.py -R timescales -p
	src/plots/gs_Psi_behaviour.py -R timescales
	src/plots/LAI_PLC_biomass.py avg_LAI timescales

clean:
	@echo "Cleaning up all but 10 most recent logs..."
	@ls -tp src/tmp/log.o*| tail -n +11| xargs -r rm --
