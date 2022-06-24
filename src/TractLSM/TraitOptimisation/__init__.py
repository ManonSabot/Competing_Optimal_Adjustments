try:
    from optimise_kmax import optimal_kmax
    from optimise_JV import ini_N_pools
    from optimise_JV import optimal_JV

except (ImportError, ModuleNotFoundError):
    from TractLSM.TraitOptimisation.optimise_kmax import optimal_kmax
    from TractLSM.TraitOptimisation.optimise_JV import ini_N_pools
    from TractLSM.TraitOptimisation.optimise_JV import optimal_JV
