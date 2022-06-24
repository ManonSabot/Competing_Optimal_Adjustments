try:
    from USO import solve_uso
    from ProfitMax import maximise_profit

except (ImportError, ModuleNotFoundError):
    from TractLSM.CH2OCoupler.USO import solve_uso
    from TractLSM.CH2OCoupler.ProfitMax import maximise_profit
