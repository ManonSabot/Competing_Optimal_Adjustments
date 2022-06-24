# -*- coding: utf-8 -*-

"""
Hypothetising short-term (N days) coordination between the Vmax25 and
Jmax25 to maximise An for a given amount of leaf nitrogen.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
-----------
* Collatz et al. (1991). Regulation of stomatal conductance and
  transpiration: a physiological model of canopy processes. Agric. For.
  Meteorol, 54, 107-136.
* Evans, J. R. (1989a). Photosynthesis and nitrogen relationships in
  leaves of C3 plants. Oecologia, 78(1), 9-19.
* Evans, J. R. (1989b). Photosynthesis—the dependence on nitrogen
  partitioning. In ‘Causes and Consequences of Variation in Growth Rate
  and Productivity of Higher Plants’. (Eds H. Lambers, ML Cambridge, H.
  Konings and TL Pons.), 159–174.
* Farquhar, G. D., von Caemmerer, S. V., & Berry, J. A. (1980). A
  biochemical model of photosynthetic CO2 assimilation in leaves of C3
  species. Planta, 149(1), 78-90.
* Medlyn, B. E. (1996). The optimal allocation of nitrogen within the C3
  photosynthetic system at elevated CO2. Functional Plant Biology,
  23(5), 593-603.

"""

__title__ = "Optimise J:V ratio"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (06.10.2019)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators
from sympy import Symbol  # express functions symbolically
from sympy import nonlinsolve  # solve multivariate systems of eqns
from sympy.utilities.lambdify import lambdify  # expression to function

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC import calc_photosynthesis  # photosynthesis
from TractLSM.SPAC.leaf import arrhen


# ======================================================================

def other_N_pools(p, Ntot, Nc, Ne):

    """
    Calculates the amount of remaining nitrogen involved in
    photosynthesis that is neither from the electron-transport system
    nor from the chlorophyll-protein complexes. This is based on the
    work of Evans (1989a), Evans (1989b), and Medlyn (1996).

    Arguments:
    ----------
    p: pandas series
        model's default parameter values

    Ntot: float
        total amount of nitrogen involved in photosynthesis [mol m-2]

    Nc: float
        amount of nitrogen in chlorophyll [mol m-2]

    Ne: float
        amount of nitrogen in electron transport components [mol m-2]

    Returns:
    --------
    Nr: float
        amount of nitrogen in Rubisco [mol m-2]

    Ns: float
        amount of nitrogen in soluble protein other than Rubisco
        [mol m-2]

    """

    Ns = p.ks * p.JV * p.Vmax25
    Nr = np.maximum(0., Ntot - Nc - Ne - Ns)

    return Nr, Ns


def ini_N_pools(p):

    """
    Initialises the amount of nitrogen in the respective pools involved
    in photosynthesis. This is done by solving the system of equations
    described in Medlyn (1996).

    Arguments:
    ----------
    p: pandas series
        model's default parameter values

    Returns:
    --------
    Ntot: float
        total amount of nitrogen involved in photosynthesis [mol m-2]

    Nc: float
        amount of nitrogen in chlorophyll [mol m-2]

    Ne: float
        amount of nitrogen in electron transport components [mol m-2]

    Nr: float
        amount of nitrogen in Rubisco [mol m-2]

    Ns: float
        amount of nitrogen in soluble protein other than Rubisco
        [mol m-2]

    """

    # solve for Nc and Ne, given the input parameters
    Nc = p.ccN * (1. - p.tau_l - p.albedo_l) / (p.tau_l + p.albedo_l)
    Ne = (p.Vmax25 * p.JV - p.bj * Nc) / p.aj

    # Ntot is given by Nc + Ne + Ns + Nr
    Ntot = (Nc + Ne + p.ks * (p.aj * Ne + p.bj * Nc) +
            p.Vmax25 / (p.kcat / (cst.MRub * cst.NR)))

    # use other formulations to check Nc and Ne
    Nr, Ns = other_N_pools(p, Ntot, Nc, Ne)

    return Ntot, Nc, Ne, Nr, Ns


def coord_JV(p, Nc, Ne, Ci, Tleaf, Rleaf, photo='Farquhar'):

    """
    Coordinates the JV ratio to Vmax25 so that Aj = Ac. The photo model
    is either the classic Farquhar photosynthesis model or the Collatz
    model. The JV(Vmax25) relationship is found by solving the following
    system of equations:
      Eq1: Aj - Ac = 0
      Eq2: Ntot - Ntot(Nc, Ne, Vmax25, JV) = 0

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    Nc: float
        amount of nitrogen in chlorophyll [mol m-2]

    Ne: float
        amount of nitrogen in electron transport components [mol m-2]

    Ci: float
        intercellular CO2 concentration [Pa]

    Tleaf: float
        leaf temperature [degC]

    Rleaf: float
        leaf day respiration [μmol m-2 s-1]

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    Returns:
    --------
    The executable JV(Vcmax) function by which Aj = Ac.

    """

    # declare JV and Vmax25, which we're trying to solve for, as symbols
    JV = Symbol('JV', positive=True)
    VMAX25 = Symbol('VMAX25', positive=True)

    # the first equation in our system is Ac - Aj which is non-linear
    __, AC, AJ, __ = calc_photosynthesis(p, 0., Ci, photo, Tleaf=Tleaf,
                                         Rleaf=Rleaf, Vmax25=VMAX25, JV=JV)
    expr1 = AC - AJ

    # the second equation is simply Ntot - Ntot(Nc, Ne, Vmax25, JV)
    expr2 = p.Ntot - (Nc + Ne + p.ks * JV * VMAX25 +
                      VMAX25 / (p.kcat / (cst.MRub * cst.NR)))

    # we solve a non-linear system of eqs and get JV(Vcmax25)
    (expr_JV, __) = next(iter(nonlinsolve([expr1, expr2], [JV, VMAX25])))

    return lambdify((VMAX25), expr_JV, 'numpy')


def optimal_JV(p, Nc, Ne, Ci, Tleaf, Rleaf, photo='Farquhar'):

    """
    Finds the optimal JV ratio for which Ac and Aj are coordinated and
    An increases. The photo model is either the classic Farquhar
    photosynthesis model or the Collatz model.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    Nc: float
        amount of nitrogen in chlorophyll [mol m-2]

    Ne: float
        amount of nitrogen in electron transport components [mol m-2]

    Ci: float
        intercellular CO2 concentration [Pa]

    Tleaf: float
        leaf temperature [degC]

    Rleaf: float
        leaf day respiration [μmol m-2 s-1]

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    Returns:
    --------
    Vmax25: float
        maximum carboxylation rate at 25 degrees [μmol m-2 s-1]

    JV: float
        unitless ratio of Jmax to Vmax at 25 degrees

    Rleaf: float
        leaf day respiration [μmol m-2 s-1]

    Nc: float
        amount of nitrogen in chlorophyll [mol m-2]

    Ne: float
        amount of nitrogen in electron transport components [mol m-2]

    Nr: float
        amount of nitrogen in Rubisco [mol m-2]

    Ns: float
        amount of nitrogen in soluble protein other than Rubisco
        [mol m-2]

    """

    try:
        # function which describes JV(Vcmax) when Aj = Ac
        opt_JV = coord_JV(p, Nc, Ne, Ci, Tleaf, Rleaf, photo=photo)

        # increase / decrease Vcmax (i.e. decrease or increase JV ratio)
        Vmax25 = np.arange(p.Vmaxmin, p.Vmaxmax, 0.1)
        JV = opt_JV(Vmax25)

        # safety checks
        mask = np.logical_and(JV > p.JVmin, JV < p.JVmax)
        Vmax25 = Vmax25[mask]
        JV = JV[mask]

        # new An
        An, Ac, Aj, __ = calc_photosynthesis(p, 0., Ci, photo, Tleaf=Tleaf,
                                             Rleaf=Rleaf, Vmax25=Vmax25, JV=JV)

        # are any of the results valid?
        if all(An < 0.) or all(An == An[0]):
            raise Exception

        # proxy a TPU limitation on An, to avoid Jmax overshoots
        Vmax = arrhen(Vmax25, p.Ev, p.Tref + conv.C_2_K, Tleaf,
                      deltaS=p.deltaSv, Hd=p.Hdv)
        mask = An <= 0.5 * Vmax

        # optimality criterion
        idx_coord = np.nanargmin(np.abs(Ac[mask] - Aj[mask]))

        # check whether the 'optimal point' is at the limit
        if (np.isclose(JV[mask][idx_coord], p.JVmin) or
           np.isclose(JV[mask][idx_coord], p.JVmax)):
            raise Exception

        # optimal point
        JV = JV[mask][idx_coord]
        Vmax25 = Vmax25[mask][idx_coord]

        # reshuffle pools
        p.Vmax25 = Vmax25
        p.JV = JV
        Ntot, Nc, Ne, Nr, Ns = ini_N_pools(p)

    except Exception:  # the coordination doesn't maximise An
        Vmax25 = p.Vmax25
        JV = p.JV
        Nr, Ns = other_N_pools(p, p.Ntot, Nc, Ne)

    # if that point is reached, An is maximised, we update Rleaf
    Rleaf = 0.015 * Vmax25

    return Vmax25, JV, Rleaf, Nc, Ne, Nr, Ns
