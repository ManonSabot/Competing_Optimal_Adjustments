# -*- coding: utf-8 -*-

"""
The profit maximisation algorithm (between carbon gain and hydraulic
cost), adapted from Sperry et al. (2017)'s hydraulic-limited stomatal
optimisation model.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
----------
* Sperry et al. (2017). Predicting stomatal responses to the environment
  from the optimization of photosynthetic gain and hydraulic cost.
  Plant, cell & environment, 40(6), 816-830.

"""

__title__ = "Profit maximisation algorithm"
__author__ = "Manon E. B. Sabot"
__version__ = "3.0 (02.10.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import sys  # check for version on the system
import numpy as np  # array manipulations, math operators
import bottleneck as bn  # faster C-compiled np for all nan operations
from sympy import Symbol  # express functions
from sympy.utilities.lambdify import lambdify  # expression to function

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC import f, Weibull_params, hydraulics
from TractLSM.SPAC import absorbed_radiation_2_leaves  # radiation
from TractLSM.SPAC import leaf_energy_balance, leaf_temperature
from TractLSM.SPAC import calc_photosynthesis, rubisco_limit  # physio

if (sys.version_info < (3, 0)):
    import multiprocessing  # multiprocessing
    from joblib import Parallel, delayed  # multiprocessing


# ======================================================================

def hydraulic_cost(p, P):

    """
    Calculates the hydraulic cost function that reflects the increasing
    damage from cavitation and greater difficulty of moving up the
    transpiration stream with decreasing values of the hydraulic
    conductance. Also calculates the associated plant vulnerability.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    P: array or sympy symbol
        leaf water potential [MPa], either an array of values from the
        soil water potential Ps to the critical water potential Pcrit
        for which cavitation of the xylem occurs, or a sympy symbol to
        use in a symbolic expression

    Returns:
    --------
    cost: array or sympy symbolic expression
        hydraulic cost [unitless]

    f(P, b, c): array or sympy symbolic expression
        vulnerability curve of the the plant [unitless]

    """

    # Weibull parameters setting the shape of the vulnerability curve
    b, c = Weibull_params(p)  # MPa, unitless

    # current maximum plant hydraulic conductance (@ saturation)
    kmax = p.kmax * f(p.Ps, b, c)  # mmol s-1 m-2 MPa-1

    # plant vulnerability curve
    VC = f(P, b, c, recovery=p.recovembo, legacy=p.legembo)

    # hydraulic conductance, from kmax @ Ps to kcrit @ Pcrit
    k = p.kmax * VC  # mmol s-1 m-2 MPa-1

    # critical percentage below which cavitation occurs
    kcrit = p.kmax * p.ratiocrit  # xylem cannot recover past this point

    # cost, from kmax @ Ps to kcrit @ Pcrit, normalized, unitless
    cost = (kmax - k) / (kmax - kcrit)

    return cost, VC


def A_trans(p, trans, Ci, Tleaf=None):

    """
    Calculates the assimilation rate given the supply function.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    trans: array or sympy symbolic expression
        transpiration [mol m-2 s-1], either an array of values depending
        on the possible leaf water potentials (P) and the Weibull
        parameters b, c or a symbolic expression depending on the sympy
        symbol for P

    Ci: float or sympy symbol
        intercellular CO2 concentration [Pa], either corresponding to a
        leaf water potential (P) for which the transpiration cost is
        minimal and the C assimilation gain is maximal, or a sympy
        symbol

    Tleaf: float
        leaf temperature [degC]

    Returns:
    --------
    Either calculates the photosynthetic gain A [umol m-2 s-1] for a
    given Ci(P) and over an array oftrans(P) values or returns a
    symbolic expression.

    """

    # get CO2 diffusive conduct.
    gc, __, __ = leaf_energy_balance(p, trans, Tleaf=Tleaf)
    A_P = conv.U * gc * (p.CO2 - Ci) / (p.Patm * conv.MILI)

    try:
        A_P[np.isclose(np.squeeze(gc), cst.zero, rtol=cst.zero,
            atol=cst.zero)] = cst.zero

    except TypeError:
        pass

    return A_P


def symbolic_solve(expr, x, y, xvals, varsol, bound_expr):

    """
    Uses a combination of symbols and matrices to find the value of
    variable (y) for which expr(y) is minimised. Being able to express
    the solution symbolically can be an interesting feature depending on
    the user's needs.

    Arguments:
    ----------
    expr: sympy symbolic expression
        single-variable expression that does not allow exact symbolic
        solving

    x: sympy symbol
        unknown variable that yields different declinations of the expr

    y: sympy symbol
        unknown variable of the expression that is being minimized for

    xvals: array or list of floats
        acceptable / physical solutions for x

    varsol: array or list of floats
        acceptable / physical solutions for y for each x

    bound_expr: sympy symbolic expression
        expression of the maximum possible value of Ci

    Returns:
    --------
    The value of solsym for which the expression is the closest to being
    solved (closest to zero).

    """

    # return function from expression
    fun = lambdify((x, y), expr, 'numpy')
    max_fun = lambdify((x, y), bound_expr, 'numpy')

    # solutions over varsol
    match = fun(np.expand_dims(xvals, axis=1), varsol)

    # closest match to ~ 0. (i.e. supply ~ demand)
    idx = bn.nanargmin(abs(match), axis=1)

    # solution with approximate minimizing
    sol = np.asarray([varsol[e, idx[e]] for e in range(len(xvals))])

    # deal with mismatches by only allowing up to 5% variation around An
    up = abs(max_fun(xvals, sol))
    mismatch = bn.nanmin(abs(match), axis=1) <= 0.05 * up
    mismatch = mismatch.astype(int)

    if all(mismatch) == 0:  # no precise enough match
        mismatch[1] = 1  # pick 1st valid value

    sol = np.ma.masked_where(idx == 0, sol)
    sol = np.ma.masked_where(mismatch == 0, sol)

    return sol


def mtx_minimize(p, trans, all_Cis, photo):

    """
    Uses matrices to find each value of Ci for which An(supply) ~
    An(demand) on the transpiration stream.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    trans: array
        transpiration [mol m-2 s-1], values depending on the possible
        leaf water potentials (P) and the Weibull parameters b, c

    all_Cis: array
        all potential Ci values over the transpiration stream (for each
        water potential, Ci values can be anywhere between a lower bound
        and Cs)

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    Returns:
    --------
    The value of Ci for which An(supply) is the closest to An(demand)
    (e.g. An(supply) - An(demand) closest to zero).

    """

    demand, __, __, __ = calc_photosynthesis(p, np.expand_dims(trans, axis=1),
                                             all_Cis, photo)
    supply = A_trans(p, np.expand_dims(trans, axis=1), all_Cis)

    # closest match to ~ 0. (i.e. supply ~ demand)
    idx = bn.nanargmin(abs(supply - demand), axis=1)

    # each Ci on the transpiration stream
    Ci = np.asarray([all_Cis[e, idx[e]] for e in range(len(trans))])
    Ci = np.ma.masked_where(idx == 0, Ci)

    return Ci


def split(a, N):

    """
    Splits a list or array into N-roughly equal parts.

    Arguments:
    ----------
    a: list or array
        list/array to be split, can contain any data type

    N: int
        number of sub-lists/arrays the input must be split into

    Returns:
    --------
    A list of N-roughly equal parts.

    """

    integ = int(len(a) / N)
    remain = int(len(a) % N)

    splitted = [a[i * integ + min(i, remain):(i + 1) * integ +
                  min(i + 1, remain)] for i in range(N)]

    return splitted


def photo_gain(p, trans, photo, res, parallel, solstep, symbolic):

    """
    Calculates the photosynthetic C gain of a plant, where the
    photosynthetic rate (A) is evaluated over the array of leaf water
    potentials (P) and, thus transpiration (E), and normalized by the
    instantaneous maximum A over the full range of E.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    trans: array
        transpiration [mol m-2 s-1], values depending on the possible
        leaf water potentials (P) and the Weibull parameters b, c

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    res: string
        either 'low' (default), 'med', or 'high' to run the optimising
        solver

    parallel: boolean or int
        True parallelises the minimisation solving of Ci over the
        system's CPUs, an int input will parallelise over N=int CPUs

    solstep: string
        'var' means the stepping increment between the lower Ci and Cs
        changes depending on Cs (e.g. there are N fixed values between
        the lower Ci and Cs), while 'fixed' means the stepping increment
        between the lower Ci and Cs is fixed (e.g. N values between the
        lower Ci and Cs is not fixed)

    symbolic: boolean
        if True, a symbolic expression of Ci will be used to solve for
        Ci

    Returns:
    --------
    gain: array
        unitless instantaneous photosynthetic gains for possible values
        of Ci minimized over the array of E

    Ci: array
        intercellular CO2 concentration [Pa] for which A(P) is minimized
        to be as close as possible to the A predicted by either the
        Collatz or the Farquhar photosynthetis model

    """

    # accounting for canopy and leaf conductances is needed further
    __, gs, gb = leaf_energy_balance(p, trans)  # mol m-2 s-1

    # ref. photosynthesis for which the dark respiration is set to 0
    A_ref, __, __, __ = calc_photosynthesis(p, trans, p.CO2, photo, Rleaf=0.)

    # Cs < Ca
    boundary_CO2 = p.Patm * conv.FROM_MILI * A_ref / (gb * conv.GbcvGb)
    Cs = np.minimum(p.CO2, p.CO2 - boundary_CO2)  # Pa

    # potential Ci values over the full range of transpirations
    if res == 'low':
        iCi = 0.1
        NCis = 500

    if res == 'med':
        iCi = 0.02
        NCis = 8000

    if res == 'high':
        iCi = 0.001
        NCis = 50000

    if solstep == 'fixed':
        Cis = np.tile(np.arange(0.1, bn.nanmax(Cs) + iCi, iCi),
                      len(trans)).reshape(len(trans), -1)
        Cis = np.ma.masked_where(Cis > np.repeat(Cs, Cis.shape[1])
                                         .reshape(len(Cs), -1) + iCi, Cis)
        Cis[np.arange(len(trans)), bn.nanargmax(Cis, axis=1)] = Cs

    else:
        Cis = np.asarray([np.linspace(0.1, Cs[e], NCis) for e in
                          range(len(trans))])

    if symbolic:  # declare the symbols used to express A(P) = A(Ci)
        CI = Symbol('CI', positive=True)
        E = Symbol('E', positive=True)

        # A over a range of leaf intercellular CO2 concentrations (Pa)
        A_Ci, __, __, __ = calc_photosynthesis(p, E, CI, photo)

        # A of the transpiration E or leaf water potentials P
        A_P = A_trans(p, E, CI)

        # expression to be solved, A(P) = A (Ci) (demand = supply)
        expr = A_Ci - A_P

    # is the optimisation solved for in parallel or not?
    Njobs = 1

    if (type(parallel) == bool) and (parallel):
        Njobs = multiprocessing.cpu_count()

    elif type(parallel) == int:
        Njobs = parallel

    if Njobs > 1:
        sub_trans = split(trans, Njobs)
        sub_Cis = split(Cis, Njobs)
        Ci = []

        if symbolic:
            Ci += (Parallel(n_jobs=Njobs)(delayed(symbolic_solve)
                   (expr, E, CI, sub_trans[e], sub_Cis[e], A_Ci)
                   for e in range(len(sub_trans))))

        else:
            Ci += (Parallel(n_jobs=Njobs)(delayed(mtx_minimize)
                   (p, sub_trans[e], sub_Cis[e], photo)
                   for e in range(len(sub_trans))))

        Ci = np.asarray([e for sub_Ci in Ci for e in sub_Ci])

    else:
        if symbolic:
            Ci = symbolic_solve(expr, E, CI, trans, Cis, A_Ci)

        else:
            Ci = mtx_minimize(p, trans, Cis, photo)

    Ci = np.ma.masked_where(np.logical_or(Ci >= Cs, np.isnan(Ci)), Ci)

    try:
        A_P = A_trans(p, trans, Ci)  # get A demand (A(P))

        # update mask
        Ci[1:] = np.ma.masked_where(np.isclose(A_P[1:], 0.), Ci[1:])
        A_P[1:] = np.ma.masked_where(np.isclose(A_P[1:], 0.), A_P[1:])

        gain = A_P / np.ma.amax(A_P[1:])  # photo gain, soil P excluded

        if np.ma.amax(A_P[1:]) < 0.:  # when resp >> An everywhere
            gain *= -1.

    except ValueError:  # if trans is "pre-opimised" for
        gain = 0.

    return gain, Ci


def maximise_profit(p, photo='Farquhar', res='low', parallel=False,
                    solstep='var', symbolic=False, scaleup=True):

    """
    Finds the instateneous profit maximisation, following the
    optimisation criterion for which, at each instant in time, the
    stomata regulate canopy gas exchange and pressure to achieve the
    maximum profit, which is the maximum difference between the
    normalised photosynthetic gain (gain) and the hydraulic cost
    function (cost). That is when d(gain)/dP = d(cost)/dP.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    res: string
        either 'low' (default), 'med', or 'high' to run the optimising
        solver

    parallel: boolean or int
        True parallelises the minimisation solving of Ci over the
        system's CPUs, an int input will parallelise over N=int CPUs

    solstep: string
        'var' means the stepping increment between the lower Ci and Cs
        changes depending on Cs (e.g. there are N fixed values between
        the lower Ci and Cs), while 'fixed' means the stepping increment
        between the lower Ci and Cs is fixed (e.g. N values between the
        lower Ci and Cs is not fixed)

    symbolic: boolean
        if True, a symbolic expression of Ci will be used to solve for
        Ci

    scaleup: boolean
        True yields canopy-scale variables whilst False will lead the
        same variables but at the tree-scale

    Returns:
    --------
    fvuln: array
        reference vulnerability point for the optimisation for each leaf

    ksc: float
        instantaneous maximum root-to-leaf hydraulic conductance
        [mmol m-2 s-1 MPa-1] at maximum profit across leaves

    E_can: float
        transpiration [mmol m-2 s-1] at maximum profit across leaves

    gs_can: float
        stomatal conductance [mol m-2 s-1] at maximum profit across
        leaves

    Pleaf_can: float
        leaf water potential [MPa] at maximum profit across leaves

    An_can: float
        net photosynthetic assimilation rate [umol m-2 s-1] at maximum
        profit across leaves

    Ci_can: float
        intercellular CO2 concentration [Pa] at maximum profit across
        leaves

    rublim_can: string
        'True' if the C assimilation is rubisco limited, 'False'
        otherwise

    Tleaf_can: float
        leaf temperature [degC] at maximum profit across leaves

    Rleaf_can: float
        day respiration rate [umol m-2 s-1] at maximum profit across
        leaves

    """

    success = True  # initial assumption: the optimisation will succeed

    # original LAI, PPFD
    LAI = p.LAI
    PPFD = p.PPFD

    # retrieve relevant sunlit / shaded fractions
    fRcan, fPPFD, fLAI, fscale2can, __ = absorbed_radiation_2_leaves(p)

    # sunlit / shaded outputs
    fvc = np.zeros(len(fPPFD))
    E = np.zeros(len(fPPFD))
    gs_can = np.zeros(len(fPPFD))
    gs_can[:] = np.nan  # make sure we have nans for averaging
    Pleaf = np.zeros(len(fPPFD))
    An = np.zeros(len(fPPFD))
    Aj = np.zeros(len(fPPFD))
    Ac = np.zeros(len(fPPFD))
    Ci_can = np.zeros(len(fPPFD))
    Ci_can[:] = np.nan  # make sure we have nans for averaging
    Tleaf = np.zeros(len(fPPFD))
    Tleaf[:] = np.nan  # make sure we have nans for averaging
    Rleaf = np.zeros(len(fPPFD))

    # hydraulics
    P, trans = hydraulics(p, res=res)
    COST, VC = hydraulic_cost(p, P)

    # apply the scaling to match the As (conductance of can, not leaf)
    trans_can = [trans * e for e in fscale2can]

    # sunlit / shaded loop, two assimilation streams
    for i in range(len(fRcan)):

        if i == 0:  # sunlit
            fvc_opt = p.fvc_sun

        else:  # shaded
            fvc_opt = p.fvc_sha

        p.Rnet = fRcan[i]
        p.PPFD = fPPFD[i]
        p.LAI = fLAI[i]
        p.scale2can = fscale2can[i]
        trans = trans_can[i]
        cost = np.ma.copy(COST)
        vc = np.ma.copy(VC)

        if p.PPFD > 50.:  # min threshold for photosynthesis
            try:
                gain, Ci = photo_gain(p, trans, photo, res, parallel, solstep,
                                      symbolic)

                # look for the most net profit
                profit = gain - cost

                # deal with edge cases by rebounding the solution
                gc, gs, gb = leaf_energy_balance(p, trans)
                profit_check = profit[1:][np.logical_and(gc[1:] > cst.zero,
                                          np.logical_and(Ci[1:] / p.CO2 < 0.95,
                                                         gs[1:] <
                                                         (conv.GbvGbc /
                                                          conv.GwvGc) * gb))]
                idx = np.isclose(profit, max(profit_check))
                idx = [list(idx).index(e) for e in idx if e]

                if idx:  # opt values
                    fvc[i] = vc[idx[0]]
                    E[i] = trans[idx[0]]
                    gs_can[i] = gs[idx[0]]
                    Pleaf[i] = P[idx[0]]
                    Tleaf[i], __ = leaf_temperature(p, trans[idx[0]])
                    Ci_can[i] = Ci[idx[0]]

                    # rubisco- or electron transport-limitation?
                    An[i], Aj[i], Ac[i], Rleaf[i] = \
                        calc_photosynthesis(p, E[i], Ci_can[i], photo)

                else:
                    success = False

            except (ValueError, TypeError):  # no opt
                success = False

            if not success:  # this is rare, use last opt vc
                idx = bn.nanargmin(abs(vc - fvc_opt))
                fvc[i] = vc[idx]
                E[i] = trans[idx]
                Pleaf[i] = P[idx]
                Tleaf[i], __ = leaf_temperature(p, trans[idx])
                __, Ci_can[i] = photo_gain(p, np.asarray([E[i]]), photo, res,
                                           False, solstep, symbolic)
                Ci_can[i] = Ci_can[0]  # a single value is returned

                if Ci_can[i] >= 0.95:  # no solve
                    Ci_can[i] = np.nan

                if (str(Ci_can[i]) == '--'):  # no valid Ci
                    Tleaf[i] = p.Tair
                    fvc[i], gs_can[i], E[i], An[i], Ci_can[i], Rleaf[i] = \
                        (fvc_opt, ) + (0., ) * 5

                __, gs_can[i], __ = leaf_energy_balance(p, E[i])

                # rubisco- or electron transport-limitation?
                An[i], Aj[i], Ac[i], Rleaf[i] = calc_photosynthesis(p, E[i],
                                                                    Ci_can[i],
                                                                    photo)

            # if the critical point has been reached, stall
            if np.isclose(fvc[i], p.ratiocrit) or np.isclose(E[i], 0.):
                Tleaf[i] = p.Tair
                gs_can[i], E[i], An[i], Ci_can[i], Rleaf[i] = (0., ) * 5

        else:
            Tleaf[i] = p.Tair
            fvc[i], gs_can[i], E[i], An[i], Ci_can[i], Rleaf[i] = \
                (fvc_opt, ) + (0., ) * 5

    # deal with no solves for Pleaf
    Pleaf[Pleaf > p.Ps] = p.Ps

    if np.isclose(np.nanmean(Pleaf), p.Ps):
        E_can, gs_can, Pleaf_can, An_can, Ci_can, rublim_can, Tleaf_can, \
            Rleaf_can = (0., ) * 8

    else:  # sum contributions from sunlit and shaded leaves
        with np.errstate(invalid='ignore'):  # nans, no warning
            if any(np.isclose(Pleaf, p.Ps)):
                Pleaf[np.isclose(Pleaf, p.Ps)] = np.nan

            # deal with no solves for gs
            gs_can[np.isclose(gs_can, 0.)] = np.nan

            E_can = np.nansum(E) * conv.MILI  # mmol m-2 s-1
            gs_can = np.nanmean(gs_can)  # mol m-2 s-1
            Pleaf_can = np.nanmean(Pleaf)  # MPa
            An_can = np.nansum(An)  # umol m-2 s-1
            Ci_can = np.nanmean(Ci_can)  # Pa
            rublim_can = rubisco_limit(np.nansum(Aj), np.nansum(Ac))
            Tleaf_can = np.nanmean(Tleaf)  # degC
            Rleaf_can = np.nansum(Rleaf)  # umol m-2 s-1

    if not scaleup:  # downscale fluxes to the tree
        E_can /= np.sum(fscale2can)
        An_can /= np.sum(fscale2can)

    # reset original all canopy / forcing LAI, PPFD
    p.LAI = LAI
    p.PPFD = PPFD

    if (any(np.isnan([E_can, gs_can, An_can, Ci_can])) or
       any(np.isclose([E_can, gs_can, An_can, Ci_can], [0., ] * 4)) or
       any([E_can < 0., gs_can < 0., Ci_can < 0.]) or
       rublim_can == 0.):
        E_can, gs_can, An_can, Ci_can, Tleaf_can, Rleaf_can = (0., ) * 6

    return (fvc, p.kmax * VC[0], E_can, gs_can, Pleaf_can, An_can, Ci_can,
            rublim_can, Tleaf_can, Rleaf_can)
