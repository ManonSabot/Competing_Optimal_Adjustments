# -*- coding: utf-8 -*-

"""
The Medlyn (USO) model, adapted for LSMs, by iteration on the air
temperature to get the leaf temperature for which the Penman-Monteith
energy balance conditions are satisfied.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
----------
* Kowalczyk, E. A., Wang, Y. P., Law, R. M., Davies, H. L., McGregor,
  J. L., & Abramowitz, G. (2006). The CSIRO Atmosphere Biosphere Land
  Exchange (CABLE) model for use in climate models and as an offline
  model. CSIRO Marine and Atmospheric Research Paper, 13, 42.
* Medlyn, B. E., Duursma, R. A., Eamus, D., Ellsworth, D. S., Prentice,
  I. C., Barton, C. V., ... & Wingate, L. (2011). Reconciling the
  optimal and empirical approaches to modelling stomatal conductance.
  Global Change Biology, 17(6), 2134-2144.
* Wang, Y. P., Kowalczyk, E., Leuning, R., Abramowitz, G., Raupach,
  M. R., Pak, B., ... & Luhar, A. (2011). Diagnosing errors in a land
  surface model (CABLE) in the time and frequency domains. Journal of
  Geophysical Research: Biogeosciences, 116(G1).

"""

__title__ = "Iterative solving with the USO model"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (02.10.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC import hydraulics  # to get LWP estimates
from TractLSM.SPAC import absorbed_radiation_2_leaves  # radiation
from TractLSM.SPAC import conductances, leaf_temperature  # energy
from TractLSM.SPAC import LH_water_vapour, vpsat, slope_vpsat, psychometric
from TractLSM.SPAC import calc_photosynthesis, rubisco_limit  # physio


# ======================================================================

def calc_trans(p, Tleaf, gs):

    """
    Calculates transpiration following Penman-Monteith at the leaf level
    accounting for effects of leaf temperature and feedback on
    evaporation.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    Tleaf: float
        leaf temperature [degC]

    gs: float
        stomatal conductance [mol m-2 s-1]

    Returns:
    --------
    trans: float
        transpiration rate [mol m-2 s-1]

    real_zero: boolean
        True if the transpiration is really zero, False if Rnet is
        negative

    gw: float
        total leaf conductance to water vapour [mol m-2 s-1]

    gb: float
        boundary layer conductance to water vapour [mol m-2 s-1]

    """

    # check that the trans value satisfies the energy balance
    real_zero = True

    # get conductances
    gw, gH, gb, __ = conductances(p, Tleaf=Tleaf, gs=gs)  # mol m-2 s-1

    # latent heat of water vapor
    Lambda = LH_water_vapour(p)  # J mol-1

    # slope of saturation vapour pressure of water vs Tair
    slp = slope_vpsat(p)  # kPa degK-1

    if np.isclose(gs, 0., rtol=cst.zero, atol=cst.zero):
        trans = cst.zero

    else:
        gamm = psychometric(p)  # psychrometric constant, kPa degK-1
        trans = (slp * p.Rnet + p.VPD * gH * cst.Cp) / (Lambda *
                                                        (slp + gamm * gH / gw))

        if trans < 0.:  # Penman-Monteith failed, non-physical trans
            real_zero = False

        trans = max(cst.zero, trans)  # mol m-2 s-1

    return trans, real_zero, gw, gb


def solve_uso(p, photo='Farquhar', threshold_conv=0.015, iter_max=40,
              scaleup=True):

    """
    Checks the energy balance by looking for convergence of the new leaf
    temperature with the leaf temperature predicted by the previous
    iteration. Then returns the corresponding An, E, Ci, etc.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    threshold_conv: float
        convergence threshold for the new leaf temperature to be in
        energy balance

    iter_max: int
        maximum number of iterations allowed on the leaf temperature
        before reaching the conclusion that the system is not energy
        balanced

    scaleup: boolean
        True yields canopy-scale variables whilst False will lead the
        same variables but at the tree-scale

    Returns:
    --------
    trans_can: float
        transpiration rate of canopy [mmol m-2 s-1] across leaves

    gs_can: float
        stomatal conductance of canopy [mol m-2 s-1] across leaves

    Pleaf_can: float
        leaf water potential [MPa] across leaves

    An_can: float
        C assimilation rate of canopy [umol m-2 s-1] across leaves

    Ci_can: float
        average intercellular CO2 concentration of canopy [Pa] across
        leaves

    rublim_can: string
        'True' if the C assimilation is rubisco limited, 'False'
        otherwise.

    Tleaf_can: float
        leaf temperature [degC] across leaves

    Rleaf_can: float
        day respiration rate [umol m-2 s-1] across leaves


    """

    # retrieve sunlit / shaded fractions
    fRcan, fPPFD, fLAI, fscale2can, fgradis = absorbed_radiation_2_leaves(p)

    # saturation vapour pressure of water at Tair
    esat_a = vpsat(p.Tair)  # kPa

    # hydraulics
    P, E = hydraulics(p)

    # sunlit / shaded outputs
    trans = np.zeros(len(fPPFD))
    gs = np.zeros(len(fPPFD))
    gs[:] = np.nan  # make sure we have nans for averaging
    An = np.zeros(len(fPPFD))
    Aj = np.zeros(len(fPPFD))
    Ac = np.zeros(len(fPPFD))
    Ci = np.zeros(len(fPPFD))
    Ci[:] = np.nan  # make sure we have nans for averaging
    Tleaf = np.zeros(len(fPPFD))
    Tleaf[:] = np.nan  # make sure we have nans for averaging
    Rleaf = np.zeros(len(fPPFD))

    # original LAI, PPFD
    LAI = p.LAI
    PPFD = p.PPFD

    # sunlit / shaded loop
    for i in range(len(fRcan)):

        p.Rnet = fRcan[i]
        p.PPFD = fPPFD[i]
        p.LAI = fLAI[i]
        p.scale2can = fscale2can[i]
        p.gradis = fgradis[i]

        if p.PPFD > 50.:  # min threshold for photosynthesis
            fw = np.maximum(cst.zero,
                            np.minimum(1., np.exp((p.Ps + p.nPs) * p.sfw)))
            Cs = p.CO2  # Pa
            Tleaf[i] = p.Tair  # deg C

            # leaf-air vpd, kPa
            Dleaf = np.maximum(0.05, p.VPD)  # gs model not valid ~0.05

            # initialise gs_over_A
            g0 = 1.e-9  # g0 ~ 0, removing it entirely introduces errors
            Cs_umol_mol = Cs * conv.MILI * conv.FROM_kPa  # umol mol-1
            gs_over_A = g0 + (1. + p.g1 * fw / (Dleaf ** 0.5)) / Cs_umol_mol

            # iter on the solution until it is stable enough
            iter = 0

            while True:

                An[i], Aj[i], Ac[i], Ci[i], Rleaf[i] = \
                    calc_photosynthesis(p, 0., Cs, photo, Tleaf=Tleaf[i],
                                        gs_over_A=gs_over_A)

                # stomatal conductance, with fw effect
                Cs_umol_mol = Cs * conv.MILI * conv.FROM_kPa
                gs_over_A = (g0 + (1. + p.g1 * fw / (Dleaf ** 0.5)) /
                             Cs_umol_mol)
                gs[i] = np.maximum(cst.zero, conv.GwvGc * gs_over_A * An[i])

                # calculate new trans, gw, gb, Tleaf
                trans[i], real_zero, gw, gb = calc_trans(p, Tleaf[i], gs[i])
                new_Tleaf, __ = leaf_temperature(p, trans[i], Tleaf=Tleaf[i],
                                                 gradis=True)

                # new Cs (in Pa)
                boundary_CO2 = (p.Patm * conv.FROM_MILI * An[i] /
                                (gb * conv.GbcvGb))
                Cs = np.maximum(cst.zero,
                                np.minimum(p.CO2, p.CO2 - boundary_CO2))

                if (np.isclose(trans[i], cst.zero, rtol=cst.zero,
                   atol=cst.zero) or np.isclose(gw, cst.zero, rtol=cst.zero,
                   atol=cst.zero) or np.isclose(gs[i], cst.zero, rtol=cst.zero,
                   atol=cst.zero)):
                    Dleaf = np.maximum(0.05, p.VPD)  # kPa

                else:
                    esat_l = vpsat(new_Tleaf)  # vpsat at new Tleaf, kPa
                    Dleaf = (esat_l - (esat_a - np.maximum(0.05, p.VPD)))

                # force stop when atm. conditions yield E < 0. (non-physical)
                if (iter < 1) and (not real_zero):
                    real_zero = None

                # check for convergence
                if ((real_zero is None) or (iter > iter_max) or ((real_zero)
                   and (abs(Tleaf[i] - new_Tleaf) <= threshold_conv) and not
                   np.isclose(gs[i], cst.zero, rtol=cst.zero, atol=cst.zero))):
                    break

                # no convergence, iterate on leaf temperature
                Tleaf[i] = new_Tleaf
                iter += 1

            if (np.isclose(trans[i], cst.zero, rtol=cst.zero, atol=cst.zero) or
               np.isclose(Ci[i], 0., rtol=cst.zero, atol=cst.zero) or
               (Ci[i] < 0.) or np.isclose(Ci[i], p.CO2, rtol=cst.zero,
                                          atol=cst.zero) or
               (Ci[i] > p.CO2) or (real_zero is None) or (not real_zero)):
                trans[i], gs[i], An[i], Ci[i] = (0., ) * 4

        else:
            trans[i], gs[i], An[i], Aj[i], Ac[i], Ci[i] = (0., ) * 6

    # sum contributions from sunlit and shaded leaves
    with np.errstate(invalid='ignore'):  # if nans, do not raise warning
        trans_can = np.nansum(trans) * conv.MILI  # mmol m-2 s-1
        gs_can = np.nanmean(gs)  # mol m-2 s-1
        An_can = np.nansum(An)  # umol m-2 s-1
        Ci_can = np.nanmean(Ci)  # Pa
        rublim_can = rubisco_limit(np.nansum(Aj), np.nansum(Ac))  # lim?
        Tleaf_can = np.nanmean(Tleaf)  # degC
        Rleaf_can = np.nansum(Rleaf)  # umol m-2 s-1

    if not scaleup:  # downscale fluxes to the tree
        trans_can /= np.sum(fscale2can)
        An_can /= np.sum(fscale2can)

    # reset original LAI, PPFD
    p.LAI = LAI
    p.PPFD = PPFD

    if any(np.isnan([trans_can, gs_can, An_can, Ci_can])):
        trans_can, gs_can, Pleaf_can, An_can, Ci_can, Tleaf_can, Rleaf_can = \
            (0., ) * 7

    else:
        Pleaf_can = 0.

        if gs_can > 0.:  # estimate Pleaf
            Pleaf_can = P[np.nanargmin(np.abs(trans_can - E * conv.MILI))]

            if np.isclose(Pleaf_can, p.Ps):
                Pleaf_can = 0.

    return (trans_can, gs_can, Pleaf_can, An_can, Ci_can, rublim_can,
            Tleaf_can, Rleaf_can)
