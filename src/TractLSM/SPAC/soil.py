# -*- coding: utf-8 -*-

"""
Simple tipping bucket soil water balance model, a proxy for soil
hydrology, with soil layer depths adapted for EucFACE.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

Reference:
-----------
* Clapp, R. B., & Hornberger, G. M. (1978). Empirical equations for some
  soil hydraulic properties. Water resources research, 14(4), 601-604.
* Cosby, B. J., Hornberger, G. M., Clapp, R. B., & Ginn, T. (1984). A
  statistical exploration of the relationships of soil moisture
  characteristics to the physical properties of soils. Water resources
  research, 20(6), 682-690.
* Duursma et al. (2008). Predicting the decline in daily maximum
  transpiration rate of two pine stands during drought based on constant
  minimum leaf water potential and plant hydraulic conductance. Tree
  Physiology, 28(2), 265-276.
* Monteith, J. L., & Unsworth, M. H. (1990). Principles of environmental
  physics. Arnold. SE, London, UK.
* Ritchie, J. T. (1972). Model for predicting evaporation from a row
  crop with incomplete cover. Water resources research, 8(5), 1204-1213.

"""

__title__ = "Tipping bucket soil water module"
__author__ = "Manon E. B. Sabot"
__version__ = "3.0 (05.10.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC import slope_vpsat, LH_water_vapour, psychometric
from TractLSM.SPAC import net_radiation  # canopy radiation


# ======================================================================

def root_distri(depths, beta):

    """
    Estimates the root distribution at depth using the equation of
    Gale & Grindal (1987)

    Arguments:
    ----------
    depths: array
        lower depth [cm] of each of six layers that match 6 averaged
        adjacent layers in the obs data

    beta: float
        root distribution parameter from Gale & Grindal (1987),
        typically a value closer to 0.9 means the roots are shallow
        whereas a value closer to 1 means the roots are deep

    Returns:
    --------
    froot: array
        fraction of roots in each layer that matches that depths array

    """

    froot = np.minimum(1., 1. - beta ** np.array(depths))
    froot[-1] = 1.  # fraction should equate to 1 in deepest layer

    for i in range(len(froot) - 1, 0, -1):

        froot[i] = froot[i] - froot[i - 1]

    return froot


def fwsoil(p, sw):

    """
    Calculates the empirical soil moisture stress factor that determines
    the stomatal conductance and soil's responses to water limitation.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    sw: float
        mean volumetric soil moisture content [m3 m-3]

    Returns:
    --------
    The empirical stomatal conductance and soil's responses to soil
    moisture stress.

    """

    fw = (sw - p.pwp) / (p.fc - p.pwp)

    return np.maximum(cst.zero, np.minimum(1., fw))


def evap_bare_soil(p):

    """
    Evaporation at the potential/equilibrium rate, where aerodynamic
    conductance is zero (i.e. winds are calm).

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    Returns:
    --------
    The evaporation of bare soil [mol m-2 s-1], using Penman eq of
    Monteith and Unsworth, 1990.

    """

    # latent heat of water vapor
    Lambda = LH_water_vapour(p)  # J mol-1

    # psychrometric constant
    gamm = psychometric(p)  # kPa deg-1

    # slope of saturation vapour pressure of water vs Tair
    slp = slope_vpsat(p)  # kPa deg-1

    # net radiation of a surface
    Rnet = net_radiation(p, surface='soil')  # W m-2

    return np.maximum(0., Rnet * slp / ((slp + gamm) * Lambda))


def soil_evap(p, sw):

    """
    Use Penman eq to calculate top soil evaporation flux. Soil
    evaporation is dependent upon soil wetness and plant cover. The net
    radiation term is scaled for the canopy cover and the impact of soil
    wetness is accounted for. As the soil dries the evaporation
    component reduces significantly.

    Key assumptions from Ritchie, 1972...
    When plant provides shade for the soil surface, evaporation will not
    be the same as bare soil evaporation. Wind speed, net radiation and
    VPD will all be lowered in proportion to the canopy density.
    Following Ritchie role of wind, VPD are assumed to be negligible and
    are therefore ignored. These assumptions are based on work with
    crops, where a fit is formed between the LAI of 5 crops types and
    the fraction of observed net radiation at the surface. Whilst the
    LAI does cover a large range, nominal 0–6, there are only 12
    measurements and only three from LAI > 3.and whether this holds for
    tree shading where the height from the soil to the base of the crown
    is larger is questionable.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    sw: float
        volumetric soil water content available for evaporation [m3 m-3]

    Returns:
    --------
    The evaporation of covered soil [mmol m-2 s-1], using Ritchie (1972)
    empirical LAI fit.

    """

    # account for litter layer and LAI cover
    evap = (np.maximum(0., (1. - p.r_soil) * evap_bare_soil(p)) *
            np.exp(-0.398 * p.LAI))

    # account for soil wetness state
    evap *= fwsoil(p, sw) * conv.MILI  # mmol m-2 s-1

    if (np.isclose(evap, 0.)) or (evap < 0.):
        evap = 0.

    return evap


def drainage(p, daily_steps, sw, volume, depth):

    """
    Updates water content in the top and lower layer of the bucket by
    applying a drainage rate, with the low layer being split in 5
    different sublayers.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    daily_steps: int
        number of timesteps in a day

    sw: array
        volume of soil water in each layer [m3]

    volume: array
        volume of each layer [m3]

    depth: array
        depth of each layer [m]

    Returns:
    --------
    sw: array
        volume of soil water in each layer [m3]

    """

    # volumetric soil water contents
    sw /= volume  # m3 m-3

    # difference in soil water between consecutive layers
    delta = np.ediff1d(sw)  # m3 m-3

    # drainage speed, last layer is limited by saturation
    speed = (p.hyds * (np.minimum(sw[:-1], sw[1:]) / p.theta_sat)
             ** (2. * p.bch + 2.))  # m s-1
    speed = np.append(speed, p.hyds * (np.minimum(sw[-1], p.theta_sat) /
                                       p.theta_sat) ** (2. * p.bch + 2.))

    # limit speed for stability, m s-1
    time_spent = (conv.SEC_2_DAY / daily_steps)
    speed[:-1] = np.minimum(speed[:-1], 0.5 * depth[:-1] / time_spent)
    speed[-1] = np.minimum(0.5 * speed[-1], 0.5 * depth[-1] / time_spent)

    # calculate the drainage flux for the upper layers
    ratio = delta[1:] / (delta[:-1] + cst.zero * np.sign(delta[:-1]))
    ratio = np.append(np.nan_to_num(ratio), 0.)  # m3 m-3
    phi = np.maximum(0., np.minimum(1., 2. * ratio))
    phi = np.maximum(phi, np.minimum(2., ratio))
    flux = speed[:-1] * (sw[:-1] + phi * (np.minimum(sw[:-1], sw[1:]) -
                         sw[:-1]))  # m s-1

    # drainage flux for the lowest layer
    flux = np.append(flux, np.maximum(0., speed[-1] * sw[-1]))  # m s-1

    # constrain fluxes by the saturation point
    flux[:-1] = np.minimum(flux[:-1], flux[1:] + (p.theta_sat - sw[1:]) *
                           depth[1:] / time_spent)

    # update water balance, constrain by saturation point
    sw[1:] = np.minimum(p.theta_sat, sw[1:] + time_spent * (flux[:-1] -
                        flux[1:]) / depth[1:])

    sw[0] -= time_spent * flux[0] / depth[0]

    # back to volumes of water
    sw *= volume  # m3

    return np.minimum(p.theta_sat * volume, sw)


def sublayer_runoff(sw, max_sw, in_sw):

    """
    Updates water content in the bucket by tipping incoming rainfall
    throughout the layers, which is a coarse estimate of below ground
    runoff.

    Arguments:
    ----------
    sw: array
        volume of soil water in each layer [m3]

    max_sw: array
        maximum volume of water in each layer (i.e. volume at
        saturation) [m3]

    in_sw: float
        incoming volume of water entering the layers [m3]

    Returns:
    --------
    sw: array
        volume of soil water in each layer [m3]

    """

    # infiltrate the sub layers with rain water
    for i in range(len(sw)):

        if in_sw > (max_sw[i] - sw[i]):  # the layer overfills!
            in_sw -= max_sw[i] - sw[i]  # less water enters next layer
            sw[i] = max_sw[i]  # saturated layer, m3

            if in_sw < 0.:  # no more infiltration

                return sw

        else:  # not all layers are saturated
            sw[i] += in_sw  # include remainder entering water, m3

    return sw


def wetness(p, daily_steps, sw0, sw1, sw2, sw3, sw4, sw5, Es, E, Tsoil):

    """
    Updates the simple bucket soil water balance by calculating the soil
    volumetric water content [m3 m-3].

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    daily_steps: int
        number of timesteps in a day

    sw0: float
        top soil layer volumetric soil moisture content from the
        previous timestep [m3 m-3]

    sw1: float
        #1 soil layer volumetric soil moisture content from the previous
        timestep [m3 m-3]

    sw2: float
        #2 soil layer volumetric soil moisture content from the previous
        timestep [m3 m-3]

    sw3: float
        #3 soil layer volumetric soil moisture content from the previous
        timestep [m3 m-3]

    sw4: float
        #4 soil layer volumetric soil moisture content from the previous
        timestep [m3 m-3]

    sw5: float
        #5 soil layer volumetric soil moisture content from the previous
        timestep [m3 m-3]

    Es: float
        soil evaporation rate from the previous timestep [mmol m-2 s-1]

    E: float
        canopy transpiration rate from the previous timestep
        [mmol m-2 s-1]

    Tsoil: float
        mean soil temperature

    Returns:
    --------
    sw: float
        volumetric soil water content [m3 m-3]

    sw0: float
        top soil layer volumetric soil moisture content [m3 m-3]

    sw1: float
        #1 soil layer volumetric soil moisture content [m3 m-3]

    sw2: float
        #2 soil layer volumetric soil moisture content [m3 m-3]

    sw3: float
        #3 soil layer volumetric soil moisture content [m3 m-3]

    sw4: float
        #4 soil layer volumetric soil moisture content [m3 m-3]

    sw5: float
        #5 soil layer volumetric soil moisture content [m3 m-3]

    sevap: float
        next step's soil evaporative rate [mmol m-2 s-1]

    """

    # unit conversions (to mm 1/2h-1)
    mmolsqrtm_2_mm = conv.FROM_MILI * cst.MH2O / cst.rho  # m3 mol-1
    Es *= mmolsqrtm_2_mm * conv.SEC_2_HLFHR
    E *= mmolsqrtm_2_mm * conv.SEC_2_HLFHR
    precip = p.precip / conv.HLFHR_2_DAY

    # deal with non-half-hourly steps!
    if daily_steps != 48:
        Es *= 48. / daily_steps
        E *= 48. / daily_steps
        precip *= 48. / daily_steps

    # root distri matching the site layer depths
    depths = [5., 15., 27.5, 62.5, 112.5, 175.]  # cm
    froot = root_distri(depths, p.root_beta)

    # tickness of each layer
    zsl = np.insert(np.diff(depths) / 100., 0, depths[0] / 100.)  # m
    zsl /= np.sum(zsl)

    # volumes of soil water in the layers
    vols = p.soil_volume * zsl  # m3
    depths = p.Zbottom * zsl  # m

    # soil moisture across layers
    sw = np.asarray([sw0, sw1, sw2, sw3, sw4, sw5])  # m3 m-3
    sw *= vols  # m3

    # litter layer = runoff + soil repellency, assumed linear
    precip = p.r_soil * precip

    if p.Tair < cst.zero:  # account for snow / frozen soil
        precip = 0.

    # evaporate the previous timestep
    itop = len(np.cumsum(depths)[np.cumsum(depths) <= p.Ztop])
    sw[0] -= Es * conv.FROM_MILI * p.ground_area  # m3

    for i in range(itop):

        # ensure we're not removing more than 5 cm worth
        if i == itop - 1:
            if sw[i] < 0.:
                dz = (np.cumsum(depths)[np.cumsum(depths) <= p.Ztop][-1] -
                      p.Ztop)
                sw[i] = np.maximum(sw[i], dz * p.ground_area)

        if sw[i] < 0.:
            sw[i + 1] += sw[i]

    sw[:itop] = np.maximum(0., np.minimum(p.theta_sat * vols[:itop],
                                          sw[:itop]))

    # preferably transpire from the root distri
    sw -= E * conv.FROM_MILI * p.ground_area * froot  # m3

    for i in range(len(sw)):  # loop over the layers

        if sw[i] < 0.:  # layer is empty, rm from nearest wettest
            if i == 0:
                sw[i + 1] += sw[i]  # m3

            elif i == (len(sw) - 1):
                j = i - 1

                while ((j > 0) and (sw[j] < 0.)):

                    j -= 1

                sw[j] += sw[i]

            else:
                if sw[i + 1] > sw[i - 1]:
                    sw[i + 1] += sw[i]  # m3

                else:
                    sw[i - 1] += sw[i]  # m3

    # restrain to the valid range
    sw = np.maximum(0., np.minimum(p.theta_sat * vols, sw))

    if Tsoil > cst.zero:  # account for snow / frozen soil
        if precip > 0.:  # deal with infiltrating rainfall
            sw = sublayer_runoff(sw, p.theta_sat * vols,
                                 precip * conv.FROM_MILI * p.ground_area)
            sw = np.maximum(0., np.minimum(p.theta_sat * vols, sw))

        # apply drainage rate
        sw = drainage(p, daily_steps, sw, vols, depths)  # m3
        sw = np.maximum(0., np.minimum(p.theta_sat * vols, sw))

        # soil evap to remove @ next water balance, mmol m-2 s-1
        if np.sum(sw[:itop]) > cst.zero:
            sevap = soil_evap(p, np.sum(sw[:itop]) / p.soil_top_volume)

        else:
            sevap = 0.

    else:
        sevap = 0.

    # overall soil moisture content
    sw_all = np.sum(sw) / p.soil_volume  # m3 m-3

    # volumetric soil moisture contents
    sw /= vols  # m3 m-3

    return sw_all, sw[0], sw[1], sw[2], sw[3], sw[4], sw[5], sevap


def water_potential(p, sw):

    """
    Calculates the soil water potential [MPa]. The parameters bch and
    Psie are estimated using the Cosby et al. (1984) regression
    relationships from the soil sand/silt/clay fractions.

    Arguments:
    ----------
    p: pandas series
        time step's met data & params

    sw: float
        volumetric soil water content [m3 m-3]

    Returns:
    --------
    The soil water potential [MPa], Ps, using Clapp and Hornberger
    eq (1978)

    """

    if (sw is not None) and (sw >= cst.zero):
        return p.Psie * (sw / p.theta_sat) ** (-p.bch)

    elif np.isclose(abs(p.Ps), 0., rtol=cst.zero, atol=cst.zero):
        return p.Psie

    else:
        return p.theta_sat * (p.Ps / p.Psie) ** (-1. / p.bch)

    return
