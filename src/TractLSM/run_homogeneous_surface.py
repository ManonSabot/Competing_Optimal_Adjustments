# -*- coding: utf-8 -*-

"""
Run the coupling models between canopy water and carbon fluxes at each
time step given by the forcing file. Adjust the photosynthetic capacity
parameters every N days (user defined). Account for legacies from
hydraulic damage every N days (user defined).
Soil hydrology is represented by a simple tipping bucket. The land
surface cover is assumed homogeneous.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Run a tractable LSM for a homogeneous surface"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (04.10.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import collections  # ordered dictionaries
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM.SPAC import canopy_intercept  # throughfall
from TractLSM.SPAC import wetness, water_potential, soil_evap  # soil
from TractLSM.SPAC import absorbed_radiation_2_leaves  # canopy rad
from TractLSM.SPAC import f, Weibull_params  # hydraulic legacies
from TractLSM.CH2OCoupler import solve_uso  # USO/Medlyn model
from TractLSM.CH2OCoupler import maximise_profit  # ProfitMax/Sperry
from TractLSM.TraitOptimisation import ini_N_pools, optimal_JV  # N leaf

try:  # support functions
    from run_utils import time_step, write_csv

except (ImportError, ModuleNotFoundError):
    from TractLSM.run_utils import time_step, write_csv


# ======================================================================

def over_time(df, step, Nsteps, dic, photo, resolution, fhydralegs, fleafNopt,
              calc_sw):

    """
    Optimisation wrapper at each time step that updates the soil
    moisture and soil water potential for each of the models before
    running them in turn: (i) the Medlyn/USO model (solve_uso), (ii)
    the Profit maximisation (maximise_profit). None of these are run
    for timesteps when PPFD < 50.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing all input data & params

    step: int
        current time step

    Nsteps: int
        total number of steps. This is necessary to know whether unit
        conversion must be based on half-hourly time steps or longer
        time steps!

    dic: dictionary
        initially empty upon input, this dictionary allows to return the
        outputs in a trackable manner. From a time-step to another, it
        also keeps in store the soil moisture and transpiration relative
        to each model, in order to accurately update the soil water
        bucket.

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    resolution: string
        either 'low' (default), 'med', or 'high' to run the optimising
        solver

    fhydralegs: string or int
        day frequency at which hydraulic legacies are computed, by
        default they are unaccounted for

    fleafNopt: string or int
        day frequency at which the leaf N optimisation module is called,
        by default it is not called

    calc_sw: bool
        if True, computes the soil hydrology, otherwise uses prescribed
        soil moisture / water potential profile

    Returns:
    --------
    Outputs a tuple of variables depending on the input dic structure.
    When PPFD is zero, a tuple of zero values is returned. If the models
    behave in a non-physical manner, zeros are returned too. Overall,
    the following variables are returned at each time step:

    An(model): float
        net photosynthetic assimilation rate [umol m-2 s-1]

    E(model): float
        transpiration rate [mmol m-2 s-1]

    Ci(model): float
        intercellular CO2 concentration [Pa]

    gs(model): float
        stomatal conductance to water vapour [mol m-2 s-1]

    Pleaf(model): float
        leaf water potential [MPa]

    Tleaf(model): float
        leaf temperature [degC]

    Rublim(model): float
        'True' if the C assimilation is rubisco limited, 'False'
        otherwise

    Vcmax25(model): float
        maximum carboxylation rate at 25 degrees [μmol m-2 s-1]

    Jmax25(model): float
        maximum electron transport rate at 25 degrees [μmol m-2 s-1]

    Nc(model): float
        amount of nitrogen in chlorophyll [mol m-2]

    Ne(model): float
        amount of nitrogen in electron transport components [mol m-2]

    Nr(model): float
        amount of nitrogen in Rubisco [mol m-2]

    Ns(model): float
        amount of nitrogen in soluble protein other than Rubisco
        [mol m-2]

    Eci(model): float
        evaporation rate from the canopy interception of rainfall
        [mmol m-2 s-1]

    Es(model): float
        soil evaporative rate [mmol m-2 s-1]

    sw(model): float
        volumetric soil water content [m3 m-3]

    Ps(model): float
        soil water potential [MPa]

    """

    # parameters & met data
    p = time_step(df, step)

    # tuple of return values
    tpl_return = ()

    # How many timesteps in a day? (for year, month, hour-based dataset)
    try:
        if step >= Nsteps - 1:  # last step of time series
            delta = p.hod - df.iloc[step - 1, df.columns.get_loc('hod')]

        else:
            if df.iloc[step + 1, df.columns.get_loc('hod')] < p.hod:
                delta = df.iloc[step + 1, df.columns.get_loc('hod')]

            else:  # during day
                delta = df.iloc[step + 1, df.columns.get_loc('hod')] - p.hod

        try:
            Dsteps = int(24. / delta)

        except Exception:  # there can be errors for the last step
            Dsteps = 48

    except Exception:  # there can be errors for the last step
        Dsteps = 48

    # canopy interception
    if (p.Tair > 0.) and (p.precip > p.can_sat):
        throughfall, Eci = canopy_intercept(p)
        p.precip = throughfall  # precip getting thru to soil

    else:
        Eci = 0.

    # average soil temperature assumed ~ average air temperature
    if ((step + 1) % Dsteps == 0) and (step > 0) and (step < Nsteps - 1):
        Tsoil = (df.iloc[step - (Dsteps - 1): step + 1,
                         df.columns.get_loc('Tair')].sum() / float(Dsteps))

    else:
        Tsoil = p.Tair

    for key in dic.keys():  # loops over the models

        if calc_sw:  # compute soil moisture state
            if step == 0:
                try:
                    dic[key]['sw'] = p.sw0  # antecedent known sw

                except AttributeError:
                    if not np.isclose(p.Ps, p.Psie):  # sw from Ps
                        dic[key]['sw'] = water_potential(p, None)

                    else:
                        dic[key]['sw'] = p.theta_sat  # saturated soil

                for layer in ['sw0', 'sw1', 'sw2', 'sw3', 'sw4', 'sw5']:

                    dic[key][layer] = dic[key]['sw']  # ini layers

                dic[key]['Tsoil'] = Tsoil  # same Tsoil for all keys

                # soil albedo?
                if dic[key]['sw0'] < 0.5 * (p.fc - p.pwp):  # 'dry soil'
                    p.albedo_s = p.albedo_ds

                else:  # 'wet soil'
                    p.albedo_s = p.albedo_ws

                __, __, __, __, __, __, __, dic[key]['Es'] = \
                    wetness(p, Dsteps, dic[key]['sw0'], dic[key]['sw1'],
                            dic[key]['sw2'], dic[key]['sw3'], dic[key]['sw4'],
                            dic[key]['sw5'], 0., 0., Tsoil)

            if ((step + 1) % Dsteps != 0) and (step > 0):
                Tsoil = dic[key]['Tsoil']  # keep same Tsoil thru day

            if step > 0:
                dic[key]['Tsoil'] = Tsoil  # same Tsoil for all keys

                # soil albedo?
                if dic[key]['sw0'] < 0.5 * (p.fc - p.pwp):  # 'dry soil'
                    p.albedo_s = p.albedo_ds

                else:  # 'wet soil'
                    p.albedo_s = p.albedo_ws

                dic[key]['sw'], dic[key]['sw0'], dic[key]['sw1'], \
                    dic[key]['sw2'], dic[key]['sw3'], dic[key]['sw4'], \
                    dic[key]['sw5'], dic[key]['Es'] = wetness(p, Dsteps,
                                                              dic[key]['sw0'],
                                                              dic[key]['sw1'],
                                                              dic[key]['sw2'],
                                                              dic[key]['sw3'],
                                                              dic[key]['sw4'],
                                                              dic[key]['sw5'],
                                                              dic[key]['Es'],
                                                              dic[key]['E'],
                                                              Tsoil)

            # soil water pot. corresponding to the key's soil moisture
            p.Ps = water_potential(p, dic[key]['sw'])
            dic[key]['Ps'] = p.Ps

        else:  # the soil moisture profile is prescribed
            dic[key]['sw0'] = p.sw0
            dic[key]['sw'] = p.sw
            dic[key]['Ps'] = p.Ps
            dic[key]['Es'] = soil_evap(p, dic[key]['sw0'])  # mmol m-2 s-1

        # optimise leaf N pools, JV and Vcmax25 every N freq days
        if fleafNopt is not None:

            # conditions for opt JV coord at peak daily PPFD
            if (step > 0) and (p.hod > 23.):
                try:
                    idx = np.argmax(dic[key]['PPFD_day'])

                    for key2 in ['PPFD', 'sc', 'Tleaf', 'Ci', 'Rleaf']:

                        dic[key]['%s_ante' % (key2)] += \
                            dic[key]['%s_day' % (key2)][idx]

                        # reset the storage dictionaries
                        dic[key]['%s_day' % (key2)] = []

                    dic[key]['count_ante'] += 1

                except ValueError:
                    pass

            if (p.doy % fleafNopt == 0.) and (p.hod < 1.):

                # original PPFD
                PPFD = p.PPFD

                if dic[key]['PPFD_ante'] > 0:
                    p.Vmax25 = dic[key]['Vmax25']
                    p.JV = dic[key]['JV']
                    p.Rlref = dic[key]['Rlref']
                    p.Ntot = (dic[key]['Nc'] + dic[key]['Ne'] +
                              dic[key]['Nr'] + dic[key]['Ns'])

                    # antecedent conditions over past N days
                    p.PPFD = dic[key]['PPFD_ante'] / dic[key]['count_ante']
                    p.scale2can = dic[key]['sc_ante'] / dic[key]['count_ante']
                    Tleaf = dic[key]['Tleaf_ante'] / dic[key]['count_ante']
                    Ci = dic[key]['Ci_ante'] / dic[key]['count_ante']
                    Rleaf = dic[key]['Rleaf_ante'] / dic[key]['count_ante']

                    # new photosynthetic optimums and N pools
                    dic[key]['Vmax25'], dic[key]['JV'], dic[key]['Rlref'], \
                        dic[key]['Nc'], dic[key]['Ne'], dic[key]['Nr'], \
                        dic[key]['Ns'] = optimal_JV(p, dic[key]['Nc'],
                                                    dic[key]['Ne'], Ci, Tleaf,
                                                    Rleaf, photo=photo)

                # reset original forcing PPFD
                p.PPFD = PPFD

                # reset the storage dictionaries
                dic[key]['count_ante'] = 0.
                dic[key]['PPFD_ante'] = 0.
                dic[key]['sc_ante'] = 0.
                dic[key]['Tleaf_ante'] = 0.
                dic[key]['Ci_ante'] = 0.
                dic[key]['Rleaf_ante'] = 0.

        # recompute hydraulic legacies every N freq days
        if fhydralegs is not None:
            if (p.doy % fhydralegs == 0.) and (p.hod < 1.):
                b, c = Weibull_params(p)  # MPa, unitless
                dic[key]['legembo'] = f(dic[key]['Psimin'], b, c)

                # reset the storage dictionary
                dic[key]['Psimin'] = 0.

    if (p.PPFD < 50.) or (p.VPD <= 0.05):  # no photo

        for key in dic.keys():

            dic[key]['E'], dic[key]['gs'], dic[key]['Pleaf'], dic[key]['An'], \
                dic[key]['Ci'], dic[key]['Rublim'], dic[key]['Tleaf'], \
                dic[key]['Rleaf'] = (0., ) * 8

    else:  # day time

        for key in dic.keys():  # call the model(s)

            # use right Ps
            p.Ps = dic[key]['Ps']

            # soil albedo?
            if dic[key]['sw0'] < 0.5 * (p.fc - p.pwp):
                p.albedo_s = p.albedo_ds

            else:
                p.albedo_s = p.albedo_ws

            if fleafNopt is not None:  # model specific changes
                p.Vmax25 = dic[key]['Vmax25']
                p.JV = dic[key]['JV']
                p.Rlref = dic[key]['Rlref']

            if fhydralegs is not None:
                p.legembo = dic[key]['legembo']

            try:
                if key == 'uso':  # USO/Medlyn model
                    dic[key]['E'], dic[key]['gs'], dic[key]['Pleaf'], \
                        dic[key]['An'], dic[key]['Ci'], dic[key]['Rublim'], \
                        dic[key]['Tleaf'], dic[key]['Rleaf'] = \
                        solve_uso(p, photo=photo)

                if key == 'pmax':  # ProfitMax/Sperry model
                    isun = df.columns.get_loc('fvc_sun')  # ini embolism
                    isha = df.columns.get_loc('fvc_sha')
                    ik = df.columns.get_loc('ksc')
                    p.fvc_sun = df.iloc[np.maximum(0, step-1), isun]
                    p.fvc_sha = df.iloc[np.maximum(0, step-1), isha]

                    fvc, ksc, dic[key]['E'], dic[key]['gs'], \
                        dic[key]['Pleaf'], dic[key]['An'], dic[key]['Ci'], \
                        dic[key]['Rublim'], dic[key]['Tleaf'], \
                        dic[key]['Rleaf'] = maximise_profit(p, photo=photo,
                                                            res=resolution)

                    # keep track of the embolism, ksc
                    df.iloc[step:np.minimum(step+Dsteps, Nsteps), isun] = \
                        fvc[0]
                    df.iloc[step:np.minimum(step+Dsteps, Nsteps), isha] = \
                        fvc[1]
                    df.iloc[step:np.minimum(step+Dsteps, Nsteps), ik] = ksc

                # JV coord: store day conditions
                if (fleafNopt is not None) and (dic[key]['Ci'] > 0.):

                    # canopy scaling / accounting for light environment
                    __, __, __, fscale2can, __ = absorbed_radiation_2_leaves(p)

                    # store relevant vars and reset every new day
                    dic[key]['PPFD_day'] += [p.PPFD]  # top of can
                    dic[key]['sc_day'] += [np.mean(fscale2can)]
                    dic[key]['Tleaf_day'] += [dic[key]['Tleaf']]
                    dic[key]['Ci_day'] += [dic[key]['Ci']]
                    dic[key]['Rleaf_day'] += [dic[key]['Rleaf']]

                # hydraulic legacies
                if fhydralegs is not None:
                    dic[key]['Psimin'] = np.minimum(dic[key]['Psimin'],
                                                    dic[key]['Pleaf'])

            except (TypeError, IndexError, ValueError):  # no solve
                dic[key]['E'], dic[key]['gs'], dic[key]['Pleaf'], \
                    dic[key]['An'], dic[key]['Ci'], dic[key]['Rublim'], \
                    dic[key]['Tleaf'], dic[key]['Rleaf'] = (0., ) * 8

    for key in dic.keys():  # model outputs

        tpl_return += (dic[key]['An'], dic[key]['E'], dic[key]['Ci'],
                       dic[key]['gs'], dic[key]['Pleaf'], dic[key]['Tleaf'],
                       dic[key]['Rublim'], dic[key]['Vmax25'],
                       dic[key]['Vmax25'] * dic[key]['JV'], dic[key]['Nc'],
                       dic[key]['Ne'], dic[key]['Nr'], dic[key]['Ns'], Eci,
                       dic[key]['Es'], dic[key]['sw'], dic[key]['Ps'], )

    return tpl_return


def run(fname, df, Nsteps, models=['USO', 'ProfitMax'], soilwater=None,
        photo='Farquhar', resolution=None, fhydralegs=None, fleafNopt=None):

    """
    Runs the profit maximisation algorithm within a simplified LSM,
    alongsite the USO model which follows traditional photosynthesis
    and transpiration coupling schemes.

    Arguments:
    ----------
    fname: string
        output filename

    df: pandas dataframe
        dataframe containing all input data & params

    Nsteps: int
        total number of time steps over which the models will be run

    models: list of strings
        names of the models to call

    soilwater: string
        either the dynamic soil hydrology scheme is run, or a
        'prescribed' soil profile is used

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    resolution: string
        either 'low' (default), 'med', or 'high' to run the optimising
        solver

    fhydralegs: string or int
        day frequency at which hydraulic legacies are computed, by
        default they are unaccounted for

    fleafNopt: string or int
        day frequency at which the leaf N optimisation module is called,
        by default it is not called

    Returns:
    --------
    df2: pandas dataframe
        dataframe of the outputs:
            An(model), E(model), Ci(model), gs(model), Pleaf(model),
            Tleaf(model), Rublim(model), ...

    """

    if resolution is None:  # hydraulic stream resolution
        resolution = 'low'

    # initialise the N storage pools
    Ntot, Nc, Ne, Nr, Ns = ini_N_pools(df.iloc[0])
    df['Ntot'] = Ntot

    # initial assumption, no embo
    df['ksc'] = df['kmax'].iloc[0]
    df['fvc_sun'] = 1.
    df['fvc_sha'] = 1.

    if fhydralegs is not None:  # hydraulic legacies
        df['recovembo'] = df['r_k']
        df['legembo'] = df['r_k']  # starting assumption

    else:  # no hydraulic legacies
        df['recovembo'] = 1.
        df['legembo'] = 1.

    # soil albedo will change depending on soil wetness
    df['albedo_s'] = df['albedo_ws'].iloc[0]

    # attributes that won't change in time
    df['soil_volume'] = df['Zbottom'].iloc[0] * df['ground_area'].iloc[0]
    df['soil_top_volume'] = df['Ztop'].iloc[0] * df['ground_area'].iloc[0]

    if soilwater is None:

        try:  # initialise the soil moisture
            df['sw0'] = df['sw'].loc[df['sw'].first_valid_index()]

        except KeyError:
            pass

        df.drop(['sw', 'Ps'], axis=1, inplace=True)
        calc_sw = True

    else:
        if len(df) - df['Ps'].count() != 0:  # Ps is missing
            df['Ps'] = water_potential(df.iloc[0], df['sw'])

        calc_sw = False

    # non time-sensitive: last valid value propagated until next valid
    df.fillna(method='ffill', inplace=True)

    # two empty dics, to structure the run setup and retrieve the output
    dic = {}  # appropriately run the models
    output_dic = collections.OrderedDict()  # unpack the output in order

    # sub-dic structures
    subdic = {'sw': None, 'sw0': None, 'sw1': None, 'sw2': None, 'sw3': None,
              'sw4': None, 'sw5': None, 'Ps': None, 'Tsoil': None, 'Es': None,
              'E': None, 'gs': None, 'Pleaf': None, 'Psimin': 0.,
              'legembo': df['legembo'].iloc[0], 'An': None, 'Ci': None,
              'Rublim': None, 'Tleaf': None, 'Rlref': df['Rlref'].iloc[0],
              'Vmax25': df['Vmax25'].iloc[0], 'JV': df['JV'].iloc[0],
              'Nc': Nc, 'Ne': Ne, 'Nr': Nr, 'Ns': Ns, 'PPFD_ante': 0.,
              'sc_ante': 0., 'Tleaf_ante': 0., 'Ci_ante': 0., 'Rleaf_ante': 0.,
              'count_ante': 0., 'PPFD_day': [], 'sc_day': [], 'Tleaf_day': [],
              'Ci_day': [], 'Rleaf_day': []}

    # for the output dic, the order of things matters!
    subdic2 = collections.OrderedDict([('An', None), ('E', None),
                                       ('Ci', None), ('gs', None),
                                       ('Pleaf', None), ('Tleaf', None),
                                       ('Rublim', None), ('Vmax25', None),
                                       ('Jmax25', None), ('Nc', None),
                                       ('Ne', None), ('Nr', None),
                                       ('Ns', None), ('Eci', None),
                                       ('Es', None), ('sw', None),
                                       ('Ps', None)])  # output

    # create dictionaries of Nones with the right structures
    if ('USO' in models) or ('USO'.lower() in models):
        dic['uso'] = subdic.copy()
        output_dic['uso'] = subdic2.copy()

    if ('ProfitMax' in models) or ('ProfitMax'.lower() in models):
        dic['pmax'] = subdic.copy()
        output_dic['pmax'] = subdic2.copy()

    # run the model(s) over the range of timesteps / the timeseries
    tpl_out = list(zip(*[over_time(df, step, Nsteps, dic, photo, resolution,
                                   fhydralegs, fleafNopt, calc_sw)
                         for step in range(Nsteps)]))

    # unpack the output tuple 17 by 17
    track = 0  # initialize

    for key in output_dic.keys():

        output_dic[key]['An'] = tpl_out[track]
        output_dic[key]['E'] = tpl_out[track + 1]
        output_dic[key]['Ci'] = tpl_out[track + 2]
        output_dic[key]['gs'] = tpl_out[track + 3]
        output_dic[key]['Pleaf'] = tpl_out[track + 4]
        output_dic[key]['Tleaf'] = tpl_out[track + 5]
        output_dic[key]['Rublim'] = tpl_out[track + 6]
        output_dic[key]['Vmax25'] = tpl_out[track + 7]
        output_dic[key]['Jmax25'] = tpl_out[track + 8]
        output_dic[key]['Nc'] = tpl_out[track + 9]
        output_dic[key]['Ne'] = tpl_out[track + 10]
        output_dic[key]['Nr'] = tpl_out[track + 11]
        output_dic[key]['Ns'] = tpl_out[track + 12]
        output_dic[key]['Eci'] = tpl_out[track + 13]
        output_dic[key]['Es'] = tpl_out[track + 14]
        output_dic[key]['sw'] = tpl_out[track + 15]
        output_dic[key]['Ps'] = tpl_out[track + 16]
        track += 17

    # save the outputs to a csv file and get the corresponding dataframe
    df2 = write_csv(fname, df, output_dic)

    return df2
