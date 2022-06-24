#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Estimates imporant evapotranspiration parameters by calibrating the
model to observations at EucFACE.

Three observational datasets stored in input/obs/ are used:
    - 'EucFACE_interception_2012_2014.csv'
    - 'EucFACE_sapflow_2012_2014.csv'
    - 'EucFACE_underET_2012_2014.csv'
see the ReadMe in input/ for information on these datasets.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along this script.

"""

__title__ = "calibrate model parameters"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (04.10.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# import general modules
import argparse  # read user input
import os  # check for files, paths
import sys  # make the TractLSM modules loadable
import pandas as pd  # read/write dataframes, csv files
import numpy as np  # array manipulations, math operators
import lmfit  # fit parameters

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM import InForcings  # met data & params
from TractLSM import conv  # unit converter
from TractLSM.Utils import get_main_dir  # locate data
from TractLSM.Utils import read_csv  # read in data files
from TractLSM.SPAC import canopy_intercept  # fit interception params
from TractLSM.CH2OCoupler import solve_uso  # fit g1, etc.
from TractLSM.CH2OCoupler import maximise_profit  # fit kmax
from TractLSM.SPAC.soil import soil_evap  # fit Es params


# ======================================================================

def main(process, ring, model='ProfitMax', project=None):

    """
    Main: calibrates evapotranspiration parameters at EucFACE and writes
          them out to text files in input/params.

    Arguments:
    ----------
    process: string
        process for which parameters should be calibrated: canopy
        interception ('intercept'), canopy transpiration ('trans'), or
        soil evaporation ('evap')

    ring: string
        EucFACE's "ring" plot to consider

    model: string
        C-H2O flux solver to use, possible choices are: 'Medlyn' or
        'ProfitMax'

    project: string
        project repository where the input files (drivers) to use are
        (with path)

    Returns:
    --------
    files of the form 'Ring.txt' in input/params/

    """

    # path to files for parameter calculation
    dir = os.path.join(os.path.dirname(get_main_dir()), 'input')
    write2 = os.path.join(dir, 'params')

    if not os.path.isdir(write2):
        os.makedirs(write2)

    # path to the drivers
    if project is not None:
        idrive = os.path.join(os.path.join(os.path.join(dir, 'projects'),
                              project), 'EucFACE%s_model_drivers.csv' % (ring))

    else:
        idrive = os.path.join(dir, 'EucFACE%s_model_drivers.csv' % (ring))

    # read in the drivers file
    try:
        idf, __ = read_csv(idrive)

    except FileNotFoundError:
        if not os.path.isdir(os.path.dirname(idrive)):
            os.makedirs(os.path.dirname(idrive))

        InForcings().run(idrive, os.path.join(dir, 'site_params.csv'))
        idf, __ = read_csv(idrive)

    if process == 'intercept':  # paths to the obs
        tobs = os.path.join(os.path.join(dir, 'obs'),
                            'EucFACE_interception_2012_2014.csv')

    elif process == 'trans':
        tobs = os.path.join(os.path.join(dir, 'obs'),
                            'EucFACE_sapflow_2012_2014.csv')

    elif process == 'evap':
        tobs = os.path.join(os.path.join(dir, 'obs'),
                            'EucFACE_underET_2012_2014.csv')

    else:
        raise NameError('Process not set up for calibration, choose another.')

    # read in the obs
    tdf = (pd.read_csv(tobs).dropna(axis=0, how='all')
             .dropna(axis=1, how='all').squeeze())

    # restrict to the ring considered
    tdf = tdf[tdf['Ring'] == ring]

    # call the calibration routines
    if process == 'intercept':
        out = estimate_interception_params(idf, tdf)

    elif process == 'trans':
        if (model != 'ProfitMax') and (model != 'Medlyn'):
            raise NameError('Please choose a valid model to calibrate.')

        out = estimate_trans_params(model, idf, tdf)

    else:
        out = estimate_evap_params(idf, tdf)

    # write out the calibrated params
    if not os.path.isfile(os.path.join(write2, '%s.txt' % (ring))):
        txt = open(os.path.join(write2, '%s.txt' % (ring)), 'w+')

    else:  # append to existing file
        txt = open(os.path.join(write2, '%s.txt' % (ring)), 'a+')

    txt.write('\n')
    txt.write(lmfit.fit_report(out))
    txt.write('\n')
    txt.close()  # close text file

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def fit_intercept_params(params, input, target):

    """
    Minimizing function between the canopy interception model and the
    observations of canopy interception.

    Arguments:
    ----------
    params: object
        parameters to calibrate, including their initialisation values

    input: recarray object or pandas series
        time step's met data & params

    target: recarray object or pandas series
        target observations to calibrate against

    Returns:
    --------
    Residuals between the simulations and the observations

    """

    for pname in params.items():  # update the model's specific param.s

        input[pname[0]] = params[pname[0]].value

    # initialise the canopy evap.
    input['Eci'] = np.nan

    for step in range(len(input)):

        p = input.iloc[step].copy()

        if p.precip >= p.can_sat:
            thru, Eci = canopy_intercept(p)

            if p.precip >= thru:
                input.iloc[step, input.columns.get_loc('Eci')] = Eci

    # day interception, mm
    sim = input.groupby('date').sum()['Eci'] * conv.mmolH2Opm2ps_2_mmphlfhr
    sim.fillna(0., inplace=True)
    sim = sim.resample('W').sum()  # weekly totals, mm

    return sim - target


def estimate_interception_params(idf, tdf):

    """
    Wrapper around the fit_intercept_params function used to calibrate
    the model to observations of daily canopy interception.

    Arguments:
    ----------
    idf: pandas dataframe
        time step's met data & params

    tdf: pandas dataframe
        observations to calibrate against

    Returns:
    --------
    out: lmfit object
        object containing the optimized parameters and several
        goodness-of-fit statistics

    """

    # non time-sensitive: last valid value propagated until next valid
    idf.fillna(method='ffill', inplace=True)

    # match dates
    idf['date'] = idf['year'] * 1000. + idf['doy']
    idf['date'] = pd.to_datetime(idf['date'], format='%Y%j')
    tdf['date'] = pd.to_datetime(tdf['Date'], dayfirst=True)
    df = idf.merge(tdf, on='date', suffixes=(None, '_y'))

    # rm missing data (-9999.)
    df = df[df['Ei'] > 0.]

    # limit the data to daytime
    df = df[df['PPFD'] > 0.]

    # upper bound minimum necessary precip for intercept?
    anc = df.groupby('date').mean()
    min = (anc.iloc[np.where(anc['Ei'] > 0.)[0]
                    [anc['Ei'][np.where(anc['Ei'] > 0.)[0]].argmin()]])
    can_sat = min.intensity * conv.HR_2_DAY  # mm d-1

    # actual input and target files for use
    input = df.copy()
    target = df.groupby('date').mean()['Ei']  # day, mm

    # declare empty parameter class
    params = lmfit.Parameters()

    # parameters to fit
    params.add('can_sat', 0., min=0., max=can_sat)
    params.add('kcanint', 0.7, min=0.5, max=0.99)

    out = lmfit.minimize(fit_intercept_params, params, args=(input, target),
                         method='differential_evolution', nan_policy='omit')

    if (np.isclose(out.params.valuesdict()['kcanint'], 0.5) or
       (out.params.valuesdict()['kcanint'] >= 0.95)):
        params = lmfit.Parameters()  # empty parameter class
        params.add('can_sat', 0., min=0., max=can_sat)
        params.add('kcanint', 0.7, vary=False)

        out = lmfit.minimize(fit_intercept_params, params,
                             args=(input, target),
                             method='differential_evolution',
                             nan_policy='omit')

    return out


def fit_trans_params(params, model, input, target):

    """
    Minimizing function between the canopy transpiration model and the
    observations of canopy transpiration.

    Arguments:
    ----------
    params: object
        parameters to calibrate, including their initialisation values

    model: string
        C-H2O flux solver to use, possible choices are: 'Medlyn' or
        'ProfitMax'

    input: recarray object or pandas series
        time step's met data & params

    target: recarray object or pandas series
        target observations to calibrate against

    Returns:
    --------
    Residuals between the simulations and the observations

    """

    for pname in params.items():  # update the model's specific param.s

        try:
            input[pname[0]] = params[pname[0]].value

        except ValueError:  # do not account for ancillary params
            pass

    # initialise E
    input['E'] = np.nan

    for step in range(len(input)):

        p = input.iloc[step].copy()

        if model == 'Medlyn':  # call the Medlyn model
            input.iloc[step, input.columns.get_loc('E')], __, __, __, __, __, \
                __, __ = solve_uso(p, photo='Farquhar', scaleup=False)

        elif model == 'ProfitMax':  # call the ProfitMax model
            __, __, input.iloc[step, input.columns.get_loc('E')], __, __, __, \
                __, __, __, __ = maximise_profit(p, photo='Farquhar',
                                                 scaleup=False)

    # day trans, mm
    sim = (input['E'].groupby(input['date']).sum() *
           conv.mmolH2Opm2ps_2_mmphlfhr)
    sim.fillna(0., inplace=True)
    sim = sim.resample('W').sum()  # weekly totals, mm

    return sim - target


def estimate_trans_params(model, idf, tdf):

    """
    Wrapper around the fit_trans_params function used to calibrate
    the model to observations of weekly canopy transpiration.

    Arguments:
    ----------
    model: string
        C-H2O flux solver to use, possible choices are: 'Medlyn' or
        'ProfitMax'

    idf: pandas dataframe
        time step's met data & params

    tdf: pandas dataframe
        observations to calibrate against

    Returns:
    --------
    out: lmfit object
        object containing the optimized parameters and several
        goodness-of-fit statistics

    """

    # non time-sensitive: last valid value propagated until next valid
    idf.fillna(method='ffill', inplace=True)

    # match dates
    idf['date'] = idf['year'] * 1000. + idf['doy']
    idf['date'] = pd.to_datetime(idf['date'], format='%Y%j')
    tdf['date'] = pd.to_datetime(tdf['Date'], dayfirst=True)
    df = idf.merge(tdf, on='date', suffixes=(None, '_y'))

    # rm missing data (-9999.)
    df = df[df['volRing'] > 0.]

    # limit the data to photosynthetic active times
    df = df[df['PPFD'] >= 50.]

    # calibrate to wettest half
    df = df[(df['volRing'] > df['volRing'].quantile(0.5))]

    # add ancillary variables / params needed for the model to run
    df['albedo_s'] = df['albedo_ws']
    df['albedo_s'].where(df['sw0'] >= 0.5 * (df['fc'] - df['pwp']),
                         df['albedo_ds'])
    df['recovembo'] = 1.
    df['legembo'] = 0.
    df['fvc_sun'] = 0.9999
    df['fvc_sha'] = 0.9999

    # actual input and target files for use
    input = df.copy()
    target = df['volRing'].groupby(df['date']).mean()  # day, mm
    target = target.resample('W').sum()  # weekly totals, mm

    # declare empty parameter class
    params = lmfit.Parameters()

    if model == 'Medlyn':  # Medlyn model params
        params.add('g1', 1., min=0.5, max=12.5)
        params.add('sfw', 0.2, min=0.01, max=5.)

    if model == 'ProfitMax':  # ProfitMax model param
        params.add('kmax', 1., min=0.1, max=5.)

    out = lmfit.minimize(fit_trans_params, params, args=(model, input, target),
                         method='differential_evolution', nan_policy='omit')

    return out


def fit_evap_params(params, input, target):

    """
    Minimizing function between the soil evaporation model and the
    observations of soil evaporation.

    Arguments:
    ----------
    params: object
        parameters to calibrate, including their initialisation values

    input: recarray object or pandas series
        time step's met data & params

    target: recarray object or pandas series
        target observations to calibrate against

    Returns:
    --------
    Residuals between the simulations and the observations

    """

    for pname in params.items():  # update the model's specific param.s

        input[pname[0]] = params[pname[0]].value

    # initialise the soil evap.
    input['Es'] = np.nan

    for step in range(len(input)):

        p = input.iloc[step].copy()
        input.iloc[step, input.columns.get_loc('Es')] = soil_evap(p, p.sw0)

    # day evap, mm
    sim = input.groupby('date').sum()['Es'] * conv.mmolH2Opm2ps_2_mmphlfhr
    sim.fillna(0., inplace=True)
    sim = sim.resample('W').sum()  # weekly totals, mm

    return np.abs(sim - target)


def estimate_evap_params(idf, tdf):

    """
    Wrapper around the fit_evap_params function used to calibrate
    the model to observations of daily soil evaporation.

    Arguments:
    ----------
    idf: pandas dataframe
        time step's met data & params

    tdf: pandas dataframe
        observations to calibrate against

    Returns:
    --------
    out: lmfit object
        object containing the optimized parameters and several
        goodness-of-fit statistics

    """

    # non time-sensitive: last valid value propagated until next valid
    idf.fillna(method='ffill', inplace=True)

    # match dates
    idf['date'] = idf['year'] * 1000. + idf['doy']
    idf['date'] = pd.to_datetime(idf['date'], format='%Y%j')
    tdf['date'] = pd.to_datetime(tdf['Date'], dayfirst=True)
    df = idf.merge(tdf, on='date', suffixes=(None, '_y'))

    # rm missing evap data (-9999.)
    df = df[df['wuTP'] > 0.]

    # limit the data to daytime
    df = df[df['PPFD'] > 0.]

    # exclude rainy days
    idx = (df.groupby('date').sum()
           [np.isclose(df.groupby('date').sum()['precip'], 0.)].index)
    df.set_index('date', inplace=True)
    df = df.loc[idx]

    # add ancillary variables / params needed for the model to run
    df['albedo_s'] = df['albedo_ws']
    df['albedo_s'].where(df['sw0'] >= 0.5 * (df['fc'] - df['pwp']),
                         df['albedo_ds'])

    # actual input and target files for use
    input = df.copy()
    target = df['wuTP'].groupby('date').mean()  # day, mm

    # declare empty parameter class
    params = lmfit.Parameters()

    # parameters to fit
    params.add('r_soil', 0.5, min=0., max=0.99)

    out = lmfit.minimize(fit_evap_params, params, args=(input, target),
                         method='differential_evolution', nan_policy='omit')

    return out


# ======================================================================

if __name__ == "__main__":

    # define the argparse settings
    description = "Calibrate key model parameters for a given process  \
                   in each ring"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-p', '--process', type=str,
                        help='process to calibrate: intercept, trans, or evap')
    parser.add_argument('-w', '--ring', type=str,
                        help='ring within which to calibrate')
    parser.add_argument('-m', '--model', type=str, help='model to calibrate',
                        default='ProfitMax')
    parser.add_argument('-R', '--project', type=str,
                        help='project where forcings are stored')
    args = parser.parse_args()

    if (args.process is None) or (args.ring is None):
        raise NotImplementedError('You must specify a process and ring')

    main(args.process, args.ring, model=args.model, project=args.project)
