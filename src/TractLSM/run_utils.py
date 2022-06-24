# -*- coding: utf-8 -*-

"""
Support functions needed to run the TractLSM model.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "useful ancillary run functions"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (11.08.2021)"
__email__ = "m.e.b.sabot@gmail.com"

# ======================================================================

# general modules
import pandas as pd  # read/write dataframes, csv files


# ======================================================================

def time_step(df, time_step):

    """
    Accesses each row / time-step of a panda dataframe

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing all input data & params

    time_step: int
        current time step

    Returns:
    --------
    A copy of the time step's data, i.e. the row (pandas series)

    """

    return df.iloc[time_step].copy()


def write_csv(fname, df, dic):

    """
    Writes a csv output file.

    Arguments:
    ----------
    fname: string
        output filename

    df: pandas dataframe
        dataframe containing all input data & params

    dic: ordered dictionary
        dictionary of the outputs returned by the models, it is read in
        order so that the csv created have the right units and headers
        consistently matching the data

    Returns:
    --------
    df2 : pandas dataframe
        dataframe of the outputs:
            An(model), E(model), Ci(model), gs(model), Pleaf(model),
            Tleaf(model), Rublim(model), ...

    Also saves the corresponding csv file, in the output/ folder under
    fname.

    """

    # timeseries for wich the optimisation has occured
    len_series = len(list(list(dic.values())[0].values())[0])

    # declare the lists for the column names, units, and data
    columns = ['year', 'doy', 'hod']
    units = ['[-]', '[-]', '[h]']
    valvars = [list(df['year'])[:len_series], list(df['doy'])[:len_series],
               list(df['hod'])[:len_series]]

    # append to the lists the content of the dictionary
    for subkey in list(dic.values())[0].keys():  # loop on output vars

        for key in dic.keys():  # loop on uso, pmax

            columns += ['%s(%s)' % (subkey, key)]

            if subkey == 'An':
                units += ['[umol m-2 s-1]']

            if subkey == 'E':
                units += ['[mmol m-2 s-1]']

            if subkey == 'Ci':
                units += ['[Pa]']

            if subkey == 'gs':
                units += ['[mol m-2 s-1]']

            if subkey == 'Pleaf':
                units += ['[MPa]']

            if subkey == 'Tleaf':
                units += ['[degC]']

            if subkey == 'Rublim':
                units += ['[-]']

            if subkey == 'Vmax25':
                units += ['[umol m-2 s-1]']

            if subkey == 'Jmax25':
                units += ['[umol m-2 s-1]']

            if subkey == 'Nc':
                units += ['[mol m-2]']

            if subkey == 'Ne':
                units += ['[mol m-2]']

            if subkey == 'Nr':
                units += ['[mol m-2]']

            if subkey == 'Ns':
                units += ['[mol m-2]']

            if subkey == 'Eci':
                units += ['[mmol m-2 s-1]']

            if subkey == 'Es':
                units += ['[mmol m-2 s-1]']

            if subkey == 'sw':
                units += ['[m3 m-3]']

            if subkey == 'Ps':
                units += ['[MPa]']

            valvars += [list(dic[key][subkey])]

    if 'pmax' in dic.keys():  # insert hydraulic outputs
        idx = [idx for idx, e in enumerate(columns) if 'Eci' in e][0]
        columns = (columns[:idx] + ['ksc(pmax)', 'VCsun(pmax)', 'VCsha(pmax)']
                   + columns[idx:])
        units = (units[:idx] + ['[mmol m-2 s-1 MPa-1]', '[-]', '[-]'] +
                 units[idx:])
        valvars = (valvars[:idx] + [list(df['ksc'])[:len_series],
                                    list(df['fvc_sun'])[:len_series],
                                    list(df['fvc_sha'])[:len_series]] +
                   valvars[idx:])

    df = (pd.DataFrame(valvars)).T
    df.columns = columns
    these_headers = list(zip(df.columns, units))
    df.columns = pd.MultiIndex.from_tuples(these_headers)

    # there can be a doubling of the last timestep
    df.drop_duplicates(inplace=True)

    # write the csv
    df.to_csv(fname, index=False, na_rep='', encoding='utf-8')

    return df
