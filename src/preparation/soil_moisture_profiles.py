#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Estimates the root-zone soil moisture by combining different
observational datasets at EucFACE.

Three observational datasets stored in input/obs/ are used:
    - 'EucFACE_sm_gap_filled.csv'
    - 'EucFACE_sm_neutron.csv'
    - 'EucFACE_soiltext.csv'
see the ReadMe in input/ for information on these datasets.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along this script.

References:
-----------
* Campbell, G. S. (1974). A simple method for determining unsaturated
  conductivity from moisture retention data. Soil science, 117(6),
  311-314.
* Cosby, B. J., Hornberger, G. M., Clapp, R. B., & Ginn, T. (1984). A
  statistical exploration of the relationships of soil moisture
  characteristics to the physical properties of soils. Water resources
  research, 20(6), 682-690.
* Gale, M. R., & Grigal, D. F. (1987). Vertical root distributions of
  northern tree species in relation to successional status. Canadian
  Journal of Forest Research, 17(8), 829-834.

"""

__title__ = "estimate root-zone soil moisture"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (04.10.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# import general modules
import os  # check for files, paths
import sys  # make the TractLSM modules loadable
import pandas as pd  # read/write dataframes, csv files
import numpy as np  # array manipulations, math operators

# own modules
from calc_model_params import Campbell_Cosby_pedotransfer
from calc_model_params import root_distri
from calc_model_params import soil_params

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from TractLSM.Utils import get_main_dir  # locate data


# ======================================================================

def main(Pswilt, root_beta, max_root_depth, write=True):

    """
    Main: estimates the root-zone soil moisture profile and the soil
          parameters of each EucFACE ring
    Arguments:
    ----------
    Pswilt: float
        soil water potential [MPa] causing the soil to wilt

    root_beta: float
        unitless extinction coefficient that sets the decline in
        cumulative root water access from the soil surface to depth

    max_root_depth: float
        maximum depth [cm] to consider when filtering the data

    write: bool
        whether to write the csv or not

    Returns:
    --------
    If write is True, a .csv for the soil profile of each EucFACE ring,
    the average across rings, and the average across ambient and
    elevated rings, stored in input/soil/.
    Otherwise, a dataframe containing the soil profiles and a dataframe
    containing the soil parameter values.

    """

    # path to files for building the soil moisture profiles
    dir = os.path.join(os.path.dirname(get_main_dir()), 'input')
    write2 = os.path.join(dir, 'soil')

    if not os.path.isdir(write2):
        os.makedirs(write2)

    # paths to the obs
    sm1 = os.path.join(os.path.join(dir, 'obs'), 'EucFACE_sm_gap_filled.csv')
    sm2 = os.path.join(os.path.join(dir, 'obs'), 'EucFACE_sm_neutron.csv')
    texture = os.path.join(os.path.join(dir, 'obs'), 'EucFACE_soiltext.csv')

    # read in the obs
    df1 = (pd.read_csv(sm1).dropna(axis=0, how='all').dropna(axis=1, how='all')
             .squeeze())
    df2 = (pd.read_csv(sm2).dropna(axis=0, how='all').dropna(axis=1, how='all')
             .squeeze())
    df3 = (pd.read_csv(texture).dropna(axis=0, how='all')
             .dropna(axis=1, how='all').squeeze())

    if write:  # save the files
        soil_moisture(df1, df2, df3, Pswilt, root_beta, max_root_depth, write2)

        return

    df1, df2 = soil_moisture(df1, df2, df3, Pswilt, root_beta, max_root_depth,
                             write2, write=False)

    return df1, df2


# ======================================================================

# ~~~ Other functions are defined here ~~~

def combine_datasets(df1, df2, max_root_depth):

    """
    Combines the different sources of soil moisture observations by
    merging their overlapping measurements at depth.

    Arguments:
    ----------
    df1: pandas dataframe
        dataframe containing a first set of soil moisture observations

    df2: pandas dataframe
        dataframe containing a second set of soil moisture observations

    max_root_depth: float
        maximum depth to consider

    Returns:
    --------
    df1: pandas dataframe
        dataframe containg merged data from the original df1 and df2s

    real: array
        a mask that differentiates the 'real' / unmodified obs from the
        interpolated or composite data

    """

    # split the lower layers by depths
    df2 = df2[['Date', 'Ring', 'Depth', 'VWC']]
    dfs = dict(tuple(df2.groupby('Depth')))

    for i, depth in enumerate(df2['Depth'].unique()):

        if depth <= max_root_depth:
            sub = dfs[depth]
            sub.rename(columns={'VWC': 'swc.theta.%d' % round(depth)},
                       inplace=True)
            sub.drop('Depth', axis=1, inplace=True)

            if i == 0:
                df2 = sub

            else:
                df2 = df2.merge(sub, how='outer', on=['Date', 'Ring'])

    df2.drop_duplicates(inplace=True)

    # add to the other soil data
    df1['Date'] = pd.to_datetime(df1['Date'], dayfirst=True)
    df2['Date'] = pd.to_datetime(df2['Date'], dayfirst=True)
    df1 = df1.merge(df2, how='outer', on=['Date', 'Ring'])
    df1.drop_duplicates(inplace=True)

    # magnitude correct theta probe data, needed for R4 after 2015
    sub = df1[df1['Ring'] == 'R4'].copy()
    ref_min = sub[pd.DatetimeIndex(sub['Date']).year == 2013].min()

    for what in ['swc.theta.5', 'swc.theta.30']:

        if (sub[pd.DatetimeIndex(sub['Date']).year > 2013][what].min() <
           ref_min.loc[what]):
            sub[what][pd.DatetimeIndex(sub['Date']).year > 2013] += \
                (ref_min.loc[what] - sub[what][pd.DatetimeIndex(sub['Date'])
                                                 .year > 2013].min())

    df1[df1['Ring'] == 'R4'] = sub

    # any leftover neg data to nans
    cols = df1.filter(like='swc.').columns.to_list()
    df1[cols].where(df1[cols] > 0., inplace=True)

    # neutron probe for 75 cm measurements
    df1['swc.theta.75'] = df1['swc.theta.75_y']
    df1.drop(['swc.tdr', 'swc.theta.75_x', 'swc.theta.75_y'], axis=1,
             inplace=True)

    # track "real data" (diff from interp or composite data)
    real = (df1[~df1.isnull().any(axis=1)].groupby(['Ring', 'Date']).mean()
            .index)

    # average adjacent layers to keep only 6 layers
    adj1 = [5, 25, 50, 100, 150]
    adj2 = [25, 30, 75, 125, 200]

    for i in range(len(adj1)):

        new = round((adj1[i] + adj2[i]) / 2., 1)

        if new.is_integer():
            new = int(new)

        df1['swc.theta.%s' % str(new)] = df1[['swc.theta.%s' % (str(adj1[i])),
                                              'swc.theta.%s' %
                                              (str(adj2[i]))]].mean(axis=1)

        if i > 0:
            df1.drop(['swc.theta.%s' % (str(adj1[i])),
                      'swc.theta.%s' % (str(adj2[i]))], axis=1, inplace=True)

    return df1.groupby(['Ring', 'Date']).mean(), real


def daily_data(df1):

    """
    Interpolates the data to ensure there are daily root-zone estimates.

    Arguments:
    ----------
    df1: pandas dataframe
        dataframe containg merged data from the original df1 and df2s

    Returns:
    --------
    df1: pandas dataframe
        dataframe containg merged and interpolated data from the
        original df1 and df2s

    """

    # interpolate the sw timeseries to fill all days/dates
    for i, ring in enumerate(df1.index.levels[0]):

        if i == 0:
            df2 = df1.loc[ring].resample('D').asfreq().interpolate()
            df2['Ring'] = ring

        else:
            df2 = df2.append(df1.loc[ring].resample('D').asfreq()
                             .interpolate())
            df2['Ring'].fillna(ring, inplace=True)

    df1 = df2  # rename to df1

    return df1


def below_pwp(df1, df2, Pswilt, depths):

    """
    Deals with water potential data below the expected soil wilting
    point in the root-zone data, using the soil texture data.

    Arguments:
    ----------
    df1: pandas dataframe
        dataframe containg merged and interpolated data from the
        original df1 and df2s

    df2: pandas dataframe
        dataframe containing the soil texture data

    Pswilt: float
        soil water potential [MPa] causing the soil to wilt

    depths: list or array
        depths at which measurements were taken

    Returns:
    --------
    df1: pandas dataframe
        dataframe containg merged, interpolated, and filtered data from
        the original df1 and df2s

    """

    # fetch texture params
    df2['Pswilt'] = Pswilt  # set the wilting point
    df2 = Campbell_Cosby_pedotransfer(df2)  # texture
    df2 = df2[df2['min_depth'] < df2['max_root_depth']]
    df2.sort_values(by=['min_depth'], inplace=True)

    # fill below "wilting point" per depth in each ring
    for ring in df1['Ring'].unique():

        texture = df2.loc[ring]  # soil texture per ring

        for j in range(len(depths)):

            depth = depths[j]

            if depth.is_integer():
                depth = int(depth)

            # soil texture parameter per depth layer
            sub = texture[texture['min_depth'] <= depths[j]]
            sub = sub[np.abs(sub['max_depth'] - depths[j]) ==
                      np.amin(np.abs(sub['max_depth'] - depths[j]))]

            if len(sub) > 1:  # equidistant depths
                sub = sub[np.abs(sub['min_depth'] - depths[j]) ==
                          np.amin(np.abs(sub['min_depth'] - depths[j]))]

            # calculate the scaler based on relative water availability
            subs = [texture.iloc[np.argmin(np.abs(texture['max_depth'] - e))]
                    for e in depths]
            theta_sat = np.asarray([e['theta_sat'] for e in subs])
            pwp = np.asarray([e['pwp'] for e in subs])
            rwa = (np.insert(np.diff(depths), 0, depths[0]) / 100. *
                   (theta_sat - pwp))
            sc = (rwa / np.sum(rwa))[j]

            # deal with values < ref pwp
            df1['swc.theta.%s' % (str(depth))][df1['Ring'] == ring] *= sc + 1.
            df1['swc.theta.%s' % (str(depth))][df1['Ring'] == ring] += \
                0.5 * sub['pwp'].values[0] * 100.

            # ensure we're within okay bounds
            df1['swc.theta.%s' % (str(depth))][df1['Ring'] == ring] = \
                np.maximum(df1['swc.theta.%s' % (str(depth))]
                              [df1['Ring'] == ring],
                           sub['pwp'].values[0] * 100.)
            df1['swc.theta.%s' % (str(depth))][df1['Ring'] == ring] = \
                np.minimum(df1['swc.theta.%s' % (str(depth))]
                              [df1['Ring'] == ring],
                           sub['theta_sat'].values[0] * 100.)

    # fill nans
    df1.interpolate(inplace=True)

    return df1


def remove_empty_lead(df):

    """
    Removes leading NaNs from the timeseries

    """

    # remove initial dates for which all values are NaNs
    idx = pd.isnull(df[['sw', 'sw0', 'Ps']]).all(1).to_numpy().nonzero()[0]

    if len(idx) > 0:  # check that the index are consecutive
        if idx[0] == 0:

            for i, e in enumerate(idx):

                if i != e:
                    idx = idx[:i]

            # rm rows
            df = df.iloc[idx[-1] + 1:]

    return df


def soil_moisture(df1, df2, df3, Pswilt, root_beta, max_root_depth, odir,
                  write=True):

    """
    Estimates the root-zone soil moisture of the different EucFACE rings
    by combining two sets of soil moisture observations at depth, as
    well as soil texture data.

    Arguments:
    ----------
    df1: pandas dataframe
        dataframe containing a first set of soil moisture observations

    df2: pandas dataframe
        dataframe containing a second set of soil moisture observations

    df3: pandas dataframe
        dataframe containing soil texture data

    Pswilt: float
        soil water potential [MPa] causing the soil to wilt

    root_beta: float
        unitless extinction coefficient that sets the decline in
        cumulative root water access from the soil surface to depth

    max_root_depth: float
        maximum depth [cm] to consider when filtering the data

    odir: string
        where to write the files to (with path)

    write: bool
        whether to write the csv or not

    Returns:
    --------
    If write is True, a .csv for the soil profile of each EucFACE ring,
    the average across rings, and the average across ambient and
    elevated rings, stored in {odir}.
    Otherwise, a dataframe containing the soil profiles and a dataframe
    containing the soil parameter values.

    """

    # remove the data from outside the rings
    df1 = df1[df1['Ring'].str.contains('Outside') == False]
    df2 = df2[df2['Ring'].str.contains('Outside') == False]
    df3 = df3[df3['Ring'] != 'Out']

    # reset the index
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    df3.reset_index(drop=True, inplace=True)

    # combine the two soil moisture datasets
    df1, real = combine_datasets(df1, df2, max_root_depth)

    # depths of the 6 layers obtained
    depths = sorted([float(e.split('swc.theta.')[1]) for e in df1.columns
                     if 'swc.theta' in e])

    # interpolate the soil moisture into daily data
    df1 = daily_data(df1)

    # fill the data below the soil wilting point
    df3['max_root_depth'] = max_root_depth
    df3['root_beta'] = root_beta
    df1 = below_pwp(df1, df3, Pswilt, depths)

    # get the relative root distribution, and a CI around
    froot = root_distri(depths, root_beta)

    # variables to build
    df1['sw'] = 0.  # overall sw
    df1['sw0'] = 0.  # topsoil sw

    for ring in df1['Ring'].unique():

        R = df1['Ring'] == ring

        # top soil soil noisture
        df1['sw0'][R] = df1['swc.theta.5'][R] / 100.

        for j in range(1, len(depths) + 1):

            depth = depths[-j]

            if depth.is_integer():
                depth = int(depth)

            # soil moisture estimates
            df1['sw'][R] += (froot[-j] * df1['swc.theta.%s' % (str(depth))][R]
                             / 100.)

    # calculate the associated soil water potentials
    df3 = soil_params(df3, depths, '', write=False)
    df1['Ps'] = 0.  # overall Ps

    for ring in df1['Ring'].unique():

        R = df1['Ring'] == ring
        texture = df3.loc[ring]  # soil texture per ring

        # calculate soil water potential
        df1['Ps'][R] = (texture['Psie'] * (df1['sw'][R] / texture['theta_sat'])
                        ** (-texture['bch']))

    # add info on "real" data (i.e., where not interpolated)
    df1['missing_sw'] = True
    df1.reset_index(inplace=True)
    df1 = df1.groupby(['Ring', 'Date']).mean()
    df1.loc[real, 'missing_sw'] = False
    df1.reset_index(level='Ring', inplace=True)

    # corresponding year and doy
    df1['year'] = df1.index.year
    df1['doy'] = df1.index.dayofyear

    # units and variables to keep
    ovars = ['year', 'doy', 'sw', 'sw0', 'Ps', 'missing_sw']
    ounits = ['[-]', '[-]', '[m3 m-3]', '[m3 m-3]', '[MPa]', '[-]']

    if write:

        for ring in df1['Ring'].unique():  # extract ring file

            df = df1[df1['Ring'] == ring].copy()
            df.reset_index(inplace=True)

            # remove initial dates for which all values are NaNs
            df = remove_empty_lead(df)

            # save df in the right format
            df = df[ovars]
            df.columns = pd.MultiIndex.from_tuples(list(zip(ovars, ounits)))
            df.to_csv(os.path.join(odir, 'EucFACE%s_sw.csv' % (ring)),
                      index=False, na_rep='', encoding='utf-8')

        # add ele/amb files and an overall file too
        amb_rings = ['R2', 'R3', 'R6']
        ele_rings = ['R1', 'R4', 'R5']

        for rings in [amb_rings, ele_rings, amb_rings + ele_rings]:

            df = df1[df1['Ring'].isin(amb_rings)].copy()
            df = df.groupby(df.index).mean()
            df.reset_index(inplace=True)

            # remove initial dates for which all values are NaNs
            df = remove_empty_lead(df)

            # save df in the right format
            df = df[ovars]
            df.columns = pd.MultiIndex.from_tuples(list(zip(ovars, ounits)))

            if rings == amb_rings:
                ring = 'amb'

            elif rings == ele_rings:
                ring = 'ele'

            else:
                ring = ''

            df.to_csv(os.path.join(odir, 'EucFACE%s_sw.csv' % (ring)),
                      index=False, na_rep='', encoding='utf-8')
        return

    df1.reset_index(inplace=True)
    df1 = remove_empty_lead(df1)

    return df1[ovars + ['Ring']], df3


# ======================================================================

if __name__ == "__main__":

    # user choices for the root distribution profile
    Pswilt = -2.  # MPa
    root_beta = 0.94  # unitless
    max_root_depth = 200.  # cm

    main(Pswilt, root_beta, max_root_depth)
