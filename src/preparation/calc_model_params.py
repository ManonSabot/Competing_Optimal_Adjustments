#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Estimates photosynthetic, plant hydraulic, and soil parameters from
observations at EucFACE.

Three observational datasets stored in input/obs/ are used:
    - 'EucFACE_photo_capacity_nitrogen_dominant_trees_2013_2020.csv'
    - 'EucFACE_water_potential_dominant_trees_2012_2013.csv'
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

__title__ = "calculate model parameters"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (04.10.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# import general modules
import os  # check for files, paths
import sys  # make the TractLSM modules loadable
import pandas as pd  # read/write dataframes, csv files
import numpy as np  # array manipulations, math operators

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM.Utils import get_main_dir  # locate data
from TractLSM.SPAC.hydraulics import Weibull_params, f  # hydraulics
from TractLSM.SPAC.soil import root_distri  # for the soil params


# ======================================================================

def main(P50, P88, Pswilt, root_beta, max_root_depth, depths):

    """
    Main: calculates photosynthetic, hydraulic, and soil parameters at
          EucFACE and writes them out to text files in input/params.

    Arguments:
    ----------
    P50: float
        leaf water potential [-MPa] causing a 50% decline in hydraulic
        conductance

    P88: float
        leaf water potential [-MPa] causing a 88% decline in hydraulic
        conductance

    Pswilt: float
        soil water potential [MPa] causing the soil to wilt

    root_beta: float
        unitless extinction coefficient that sets the decline in
        cumulative root water access from the soil surface to depth

    max_root_depth: float
        maximum depth [cm] to consider when filtering the data

    depths: array
        lower depth of each of six layers that match 6 averaged
        adjacent layers in the obs data

    Returns:
    --------
    files of the form 'Ring.txt' in input/params/

    """

    # path to files for parameter calculation
    dir = os.path.join(os.path.dirname(get_main_dir()), 'input')
    write2 = os.path.join(dir, 'params')

    if not os.path.isdir(write2):
        os.makedirs(write2)

    # paths to the obs
    fphoto = 'EucFACE_photo_capacity_nitrogen_dominant_trees_2013_2020.csv'
    photo = os.path.join(os.path.join(dir, 'obs'), fphoto)
    fhydra = 'EucFACE_water_potential_dominant_trees_2012_2013.csv'
    hydra = os.path.join(os.path.join(dir, 'obs'), fhydra)
    soil = os.path.join(os.path.join(dir, 'obs'), 'EucFACE_soiltext.csv')

    # read in the obs
    df1 = (pd.read_csv(photo).dropna(axis=0, how='all')
             .dropna(axis=1, how='all').squeeze())
    df2 = (pd.read_csv(hydra).dropna(axis=0, how='all')
             .dropna(axis=1, how='all').squeeze())
    df3 = (pd.read_csv(soil).dropna(axis=0, how='all')
           .dropna(axis=1, how='all').squeeze())

    # add the P50 and P88 to df2
    df2['P50'] = P50
    df2['P88'] = P88

    # add the Pswilt, max_root_depth, and root_beta to df3
    df3['Pswilt'] = Pswilt
    df3['max_root_depth'] = max_root_depth
    df3['root_beta'] = root_beta

    # calculate parameters and write them out to text files
    photo_params(df1, write2)
    rk_param(df2, write2)
    soil_params(df3, depths, write2)

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def photo_params(df, odir, write=True):

    """
    Performs basic filtering and averaging to retrieve reference
    photosynthetic parameters in Feb. 2013, i.e. at the time of the
    first A-Ci campaign at EucFACE.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing Vcmax, Jmax, and Rleaf data

    odir: string
        path to the directory in which to write the computed params

    write: bool
        whether to write the text files or not

    Returns:
    --------
    If write is True, a text file for each EucFACE ring, stored in odir.
    Otherwise, a dataframe containing the parameter values.

    """

    # replace negative bonkers Rleaf fits with Nans
    df['Rd25'] = df['Rd25'].where(df['Rd25'] > 0.)

    # group by campaign and ring
    df = df.groupby(['Campaign', 'Ring']).mean()

    # replace missing Rleaf with the avg Rleaf(Vcmax) across rings
    df['Rd25'].fillna((df['Rd25'] / df['Vcmax25']).mean() * df['Vcmax25'],
                      inplace=True)

    # average across campaigns for each ring
    df = df.groupby(level=1).mean()

    if write:  # loop over the rings to create the param files
        for ring in df.index.to_list():

            # write to param file
            if not os.path.isfile(os.path.join(odir, '%s.txt' % (ring))):
                txt = open(os.path.join(odir, '%s.txt' % (ring)), 'w+')

            else:  # append to existing file
                txt = open(os.path.join(odir, '%s.txt' % (ring)), 'a+')

            txt.write('\n[[Photosynthetic parameters]]\n')
            txt.write(f"    Vmax25: {df.loc[ring, 'Vcmax25']: .10f}\n")
            txt.write(f"    Jmax25: {df.loc[ring, 'Jmax25']: .10f}\n")
            JV = df.loc[ring, 'Jmax25'] / df.loc[ring, 'Vcmax25']
            txt.write(f"    JV: {JV: .10f}\n")
            txt.write(f"    Rlref: {df.loc[ring, 'Rd25']: .10f}\n")
            txt.write('\n')
            txt.close()  # close text file

        return

    return df[['Vcmax25', 'Jmax25', 'Rd25']]


def rk_param(df, odir, write=True):

    """
    Performs basic filtering and estimates the r_k parameter which
    defines the ability to recover from embolism after drought, i.e.
    the ratio of recovery of kmax.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing leaf water potential data at predawn, in
        the morning, and at midday

    odir: string
        path to the directory in which to write the computed params

    write: bool
        whether to write the textfiles or not

    Returns:
    --------
    If write is True, a text file for each EucFACE ring, stored in odir.
    Otherwise, a dataframe containing the parameter values.

    """

    # remove day x tree measurements for which LWP > predawn LWP
    if any(df[df['WP'] > df['Predawn.WP']]):
        dates = df[df['WP'] > df['Predawn.WP']]['Date'].to_list()
        trees = df[df['WP'] > df['Predawn.WP']]['Tree'].to_list()

        for i in range(len(dates)):

            df.drop(df[np.logical_and(df['Date'] == dates[i],
                                      df['Tree'] == trees[i])].index,
                    inplace=True)

    df.reset_index(inplace=True)  # reset the index after removal

    # calculate the conductivity for every set of measurements
    b, c = Weibull_params(df.iloc[0])
    df['VC'] = f(df['WP'], b, c)

    # estimate each tree's ability to recover conductivity
    df['rk'] = np.nan

    for tree in df['Tree'].unique():

        sub = df[df['Tree'] == tree]

        # three different dates needed to proxy xylem statuses
        if len(sub['Date'].unique()) > 2:
            unaltered = sub.groupby('Date')['VC'].max().max()
            recovered = sub.groupby('Date')['VC'].max().nlargest(2).min()
            damaged = sub.groupby('Date')['VC'].min().min()
            df.loc[sub.index, 'rk'] = ((damaged - recovered) /
                                       (damaged - unaltered))

    # group by ring
    df = df.groupby('Ring').mean()

    if write:  # loop over the rings to create the param files
        for ring in df.index.to_list():

            # write to param file
            if not os.path.isfile(os.path.join(odir, '%s.txt' % (ring))):
                txt = open(os.path.join(odir, '%s.txt' % (ring)), 'w+')

            else:  # append to existing file
                txt = open(os.path.join(odir, '%s.txt' % (ring)), 'a+')

            txt.write('[[Hydaulic parameters]]\n')
            txt.write(f"    r_k: {df.loc[ring, 'rk']: .10f}\n")
            txt.write('\n')
            txt.close()  # close text file

        return

    return df['rk']


def Campbell_Cosby_pedotransfer(df):

    """
    Estimates reference soil texture and soil hydraulic parameters from
    the pedotransfer functions of Campbell (1974) and of
    Cosby et al. (1984).

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing sand, clay, and silt contents at depth

    Returns:
    --------
    df: pandas dataframe
        dataframe containing soil texture and soil hydraulic parameters

    """

    # constrain data where the material contents overshoot 100%
    where = (100. - df['Sand_%'] - df['Clay_%'] - df['Silt_%']) < 0.
    df['Clay_%'][where] = df['Clay_%'][where] + (100. - df['Sand_%'] -
                                                 df['Clay_%'] - df['Silt_%'])

    # fracs of sand (SA), clay (CL), silt (SI), & soil organic C (SOC)
    SA = df['Sand_%'] / 100.
    CL = df['Clay_%'] / 100.
    SI = df['Silt_%'] / 100.
    SOC = np.maximum(0., 1. - SA - CL - SI)

    # Clapp and Hornberger index, unitless
    df['bch'] = (1. - SOC) * (3.1 + 15.7 * CL - 0.3 * SA) + 3. * SOC

    # saturation hydraulic conductivity, mm s-1
    df['hyds'] = ((1. - SOC) * (0.00706 * 10. ** (-0.6 + 1.26 * SA -
                                                  0.64 * CL)) +
                  SOC * 10. ** (-4.))

    # soil moisture at saturation, m3 m-3
    df['theta_sat'] = ((1. - SOC) * (0.505 - 0.142 * SA - 0.037 * CL) +
                       0.6 * SOC)

    # field capacity, m3 m-3
    water = (1. - SOC) * (0.02 + 0.018 * CL) + 0.15 * SOC
    df['fc'] = (water + (df['theta_sat'] - water) * ((1.157407 * 10. ** (-6.) /
                df['hyds']) ** (1. / (2. * df['bch'] + 3.))))

    # air entry point water potential, MPa
    df['Psie'] = -0.033 * (df['theta_sat'] / df['fc']) ** (-df['bch'])

    # set soil wilting pt at Pswilt
    df['pwp'] = df['theta_sat'] * (df['Pswilt'] / df['Psie']) ** (-1. /
                                                                  df['bch'])

    # correct unit for the soil hydraulic conductivity
    df['hyds'] *= 1.e-3  # m s-1

    # min depth and max depth instead of depth intervals
    df = df.groupby(['Ring', 'Depth_interval_cm']).mean()
    df['min_depth'] = [int(e.split('-')[0]) for e in
                       df.index.get_level_values(1).to_list()]
    df['max_depth'] = [int(e.split('-')[1]) for e in
                       df.index.get_level_values(1).to_list()]
    df.index = df.index.get_level_values(0)

    return df


def soil_params(df, depths, odir, write=True):

    """
    Performs basic filtering and estimates some soil / hydrological
    parameters commonly used by LSMs.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing leaf water potential data at predawn, in
        the morning, and at midday

    depths: array
        lower depth of each of six layers that match 6 averaged
        adjacent layers in the obs data

    odir: string
        path to the directory in which to write the computed params

    write: bool
        whether to write the textfiles or not

    Returns:
    --------
    If write is True, a text file for each EucFACE ring, stored in odir.
    Otherwise, a dataframe containing the parameter values.

    """

    # remove the data from outside the rings
    df = df[df['Ring'] != 'Out']
    df.reset_index(inplace=True)  # reset the index after removal

    # soil texture parameters
    df = Campbell_Cosby_pedotransfer(df)

    # limit the data at depth
    df = df[df['min_depth'] < df['max_root_depth']]
    df.sort_values(by=['min_depth'], inplace=True)

    # root distribution?
    froot = root_distri(depths, df['root_beta'].iloc[0])

    # relative water availability at depth
    rwa = np.zeros((len(df.index.unique()), len(depths)))

    # root fraction matching texture at depth
    df['froot'] = 0.

    for i, ring in enumerate(df.index.unique().sort_values()):

        # soil texture parameters per ring
        texture = df.loc[ring]

        for j in range(len(depths)):

            # soil texture parameter per depth layer
            sub = texture[texture['min_depth'] <= depths[j]]
            sub = sub[np.abs(sub['max_depth'] - depths[j]) ==
                      np.amin(np.abs(sub['max_depth'] - depths[j]))]

            if len(sub) > 1:  # equidistant depths
                sub = sub[np.abs(sub['min_depth'] - depths[j]) ==
                          np.amin(np.abs(sub['min_depth'] - depths[j]))]

            # calculate the relative water availability
            rwa[i, j] = (np.insert(np.diff(depths), 0, depths[0])[j] / 100. *
                         (sub['theta_sat'].values - sub['pwp'].values))

            # assign root fraction to texture at depth
            idx = (np.arange(len(df))[df.index.get_loc(ring)]
                   [texture['min_depth'] == sub['min_depth'].values[0]])
            df.iloc[idx, -1] = froot[j]

    # total rwa per ring
    rwa = np.sum(rwa, axis=1)

    # scale to account for where the roots are
    df = df[df['froot'] > 0.]
    df['fTOT'] = df['froot'].groupby(df.index).sum()
    df['froot'] = df['froot'].divide(df['fTOT'])
    df[df.columns.difference(['froot'])] = (df[df.columns
                                                 .difference(['froot'])]
                                            .multiply(df['froot'],
                                                      axis='index')
                                            .groupby(df.index).sum())
    df = df.groupby(df.index).mean()

    if write:  # loop over the rings to create the param files
        for i, ring in enumerate(df.index.sort_values()):

            # write to param file
            if not os.path.isfile(os.path.join(odir, '%s.txt' % (ring))):
                txt = open(os.path.join(odir, '%s.txt' % (ring)), 'w+')

            else:  # append to existing file
                txt = open(os.path.join(odir, '%s.txt' % (ring)), 'a+')

            txt.write('[[Soil parameters]]\n')
            txt.write(f"    Ztop: {(depths[0] / 100.): .10f}\n")
            txt.write(f"    Zbottom: {rwa[i]: .10f}\n")
            txt.write(f"    bch: {df.loc[ring, 'bch']: .10f}\n")
            txt.write(f"    hyds: {df.loc[ring, 'hyds']: .10f}\n")
            txt.write(f"    Psie: {df.loc[ring, 'Psie']: .10f}\n")
            txt.write(f"    theta_sat: {df.loc[ring, 'theta_sat']: .10f}\n")
            txt.write(f"    fc: {df.loc[ring, 'fc']: .10f}\n")
            txt.write(f"    pwp: {df.loc[ring, 'pwp']: .10f}\n")
            txt.write('\n')
            txt.close()  # close text file

        return

    return df[['bch', 'hyds', 'Psie', 'theta_sat', 'fc', 'pwp']]


# ======================================================================

if __name__ == "__main__":

    # necessary hydraulics info
    P50 = 4.07324814  # -MPa
    P88 = 5.495033637  # -MPa

    # user choices for the soil parameter calculations
    Pswilt = -2.  # MPa
    root_beta = 0.94  # unitless
    max_root_depth = 200.  # cm

    # depths over which to integrate the params
    depths = [5., 15., 27.5, 62.5, 112.5, 175.]

    main(P50, P88, Pswilt, root_beta, max_root_depth, depths)
