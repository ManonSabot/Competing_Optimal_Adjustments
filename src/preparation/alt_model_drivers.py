#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that alters parameters or variables of the standard model input
files. This is useful for, e.g., a sensitivity analysis.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along this script.

"""

__title__ = "perturb model drivers"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (12.10.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# import general modules
import argparse  # read user input
import os  # check for files, paths
import sys  # make the TractLSM modules loadable
import pandas as pd  # read/write dataframes, csv files
import numpy as np  # array manipulations, math operators

# own modules
from soil_moisture_profiles import main as roots

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from TractLSM.Utils import get_main_dir  # locate data
from TractLSM.Utils import read_csv  # read in data files


# ======================================================================

def main(project, var):

    """
    Main function: takes forcing files already meeting the model's
                   requirements and updates some of the parameters /
                   variables in a systematic fashion.

    Arguments:
    ----------
    project: string
        project repository where the input files (drivers) to alter are

    var: string
        variable or parameter to alter

    Returns:
    --------
    Saves the new data input files in input/projects/{project2}.

    """

    # path to files for parameter calculation
    dir = os.path.join(os.path.dirname(get_main_dir()), 'input')

    # path to the drivers
    if project is not None:
        idir = os.path.join(os.path.join(dir, 'projects'), project)

    else:
        idir = dir

    if 'root' in var.lower():
        project2 = 'root_distri'

    elif 'lai' in var.lower():
        project2 = 'avg_LAI'

    elif ('vcmax' in var.lower()) or ('vmax' in var.lower()):
        project2 = 'avg_aVcmax'

    if 'root' in var.lower():
        write2up = os.path.join(os.path.join(dir, 'projects'),
                                'upper_%s' % (project2))
        write2down = os.path.join(os.path.join(dir, 'projects'),
                                  'lower_%s' % (project2))

        if not os.path.isdir(write2up):
            os.makedirs(write2up)

        if not os.path.isdir(write2down):
            os.makedirs(write2down)

    else:
        write2 = os.path.join(os.path.join(dir, 'projects'), project2)

        if not os.path.isdir(write2):
            os.makedirs(write2)

    files = os.listdir(idir)
    files = [e for e in files if e.endswith('model_drivers.csv')]

    if 'root' in var.lower():
        alt_root_distri(files, idir, write2up, write2down)

    elif 'lai' in var.lower():
        alt_LAI(files, idir, write2)

    elif ('vcmax' in var.lower()) or ('vmax' in var.lower()):
        alt_Vcmax(files, idir, write2)

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def alt_root_distri(files, idir, odir1, odir2):

    """
    Perturbs the soil parameters by testing alternative two alternative
    root distributions: slightly shallower vs. slightly deeper roots.

    Arguments:
    ----------
    files: list
        name of the input files to consider in idir

    idir: string
        project repository where the input files (drivers) to alter are
        (with path)

    odir1: string
        project repository where the new altered shallower root distri
        input files (drivers) are to be stored (with path)

    odir2: string
        project repository where the new altered deeper root distri
        input files (drivers) are to be stored (with path)

    Returns:
    --------
    Saves the new data input files in input/projects/{odir1/odir2}.

    """

    # user choices for the soil parameter calculations
    Pswilt = -2.  # MPa
    max_root_depth = 200.  # cm

    df1_up, df2_up = roots(Pswilt, 0.94 - 0.02, max_root_depth, write=False)
    df1_down, df2_down = roots(Pswilt, 0.94 + 0.02, max_root_depth,
                               write=False)

    for file in files:

        # read in the existing file
        df, headers = read_csv(os.path.join(idir, file))

        # empty the columns we're interested in
        df[['sw', 'sw0', 'Ps']] = np.nan

        # populate the moisture data by doy within each year
        ring = file.split('_')[0].split('EucFACE')[1]
        df_up = df.merge(df1_up[df1_up['Ring'] == ring], how='left',
                         on=['year', 'doy'], suffixes=(None, '_y'))
        df_up[['sw', 'sw0', 'Ps']] = df_up[['sw_y', 'sw0_y', 'Ps_y']].bfill()
        df_down = df.merge(df1_down[df1_down['Ring'] == ring], how='left',
                           on=['year', 'doy'], suffixes=(None, '_y'))
        df_down[['sw', 'sw0', 'Ps']] = (df_down[['sw_y', 'sw0_y', 'Ps_y']]
                                        .bfill())

        # populate the texture parameters
        df_up.loc[0, df2_up.loc[ring].index] = df2_up.loc[ring]
        df_down.loc[0, df2_down.loc[ring].index] = df2_down.loc[ring]

        # select the columns and reapply multi-index
        df_up = df_up[headers.get_level_values(0)]
        df_up.columns = pd.MultiIndex.from_tuples(headers)
        df_down = df_down[headers.get_level_values(0)]
        df_down.columns = pd.MultiIndex.from_tuples(headers)

        # save file in relevant project folder
        df_up.to_csv(os.path.join(odir1, file), index=False, na_rep='',
                     encoding='utf-8')
        df_down.to_csv(os.path.join(odir2, file), index=False, na_rep='',
                       encoding='utf-8')

    return


def alt_LAI(files, idir, odir):

    """
    Perturbs the LAI of each year by replacing it by an average
    multi-year phenology.

    Arguments:
    ----------
    files: list
        name of the input files to consider in idir

    idir: string
        project repository where the input files (drivers) to alter are
        (with path)

    odir: string
        project repository where the new altered input files (drivers)
        are to be stored (with path)

    Returns:
    --------
    Saves the new data input files in input/projects/{odir}.

    """

    for file in files:

        # read in the existing file
        df, headers = read_csv(os.path.join(idir, file))

        # build an average phenology
        LAI_clim = df[['doy', 'LAI']].groupby('doy').mean()

        # smooth 'end-of-year' to 'beginning-of-year' transition
        smooth = LAI_clim.iloc[-15:-1].append(LAI_clim.iloc[:15])
        LAI_clim.loc[smooth.index] = smooth.rolling(7, min_periods=1).mean()

        # fix day 366 which is completely off
        LAI_clim.iloc[-1] = 0.5 * (LAI_clim.iloc[0] + LAI_clim.iloc[-2])

        # now repeat this often enough for each year
        days = df.groupby('year').agg({'doy': 'nunique'})
        LAI_clim = [LAI_clim.iloc[:days['doy'].iloc[i]] for i in
                    range(len(days))]
        LAI_clim = LAI_clim[0].append(LAI_clim[1:], ignore_index=True)
        LAI_clim = LAI_clim.loc[LAI_clim.index.repeat(48)]

        # assign the new LAI climatology
        df['LAI'] = LAI_clim.values

        # save file in relevant project folder
        df.columns = pd.MultiIndex.from_tuples(headers)
        df.to_csv(os.path.join(odir, file), index=False, na_rep='',
                  encoding='utf-8')

    return


def alt_Vcmax(files, idir, odir):

    """
    Perturbs the Vcmax and Jmax parameters of each ring by replacing
    them by aCO2 averages of Vcmax and Jmax.

    Arguments:
    ----------
    files: list
        name of the input files to consider in idir

    idir: string
        project repository where the input files (drivers) to alter are
        (with path)

    odir: string
        project repository where the new altered input files (drivers)
        are to be stored (with path)

    Returns:
    --------
    Saves the new data input files in input/projects/{odir}.

    """

    # first calculate the average params
    aVcmax = 0.
    aJV = 0.

    for file in [e for e in files if e.split('EucFACE')[1].split('_')[0] in
                 ['R2', 'R3', 'R6']]:  # ambient rings

        # read in the file
        df, __ = read_csv(os.path.join(idir, file))

        # put param value aside
        aVcmax += df.loc[0, 'Vmax25'] / 3.
        aJV += df.loc[0, 'JV'] / 3.

    # now populate files with new param values
    for file in files:

        # read in the existing file
        df, headers = read_csv(os.path.join(idir, file))

        # assign the new param values
        df.loc[0, 'Vmax25'] = aVcmax
        df.loc[0, 'JV'] = aJV

        # save file in relevant project folder
        df.columns = pd.MultiIndex.from_tuples(headers)
        df.to_csv(os.path.join(odir, file), index=False, na_rep='',
                  encoding='utf-8')

    return


# ======================================================================

if __name__ == "__main__":

    # define the argparse settings
    description = "Vary model drivers in each ring"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-v', '--variable', type=str,
                        help='variable to alter: root_distri/root(s) or LAI')
    parser.add_argument('-R', '--project', type=str,
                        help='project where reference forcings are stored')
    args = parser.parse_args()

    if args.variable is None:
        raise NotImplementedError('You must specify a variable to alter')

    main(args.project, args.variable)
