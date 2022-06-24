#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code that assesses the contributions from various environmental drivers
to the modelled Np, Vcmax, and Jmax, using dominance-analysis.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "drivers of Np, Vcmax, and Jmax"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (14.01.2022)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# import general modules
import argparse  # read in the user input
import os  # check for files, paths
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files
from dominance_analysis import Dominance
from analysis_utils import concat_rings

# own modules
from analysis_utils import dirpath


# ======================================================================

def main(project, periods, predictors, predicted):

    """
    Main function: Generates '(project_)N_leaf_photo_drivers.csv' in the
                   output directory, with information on the main
                   drivers of variance in simulated Np, Vcmax, and Jmax.

    Arguments:
    ----------
    project: string
        project repository where the files to analyse are (with path)

    periods: list
        time periods to consider in the analysis, defined based on
        whether LAI is high or low

    predictors: list
        environmental drivers to consider

    predicted: list
        simulated variables influenced by the environmental drivers

    Returns:
    --------
    '(project_)N_leaf_photo_drivers.csv' in the output directory

    """

    # paths
    best = os.path.join(os.path.dirname(os.path.dirname(project)),
                        '%s_ranked.csv' % (os.path.basename(project)))

    # what is the best file
    best = (pd.read_csv(best).dropna(axis=0, how='all')
              .dropna(axis=1, how='all').squeeze())
    tag = best.iloc[0, 0]

    # combine all relevant input and output files
    fins = concat_rings(project.replace('output', 'input'),
                        endtag='model_drivers')
    sims = concat_rings(project, endtag=tag, keyword='DSH')

    # analyse the data
    fins, sims = prepare_data(fins, sims, tag)
    perform_dominance_analysis(fins, sims, periods, predictors, predicted,
                               project=project)

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def prepare_data(fins, sims, tag):

    """
    Makes the environmental drivers representative of what is used in
    the model to estimates Np, Vcmax, and Jmax: midday conditions
    averaged over a period of N prior days (defined by the tag).

    Arguments:
    ----------
    fins: pandas dataframe
        dataframes containing the model input data (drivers)

    sims: pandas dataframe
        dataframes containing the model simulation output

    tag: string
        tag specific to the best model configuration

    Returns:
    --------
    fins: pandas dataframe
        dataframes containing the model input data (drivers)

    sims: pandas dataframe
        dataframes containing the model simulation output

    """

    # reduce inputs outputs to one per day, at max PPFD
    idx = (fins.groupby(['Ring', 'year', 'doy'])['PPFD'].transform(max) ==
           fins.PPFD)
    fins = fins[idx]
    sims = sims[idx]

    # use simulation outputs of Ps as inputs
    for mod in np.unique([ee.split("(")[1].split(")")[0]
                          for ee in [e for e in sims.columns if '(' in e]]):

        fins['Ps(%s)' % (mod)] = sims['Ps(%s)' % (mod)]

    # conditions: make sure they match what is used in the model
    fins.set_index(['Ring', 'year', 'doy'], inplace=True)
    fins = (fins.groupby([fins.index.get_level_values(0),
                          fins.index.get_level_values(1),
                          fins.index.get_level_values(2) //
                          int(tag.split('JV')[1]) * int(tag.split('JV')[1])])
            .mean())
    sims.set_index(['Ring', 'year', 'doy'], inplace=True)
    sims = (sims.groupby([sims.index.get_level_values(0),
                          sims.index.get_level_values(1),
                          sims.index.get_level_values(2) //
                          int(tag.split('JV')[1]) * int(tag.split('JV')[1])])
            .mean())

    # shift the sims by one week backwars and rm last week of forcing
    for ring in fins.index.levels[0]:

        fins.drop((ring, ) + fins.loc[ring].head(1).index.values[0],
                  inplace=True)
        sims.drop((ring, ) + sims.loc[ring].tail(1).index.values[0],
                  inplace=True)

    fins.reset_index(inplace=True)
    sims.reset_index(inplace=True)

    # calculate the total Np
    for mod in np.unique([ee.split("(")[1].split(")")[0]
                          for ee in [e for e in sims.columns if '(' in e]]):

        sims['Np(%s)' % (mod)] = (sims['Nc(%s)' % (mod)] +
                                  sims['Ne(%s)' % (mod)] +
                                  sims['Nr(%s)' % (mod)] +
                                  sims['Ns(%s)' % (mod)])

    return fins, sims


def perform_dominance_analysis(fins, sims, periods, predictors, predicted,
                               project=None):

    """
    Estimates the relative dominant role of drivers of variability in
    modelled Np, Vcmax, and Jmax, for each of EucFACE's "ring" plots and
    for different time periods, defined by LAI thresholds.

    Arguments:
    ----------
    fins: pandas dataframe
        dataframes containing the model input data (drivers)

    sims: pandas dataframe
        dataframes containing the model simulation output

    periods: list
        time periods to consider in the analysis, defined based on
        whether LAI is high or low

    predictors: list
        environmental drivers to consider

    predicted: list
        simulated variables influenced by the environmental drivers

    project: string
        project repository where the files to analyse are (with path)

    Returns:
    --------
    '(project_)N_leaf_photo_drivers.csv' in the output directory

    """

    for ring in fins['Ring'].unique():

        fin = fins[fins['Ring'] == ring]
        sim = sims[sims['Ring'] == ring]

        for mod in np.unique([ee.split("(")[1].split(")")[0] for ee in
                              [e for e in sims.columns if '(' in e]]):

            for period in periods:

                if period == 'dry':  # arbitrary threshold
                    mask = fin['LAI'] < 0.7

                elif period == 'wet':
                    mask = fin['LAI'] > 0.7

                else:
                    mask = [True, ] * len(fin)

                for y in predicted:

                    # dominant analysis
                    data = (fin[predictors][mask]
                            .join(sim[['%s(%s)' % (y, mod)]][mask]))
                    corr_mtx = data.corr()

                    try:  # dominant analysis
                        dom_reg = Dominance(data=corr_mtx,
                                            target='%s(%s)' % (y, mod),
                                            data_format=1)  # problem
                        dom_reg.incremental_rsquare()
                        DA = dom_reg.dominance_stats()
                        DA['ring'] = ring
                        DA['model'] = mod
                        DA['period'] = period
                        DA['var'] = y
                        DA.reset_index(inplace=True)
                        DA.rename(columns={'index': 'predictor'}, inplace=True)

                        try:
                            df = df.append(DA, ignore_index=True)

                        except UnboundLocalError:
                            df = DA.copy()

                    except Exception:
                        pass

    # save file
    fdrivers = os.path.join(os.path.dirname(os.path.dirname(project)),
                            '%s_N_leaf_photo_drivers.csv' %
                            (os.path.basename(project)))
    df.to_csv(fdrivers, na_rep='', index=False, encoding='utf-8')

    return


# ======================================================================

if __name__ == "__main__":

    # define the argparse settings to read run set up file
    description = ''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-R', '--project', type=str,
                        help='folder containing the files to analyse')
    args = parser.parse_args()

    # user input
    periods = ['all', 'dry']  # periods to consider (LAI high or low?)
    predictors = ['PPFD', 'Tair', 'VPD', 'CO2', 'LAI', 'Ps']
    predicted = ['Vmax25', 'Jmax25', 'Np']

    main(dirpath(args.project), periods, predictors, predicted)
