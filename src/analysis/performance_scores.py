#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code that computes the performance of the different model configurations
and that ranks them. The logic is inspired by PLUMBER.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
----------
* Best, M. J., Abramowitz, G., Johnson, H. R., Pitman, A. J., Balsamo,
  G., Boone, A., ... & Ek, M. (2015). The plumbing of land surface
  models: benchmarking model performance. Journal of Hydrometeorology,
  16(3), 1425-1442.
* Perkins, S. E., Pitman, A. J., Holbrook, N. J., & McAneney, J. (2007).
  Evaluation of the AR4 climate models’ simulated daily maximum
  temperature, minimum temperature, and precipitation over Australia
  using probability density functions. Journal of climate, 20(17),
  4356-4376.
* Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of
  forecast accuracy. International journal of forecasting, 22(4),
  679-688.

"""

__title__ = "performance of the model configurations"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (08.10.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# import general modules
import argparse  # read in the user input
import os  # check for files, paths
import sys  # check for files, paths
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files
from scipy import stats  # compute scores
from analysis_utils import concat_rings

# own modules
from analysis_utils import dirpath

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from TractLSM import conv  # unit converter
from TractLSM.Utils import read_csv  # read in data files


# ======================================================================

def main(project):

    """
    Main function: Generates '(project_)E_perf.csv' in the output
                   directory, which contains information on the
                   performance of different configurations of the model,
                   in the form of statistical metrics. The logic is
                   based on PLUMBER (Best et al., 2015). Also generates
                   '(project_)ranked.csv' which gives the best to worst
                   model configurations in order.

    Arguments:
    ----------
    project: string
        project repository where the files to analyse are (with path)

    Returns:
    --------
    '(project_)E_perf.csv' and '(project_)ranked.csv' in the output
    directory

    """

    # observational data against which to evaluate
    idir = os.path.join(os.path.dirname(os.path.dirname(project))
                        .replace('output', 'input'), 'obs')
    trans = (pd.read_csv(os.path.join(idir, 'EucFACE_sapflow_2012_2014.csv'))
             .dropna(axis=0, how='all').dropna(axis=1, how='all').squeeze())

    # combine all relevant input files
    fins = concat_rings(project.replace('output', 'input'),
                        endtag='model_drivers')
    fins = fins[fins[fins.count()[fins.count() > len(fins['Ring'].unique())]
                .index.to_list()].columns]

    # now go over the outputs
    files = [e for e in os.listdir(project)
             if ('DSH' in e and e.endswith('.csv'))]

    for file in files:

        sim, __ = read_csv(os.path.join(project, file))
        ring = file.split('_')[0].split('EucFACE')[1]
        performance_scores(fins[fins['Ring'] == ring].copy(), sim,
                           trans[trans['Ring'] == ring].copy(),
                           os.path.join(project, file))

    # rank the performance
    ranks(project)

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def match_dates(drivers, sim, obs):

    """
    Matches overlapping dates between different dataframes and keeps
    these dates only.

    Arguments:
    ----------
    drivers: pandas dataframe
        dataframe containing the model input data (drivers)

    sim: pandas dataframe
        dataframe containing the model simulation output

    obs: pandas dataframe
        dataframe containing observations to match

    Returns:
    --------
    drivers: pandas dataframe
        dataframe containing matched model input data (drivers)

    sim: pandas dataframe
        dataframe containing matched model simulation output

    obs: pandas dataframe
        dataframe containing matched observations

    """

    # match dates / time periods
    sim['Date'] = pd.to_datetime(sim['year'] * 1000. + sim['doy'],
                                 format='%Y%j')
    sim.set_index('Date', inplace=True)

    drivers.set_index('Date', inplace=True)
    drivers = drivers[:len(sim)]

    obs['Date'] = pd.to_datetime(obs['Date'], dayfirst=True)
    obs.set_index('Date', inplace=True)
    obs = obs.loc[obs.index.isin(sim.index.unique())]

    return drivers, sim, obs


def prepare_data(drivers, sim, obs):

    """
    Converts the transpiration data to a single unit in the obs and sim.
    The obs are also scaled by LAI, from the tree-level up to the
    stand-level.

    Arguments:
    ----------
    drivers: pandas dataframe
        dataframe containing the model input data (drivers)

    sim: pandas dataframe
        dataframe containing the model simulation output

    obs: pandas dataframe
        dataframe containing observations to match

    Returns:
    --------
    sim: pandas dataframe
        dataframe containing the model simulation output

    obs: pandas dataframe
        dataframe containing observations to match

    """

    # evaluate trans: we must scale the obs by LAI
    sc = drivers.loc[drivers.index.isin(obs.index.unique()), 'LAI']
    obs = (obs['volRing'].groupby(obs.index).mean() *
           sc.groupby(sc.index).mean())
    sim = ((sim[sim.filter(like='E(').columns.to_list()].groupby(sim.index)
            .sum()) * conv.mmolH2Opm2ps_2_mmphlfhr).loc[obs.index]
    obs = obs.loc[sim.index]

    return sim, obs


def similarity_skill(obs, sim):

    """
    Computes the PDF overlap score of observations and model simulations
    of these observations, as per Perkins et al. (2007).

    Arguments:
    ----------
    obs: pandas dataframe
        dataframe containing observations

    sim: pandas dataframe
        dataframe containing model outputs

    Returns:
    --------
    The PDF overlap score of obs and sim

    """

    bins = int(len(obs) / 10)  # 10 for the ref, can play with that

    if bins < 10:
        bins = 5

    bin_range = (min(np.amin(obs), np.amin(sim)),
                 max(np.amax(obs), np.amax(sim)))
    obs_pdf, _ = np.histogram(obs, bins, range=bin_range)
    sim_pdf, _ = np.histogram(sim, bins, range=bin_range)

    return np.sum(np.minimum(obs_pdf, sim_pdf)) / len(obs)


def relative_sample_sd(obs, sim):

    """
    Computes the ratio of modelled to observed sample standard
    deviation, which measures relative variability, as given by Best et
    al. (2015).

    Arguments:
    ----------
    obs: pandas dataframe
        dataframe containing observations

    sim: pandas dataframe
        dataframe containing model outputs

    Returns:
    --------
    The relative variability score of obs and sim

    """

    obs_ssd = (np.sum((obs - np.mean(obs)) ** 2.) / (len(obs) - 1)) ** 0.5
    sim_ssd = (np.sum((sim - np.mean(sim)) ** 2.) / (len(sim) - 1)) ** 0.5

    return sim_ssd / obs_ssd


def mean_abs_scaled_err(obs, sim):

    """
    Computes the mean absolute scaled error between model simulations
    and associated observations, which measures accuracy, as per Hyndman
    & Koehler (2006).

    Arguments:
    ----------
    obs: pandas dataframe
        dataframe containing observations

    sim: pandas dataframe
        dataframe containing model outputs

    Returns:
    --------
    The mean absolute scaled error between obs and sim

    """

    return np.mean(np.abs(obs - sim)) / np.mean(np.abs(np.diff(obs)))


def performance_scores(drivers, sim, obs, fname):

    """
    Computes statistical measures of performance for different
    model configurations (logic based on PLUMBER; Best et al., 2015).

    Arguments:
    ----------
    drivers: pandas dataframe
        dataframe containing the model input data (drivers)

    sim: pandas dataframe
        dataframe containing the model simulation output

    obs: pandas dataframe
        dataframe containing observations to match

    fname: string
        simulation file name and path

    Returns:
    --------
    '(project_)E_perf.csv' in the output directory

    """

    # is the performance score file already present?
    odir = os.path.dirname(os.path.dirname(os.path.dirname(fname)))
    fperf = os.path.join(odir, '%s_E_perf.csv' %
                         (os.path.basename(os.path.dirname(fname))))

    if os.path.isfile(fperf):
        fp = (pd.read_csv(fperf).dropna(axis=0, how='all')
                .dropna(axis=1, how='all'))
        fp.set_index('fname')

    else:
        fp = (pd.DataFrame([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan]).T)

        # add JV and Leg for easier filtering in other functions
        fp.columns = ['ring', 'tag', 'model', 'similarity', 'variability',
                      'accuracy', 'fname']
        fp.set_index('fname')

    # collect info that will be used to populate fp
    fname = os.path.basename(fname)
    ring = fname.split('_')[0].split('EucFACE')[1]
    tag = fname.split('_')[1].split('.csv')[0]

    # match the data to evaluate (by dates of availability)
    drivers, sim, obs = match_dates(drivers, sim, obs)

    # convert the data to comparable units
    simsim, obs = prepare_data(drivers, sim, obs)

    # now calculate the performance metrics
    obs = obs.to_numpy().flatten()

    # loop in case there are several models
    models = [e.split('(')[1].split(')')[0] for e in simsim.columns]

    for j in range(len(models)):

        sim = simsim.iloc[:, j].to_numpy().flatten()
        mask = np.logical_and(np.isfinite(obs), np.isfinite(sim))
        sss = similarity_skill(obs[mask], sim[mask])
        rssd = relative_sample_sd(obs[mask], sim[mask])
        mase = mean_abs_scaled_err(obs[mask], sim[mask])
        fp = fp.append(pd.Series({'ring': ring,
                                  'tag': tag,
                                  'model': models[j],
                                  'similarity': sss,
                                  'variability': rssd,
                                  'accuracy': mase,
                                  'fname': fname},
                       name=fname))

    # save file
    fp.dropna(how='all', inplace=True)
    fp.drop_duplicates(inplace=True)
    fp.to_csv(fperf, index=False, na_rep='', encoding='utf-8')

    return


def qranks(x):

    """
    Calculates quantile ranks for values in x.

    Arguments:
    ----------
    x: array or pandas series
        data to rank

    Returns:
    --------
    The quantile ranks of the data

    """

    return np.array([stats.percentileofscore(x.values, e, 'weak') / 100.
                     for e in x.values])


def ranks(project):

    """
    Computes the quantile ranks of different configurations of the
    model (logic based on PLUMBER; Best et al., 2015).

    Arguments:
    ----------
    project: string
        project repository where the files to analyse are (with path)

    Returns:
    --------
    '(project_)ranked.csv' in the output directory

    """

    fperf = os.path.join(os.path.dirname(os.path.dirname(project)),
                         '%s_E_perf.csv' % (os.path.basename(project)))

    df = (pd.read_csv(fperf).dropna(axis=0, how='all')
            .dropna(axis=1, how='all').squeeze())
    cols = df.columns.to_list()

    # rank by ring and variable
    df['similarity0'] = 1. - df['similarity']  # reverse order
    df['variability0'] = (1. - df['variability']).abs()  # distance to 1
    df['rank'] = (df.groupby(['ring', 'model'])['similarity0']
                    .transform(qranks) +
                  df.groupby(['ring', 'model'])['variability0']
                    .transform(qranks) +
                  df.groupby(['ring', 'model'])['accuracy']
                    .transform(qranks)) / 3.

    # add ranks to perf file
    (df[cols[:cols.index('fname')] + ['rank'] + cols[cols.index('fname'):]]
     .to_csv(fperf, index=False, na_rep='', encoding='utf-8'))

    # only keep the pmax mode, the trans and ambient rings for ranking
    df = df[df['model'] == 'pmax']
    df = df[df['ring'].isin(['R2', 'R3', 'R6'])]

    # more obvious tags
    tags = [e.replace('P-DSH-F-Rlow', 'default') for e in df['tag']]
    tags = [e.replace('default-', '') for e in tags]
    df['tag'] = np.asarray(tags)

    # compute average ranks
    df = df.groupby('tag').mean().sort_values(by='rank')['rank']
    df.index.name = None

    # save ranking file
    frank = os.path.join(os.path.dirname(os.path.dirname(project)),
                         '%s_ranked.csv' % (os.path.basename(project)))
    df.to_csv(frank, na_rep='', encoding='utf-8')

    return


# ======================================================================

if __name__ == "__main__":

    # define the argparse settings to read run set up file
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-R', '--project', type=str,
                        help='folder containing the files to analyse')
    args = parser.parse_args()

    # execute main
    main(dirpath(args.project))
