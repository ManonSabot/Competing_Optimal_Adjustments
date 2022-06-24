#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that plots the temporal model simulations of root-zone soil
moisture, and contrasts them to observations at depth.

This file is part of the TractLSM project.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "plot the aCO2 vs. eCO2 simulated soil moisture"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (03.05.2022)"
__email__ = "m.e.b.sabot@gmail.com"

# ======================================================================

# import general modules
import argparse  # read in the user input
import os  # check for files, paths
import sys  # check for files, paths
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files

# plotting modules
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # tick locators
import string  # automate subplot lettering

# own modules
from plot_utils import default_plt_setup
from plot_utils import amb_ele
from plot_utils import correct_timeseriesticks
from plot_utils import render_ylabels

# make sure that modules can be loaded from other directories
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from analysis.analysis_utils import dirpath  # locate data
from analysis.analysis_utils import concat_rings  # concatenate dfs


# ======================================================================

def main(project):

    """
    Main function: plot of the time-varying root-zone soil moisture

    Arguments:
    ----------
    project: string
        project repository where the files containing the data to plot
        are (with path)

    Returns:
    --------
    '(project_)soil_moisture.png' in the figure directory

    """

    # paths
    dir = os.path.dirname(os.path.dirname(project))
    best = os.path.join(dir, '%s_ranked.csv' % (os.path.basename(project)))

    # read in the obs files
    sm1 = os.path.join(os.path.join(dir.replace('output', 'input'), 'obs'),
                       'EucFACE_sm_gap_filled.csv')
    sm1 = (pd.read_csv(sm1).dropna(axis=0, how='all').dropna(axis=1, how='all')
             .squeeze())
    sm2 = os.path.join(os.path.join(dir.replace('output', 'input'), 'obs'),
                       'EucFACE_sm_neutron.csv')
    sm2 = (pd.read_csv(sm2).dropna(axis=0, how='all').dropna(axis=1, how='all')
             .squeeze())

    # what is the best file
    best = (pd.read_csv(best).dropna(axis=0, how='all')
              .dropna(axis=1, how='all').squeeze())
    tag = best.iloc[0, 0]

    # now open drivers + best files
    fins = concat_rings(project.replace('output', 'input'),
                        endtag='model_drivers')
    sims = concat_rings(project, endtag=tag, keyword='DSH')

    # sort out the obs data
    sm1, sm2 = prepare_data(sm1, sm2, sims['Date'].unique())

    # figure name (inc. path)
    figure = os.path.join(os.path.dirname(project).replace('projects',
                                                           'figures'),
                          '%s_soil_moisture.png' % (os.path.basename(project)))

    # plot
    plot_soil_moisture(fins, sims, sm1, sm2, figure)

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def prepare_data(obs1, obs2, valid_dates):

    """
    Filters the data to only keep observations from inside EucFACE's
    "ring" plots, at dates when simulation outputs were generated.

    Arguments:
    ----------
    obs1: pandas dataframe
        dataframe containing a first set of soil moisture observations

    obs2: pandas dataframe
        dataframe containing alternative observations of soil moisture

    valid_dates: array
        dates to keep

    Returns:
    --------
    obs1: pandas dataframe
        dataframe containing a first set of soil moisture observations
        at depth

    obs2: pandas dataframe
        dataframe containing alternative observations of soil moisture
        at depth

    """

    # remove the data that's from outside the rings
    obs1 = obs1[obs1['Ring'].str.contains('Outside') == False]
    obs2 = obs2[obs2['Ring'].str.contains('Outside') == False]

    # average sw profiles
    obs1['sw'] = (obs1['swc.theta.5'] + obs1['swc.theta.30'] +
                  obs1['swc.theta.75']) / 300.
    obs1['sw'].where(obs1['sw'] > 0., inplace=True)
    obs2 = obs2.where(obs2['Depth'] <= 175).dropna()
    obs2['sw'] = obs2['VWC'] / 100.
    obs2['sw'].where(obs2['sw'] > 0., inplace=True)

    # dates to datetime
    obs1['Date'] = pd.to_datetime(obs1['Date'], dayfirst=True)
    obs2['Date'] = pd.to_datetime(obs2['Date'], dayfirst=True)
    obs1 = obs1[obs1['Date'].isin(valid_dates)]
    obs2 = obs2[obs2['Date'].isin(valid_dates)]

    return obs1, obs2


def sample_max_weekly(fin, sim, obs1, obs2):

    """
    Resamples the data to only keep the maximum soil moisture per week
    in each of EucFACE's "ring" plots, for each dataset.

    Arguments:
    ----------
    fin: pandas dataframe
        dataframe containing observation-based estimates of root-zone
        soil moisture

    sim: pandas dataframe
        dataframe containing modelled estimates of root-zone soil
        moisture

    obs1: pandas dataframe
        dataframe containing a first set of soil moisture observations
        at depth

    obs2: pandas dataframe
        dataframe containing alternative observations of soil moisture
        at depth

    Returns:
    --------
    fin: pandas dataframe
        dataframe containing observation-based estimates of root-zone
        soil moisture

    sim: pandas dataframe
        dataframe containing modelled estimates of root-zone soil
        moisture

    obs1: pandas dataframe
        dataframe containing a first set of soil moisture observations
        at depth

    obs2: pandas dataframe
        dataframe containing alternative observations of soil moisture
        at depth

    """

    # daily max per ring
    fin = fin.groupby(['Date', 'Ring'])['sw'].max()
    sim = sim.groupby(['Date', 'Ring'])['sw(pmax)'].max()
    obs1 = obs1.groupby(['Date', 'Ring'])['sw'].max()
    obs2 = obs2.groupby(['Date', 'Ring'])['sw'].max()

    # weekly max per ring
    fin = fin.groupby([pd.Grouper(level='Date', freq='W'),
                       pd.Grouper(level='Ring')]).max()
    sim = sim.groupby([pd.Grouper(level='Date', freq='W'),
                       pd.Grouper(level='Ring')]).max()
    obs1 = obs1.groupby([pd.Grouper(level='Date', freq='W'),
                         pd.Grouper(level='Ring')]).max()
    obs2 = obs2.groupby([pd.Grouper(level='Date', freq='W'),
                         pd.Grouper(level='Ring')]).max()

    return fin, sim, obs1, obs2


def plot_soil_moisture(fins, sims, obs1, obs2, fpath):

    """
    Plots the average and range of time-varying modelled root-zone
    volumetric soil moisture across the aCO2 vs. eCO2 rings, and
    constrasts it to observation-based estimates of root-zone soil
    moisture, as well as to observations of soil moisture down to 75 cm
    and 150 cm.

    Arguments:
    ----------
    fins: pandas dataframe
        dataframe containing observation-based estimates of root-zone
        soil moisture

    sims: pandas dataframe
        dataframe containing modelled estimates of root-zone soil
        moisture

    obs1: pandas dataframe
        dataframe containing a first set of soil moisture observations
        at depth

    obs2: pandas dataframe
        dataframe containing alternative observations of soil moisture
        at depth

    fpath: string
        name of the figure to produce, including path

    Returns:
    --------
    '(project_)soil_moisture.png' in the figure directory

    """

    # declare figure
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, sharey=True)
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.1)

    # separate ambient from elevated rings
    for i, rings in enumerate(amb_ele()):

        fin = fins[fins['Ring'].isin(rings)].copy()
        sim = sims[sims['Ring'].isin(rings)].copy()
        sw1 = obs1[obs1['Ring'].isin(rings)].copy()
        sw2 = obs2[obs2['Ring'].isin(rings)].copy()

        # only keep the weekly max data
        fin, sim, sw1, sw2 = sample_max_weekly(fin, sim, sw1, sw2)

        # dates to datetime int values, necessary for plotting
        fin.index = fin.index.set_levels(fin.index.levels[0].astype(np.int64),
                                         level=0)
        sim.index = sim.index.set_levels(sim.index.levels[0].astype(np.int64),
                                         level=0)
        sw1.index = sw1.index.set_levels(sw1.index.levels[0].astype(np.int64),
                                         level=0)
        sw2.index = sw2.index.set_levels(sw2.index.levels[0].astype(np.int64),
                                         level=0)

        # mean, min, max root-zone or sim
        fin = fin.groupby(fin.index.get_level_values(0)).agg(['mean', 'min',
                                                              'max'])
        sim = sim.groupby(sim.index.get_level_values(0)).agg(['mean', 'min',
                                                              'max'])

        # obs average at depth
        sw1 = sw1.groupby(sw1.index.get_level_values(0)).mean()
        sw2 = sw2.groupby(sw2.index.get_level_values(0)).mean()

        # plot the sim
        axes[i].fill_between(sim.index, sim['min'], sim['max'], alpha=0.4)
        axes[i].plot(sim.index, sim['mean'], lw=2., label='Sim.')

        # plot the estimated 'obs' root-zone sw
        axes[i].fill_between(fin.index, fin['min'], fin['max'], alpha=0.4,
                             zorder=-1)
        axes[i].plot(fin.index, fin['mean'], lw=2., label='Root-zone',
                     zorder=-1)

        # plot the ref obs
        axes[i].plot(sw1.index, sw1, label='75 cm')
        axes[i].plot(sw2.index, sw2, label='150 cm')

    # add legend
    axes[1].legend(handletextpad=0.4, ncol=4, bbox_to_anchor=(1., 2.1),
                   loc=1)

    for j, ax in enumerate(axes):  # format ticks, label subplots

        correct_timeseriesticks(ax, sims)
        ax.yaxis.set_major_locator(MaxNLocator(3))

    # label axes
    render_ylabels(axes[0], r'$\theta$', r'm$^{3}$ m$^{-3}$',
                   fs=plt.rcParams['axes.labelsize'])
    render_ylabels(axes[1], r'$\theta$', r'm$^{3}$ m$^{-3}$',
                   fs=plt.rcParams['axes.labelsize'])

    # label plots
    axes[0].text(0.01, 0.9, r'\textbf{(%s)} %s' % (string.ascii_lowercase[0],
                                                   r'aCO$_2$'),
                 transform=axes[0].transAxes)
    axes[1].text(0.01, 0.9, r'\textbf{(%s)} %s' % (string.ascii_lowercase[1],
                                                   r'eCO$_2$'),
                 transform=axes[1].transAxes)

    if not os.path.isdir(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))

    plt.savefig(fpath)
    plt.close()

    return


# ======================================================================

if __name__ == "__main__":

    # define the argparse settings to read run set up file
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-R', '--project', type=str,
                        help='folder containing the files to analyse')
    args = parser.parse_args()

    # default setup
    default_plt_setup(colours=['#67d2ec', '#024c5f', '#a777d4', '#ef9f08'])

    # specific setup
    plt.rcParams['axes.labelsize'] = plt.rcParams['axes.titlesize']
    plt.rcParams['legend.fontsize'] -= 2.
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.borderpad'] = 0.01
    plt.rcParams['legend.facecolor'] = 'w'
    plt.rcParams['legend.edgecolor'] = 'w'

    # user input
    main(dirpath(args.project))
